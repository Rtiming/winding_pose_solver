from __future__ import annotations

import atexit
import io
import math
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import redirect_stdout
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Sequence

from app_settings import build_app_runtime_settings
from src.core.collab_models import ProfileEvaluationRequest
from src.core.frame_math import load_centerline_dataset
from src.core.motion_settings import motion_settings_to_dict
from src.core.pose_solver import solve_tool_poses_from_dataset
from src.robodk_runtime.eval_worker import evaluate_request, open_offline_ik_station_context
from src.search.global_search import _extract_row_labels


@dataclass(frozen=True)
class SweepEnvironment:
    validation_centerline_csv: Path
    target_frame_rotation_xyz_deg: tuple[float, float, float]
    enable_custom_smoothing_and_pose_selection: bool
    robot_name: str
    frame_name: str
    program_name: str
    ik_backend: str
    append_start_as_terminal: bool = False
    local_parallel_workers: int = 1
    local_parallel_min_batch_size: int = 999999
    artifact_root: Path = Path("artifacts/tmp/origin_sweep")


@dataclass(frozen=True)
class SweepCase:
    x_mm: float
    y_mm: float
    z_mm: float


@dataclass(frozen=True)
class EvalConfig:
    strategy: str
    run_window_repair: bool
    run_inserted_repair: bool


@dataclass(frozen=True)
class AdaptiveSweepConfig:
    x_mm: float
    start_y_mm: float
    start_z_mm: float
    step_y_mm: float
    step_z_mm: float
    min_step_y_mm: float
    min_step_z_mm: float
    max_iters: int
    include_diagonal: bool
    min_y_mm: float | None = None
    max_y_mm: float | None = None
    min_z_mm: float | None = None
    max_z_mm: float | None = None
    stop_on_first_valid: bool = True


@dataclass(frozen=True)
class CandidateSweepConfig:
    x_mm: float
    seed_y_mm: float
    seed_z_mm: float
    radius_mm: float
    coarse_step_mm: float
    fine_radius_mm: float
    fine_step_mm: float
    confirm_top_n: int
    min_y_mm: float | None = None
    max_y_mm: float | None = None
    min_z_mm: float | None = None
    max_z_mm: float | None = None


@dataclass(frozen=True)
class SmartSquareSweepConfig:
    x_mm: float
    center_y_mm: float
    center_z_mm: float
    half_span_mm: float
    initial_step_mm: float
    min_step_mm: float
    max_iters: int
    beam_width: int
    diagonal_policy: str = "conditional"
    polish_step_mm: float = 0.0
    validation_grid_step_mm: float = 0.0
    min_y_mm: float | None = None
    max_y_mm: float | None = None
    min_z_mm: float | None = None
    max_z_mm: float | None = None


@dataclass(frozen=True)
class SweepResult:
    request_id: str
    x_mm: float
    y_mm: float
    z_mm: float
    status: str
    timing_seconds: float
    invalid_row_count: int
    ik_empty_row_count: int
    config_switches: int
    bridge_like_segments: int
    big_circle_step_count: int
    branch_flip_ratio: float
    worst_joint_step_deg: float
    mean_joint_step_deg: float
    total_cost: float
    pose_build_seconds: float = 0.0
    ik_collection_seconds: float = 0.0
    dp_path_selection_seconds: float = 0.0
    window_repair_seconds: float = 0.0
    inserted_repair_seconds: float = 0.0

    def rank_key(self) -> tuple[float, ...]:
        return (
            0.0 if self.status == "valid" else 1.0,
            float(self.invalid_row_count),
            float(self.ik_empty_row_count),
            float(self.bridge_like_segments),
            float(self.big_circle_step_count),
            float(self.branch_flip_ratio),
            float(self.worst_joint_step_deg),
            float(self.mean_joint_step_deg),
            float(self.config_switches),
            float(self.total_cost),
            float(self.timing_seconds),
        )


def parse_float_list(raw: str) -> tuple[float, ...]:
    values: list[float] = []
    for token in raw.split(","):
        normalized = token.strip()
        if not normalized:
            continue
        values.append(float(normalized))
    if not values:
        raise ValueError("Expected at least one numeric value.")
    return tuple(values)


def format_value(value: float) -> str:
    if abs(value - round(value)) <= 1e-9:
        return str(int(round(value)))
    text = f"{value:.3f}".rstrip("0").rstrip(".")
    return text.replace("-", "m").replace(".", "p")


def case_key(case: SweepCase) -> tuple[float, float]:
    return (round(float(case.y_mm), 6), round(float(case.z_mm), 6))


def case_key_result(result: SweepResult) -> tuple[float, float]:
    return (round(float(result.y_mm), 6), round(float(result.z_mm), 6))


def build_cases(
    *,
    x_mm: float,
    y_values: Sequence[float],
    z_values: Sequence[float],
) -> tuple[SweepCase, ...]:
    return tuple(
        SweepCase(x_mm=float(x_mm), y_mm=float(y_mm), z_mm=float(z_mm))
        for y_mm in y_values
        for z_mm in z_values
    )


def generate_radius_axis_values(
    *,
    center_mm: float,
    radius_mm: float,
    step_mm: float,
) -> tuple[float, ...]:
    if radius_mm < 0.0:
        raise ValueError("radius_mm must be non-negative.")
    if step_mm <= 0.0:
        raise ValueError("step_mm must be positive.")
    max_index = int(math.floor(float(radius_mm) / float(step_mm) + 1e-9))
    values = [float(center_mm) + float(index) * float(step_mm) for index in range(-max_index, max_index + 1)]
    if not any(abs(value - float(center_mm)) <= 1e-9 for value in values):
        values.append(float(center_mm))
    return tuple(sorted(values))


def _value_within_optional_bounds(
    value: float,
    *,
    lower_bound: float | None,
    upper_bound: float | None,
) -> bool:
    if lower_bound is not None and value < float(lower_bound) - 1e-9:
        return False
    if upper_bound is not None and value > float(upper_bound) + 1e-9:
        return False
    return True


def _case_within_optional_bounds(
    case: SweepCase,
    *,
    min_y_mm: float | None,
    max_y_mm: float | None,
    min_z_mm: float | None,
    max_z_mm: float | None,
) -> bool:
    return _value_within_optional_bounds(
        float(case.y_mm),
        lower_bound=min_y_mm,
        upper_bound=max_y_mm,
    ) and _value_within_optional_bounds(
        float(case.z_mm),
        lower_bound=min_z_mm,
        upper_bound=max_z_mm,
    )


def _case_within_optional_radius(
    case: SweepCase,
    *,
    center_y_mm: float | None,
    center_z_mm: float | None,
    radius_mm: float | None,
) -> bool:
    if radius_mm is None:
        return True
    if center_y_mm is None or center_z_mm is None:
        raise ValueError("center_y_mm and center_z_mm are required when radius_mm is set.")
    if radius_mm < 0.0:
        raise ValueError("radius_mm must be non-negative.")
    dy_mm = float(case.y_mm) - float(center_y_mm)
    dz_mm = float(case.z_mm) - float(center_z_mm)
    return dy_mm * dy_mm + dz_mm * dz_mm <= float(radius_mm) * float(radius_mm) + 1e-9


def _deduplicate_cases(cases: Sequence[SweepCase]) -> tuple[SweepCase, ...]:
    unique_cases: list[SweepCase] = []
    seen: set[tuple[float, float]] = set()
    for case in cases:
        key = case_key(case)
        if key in seen:
            continue
        seen.add(key)
        unique_cases.append(case)
    return tuple(unique_cases)


def _filter_cases(
    cases: Sequence[SweepCase],
    *,
    min_y_mm: float | None,
    max_y_mm: float | None,
    min_z_mm: float | None,
    max_z_mm: float | None,
    center_y_mm: float | None = None,
    center_z_mm: float | None = None,
    radius_mm: float | None = None,
) -> tuple[SweepCase, ...]:
    return _deduplicate_cases(
        tuple(
            case
            for case in cases
            if _case_within_optional_bounds(
                case,
                min_y_mm=min_y_mm,
                max_y_mm=max_y_mm,
                min_z_mm=min_z_mm,
                max_z_mm=max_z_mm,
            )
            and _case_within_optional_radius(
                case,
                center_y_mm=center_y_mm,
                center_z_mm=center_z_mm,
                radius_mm=radius_mm,
            )
        )
    )


def _smart_square_bounds(
    config: SmartSquareSweepConfig,
) -> tuple[float, float, float, float]:
    if config.half_span_mm < 0.0:
        raise ValueError("Smart-square half_span_mm must be non-negative.")
    min_y = float(config.center_y_mm) - float(config.half_span_mm)
    max_y = float(config.center_y_mm) + float(config.half_span_mm)
    min_z = float(config.center_z_mm) - float(config.half_span_mm)
    max_z = float(config.center_z_mm) + float(config.half_span_mm)
    if config.min_y_mm is not None:
        min_y = max(min_y, float(config.min_y_mm))
    if config.max_y_mm is not None:
        max_y = min(max_y, float(config.max_y_mm))
    if config.min_z_mm is not None:
        min_z = max(min_z, float(config.min_z_mm))
    if config.max_z_mm is not None:
        max_z = min(max_z, float(config.max_z_mm))
    if min_y > max_y or min_z > max_z:
        raise ValueError(
            "Smart-square search bounds are empty after applying optional limits: "
            f"y=[{min_y},{max_y}], z=[{min_z},{max_z}]."
        )
    return min_y, max_y, min_z, max_z


def _clamp(value: float, lower: float, upper: float) -> float:
    return min(max(float(value), float(lower)), float(upper))


def _smart_square_probe_cases(
    *,
    x_mm: float,
    center_y_mm: float,
    center_z_mm: float,
    min_y_mm: float,
    max_y_mm: float,
    min_z_mm: float,
    max_z_mm: float,
) -> tuple[SweepCase, ...]:
    y_mid = _clamp(center_y_mm, min_y_mm, max_y_mm)
    z_mid = _clamp(center_z_mm, min_z_mm, max_z_mm)
    y_values = (min_y_mm, y_mid, max_y_mm)
    z_values = (min_z_mm, z_mid, max_z_mm)
    return _deduplicate_cases(
        tuple(
            SweepCase(x_mm=float(x_mm), y_mm=float(y_mm), z_mm=float(z_mm))
            for y_mm in y_values
            for z_mm in z_values
        )
    )


def _smart_square_axis_cases(
    *,
    x_mm: float,
    center_y_mm: float,
    center_z_mm: float,
    step_mm: float,
    min_y_mm: float,
    max_y_mm: float,
    min_z_mm: float,
    max_z_mm: float,
) -> tuple[SweepCase, ...]:
    step = float(step_mm)
    if step <= 0.0:
        raise ValueError("Smart-square neighbor step must be positive.")
    offsets = ((0.0, 0.0), (step, 0.0), (-step, 0.0), (0.0, step), (0.0, -step))
    return _deduplicate_cases(
        tuple(
            SweepCase(
                x_mm=float(x_mm),
                y_mm=_clamp(float(center_y_mm) + dy_mm, min_y_mm, max_y_mm),
                z_mm=_clamp(float(center_z_mm) + dz_mm, min_z_mm, max_z_mm),
            )
            for dy_mm, dz_mm in offsets
        )
    )


def _smart_square_diagonal_cases(
    *,
    x_mm: float,
    center_y_mm: float,
    center_z_mm: float,
    step_mm: float,
    min_y_mm: float,
    max_y_mm: float,
    min_z_mm: float,
    max_z_mm: float,
) -> tuple[SweepCase, ...]:
    step = float(step_mm)
    if step <= 0.0:
        raise ValueError("Smart-square neighbor step must be positive.")
    offsets = (
        (step, step),
        (step, -step),
        (-step, step),
        (-step, -step),
    )
    return _deduplicate_cases(
        tuple(
            SweepCase(
                x_mm=float(x_mm),
                y_mm=_clamp(float(center_y_mm) + dy_mm, min_y_mm, max_y_mm),
                z_mm=_clamp(float(center_z_mm) + dz_mm, min_z_mm, max_z_mm),
            )
            for dy_mm, dz_mm in offsets
        )
    )


def _smart_square_axis_values(
    *,
    min_mm: float,
    max_mm: float,
    step_mm: float,
) -> tuple[float, ...]:
    if step_mm <= 0.0:
        raise ValueError("Smart-square validation grid step must be positive.")
    values: list[float] = []
    value = float(min_mm)
    while value <= float(max_mm) + 1e-9:
        values.append(min(value, float(max_mm)))
        value += float(step_mm)
    if not values or abs(values[-1] - float(max_mm)) > 1e-9:
        values.append(float(max_mm))
    return tuple(sorted(set(round(value, 6) for value in values)))


def _outside_square_ring_cases(
    *,
    x_mm: float,
    center_y_mm: float,
    center_z_mm: float,
    inner_half_span_mm: float,
    ring_margin_mm: float,
    edge_step_mm: float,
    min_y_mm: float | None = None,
    max_y_mm: float | None = None,
    min_z_mm: float | None = None,
    max_z_mm: float | None = None,
) -> tuple[SweepCase, ...]:
    if ring_margin_mm <= 0.0:
        raise ValueError("Outside-square ring margin must be positive.")
    if edge_step_mm <= 0.0:
        raise ValueError("Outside-square edge step must be positive.")

    outer_half_span = float(inner_half_span_mm) + float(ring_margin_mm)
    outer_min_y = float(center_y_mm) - outer_half_span
    outer_max_y = float(center_y_mm) + outer_half_span
    outer_min_z = float(center_z_mm) - outer_half_span
    outer_max_z = float(center_z_mm) + outer_half_span
    y_values = _smart_square_axis_values(
        min_mm=outer_min_y,
        max_mm=outer_max_y,
        step_mm=float(edge_step_mm),
    )
    z_values = _smart_square_axis_values(
        min_mm=outer_min_z,
        max_mm=outer_max_z,
        step_mm=float(edge_step_mm),
    )

    cases: list[SweepCase] = []
    for y_mm in y_values:
        cases.append(SweepCase(x_mm=float(x_mm), y_mm=float(y_mm), z_mm=outer_min_z))
        cases.append(SweepCase(x_mm=float(x_mm), y_mm=float(y_mm), z_mm=outer_max_z))
    for z_mm in z_values:
        cases.append(SweepCase(x_mm=float(x_mm), y_mm=outer_min_y, z_mm=float(z_mm)))
        cases.append(SweepCase(x_mm=float(x_mm), y_mm=outer_max_y, z_mm=float(z_mm)))

    return _deduplicate_cases(
        tuple(
            case
            for case in cases
            if outside_square_distance_yz_mm(
                case,
                center_y_mm=float(center_y_mm),
                center_z_mm=float(center_z_mm),
                half_span_mm=float(inner_half_span_mm),
            )
            > 1e-9
            and _case_within_optional_bounds(
                case,
                min_y_mm=min_y_mm,
                max_y_mm=max_y_mm,
                min_z_mm=min_z_mm,
                max_z_mm=max_z_mm,
            )
        )
    )


def distance_from_seed_yz_mm(
    result: SweepResult,
    *,
    seed_y_mm: float,
    seed_z_mm: float,
) -> float:
    return math.hypot(float(result.y_mm) - float(seed_y_mm), float(result.z_mm) - float(seed_z_mm))


def outside_square_distance_yz_mm(
    result_or_case: SweepResult | SweepCase,
    *,
    center_y_mm: float,
    center_z_mm: float,
    half_span_mm: float,
) -> float:
    dy_out = max(0.0, abs(float(result_or_case.y_mm) - float(center_y_mm)) - float(half_span_mm))
    dz_out = max(0.0, abs(float(result_or_case.z_mm) - float(center_z_mm)) - float(half_span_mm))
    return math.hypot(dy_out, dz_out)


def result_passes_official_gate(result: SweepResult, *, worst_step_limit_deg: float = 60.0) -> bool:
    return (
        str(result.status) == "valid"
        and int(result.invalid_row_count) == 0
        and int(result.ik_empty_row_count) == 0
        and int(result.bridge_like_segments) == 0
        and int(result.big_circle_step_count) == 0
        and float(result.worst_joint_step_deg) <= float(worst_step_limit_deg) + 1e-9
    )


def select_nearest_official_outside_results(
    results: Sequence[SweepResult],
    *,
    center_y_mm: float,
    center_z_mm: float,
    half_span_mm: float,
    top_k: int,
    min_separation_mm: float,
) -> tuple[SweepResult, ...]:
    ordered = sorted(
        (
            result
            for result in results
            if result_passes_official_gate(result)
            and outside_square_distance_yz_mm(
                result,
                center_y_mm=center_y_mm,
                center_z_mm=center_z_mm,
                half_span_mm=half_span_mm,
            )
            > 1e-9
        ),
        key=lambda result: (
            outside_square_distance_yz_mm(
                result,
                center_y_mm=center_y_mm,
                center_z_mm=center_z_mm,
                half_span_mm=half_span_mm,
            ),
            float(result.branch_flip_ratio),
            float(result.worst_joint_step_deg),
            float(result.mean_joint_step_deg),
            distance_from_seed_yz_mm(
                result,
                seed_y_mm=center_y_mm,
                seed_z_mm=center_z_mm,
            ),
        ),
    )
    selected: list[SweepResult] = []
    for result in ordered:
        if len(selected) >= max(0, int(top_k)):
            break
        if any(
            math.hypot(float(result.y_mm) - float(existing.y_mm), float(result.z_mm) - float(existing.z_mm))
            < float(min_separation_mm) - 1e-9
            for existing in selected
        ):
            continue
        selected.append(result)
    if len(selected) < max(0, int(top_k)):
        for result in ordered:
            if len(selected) >= max(0, int(top_k)):
                break
            if result in selected:
                continue
            selected.append(result)
    return tuple(selected)


def candidate_rank_key(
    result: SweepResult,
    *,
    seed_y_mm: float,
    seed_z_mm: float,
) -> tuple[float, ...]:
    return (
        0.0 if result.status == "valid" else 1.0,
        float(result.invalid_row_count),
        float(result.ik_empty_row_count),
        float(result.bridge_like_segments),
        float(result.big_circle_step_count),
        float(result.branch_flip_ratio),
        float(result.worst_joint_step_deg),
        float(result.mean_joint_step_deg),
        float(result.config_switches),
        float(result.total_cost),
        distance_from_seed_yz_mm(result, seed_y_mm=seed_y_mm, seed_z_mm=seed_z_mm),
        float(result.timing_seconds),
    )


def select_diverse_top_results(
    results: Sequence[SweepResult],
    *,
    seed_y_mm: float,
    seed_z_mm: float,
    top_k: int,
    min_separation_mm: float,
    pre_sorted: bool = False,
) -> tuple[SweepResult, ...]:
    limit = max(0, int(top_k))
    if limit == 0:
        return ()
    selected: list[SweepResult] = []
    if pre_sorted:
        ranked_results = results
    else:
        rank_result = _make_candidate_ranker(seed_y_mm=seed_y_mm, seed_z_mm=seed_z_mm)
        ranked_results = sorted(results, key=rank_result)
    for result in ranked_results:
        if len(selected) >= limit:
            break
        if any(
            math.hypot(float(result.y_mm) - float(existing.y_mm), float(result.z_mm) - float(existing.z_mm))
            < float(min_separation_mm) - 1e-9
            for existing in selected
        ):
            continue
        selected.append(result)
    return tuple(selected)


def generate_adaptive_neighbors(
    *,
    x_mm: float,
    center_y_mm: float,
    center_z_mm: float,
    step_y_mm: float,
    step_z_mm: float,
    include_diagonal: bool,
    min_y_mm: float | None = None,
    max_y_mm: float | None = None,
    min_z_mm: float | None = None,
    max_z_mm: float | None = None,
) -> tuple[SweepCase, ...]:
    offsets: list[tuple[float, float]] = [
        (0.0, 0.0),
        (step_y_mm, 0.0),
        (-step_y_mm, 0.0),
        (0.0, step_z_mm),
        (0.0, -step_z_mm),
    ]
    if include_diagonal:
        offsets.extend(
            (
                (step_y_mm, step_z_mm),
                (step_y_mm, -step_z_mm),
                (-step_y_mm, step_z_mm),
                (-step_y_mm, -step_z_mm),
            )
        )
    cases: list[SweepCase] = []
    for dy_mm, dz_mm in offsets:
        case = SweepCase(
            x_mm=float(x_mm),
            y_mm=float(center_y_mm + dy_mm),
            z_mm=float(center_z_mm + dz_mm),
        )
        if _case_within_optional_bounds(
            case,
            min_y_mm=min_y_mm,
            max_y_mm=max_y_mm,
            min_z_mm=min_z_mm,
            max_z_mm=max_z_mm,
        ):
            cases.append(case)
    return _deduplicate_cases(tuple(cases))


def is_better(lhs: SweepResult, rhs: SweepResult) -> bool:
    return lhs.rank_key() < rhs.rank_key()


def print_result_table(results: Sequence[SweepResult]) -> None:
    print(
        "request_id,status,y_mm,z_mm,worst_joint_step_deg,mean_joint_step_deg,"
        "config_switches,bridge_like_segments,big_circle_step_count,branch_flip_ratio,ik_empty_rows,time_s,pose_s,ik_s,dp_s,"
        "window_repair_s,inserted_repair_s,total_cost"
    )
    for result in results:
        print(
            f"{result.request_id},{result.status},{result.y_mm:.3f},{result.z_mm:.3f},"
            f"{result.worst_joint_step_deg:.6f},{result.mean_joint_step_deg:.6f},"
            f"{result.config_switches},{result.bridge_like_segments},"
            f"{result.big_circle_step_count},{result.branch_flip_ratio:.6f},"
            f"{result.ik_empty_row_count},{result.timing_seconds:.3f},"
            f"{result.pose_build_seconds:.3f},{result.ik_collection_seconds:.3f},"
            f"{result.dp_path_selection_seconds:.3f},{result.window_repair_seconds:.3f},"
            f"{result.inserted_repair_seconds:.3f},{result.total_cost:.6f}"
        )


def _make_candidate_ranker(
    *,
    seed_y_mm: float,
    seed_z_mm: float,
):
    cached_keys: dict[SweepResult, tuple[float, ...]] = {}

    def rank_result(result: SweepResult) -> tuple[float, ...]:
        cached_key = cached_keys.get(result)
        if cached_key is None:
            cached_key = candidate_rank_key(
                result,
                seed_y_mm=seed_y_mm,
                seed_z_mm=seed_z_mm,
            )
            cached_keys[result] = cached_key
        return cached_key

    return rank_result


def evaluate_cases_parallel(
    *,
    cases: Sequence[SweepCase],
    environment: SweepEnvironment,
    eval_config: EvalConfig,
    workers: int,
    prefix: str,
) -> tuple[SweepResult, ...]:
    case_list = tuple(cases)
    if not case_list:
        return ()

    worker_count = max(1, int(workers))
    if worker_count == 1 or len(case_list) == 1:
        completed: list[SweepResult] = []
        for finished_index, case in enumerate(case_list, start=1):
            result = _evaluate_case(case=case, environment=environment, eval_config=eval_config)
            completed.append(result)
            _print_progress(prefix, finished_index, len(case_list), result)
        return tuple(completed)

    batch_count = min(len(case_list), worker_count * 4)
    case_batches = _split_case_batches(case_list, batch_count=batch_count)
    completed: list[SweepResult] = []
    finished_index = 0
    executor = _get_shared_process_pool(worker_count)
    futures = [
        executor.submit(_evaluate_case_batch_entry, case_batch, environment, eval_config)
        for case_batch in case_batches
    ]
    for future in as_completed(futures):
        for result in future.result():
            finished_index += 1
            completed.append(result)
            _print_progress(prefix, finished_index, len(case_list), result)
    return tuple(completed)


_SHARED_PROCESS_POOLS: dict[int, ProcessPoolExecutor] = {}
_SHARED_PROCESS_POOL_SHUTDOWN_REGISTERED = False


def _get_shared_process_pool(worker_count: int) -> ProcessPoolExecutor:
    global _SHARED_PROCESS_POOL_SHUTDOWN_REGISTERED
    normalized_count = max(1, int(worker_count))
    executor = _SHARED_PROCESS_POOLS.get(normalized_count)
    if executor is None:
        executor = ProcessPoolExecutor(max_workers=normalized_count)
        _SHARED_PROCESS_POOLS[normalized_count] = executor
    if not _SHARED_PROCESS_POOL_SHUTDOWN_REGISTERED:
        atexit.register(_shutdown_shared_process_pools)
        _SHARED_PROCESS_POOL_SHUTDOWN_REGISTERED = True
    return executor


def _shutdown_shared_process_pools() -> None:
    for executor in tuple(_SHARED_PROCESS_POOLS.values()):
        executor.shutdown(wait=True, cancel_futures=False)
    _SHARED_PROCESS_POOLS.clear()


def run_grid_sweep(
    *,
    x_mm: float,
    y_values: Sequence[float],
    z_values: Sequence[float],
    environment: SweepEnvironment,
    eval_config: EvalConfig,
    workers: int,
    prefix: str = "[sweep]",
    min_y_mm: float | None = None,
    max_y_mm: float | None = None,
    min_z_mm: float | None = None,
    max_z_mm: float | None = None,
    center_y_mm: float | None = None,
    center_z_mm: float | None = None,
    radius_mm: float | None = None,
) -> tuple[dict[tuple[float, float], SweepResult], tuple[float, ...], tuple[float, ...]]:
    y_series = tuple(float(value) for value in y_values)
    z_series = tuple(float(value) for value in z_values)
    cases = tuple(
        case
        for case in build_cases(x_mm=float(x_mm), y_values=y_series, z_values=z_series)
        if _case_within_optional_bounds(
            case,
            min_y_mm=min_y_mm,
            max_y_mm=max_y_mm,
            min_z_mm=min_z_mm,
            max_z_mm=max_z_mm,
        )
        and _case_within_optional_radius(
            case,
            center_y_mm=center_y_mm,
            center_z_mm=center_z_mm,
            radius_mm=radius_mm,
        )
    )
    print(
        f"{prefix} mode=grid cases={len(cases)}, workers={workers}, "
        f"strategy={eval_config.strategy}, "
        f"window_repair={eval_config.run_window_repair}, "
        f"inserted_repair={eval_config.run_inserted_repair}, "
        f"x={float(x_mm):.3f}, y={y_series}, z={z_series}, "
        f"bounds=(y:[{min_y_mm},{max_y_mm}], z:[{min_z_mm},{max_z_mm}]), "
        f"radius=(center:[{center_y_mm},{center_z_mm}], r:{radius_mm})"
    )
    completed = evaluate_cases_parallel(
        cases=cases,
        environment=environment,
        eval_config=eval_config,
        workers=workers,
        prefix=prefix,
    )
    result_map = {case_key_result(result): result for result in completed}
    return result_map, y_series, z_series


def run_candidate_sweep(
    *,
    config: CandidateSweepConfig,
    environment: SweepEnvironment,
    eval_config: EvalConfig,
    workers: int,
    prefix: str = "[candidates]",
) -> tuple[dict[tuple[float, float], SweepResult], dict[str, object]]:
    if config.radius_mm < 0.0:
        raise ValueError("Candidate search radius_mm must be non-negative.")
    if config.coarse_step_mm <= 0.0:
        raise ValueError("Candidate search coarse_step_mm must be positive.")
    if config.fine_radius_mm < 0.0:
        raise ValueError("Candidate search fine_radius_mm must be non-negative.")
    if config.fine_radius_mm > 0.0 and config.fine_step_mm <= 0.0:
        raise ValueError("Candidate search fine_step_mm must be positive when fine_radius_mm is enabled.")

    seed_case = SweepCase(
        x_mm=float(config.x_mm),
        y_mm=float(config.seed_y_mm),
        z_mm=float(config.seed_z_mm),
    )
    seed_within_bounds = _case_within_optional_bounds(
        seed_case,
        min_y_mm=config.min_y_mm,
        max_y_mm=config.max_y_mm,
        min_z_mm=config.min_z_mm,
        max_z_mm=config.max_z_mm,
    )
    if not seed_within_bounds:
        print(
            f"{prefix} warning: seed origin is outside the configured search bounds; "
            f"baseline will not be included unless evaluated elsewhere. "
            f"seed=({config.x_mm:.3f}, {config.seed_y_mm:.3f}, {config.seed_z_mm:.3f}), "
            f"bounds=(y:[{config.min_y_mm},{config.max_y_mm}], "
            f"z:[{config.min_z_mm},{config.max_z_mm}])."
        )

    coarse_y_values = generate_radius_axis_values(
        center_mm=float(config.seed_y_mm),
        radius_mm=float(config.radius_mm),
        step_mm=float(config.coarse_step_mm),
    )
    coarse_z_values = generate_radius_axis_values(
        center_mm=float(config.seed_z_mm),
        radius_mm=float(config.radius_mm),
        step_mm=float(config.coarse_step_mm),
    )
    coarse_cases = _filter_cases(
        build_cases(
            x_mm=float(config.x_mm),
            y_values=coarse_y_values,
            z_values=coarse_z_values,
        ),
        min_y_mm=config.min_y_mm,
        max_y_mm=config.max_y_mm,
        min_z_mm=config.min_z_mm,
        max_z_mm=config.max_z_mm,
        center_y_mm=float(config.seed_y_mm),
        center_z_mm=float(config.seed_z_mm),
        radius_mm=float(config.radius_mm),
    )
    print(
        f"{prefix} mode=candidates stage=coarse cases={len(coarse_cases)}, workers={workers}, "
        f"strategy={eval_config.strategy}, "
        f"window_repair={eval_config.run_window_repair}, "
        f"inserted_repair={eval_config.run_inserted_repair}, "
        f"seed=({config.x_mm:.3f},{config.seed_y_mm:.3f},{config.seed_z_mm:.3f}), "
        f"radius={config.radius_mm:.3f}, coarse_step={config.coarse_step_mm:.3f}, "
        f"bounds=(y:[{config.min_y_mm},{config.max_y_mm}], z:[{config.min_z_mm},{config.max_z_mm}])"
    )

    all_results_by_key: dict[tuple[float, float], SweepResult] = {}
    rank_result = _make_candidate_ranker(
        seed_y_mm=float(config.seed_y_mm),
        seed_z_mm=float(config.seed_z_mm),
    )
    for result in evaluate_cases_parallel(
        cases=coarse_cases,
        environment=environment,
        eval_config=eval_config,
        workers=workers,
        prefix=f"{prefix} coarse",
    ):
        all_results_by_key[case_key_result(result)] = result

    coarse_ordered = sorted(all_results_by_key.values(), key=rank_result)
    refine_center_count = min(max(0, int(config.confirm_top_n)), len(coarse_ordered))
    refine_centers = tuple(coarse_ordered[:refine_center_count])

    fine_cases: list[SweepCase] = []
    if config.fine_radius_mm > 0.0 and refine_centers:
        for center_result in refine_centers:
            fine_y_values = generate_radius_axis_values(
                center_mm=float(center_result.y_mm),
                radius_mm=float(config.fine_radius_mm),
                step_mm=float(config.fine_step_mm),
            )
            fine_z_values = generate_radius_axis_values(
                center_mm=float(center_result.z_mm),
                radius_mm=float(config.fine_radius_mm),
                step_mm=float(config.fine_step_mm),
            )
            local_cases = _filter_cases(
                build_cases(
                    x_mm=float(config.x_mm),
                    y_values=fine_y_values,
                    z_values=fine_z_values,
                ),
                min_y_mm=config.min_y_mm,
                max_y_mm=config.max_y_mm,
                min_z_mm=config.min_z_mm,
                max_z_mm=config.max_z_mm,
                center_y_mm=float(center_result.y_mm),
                center_z_mm=float(center_result.z_mm),
                radius_mm=float(config.fine_radius_mm),
            )
            global_cases = _filter_cases(
                local_cases,
                min_y_mm=config.min_y_mm,
                max_y_mm=config.max_y_mm,
                min_z_mm=config.min_z_mm,
                max_z_mm=config.max_z_mm,
                center_y_mm=float(config.seed_y_mm),
                center_z_mm=float(config.seed_z_mm),
                radius_mm=float(config.radius_mm),
            )
            fine_cases.extend(
                case for case in global_cases if case_key(case) not in all_results_by_key
            )
    pending_fine_cases = _deduplicate_cases(fine_cases)
    print(
        f"{prefix} mode=candidates stage=fine centers={len(refine_centers)}, "
        f"pending_cases={len(pending_fine_cases)}, fine_radius={config.fine_radius_mm:.3f}, "
        f"fine_step={config.fine_step_mm:.3f}"
    )
    if pending_fine_cases:
        for result in evaluate_cases_parallel(
            cases=pending_fine_cases,
            environment=environment,
            eval_config=eval_config,
            workers=workers,
            prefix=f"{prefix} fine",
        ):
            all_results_by_key[case_key_result(result)] = result

    seed_key = case_key(seed_case)
    trace: dict[str, object] = {
        "seed_key": {"y_mm": float(config.seed_y_mm), "z_mm": float(config.seed_z_mm)},
        "coarse_case_count": len(coarse_cases),
        "fine_center_count": len(refine_centers),
        "fine_pending_case_count": len(pending_fine_cases),
        "evaluated_case_count": len(all_results_by_key),
        "seed_within_search_bounds": seed_within_bounds,
        "baseline_found": seed_key in all_results_by_key,
        "refine_centers": [
            {
                "y_mm": float(result.y_mm),
                "z_mm": float(result.z_mm),
                "status": result.status,
                "ik_empty_row_count": int(result.ik_empty_row_count),
                "config_switches": int(result.config_switches),
                "bridge_like_segments": int(result.bridge_like_segments),
                "big_circle_step_count": int(result.big_circle_step_count),
                "branch_flip_ratio": float(result.branch_flip_ratio),
                "worst_joint_step_deg": float(result.worst_joint_step_deg),
            }
            for result in refine_centers
        ],
    }
    return all_results_by_key, trace


def run_smart_square_sweep(
    *,
    config: SmartSquareSweepConfig,
    environment: SweepEnvironment,
    eval_config: EvalConfig,
    workers: int,
    prefix: str = "[smart-square]",
) -> tuple[dict[tuple[float, float], SweepResult], dict[str, object]]:
    if config.initial_step_mm <= 0.0:
        raise ValueError("Smart-square initial_step_mm must be positive.")
    if config.min_step_mm <= 0.0:
        raise ValueError("Smart-square min_step_mm must be positive.")
    if config.beam_width <= 0:
        raise ValueError("Smart-square beam_width must be positive.")

    min_y_mm, max_y_mm, min_z_mm, max_z_mm = _smart_square_bounds(config)
    all_results_by_key: dict[tuple[float, float], SweepResult] = {}
    trace: dict[str, object] = {
        "algorithm": "beam_pattern_search",
        "bounds": {
            "min_y_mm": min_y_mm,
            "max_y_mm": max_y_mm,
            "min_z_mm": min_z_mm,
            "max_z_mm": max_z_mm,
        },
        "iterations": [],
        "polish": None,
        "validation_grid": None,
    }

    rank_result = _make_candidate_ranker(
        seed_y_mm=float(config.center_y_mm),
        seed_z_mm=float(config.center_z_mm),
    )
    probe_cases = _smart_square_probe_cases(
        x_mm=float(config.x_mm),
        center_y_mm=float(config.center_y_mm),
        center_z_mm=float(config.center_z_mm),
        min_y_mm=min_y_mm,
        max_y_mm=max_y_mm,
        min_z_mm=min_z_mm,
        max_z_mm=max_z_mm,
    )
    print(
        f"{prefix} mode=smart-square stage=probe cases={len(probe_cases)}, workers={workers}, "
        f"strategy={eval_config.strategy}, "
        f"window_repair={eval_config.run_window_repair}, "
        f"inserted_repair={eval_config.run_inserted_repair}, "
        f"center=({config.x_mm:.3f},{config.center_y_mm:.3f},{config.center_z_mm:.3f}), "
        f"square=({min_y_mm:.3f}:{max_y_mm:.3f}, {min_z_mm:.3f}:{max_z_mm:.3f}), "
        f"initial_step={config.initial_step_mm:.3f}, min_step={config.min_step_mm:.3f}, "
        f"beam_width={config.beam_width}, max_iters={config.max_iters}"
    )
    for result in evaluate_cases_parallel(
        cases=probe_cases,
        environment=environment,
        eval_config=eval_config,
        workers=workers,
        prefix=f"{prefix} probe",
    ):
        all_results_by_key[case_key_result(result)] = result

    step_mm = float(config.initial_step_mm)
    max_iters = max(1, int(config.max_iters))
    beam_width = max(1, int(config.beam_width))
    diagonal_policy = str(config.diagonal_policy).strip().lower()
    if diagonal_policy not in {"always", "conditional", "never"}:
        raise ValueError(
            "Smart-square diagonal_policy must be one of: always, conditional, never."
        )
    for iteration_index in range(max_iters):
        ordered = sorted(all_results_by_key.values(), key=rank_result)
        if not ordered:
            break
        centers = select_diverse_top_results(
            ordered,
            seed_y_mm=float(config.center_y_mm),
            seed_z_mm=float(config.center_z_mm),
            top_k=beam_width,
            min_separation_mm=max(float(config.min_step_mm), step_mm * 0.5),
            pre_sorted=True,
        )
        if not centers:
            centers = tuple(ordered[:beam_width])

        axis_cases: list[SweepCase] = []
        for center_result in centers:
            axis_cases.extend(
                _smart_square_axis_cases(
                    x_mm=float(config.x_mm),
                    center_y_mm=float(center_result.y_mm),
                    center_z_mm=float(center_result.z_mm),
                    step_mm=step_mm,
                    min_y_mm=min_y_mm,
                    max_y_mm=max_y_mm,
                    min_z_mm=min_z_mm,
                    max_z_mm=max_z_mm,
                )
            )
        pending_axis_cases = tuple(
            case
            for case in _deduplicate_cases(axis_cases)
            if case_key(case) not in all_results_by_key
        )
        if pending_axis_cases:
            for result in evaluate_cases_parallel(
                cases=pending_axis_cases,
                environment=environment,
                eval_config=eval_config,
                workers=workers,
                prefix=f"{prefix} iter {iteration_index + 1} axis",
            ):
                all_results_by_key[case_key_result(result)] = result

        best_after_axis = min(all_results_by_key.values(), key=rank_result)
        diagonal_needed = diagonal_policy == "always"
        if diagonal_policy == "conditional":
            diagonal_needed = not (
                best_after_axis.status == "valid"
                and best_after_axis.bridge_like_segments == 0
                and best_after_axis.big_circle_step_count == 0
                and best_after_axis.worst_joint_step_deg <= 60.0 + 1e-9
            )
        pending_diagonal_cases: tuple[SweepCase, ...] = ()
        if diagonal_needed:
            diagonal_cases: list[SweepCase] = []
            for center_result in centers:
                diagonal_cases.extend(
                    _smart_square_diagonal_cases(
                        x_mm=float(config.x_mm),
                        center_y_mm=float(center_result.y_mm),
                        center_z_mm=float(center_result.z_mm),
                        step_mm=step_mm,
                        min_y_mm=min_y_mm,
                        max_y_mm=max_y_mm,
                        min_z_mm=min_z_mm,
                        max_z_mm=max_z_mm,
                    )
                )
            pending_diagonal_cases = tuple(
                case
                for case in _deduplicate_cases(diagonal_cases)
                if case_key(case) not in all_results_by_key
            )
        print(
            f"{prefix} iter={iteration_index + 1}, step={step_mm:.3f}, "
            f"centers={len(centers)}, axis_pending={len(pending_axis_cases)}, "
            f"diagonal_pending={len(pending_diagonal_cases)}, policy={diagonal_policy}"
        )
        if pending_diagonal_cases:
            for result in evaluate_cases_parallel(
                cases=pending_diagonal_cases,
                environment=environment,
                eval_config=eval_config,
                workers=workers,
                prefix=f"{prefix} iter {iteration_index + 1} diagonal",
            ):
                all_results_by_key[case_key_result(result)] = result

        best_after = min(all_results_by_key.values(), key=rank_result)
        iteration_payload = {
            "iteration": iteration_index + 1,
            "step_mm": step_mm,
            "center_count": len(centers),
            "axis_pending_case_count": len(pending_axis_cases),
            "diagonal_pending_case_count": len(pending_diagonal_cases),
            "pending_case_count": len(pending_axis_cases) + len(pending_diagonal_cases),
            "evaluated_case_count": len(all_results_by_key),
            "best_y_mm": float(best_after.y_mm),
            "best_z_mm": float(best_after.z_mm),
            "best_status": str(best_after.status),
            "best_ik_empty_row_count": int(best_after.ik_empty_row_count),
            "best_config_switches": int(best_after.config_switches),
            "best_bridge_like_segments": int(best_after.bridge_like_segments),
            "best_big_circle_step_count": int(best_after.big_circle_step_count),
            "best_branch_flip_ratio": float(best_after.branch_flip_ratio),
            "best_worst_joint_step_deg": float(best_after.worst_joint_step_deg),
            "centers": [
                {
                    "y_mm": float(result.y_mm),
                    "z_mm": float(result.z_mm),
                    "status": str(result.status),
                    "big_circle_step_count": int(result.big_circle_step_count),
                    "worst_joint_step_deg": float(result.worst_joint_step_deg),
                }
                for result in centers
            ],
            "diagonal_policy": diagonal_policy,
        }
        trace["iterations"].append(iteration_payload)  # type: ignore[index]

        if step_mm <= float(config.min_step_mm) + 1e-9:
            break
        next_step_mm = max(float(config.min_step_mm), step_mm * 0.5)
        if abs(next_step_mm - step_mm) <= 1e-9:
            break
        step_mm = next_step_mm

    polish_step_mm = max(0.0, float(config.polish_step_mm))
    if polish_step_mm > 0.0 and all_results_by_key:
        best_before_polish = min(all_results_by_key.values(), key=rank_result)
        polish_cases = _deduplicate_cases(
            (
                *_smart_square_axis_cases(
                    x_mm=float(config.x_mm),
                    center_y_mm=float(best_before_polish.y_mm),
                    center_z_mm=float(best_before_polish.z_mm),
                    step_mm=polish_step_mm,
                    min_y_mm=min_y_mm,
                    max_y_mm=max_y_mm,
                    min_z_mm=min_z_mm,
                    max_z_mm=max_z_mm,
                ),
                *_smart_square_diagonal_cases(
                    x_mm=float(config.x_mm),
                    center_y_mm=float(best_before_polish.y_mm),
                    center_z_mm=float(best_before_polish.z_mm),
                    step_mm=polish_step_mm,
                    min_y_mm=min_y_mm,
                    max_y_mm=max_y_mm,
                    min_z_mm=min_z_mm,
                    max_z_mm=max_z_mm,
                ),
            )
        )
        pending_polish_cases = tuple(
            case
            for case in polish_cases
            if case_key(case) not in all_results_by_key
        )
        print(
            f"{prefix} stage=polish step={polish_step_mm:.3f}, "
            f"center=({best_before_polish.y_mm:.3f},{best_before_polish.z_mm:.3f}), "
            f"pending_cases={len(pending_polish_cases)}"
        )
        for result in evaluate_cases_parallel(
            cases=pending_polish_cases,
            environment=environment,
            eval_config=eval_config,
            workers=workers,
            prefix=f"{prefix} polish",
        ):
            all_results_by_key[case_key_result(result)] = result
        best_after_polish = min(all_results_by_key.values(), key=rank_result)
        trace["polish"] = {
            "step_mm": polish_step_mm,
            "pending_case_count": len(pending_polish_cases),
            "best_y_mm_before": float(best_before_polish.y_mm),
            "best_z_mm_before": float(best_before_polish.z_mm),
            "best_y_mm_after": float(best_after_polish.y_mm),
            "best_z_mm_after": float(best_after_polish.z_mm),
            "best_worst_joint_step_deg_after": float(best_after_polish.worst_joint_step_deg),
            "best_branch_flip_ratio_after": float(best_after_polish.branch_flip_ratio),
        }

    if config.validation_grid_step_mm > 0.0:
        validation_y_values = _smart_square_axis_values(
            min_mm=min_y_mm,
            max_mm=max_y_mm,
            step_mm=float(config.validation_grid_step_mm),
        )
        validation_z_values = _smart_square_axis_values(
            min_mm=min_z_mm,
            max_mm=max_z_mm,
            step_mm=float(config.validation_grid_step_mm),
        )
        validation_cases = tuple(
            case
            for case in build_cases(
                x_mm=float(config.x_mm),
                y_values=validation_y_values,
                z_values=validation_z_values,
            )
            if case_key(case) not in all_results_by_key
        )
        print(
            f"{prefix} stage=validation_grid step={config.validation_grid_step_mm:.3f}, "
            f"pending_cases={len(validation_cases)}"
        )
        for result in evaluate_cases_parallel(
            cases=validation_cases,
            environment=environment,
            eval_config=eval_config,
            workers=workers,
            prefix=f"{prefix} validation",
        ):
            all_results_by_key[case_key_result(result)] = result
        trace["validation_grid"] = {
            "step_mm": float(config.validation_grid_step_mm),
            "y_count": len(validation_y_values),
            "z_count": len(validation_z_values),
            "pending_case_count": len(validation_cases),
        }

    trace["evaluated_case_count"] = len(all_results_by_key)
    return all_results_by_key, trace


def run_outside_square_fallback(
    *,
    existing_results_by_key: dict[tuple[float, float], SweepResult],
    config: SmartSquareSweepConfig,
    environment: SweepEnvironment,
    eval_config: EvalConfig,
    workers: int,
    target_count: int,
    max_rings: int,
    ring_step_mm: float,
    edge_step_mm: float,
    min_separation_mm: float,
    prefix: str = "[outside-square]",
) -> tuple[dict[tuple[float, float], SweepResult], dict[str, object]]:
    """Evaluate expanding perimeter rings outside the configured square.

    This is a fallback path for the case where the configured square has no
    official deliverable.  It samples only the square perimeters just outside
    the requested region, so the first passing candidates are distance-near
    alternatives rather than arbitrary far-away optima.
    """

    if target_count <= 0:
        return dict(existing_results_by_key), {"triggered": False, "reason": "target_count<=0"}
    if max_rings <= 0:
        return dict(existing_results_by_key), {"triggered": False, "reason": "max_rings<=0"}
    if ring_step_mm <= 0.0:
        raise ValueError("Outside-square ring_step_mm must be positive.")
    if edge_step_mm <= 0.0:
        raise ValueError("Outside-square edge_step_mm must be positive.")

    all_results_by_key = dict(existing_results_by_key)
    trace: dict[str, object] = {
        "triggered": True,
        "target_count": int(target_count),
        "max_rings": int(max_rings),
        "ring_step_mm": float(ring_step_mm),
        "edge_step_mm": float(edge_step_mm),
        "rings": [],
        "selected": [],
    }

    for ring_index in range(1, max(1, int(max_rings)) + 1):
        ring_margin_mm = float(ring_step_mm) * ring_index
        ring_cases = _outside_square_ring_cases(
            x_mm=float(config.x_mm),
            center_y_mm=float(config.center_y_mm),
            center_z_mm=float(config.center_z_mm),
            inner_half_span_mm=float(config.half_span_mm),
            ring_margin_mm=ring_margin_mm,
            edge_step_mm=float(edge_step_mm),
            min_y_mm=config.min_y_mm,
            max_y_mm=config.max_y_mm,
            min_z_mm=config.min_z_mm,
            max_z_mm=config.max_z_mm,
        )
        pending_cases = tuple(
            case
            for case in ring_cases
            if case_key(case) not in all_results_by_key
        )
        print(
            f"{prefix} ring={ring_index}, margin={ring_margin_mm:.3f}, "
            f"edge_step={float(edge_step_mm):.3f}, pending_cases={len(pending_cases)}"
        )
        for result in evaluate_cases_parallel(
            cases=pending_cases,
            environment=environment,
            eval_config=eval_config,
            workers=workers,
            prefix=f"{prefix} ring {ring_index}",
        ):
            all_results_by_key[case_key_result(result)] = result

        nearest_official = select_nearest_official_outside_results(
            tuple(all_results_by_key.values()),
            center_y_mm=float(config.center_y_mm),
            center_z_mm=float(config.center_z_mm),
            half_span_mm=float(config.half_span_mm),
            top_k=int(target_count),
            min_separation_mm=float(min_separation_mm),
        )
        ring_payload = {
            "ring_index": ring_index,
            "ring_margin_mm": ring_margin_mm,
            "pending_case_count": len(pending_cases),
            "nearest_official_count": len(nearest_official),
            "nearest_official": [
                {
                    "x_mm": float(result.x_mm),
                    "y_mm": float(result.y_mm),
                    "z_mm": float(result.z_mm),
                    "outside_square_distance_mm": outside_square_distance_yz_mm(
                        result,
                        center_y_mm=float(config.center_y_mm),
                        center_z_mm=float(config.center_z_mm),
                        half_span_mm=float(config.half_span_mm),
                    ),
                    "worst_joint_step_deg": float(result.worst_joint_step_deg),
                    "mean_joint_step_deg": float(result.mean_joint_step_deg),
                    "branch_flip_ratio": float(result.branch_flip_ratio),
                }
                for result in nearest_official
            ],
        }
        trace["rings"].append(ring_payload)  # type: ignore[index]
        if len(nearest_official) >= int(target_count):
            break

    selected = select_nearest_official_outside_results(
        tuple(all_results_by_key.values()),
        center_y_mm=float(config.center_y_mm),
        center_z_mm=float(config.center_z_mm),
        half_span_mm=float(config.half_span_mm),
        top_k=int(target_count),
        min_separation_mm=float(min_separation_mm),
    )
    trace["selected"] = [
        {
            "x_mm": float(result.x_mm),
            "y_mm": float(result.y_mm),
            "z_mm": float(result.z_mm),
            "outside_square_distance_mm": outside_square_distance_yz_mm(
                result,
                center_y_mm=float(config.center_y_mm),
                center_z_mm=float(config.center_z_mm),
                half_span_mm=float(config.half_span_mm),
            ),
            "worst_joint_step_deg": float(result.worst_joint_step_deg),
            "mean_joint_step_deg": float(result.mean_joint_step_deg),
            "branch_flip_ratio": float(result.branch_flip_ratio),
        }
        for result in selected
    ]
    return all_results_by_key, trace


def run_adaptive_sweep(
    *,
    config: AdaptiveSweepConfig,
    environment: SweepEnvironment,
    eval_config: EvalConfig,
    workers: int,
    prefix: str = "[adaptive]",
) -> tuple[dict[tuple[float, float], SweepResult], list[dict[str, float | int | str | bool]]]:
    center_y_mm = float(config.start_y_mm)
    center_z_mm = float(config.start_z_mm)
    step_y_mm = float(config.step_y_mm)
    step_z_mm = float(config.step_z_mm)
    min_step_y_mm = float(config.min_step_y_mm)
    min_step_z_mm = float(config.min_step_z_mm)
    max_iters = max(1, int(config.max_iters))

    print(
        f"{prefix} mode=adaptive workers={workers}, strategy={eval_config.strategy}, "
        f"window_repair={eval_config.run_window_repair}, "
        f"inserted_repair={eval_config.run_inserted_repair}, "
        f"x={float(config.x_mm):.3f}, start=({center_y_mm:.3f},{center_z_mm:.3f}), "
        f"step=({step_y_mm:.3f},{step_z_mm:.3f}), "
        f"min_step=({min_step_y_mm:.3f},{min_step_z_mm:.3f}), "
        f"max_iters={max_iters}, diagonal={config.include_diagonal}, "
        f"bounds=(y:[{config.min_y_mm},{config.max_y_mm}], z:[{config.min_z_mm},{config.max_z_mm}])"
    )

    start_case = SweepCase(
        x_mm=float(config.x_mm),
        y_mm=center_y_mm,
        z_mm=center_z_mm,
    )
    if not _case_within_optional_bounds(
        start_case,
        min_y_mm=config.min_y_mm,
        max_y_mm=config.max_y_mm,
        min_z_mm=config.min_z_mm,
        max_z_mm=config.max_z_mm,
    ):
        raise ValueError(
            "Adaptive sweep start origin is outside the configured hard bounds: "
            f"start=({center_y_mm:.3f}, {center_z_mm:.3f}), "
            f"bounds=(y:[{config.min_y_mm},{config.max_y_mm}], "
            f"z:[{config.min_z_mm},{config.max_z_mm}])."
        )

    all_results_by_key: dict[tuple[float, float], SweepResult] = {}
    adaptive_trace: list[dict[str, float | int | str | bool]] = []
    best_result: SweepResult | None = None

    for iteration_index in range(max_iters):
        neighbors = generate_adaptive_neighbors(
            x_mm=float(config.x_mm),
            center_y_mm=center_y_mm,
            center_z_mm=center_z_mm,
            step_y_mm=step_y_mm,
            step_z_mm=step_z_mm,
            include_diagonal=config.include_diagonal,
            min_y_mm=config.min_y_mm,
            max_y_mm=config.max_y_mm,
            min_z_mm=config.min_z_mm,
            max_z_mm=config.max_z_mm,
        )
        pending_cases = tuple(
            case for case in neighbors if case_key(case) not in all_results_by_key
        )
        if pending_cases:
            completed = evaluate_cases_parallel(
                cases=pending_cases,
                environment=environment,
                eval_config=eval_config,
                workers=workers,
                prefix=f"[adaptive iter {iteration_index + 1}]",
            )
            for result in completed:
                all_results_by_key[case_key_result(result)] = result

        local_results = tuple(
            all_results_by_key[case_key(case)]
            for case in neighbors
            if case_key(case) in all_results_by_key
        )
        if not local_results:
            break

        local_best = min(local_results, key=lambda item: item.rank_key())
        if best_result is None or is_better(local_best, best_result):
            best_result = local_best

        improved_center = (
            abs(local_best.y_mm - center_y_mm) > 1e-9
            or abs(local_best.z_mm - center_z_mm) > 1e-9
        )
        adaptive_trace.append(
            {
                "iteration": iteration_index + 1,
                "center_y_mm": center_y_mm,
                "center_z_mm": center_z_mm,
                "step_y_mm": step_y_mm,
                "step_z_mm": step_z_mm,
                "local_best_y_mm": local_best.y_mm,
                "local_best_z_mm": local_best.z_mm,
                "local_best_status": local_best.status,
                "local_best_worst_joint_step_deg": local_best.worst_joint_step_deg,
                "improved_center": improved_center,
            }
        )
        print(
            f"{prefix} iter={iteration_index + 1}, center=({center_y_mm:.3f},{center_z_mm:.3f}), "
            f"step=({step_y_mm:.3f},{step_z_mm:.3f}), best=({local_best.y_mm:.3f},{local_best.z_mm:.3f}), "
            f"status={local_best.status}, worst={local_best.worst_joint_step_deg:.3f}"
        )

        if local_best.status == "valid" and bool(config.stop_on_first_valid):
            center_y_mm = local_best.y_mm
            center_z_mm = local_best.z_mm
            break

        if improved_center:
            center_y_mm = local_best.y_mm
            center_z_mm = local_best.z_mm
            continue

        step_y_mm = max(min_step_y_mm, step_y_mm * 0.5)
        step_z_mm = max(min_step_z_mm, step_z_mm * 0.5)
        if step_y_mm <= min_step_y_mm + 1e-9 and step_z_mm <= min_step_z_mm + 1e-9:
            break

    return all_results_by_key, adaptive_trace


def _evaluate_case_batch_entry(
    case_batch: Sequence[SweepCase],
    environment: SweepEnvironment,
    eval_config: EvalConfig,
) -> tuple[SweepResult, ...]:
    return tuple(
        _evaluate_case(case=case, environment=environment, eval_config=eval_config)
        for case in case_batch
    )


_REQUIRED_POSE_COLUMNS = (
    "x_mm",
    "y_mm",
    "z_mm",
    "r11",
    "r12",
    "r13",
    "r21",
    "r22",
    "r23",
    "r31",
    "r32",
    "r33",
)
_OPTIONAL_POSE_LABEL_COLUMNS = ("source_row", "index")


def _evaluate_case(
    *,
    case: SweepCase,
    environment: SweepEnvironment,
    eval_config: EvalConfig,
) -> SweepResult:
    request_id = f"origin_y{format_value(case.y_mm)}_z{format_value(case.z_mm)}"
    csv_name = f"tool_poses_frame2_{request_id}.csv"
    pose_csv_path = environment.artifact_root / csv_name
    with redirect_stdout(io.StringIO()):
        runtime_settings = build_app_runtime_settings(
            validation_centerline_csv=environment.validation_centerline_csv,
            tool_poses_frame2_csv=pose_csv_path,
            append_start_as_terminal=environment.append_start_as_terminal,
            target_frame_origin_mm=(case.x_mm, case.y_mm, case.z_mm),
            target_frame_rotation_xyz_deg=environment.target_frame_rotation_xyz_deg,
            enable_custom_smoothing_and_pose_selection=environment.enable_custom_smoothing_and_pose_selection,
            robot_name=environment.robot_name,
            frame_name=environment.frame_name,
            program_name=environment.program_name,
            ik_backend=environment.ik_backend,
            local_parallel_workers=environment.local_parallel_workers,
            local_parallel_min_batch_size=environment.local_parallel_min_batch_size,
        )

        request = _build_sweep_profile_evaluation_request(
            case=case,
            runtime_settings=runtime_settings,
            request_id=request_id,
            eval_config=eval_config,
        )

        context = _get_worker_offline_context(
            robot_name=request.robot_name,
            frame_name=request.frame_name,
        )
        result, _search = evaluate_request(request, context)

    return SweepResult(
        request_id=request_id,
        x_mm=case.x_mm,
        y_mm=case.y_mm,
        z_mm=case.z_mm,
        status=result.status,
        timing_seconds=float(result.timing_seconds),
        invalid_row_count=int(result.invalid_row_count),
        ik_empty_row_count=int(result.ik_empty_row_count),
        config_switches=int(result.config_switches),
        bridge_like_segments=int(result.bridge_like_segments),
        big_circle_step_count=int(getattr(result, "big_circle_step_count", 0)),
        branch_flip_ratio=float(getattr(result, "branch_flip_ratio", 0.0)),
        worst_joint_step_deg=float(result.worst_joint_step_deg),
        mean_joint_step_deg=float(result.mean_joint_step_deg),
        total_cost=float(result.total_cost),
        pose_build_seconds=_metadata_seconds(
            result.metadata,
            "origin_sweep_timing",
            "pose_build_seconds",
        ),
        ik_collection_seconds=_profile_seconds(result.profiling, "ik_collection"),
        dp_path_selection_seconds=_profile_seconds(result.profiling, "dp_path_selection"),
        window_repair_seconds=_profile_seconds(result.profiling, "window_repair"),
        inserted_repair_seconds=_profile_seconds(result.profiling, "inserted_repair"),
    )


def _build_sweep_profile_evaluation_request(
    *,
    case: SweepCase,
    runtime_settings,
    request_id: str,
    eval_config: EvalConfig,
) -> ProfileEvaluationRequest:
    pose_build_started = perf_counter()
    centerline_dataset = _get_worker_centerline_dataset(runtime_settings)
    pose_frame = solve_tool_poses_from_dataset(
        centerline_dataset,
        None,
        target_frame_origin_mm=runtime_settings.target_frame_origin_mm,
        target_frame_rotation_xyz_deg=runtime_settings.target_frame_rotation_xyz_deg,
        verify_solution=runtime_settings.enable_solver_verification,
        verification_row_ids=list(runtime_settings.verification_row_ids)
        if runtime_settings.verification_row_ids is not None
        else None,
        verification_tolerance=runtime_settings.verification_tolerance,
    )
    pose_build_seconds = perf_counter() - pose_build_started
    pose_rows = _pose_rows_from_dataframe(pose_frame)
    row_labels = _extract_row_labels(pose_rows)
    zero_profile = tuple((0.0, 0.0) for _ in pose_rows)
    return ProfileEvaluationRequest(
        request_id=request_id,
        robot_name=runtime_settings.robot_name,
        frame_name=runtime_settings.frame_name,
        motion_settings=motion_settings_to_dict(runtime_settings.motion_settings),
        reference_pose_rows=pose_rows,
        frame_a_origin_yz_profile_mm=zero_profile,
        row_labels=row_labels,
        inserted_flags=tuple(False for _ in pose_rows),
        strategy=eval_config.strategy,
        run_window_repair=eval_config.run_window_repair,
        run_inserted_repair=eval_config.run_inserted_repair,
        include_pose_rows_in_result=False,
        create_program=False,
        metadata={
            "origin_sweep": {
                "x_mm": case.x_mm,
                "y_mm": case.y_mm,
                "z_mm": case.z_mm,
                "strategy": eval_config.strategy,
                "run_window_repair": eval_config.run_window_repair,
                "run_inserted_repair": eval_config.run_inserted_repair,
            },
            "origin_sweep_timing": {
                "pose_build_seconds": round(float(pose_build_seconds), 6),
                "pose_csv_materialized": False,
                "pose_row_count": len(pose_rows),
            },
        },
    )


def _pose_rows_from_dataframe(pose_frame: Any) -> tuple[dict[str, float], ...]:
    pose_rows: list[dict[str, float]] = []
    for row_number, raw_row in enumerate(pose_frame.to_dict(orient="records"), start=1):
        if not _dataframe_pose_row_is_valid(raw_row):
            continue
        row: dict[str, float] = {}
        for column in _REQUIRED_POSE_COLUMNS:
            value = float(raw_row[column])
            if not math.isfinite(value):
                raise ValueError(f"Non-finite pose value for {column!r} at generated row {row_number}.")
            row[column] = value
        for column in _OPTIONAL_POSE_LABEL_COLUMNS:
            value = raw_row.get(column)
            if value is None:
                continue
            try:
                numeric_value = float(value)
            except (TypeError, ValueError):
                continue
            if math.isfinite(numeric_value):
                row[column] = numeric_value
        pose_rows.append(row)
    if not pose_rows:
        raise ValueError("Pose solver did not produce any valid pose rows.")
    return tuple(pose_rows)


def _dataframe_pose_row_is_valid(raw_row: dict[str, Any]) -> bool:
    value = raw_row.get("valid", True)
    if isinstance(value, str):
        return value.strip().lower() not in {"0", "false", "no"}
    try:
        if isinstance(value, float) and math.isnan(value):
            return False
        return bool(value)
    except Exception:
        return False


def _metadata_seconds(
    metadata: dict[str, Any],
    section_name: str,
    value_name: str,
) -> float:
    section = metadata.get(section_name, {})
    if not isinstance(section, dict):
        return 0.0
    try:
        return float(section.get(value_name, 0.0))
    except (TypeError, ValueError):
        return 0.0


def _profile_seconds(
    profile: dict[str, dict[str, float | int]],
    section_name: str,
) -> float:
    section = profile.get(section_name, {})
    try:
        return float(section.get("seconds", 0.0))
    except (TypeError, ValueError, AttributeError):
        return 0.0


_WORKER_OFFLINE_CONTEXT_CACHE: dict[tuple[str, str], object] = {}
_WORKER_CENTERLINE_DATASET_CACHE: dict[tuple[str, int, int, object, bool], object] = {}


def _get_worker_centerline_dataset(runtime_settings):
    """Return a per-process parsed centerline dataset for repeated origin cases."""

    centerline_path = Path(runtime_settings.validation_centerline_csv)
    centerline_stat = centerline_path.stat()
    cache_key = (
        str(centerline_path.resolve()),
        int(centerline_stat.st_mtime_ns),
        int(centerline_stat.st_size),
        runtime_settings.frame_build_options,
        bool(runtime_settings.append_start_as_terminal),
    )
    cached = _WORKER_CENTERLINE_DATASET_CACHE.get(cache_key)
    if cached is not None:
        return cached
    dataset = load_centerline_dataset(
        runtime_settings.validation_centerline_csv,
        require_boundaries=False,
        build_options=runtime_settings.frame_build_options,
        append_start_as_terminal=runtime_settings.append_start_as_terminal,
    )
    _WORKER_CENTERLINE_DATASET_CACHE[cache_key] = dataset
    return dataset


def _get_worker_offline_context(*, robot_name: str, frame_name: str):
    cache_key = (str(robot_name), str(frame_name))
    cached = _WORKER_OFFLINE_CONTEXT_CACHE.get(cache_key)
    if cached is not None:
        return cached
    context = open_offline_ik_station_context(
        robot_name=robot_name,
        frame_name=frame_name,
    )
    _WORKER_OFFLINE_CONTEXT_CACHE[cache_key] = context
    return context


def _split_case_batches(
    case_list: Sequence[SweepCase],
    *,
    batch_count: int,
) -> tuple[tuple[SweepCase, ...], ...]:
    if not case_list:
        return ()
    safe_batch_count = max(1, int(batch_count))
    if safe_batch_count == 1 or len(case_list) == 1:
        return (tuple(case_list),)
    batch_size = max(1, math.ceil(len(case_list) / safe_batch_count))
    return tuple(
        tuple(case_list[index : index + batch_size])
        for index in range(0, len(case_list), batch_size)
    )


def _print_progress(
    prefix: str,
    finished_index: int,
    total_count: int,
    result: SweepResult,
) -> None:
    print(
        f"{prefix} {finished_index:>2}/{total_count} "
        f"y={result.y_mm:.3f}, z={result.z_mm:.3f}, status={result.status}, "
        f"worst={result.worst_joint_step_deg:.3f}, time={result.timing_seconds:.3f}s",
        flush=True,
    )
