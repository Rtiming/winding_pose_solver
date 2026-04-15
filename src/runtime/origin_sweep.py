from __future__ import annotations

import io
import math
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import redirect_stdout
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from app_settings import build_app_runtime_settings
from src.robodk_runtime.eval_worker import evaluate_request, open_offline_ik_station_context
from src.runtime.request_builder import build_profile_evaluation_request


@dataclass(frozen=True)
class SweepEnvironment:
    validation_centerline_csv: Path
    target_frame_rotation_xyz_deg: tuple[float, float, float]
    enable_custom_smoothing_and_pose_selection: bool
    robot_name: str
    frame_name: str
    program_name: str
    ik_backend: str
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
    worst_joint_step_deg: float
    mean_joint_step_deg: float
    total_cost: float

    def rank_key(self) -> tuple[float, ...]:
        return (
            0.0 if self.status == "valid" else 1.0,
            float(self.invalid_row_count),
            float(self.ik_empty_row_count),
            float(self.config_switches),
            float(self.bridge_like_segments),
            float(self.worst_joint_step_deg),
            float(self.mean_joint_step_deg),
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
    return _deduplicate_cases(
        tuple(
        SweepCase(
            x_mm=float(x_mm),
            y_mm=float(center_y_mm + dy_mm),
            z_mm=float(center_z_mm + dz_mm),
        )
        for dy_mm, dz_mm in offsets
        if _case_within_optional_bounds(
            SweepCase(
                x_mm=float(x_mm),
                y_mm=float(center_y_mm + dy_mm),
                z_mm=float(center_z_mm + dz_mm),
            ),
            min_y_mm=min_y_mm,
            max_y_mm=max_y_mm,
            min_z_mm=min_z_mm,
            max_z_mm=max_z_mm,
        )
    ))


def is_better(lhs: SweepResult, rhs: SweepResult) -> bool:
    return lhs.rank_key() < rhs.rank_key()


def print_result_table(results: Sequence[SweepResult]) -> None:
    print(
        "request_id,status,y_mm,z_mm,worst_joint_step_deg,mean_joint_step_deg,"
        "config_switches,bridge_like_segments,ik_empty_rows,time_s,total_cost"
    )
    for result in results:
        print(
            f"{result.request_id},{result.status},{result.y_mm:.3f},{result.z_mm:.3f},"
            f"{result.worst_joint_step_deg:.6f},{result.mean_joint_step_deg:.6f},"
            f"{result.config_switches},{result.bridge_like_segments},"
            f"{result.ik_empty_row_count},{result.timing_seconds:.3f},{result.total_cost:.6f}"
        )


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
    if worker_count == 1:
        completed: list[SweepResult] = []
        for finished_index, case in enumerate(case_list, start=1):
            result = _evaluate_case(case=case, environment=environment, eval_config=eval_config)
            completed.append(result)
            _print_progress(prefix, finished_index, len(case_list), result)
        return tuple(completed)

    batch_count = min(len(case_list), worker_count * 2)
    case_batches = _split_case_batches(case_list, batch_count=batch_count)
    completed: list[SweepResult] = []
    finished_index = 0
    with ProcessPoolExecutor(max_workers=worker_count) as executor:
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
    )
    print(
        f"{prefix} mode=grid cases={len(cases)}, workers={workers}, "
        f"strategy={eval_config.strategy}, "
        f"window_repair={eval_config.run_window_repair}, "
        f"inserted_repair={eval_config.run_inserted_repair}, "
        f"x={float(x_mm):.3f}, y={y_series}, z={z_series}, "
        f"bounds=(y:[{min_y_mm},{max_y_mm}], z:[{min_z_mm},{max_z_mm}])"
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

        if local_best.status == "valid":
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


def _evaluate_case(
    *,
    case: SweepCase,
    environment: SweepEnvironment,
    eval_config: EvalConfig,
) -> SweepResult:
    request_id = f"origin_y{format_value(case.y_mm)}_z{format_value(case.z_mm)}"
    csv_name = f"tool_poses_frame2_{request_id}.csv"
    pose_csv_path = environment.artifact_root / csv_name
    pose_csv_path.parent.mkdir(parents=True, exist_ok=True)
    with redirect_stdout(io.StringIO()):
        runtime_settings = build_app_runtime_settings(
            validation_centerline_csv=environment.validation_centerline_csv,
            tool_poses_frame2_csv=pose_csv_path,
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

        request = build_profile_evaluation_request(
            runtime_settings,
            request_id=request_id,
            strategy=eval_config.strategy,
            refresh_csv=True,
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
                }
            },
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
        worst_joint_step_deg=float(result.worst_joint_step_deg),
        mean_joint_step_deg=float(result.mean_joint_step_deg),
        total_cost=float(result.total_cost),
    )


_WORKER_OFFLINE_CONTEXT_CACHE: dict[tuple[str, str], object] = {}


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
    batch_size = max(1, math.ceil(len(case_list) / safe_batch_count))
    return tuple(
        tuple(case_list[start_index : start_index + batch_size])
        for start_index in range(0, len(case_list), batch_size)
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
        f"worst={result.worst_joint_step_deg:.3f}, time={result.timing_seconds:.3f}s"
    )
