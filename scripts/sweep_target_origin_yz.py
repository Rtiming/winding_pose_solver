from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path
from time import perf_counter

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from main import (
    APPEND_CENTERLINE_START_AS_TERMINAL,
    ENABLE_CUSTOM_SMOOTHING_AND_POSE_SELECTION,
    FRAME_NAME,
    IK_BACKEND,
    PROGRAM_NAME,
    ROBOT_NAME,
    TARGET_FRAME_A_ORIGIN_IN_FRAME2_MM,
    TARGET_FRAME_A_ROTATION_IN_FRAME2_XYZ_DEG,
    VALIDATION_CENTERLINE_CSV,
)
from src.runtime.origin_sweep import (
    AdaptiveSweepConfig,
    CandidateSweepConfig,
    EvalConfig,
    SmartSquareSweepConfig,
    SweepEnvironment,
    candidate_rank_key,
    distance_from_seed_yz_mm,
    generate_radius_axis_values,
    outside_square_distance_yz_mm,
    parse_float_list,
    print_result_table,
    run_adaptive_sweep,
    run_candidate_sweep,
    run_grid_sweep,
    run_outside_square_fallback,
    run_smart_square_sweep,
    select_diverse_top_results,
    select_nearest_official_outside_results,
)


def _build_environment(
    *,
    local_parallel_workers: int,
    local_parallel_min_batch_size: int,
) -> SweepEnvironment:
    return SweepEnvironment(
        validation_centerline_csv=VALIDATION_CENTERLINE_CSV,
        target_frame_rotation_xyz_deg=TARGET_FRAME_A_ROTATION_IN_FRAME2_XYZ_DEG,
        enable_custom_smoothing_and_pose_selection=ENABLE_CUSTOM_SMOOTHING_AND_POSE_SELECTION,
        robot_name=ROBOT_NAME,
        frame_name=FRAME_NAME,
        program_name=PROGRAM_NAME,
        ik_backend=IK_BACKEND,
        append_start_as_terminal=APPEND_CENTERLINE_START_AS_TERMINAL,
        local_parallel_workers=int(local_parallel_workers),
        local_parallel_min_batch_size=int(local_parallel_min_batch_size),
    )


def _parse_origin(raw: str | None) -> tuple[float, float, float]:
    if raw is None:
        return tuple(float(value) for value in TARGET_FRAME_A_ORIGIN_IN_FRAME2_MM)
    values = parse_float_list(raw)
    if len(values) != 3:
        raise ValueError("--seed-origin must contain exactly three comma-separated values: X,Y,Z.")
    return (float(values[0]), float(values[1]), float(values[2]))


def _result_payload_with_distance(
    result,
    *,
    seed_y_mm: float,
    seed_z_mm: float,
    square_half_span_mm: float | None = None,
) -> dict[str, object]:
    payload = asdict(result)
    payload["distance_from_seed_yz_mm"] = distance_from_seed_yz_mm(
        result,
        seed_y_mm=float(seed_y_mm),
        seed_z_mm=float(seed_z_mm),
    )
    if square_half_span_mm is not None:
        payload["outside_square_distance_mm"] = outside_square_distance_yz_mm(
            result,
            center_y_mm=float(seed_y_mm),
            center_z_mm=float(seed_z_mm),
            half_span_mm=float(square_half_span_mm),
        )
    return payload


def _print_recommended_constants(
    results,
    *,
    seed_y_mm: float,
    seed_z_mm: float,
    top_k: int,
    min_separation_mm: float,
    preselected: tuple | None = None,
) -> tuple:
    recommended = (
        tuple(preselected)
        if preselected is not None
        else select_diverse_top_results(
            results,
            seed_y_mm=float(seed_y_mm),
            seed_z_mm=float(seed_z_mm),
            top_k=int(top_k),
            min_separation_mm=float(min_separation_mm),
        )
    )
    if not recommended:
        return recommended

    print(
        f"[sweep] top {len(recommended)} candidate origins "
        f"(min_separation={float(min_separation_mm):.3f}mm):"
    )
    for index, result in enumerate(recommended, start=1):
        distance_mm = distance_from_seed_yz_mm(
            result,
            seed_y_mm=float(seed_y_mm),
            seed_z_mm=float(seed_z_mm),
        )
        print(
            f"[sweep] #{index} TARGET_FRAME_A_ORIGIN_IN_FRAME2_MM = "
            f"({result.x_mm:.3f}, {result.y_mm:.3f}, {result.z_mm:.3f}) "
            f"distance={distance_mm:.3f}mm status={result.status} "
            f"empty={result.ik_empty_row_count} switches={result.config_switches} "
            f"bridge={result.bridge_like_segments} big_circle={result.big_circle_step_count} "
            f"ratio={result.branch_flip_ratio:.3f} worst={result.worst_joint_step_deg:.3f} "
            f"mean={result.mean_joint_step_deg:.6f}"
        )
    return recommended


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Parallel Y/Z sweep for TARGET_FRAME_A_ORIGIN_IN_FRAME2_MM under local six_axis_ik evaluation."
    )
    parser.add_argument(
        "--mode",
        choices=("grid", "adaptive", "candidates", "smart-square"),
        default="grid",
    )
    parser.add_argument("--x", type=float, default=float(TARGET_FRAME_A_ORIGIN_IN_FRAME2_MM[0]))
    parser.add_argument(
        "--seed-origin",
        type=str,
        default=None,
        help="Seed origin as X,Y,Z in mm. Candidate mode fixes X to this value.",
    )
    parser.add_argument(
        "--y-values",
        type=str,
        default="-420,-410,-400,-390,-380",
        help="Comma-separated Y values in mm.",
    )
    parser.add_argument(
        "--z-values",
        type=str,
        default="1080,1100,1120",
        help="Comma-separated Z values in mm.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, min(4, (os.cpu_count() or 4) // 2)),
        help="Number of outer processes for independent origin cases.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "tmp" / "origin_sweep_results.json",
    )
    parser.add_argument(
        "--strategy",
        choices=("full_search", "exact_profile"),
        default="full_search",
        help="Evaluation strategy for each origin case.",
    )
    parser.add_argument(
        "--inner-workers",
        type=int,
        default=None,
        help=(
            "Workers for exact-profile repair inside one origin case. "
            "Default: auto-enable when --workers=1, disable when outer workers > 1."
        ),
    )
    parser.add_argument(
        "--inner-min-batch-size",
        type=int,
        default=None,
        help=(
            "Minimum number of repair candidate profiles before inner parallel "
            "evaluation starts. Default: 8 when inner parallel is enabled."
        ),
    )
    parser.add_argument(
        "--skip-window-repair",
        action="store_true",
        help="Disable local window repair in per-case evaluation.",
    )
    parser.add_argument(
        "--skip-inserted-repair",
        action="store_true",
        help="Disable inserted transition repair in per-case evaluation.",
    )
    parser.add_argument("--start-y", type=float, default=float(TARGET_FRAME_A_ORIGIN_IN_FRAME2_MM[1]))
    parser.add_argument("--start-z", type=float, default=float(TARGET_FRAME_A_ORIGIN_IN_FRAME2_MM[2]))
    parser.add_argument("--step-y", type=float, default=20.0)
    parser.add_argument("--step-z", type=float, default=20.0)
    parser.add_argument("--min-step-y", type=float, default=5.0)
    parser.add_argument("--min-step-z", type=float, default=5.0)
    parser.add_argument("--max-iters", type=int, default=8)
    parser.add_argument("--no-diagonal", action="store_true")
    parser.add_argument(
        "--continue-after-valid",
        action="store_true",
        help="Keep optimizing quality after finding a reachable origin instead of stopping immediately.",
    )
    parser.add_argument("--min-y", type=float, default=None, help="Hard lower bound for origin Y.")
    parser.add_argument("--max-y", type=float, default=None, help="Hard upper bound for origin Y.")
    parser.add_argument("--min-z", type=float, default=None, help="Hard lower bound for origin Z.")
    parser.add_argument("--max-z", type=float, default=None, help="Hard upper bound for origin Z.")
    parser.add_argument(
        "--radius-mm",
        type=float,
        default=None,
        help="Circular Y/Z search radius around the seed/center. Grid mode uses it as a filter.",
    )
    parser.add_argument(
        "--radius-step-mm",
        type=float,
        default=80.0,
        help="Grid spacing when --radius-mm is used in grid mode.",
    )
    parser.add_argument(
        "--coarse-step-mm",
        type=float,
        default=80.0,
        help="Candidate mode coarse Y/Z grid spacing.",
    )
    parser.add_argument(
        "--fine-radius-mm",
        type=float,
        default=40.0,
        help="Candidate mode local refinement radius around coarse winners.",
    )
    parser.add_argument(
        "--fine-step-mm",
        type=float,
        default=10.0,
        help="Candidate mode local refinement Y/Z grid spacing.",
    )
    parser.add_argument(
        "--confirm-top-n",
        type=int,
        default=16,
        help="Candidate mode coarse winners to refine.",
    )
    parser.add_argument(
        "--square-size-mm",
        type=float,
        default=300.0,
        help="Smart-square mode Y/Z square side length centered on --seed-origin.",
    )
    parser.add_argument(
        "--smart-initial-step-mm",
        type=float,
        default=75.0,
        help="Smart-square mode first neighborhood step.",
    )
    parser.add_argument(
        "--smart-min-step-mm",
        type=float,
        default=10.0,
        help="Smart-square mode final neighborhood step.",
    )
    parser.add_argument(
        "--smart-max-iters",
        type=int,
        default=5,
        help="Smart-square mode maximum beam/pattern-search iterations.",
    )
    parser.add_argument(
        "--beam-width",
        type=int,
        default=4,
        help="Smart-square mode number of diverse promising centers to refine each iteration.",
    )
    parser.add_argument(
        "--smart-diagonal-policy",
        choices=("conditional", "always", "never"),
        default="conditional",
        help=(
            "Smart-square mode diagonal-neighbor policy: conditional skips the "
            "diagonals once a clean valid axis candidate appears."
        ),
    )
    parser.add_argument(
        "--smart-polish-step-mm",
        type=float,
        default=0.0,
        help=(
            "Optional final local polish step around the current best smart-square "
            "origin. 0 disables polish."
        ),
    )
    parser.add_argument(
        "--validation-grid-step-mm",
        type=float,
        default=0.0,
        help=(
            "Optional smart-square coarse validation grid step. "
            "0 disables validation grid evaluation."
        ),
    )
    parser.add_argument("--top-k", type=int, default=8, help="Number of diverse candidate constants to print.")
    parser.add_argument(
        "--min-separation-mm",
        type=float,
        default=20.0,
        help="Minimum Y/Z distance between printed candidate origins.",
    )
    parser.add_argument(
        "--outside-fallback-count",
        type=int,
        default=3,
        help=(
            "If smart-square finds no official deliverable inside the square, "
            "search for this many nearest official candidates outside the square."
        ),
    )
    parser.add_argument(
        "--outside-fallback-max-rings",
        type=int,
        default=6,
        help="Maximum number of expanding outside-square perimeter rings to evaluate.",
    )
    parser.add_argument(
        "--outside-fallback-ring-step-mm",
        type=float,
        default=25.0,
        help="Distance between outside-square fallback rings. 0 uses the smart minimum step.",
    )
    parser.add_argument(
        "--outside-fallback-edge-step-mm",
        type=float,
        default=75.0,
        help="Spacing along each outside-square fallback ring edge. 0 uses half the smart initial step.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    outer_workers = max(1, int(args.workers))
    if args.inner_workers is None:
        inner_workers = 0 if outer_workers == 1 else 1
    else:
        inner_workers = max(0, int(args.inner_workers))
    if args.inner_min_batch_size is None:
        inner_min_batch_size = 8 if inner_workers != 1 else 999999
    else:
        inner_min_batch_size = max(1, int(args.inner_min_batch_size))

    environment = _build_environment(
        local_parallel_workers=inner_workers,
        local_parallel_min_batch_size=inner_min_batch_size,
    )
    eval_config = EvalConfig(
        strategy=str(args.strategy),
        run_window_repair=not bool(args.skip_window_repair),
        run_inserted_repair=not bool(args.skip_inserted_repair),
    )

    seed_x_mm, seed_y_mm, seed_z_mm = _parse_origin(args.seed_origin)
    min_y_mm = None if args.min_y is None else float(args.min_y)
    max_y_mm = None if args.max_y is None else float(args.max_y)
    min_z_mm = None if args.min_z is None else float(args.min_z)
    max_z_mm = None if args.max_z is None else float(args.max_z)

    start_ts = perf_counter()
    adaptive_trace: list[dict[str, float | int | str | bool]] = []
    candidate_trace: dict[str, object] = {}
    y_values: tuple[float, ...] | None = None
    z_values: tuple[float, ...] | None = None
    run_x_mm = float(args.x)
    if args.mode == "grid":
        if args.radius_mm is None:
            y_values = parse_float_list(args.y_values)
            z_values = parse_float_list(args.z_values)
            radius_center_y_mm = None
            radius_center_z_mm = None
        else:
            y_values = generate_radius_axis_values(
                center_mm=seed_y_mm,
                radius_mm=float(args.radius_mm),
                step_mm=float(args.radius_step_mm),
            )
            z_values = generate_radius_axis_values(
                center_mm=seed_z_mm,
                radius_mm=float(args.radius_mm),
                step_mm=float(args.radius_step_mm),
            )
            radius_center_y_mm = seed_y_mm
            radius_center_z_mm = seed_z_mm
        all_results_by_key, y_values, z_values = run_grid_sweep(
            x_mm=run_x_mm,
            y_values=y_values,
            z_values=z_values,
            environment=environment,
            eval_config=eval_config,
            workers=outer_workers,
            prefix="[sweep]",
            min_y_mm=min_y_mm,
            max_y_mm=max_y_mm,
            min_z_mm=min_z_mm,
            max_z_mm=max_z_mm,
            center_y_mm=radius_center_y_mm,
            center_z_mm=radius_center_z_mm,
            radius_mm=None if args.radius_mm is None else float(args.radius_mm),
        )
        adaptive_trace = []
    elif args.mode == "adaptive":
        all_results_by_key, adaptive_trace = run_adaptive_sweep(
            config=AdaptiveSweepConfig(
                x_mm=run_x_mm,
                start_y_mm=float(args.start_y),
                start_z_mm=float(args.start_z),
                step_y_mm=float(args.step_y),
                step_z_mm=float(args.step_z),
                min_step_y_mm=float(args.min_step_y),
                min_step_z_mm=float(args.min_step_z),
                max_iters=max(1, int(args.max_iters)),
                include_diagonal=not bool(args.no_diagonal),
                min_y_mm=min_y_mm,
                max_y_mm=max_y_mm,
                min_z_mm=min_z_mm,
                max_z_mm=max_z_mm,
                stop_on_first_valid=not bool(args.continue_after_valid),
            ),
            environment=environment,
            eval_config=eval_config,
            workers=outer_workers,
            prefix="[adaptive]",
        )
    elif args.mode == "candidates":
        if args.radius_mm is None:
            raise ValueError("--mode candidates requires --radius-mm.")
        run_x_mm = seed_x_mm
        all_results_by_key, candidate_trace = run_candidate_sweep(
            config=CandidateSweepConfig(
                x_mm=seed_x_mm,
                seed_y_mm=seed_y_mm,
                seed_z_mm=seed_z_mm,
                radius_mm=float(args.radius_mm),
                coarse_step_mm=float(args.coarse_step_mm),
                fine_radius_mm=float(args.fine_radius_mm),
                fine_step_mm=float(args.fine_step_mm),
                confirm_top_n=max(0, int(args.confirm_top_n)),
                min_y_mm=min_y_mm,
                max_y_mm=max_y_mm,
                min_z_mm=min_z_mm,
                max_z_mm=max_z_mm,
            ),
            environment=environment,
            eval_config=eval_config,
            workers=outer_workers,
            prefix="[candidates]",
        )
    else:
        run_x_mm = seed_x_mm
        if args.square_size_mm <= 0.0:
            raise ValueError("--square-size-mm must be positive.")
        smart_square_config = SmartSquareSweepConfig(
            x_mm=seed_x_mm,
            center_y_mm=seed_y_mm,
            center_z_mm=seed_z_mm,
            half_span_mm=float(args.square_size_mm) * 0.5,
            initial_step_mm=float(args.smart_initial_step_mm),
            min_step_mm=float(args.smart_min_step_mm),
            max_iters=max(1, int(args.smart_max_iters)),
            beam_width=max(1, int(args.beam_width)),
            diagonal_policy=str(args.smart_diagonal_policy),
            polish_step_mm=max(0.0, float(args.smart_polish_step_mm)),
            validation_grid_step_mm=max(0.0, float(args.validation_grid_step_mm)),
            min_y_mm=min_y_mm,
            max_y_mm=max_y_mm,
            min_z_mm=min_z_mm,
            max_z_mm=max_z_mm,
        )
        all_results_by_key, candidate_trace = run_smart_square_sweep(
            config=smart_square_config,
            environment=environment,
            eval_config=eval_config,
            workers=outer_workers,
            prefix="[smart-square]",
        )
        official_inside = [
            result
            for result in all_results_by_key.values()
            if outside_square_distance_yz_mm(
                result,
                center_y_mm=seed_y_mm,
                center_z_mm=seed_z_mm,
                half_span_mm=smart_square_config.half_span_mm,
            )
            <= 1e-9
            and result.status == "valid"
            and result.invalid_row_count == 0
            and result.ik_empty_row_count == 0
            and result.bridge_like_segments == 0
            and result.big_circle_step_count == 0
            and result.worst_joint_step_deg <= 60.0 + 1e-9
        ]
        if not official_inside and int(args.outside_fallback_count) > 0:
            ring_step_mm = (
                float(args.outside_fallback_ring_step_mm)
                if float(args.outside_fallback_ring_step_mm) > 0.0
                else float(args.smart_min_step_mm)
            )
            edge_step_mm = (
                float(args.outside_fallback_edge_step_mm)
                if float(args.outside_fallback_edge_step_mm) > 0.0
                else max(float(args.smart_min_step_mm), float(args.smart_initial_step_mm) * 0.5)
            )
            all_results_by_key, outside_trace = run_outside_square_fallback(
                existing_results_by_key=all_results_by_key,
                config=smart_square_config,
                environment=environment,
                eval_config=eval_config,
                workers=outer_workers,
                target_count=max(0, int(args.outside_fallback_count)),
                max_rings=max(0, int(args.outside_fallback_max_rings)),
                ring_step_mm=ring_step_mm,
                edge_step_mm=edge_step_mm,
                min_separation_mm=float(args.min_separation_mm),
                prefix="[outside-square]",
            )
            candidate_trace["outside_fallback"] = outside_trace
        else:
            candidate_trace["outside_fallback"] = {
                "triggered": False,
                "reason": "official_inside_found" if official_inside else "disabled",
            }

    if args.mode in {"candidates", "smart-square"} or args.radius_mm is not None:
        ordered = tuple(
            sorted(
                all_results_by_key.values(),
                key=lambda item: candidate_rank_key(
                    item,
                    seed_y_mm=seed_y_mm,
                    seed_z_mm=seed_z_mm,
                ),
            )
        )
    else:
        ordered = tuple(sorted(all_results_by_key.values(), key=lambda item: item.rank_key()))
    best = ordered[0] if ordered else None
    elapsed = perf_counter() - start_ts
    baseline = all_results_by_key.get((round(seed_y_mm, 6), round(seed_z_mm, 6)))
    square_half_span_mm = float(args.square_size_mm) * 0.5 if args.mode == "smart-square" else None
    outside_fallback_trace = (
        candidate_trace.get("outside_fallback", {})
        if isinstance(candidate_trace, dict)
        else {}
    )
    if isinstance(outside_fallback_trace, dict) and outside_fallback_trace.get("selected"):
        recommended = select_nearest_official_outside_results(
            ordered,
            center_y_mm=seed_y_mm,
            center_z_mm=seed_z_mm,
            half_span_mm=float(square_half_span_mm),
            top_k=max(int(args.top_k), int(args.outside_fallback_count)),
            min_separation_mm=float(args.min_separation_mm),
        )
    else:
        recommended = select_diverse_top_results(
            ordered,
            seed_y_mm=seed_y_mm,
            seed_z_mm=seed_z_mm,
            top_k=int(args.top_k),
            min_separation_mm=float(args.min_separation_mm),
        )

    payload: dict[str, object] = {
        "mode": str(args.mode),
        "x_mm": run_x_mm,
        "seed_origin_mm": [seed_x_mm, seed_y_mm, seed_z_mm],
        "workers": outer_workers,
        "inner_workers": inner_workers,
        "inner_min_batch_size": inner_min_batch_size,
        "strategy": eval_config.strategy,
        "run_window_repair": eval_config.run_window_repair,
        "run_inserted_repair": eval_config.run_inserted_repair,
        "elapsed_seconds": elapsed,
        "evaluated_cases": len(ordered),
        "adaptive_trace": adaptive_trace,
        "candidate_trace": candidate_trace,
        "best": (
            _result_payload_with_distance(
                best,
                seed_y_mm=seed_y_mm,
                seed_z_mm=seed_z_mm,
                square_half_span_mm=square_half_span_mm,
            )
            if best is not None
            else None
        ),
        "baseline": (
            _result_payload_with_distance(
                baseline,
                seed_y_mm=seed_y_mm,
                seed_z_mm=seed_z_mm,
                square_half_span_mm=square_half_span_mm,
            )
            if baseline is not None
            else None
        ),
        "recommended": [
            _result_payload_with_distance(
                item,
                seed_y_mm=seed_y_mm,
                seed_z_mm=seed_z_mm,
                square_half_span_mm=square_half_span_mm,
            )
            for item in recommended
        ],
        "results": [
            _result_payload_with_distance(
                item,
                seed_y_mm=seed_y_mm,
                seed_z_mm=seed_z_mm,
                square_half_span_mm=square_half_span_mm,
            )
            for item in ordered
        ],
    }
    if args.mode == "grid":
        payload["y_values"] = y_values if y_values is not None else parse_float_list(args.y_values)
        payload["z_values"] = z_values if z_values is not None else parse_float_list(args.z_values)
    elif args.mode == "adaptive":
        payload["start_y"] = float(args.start_y)
        payload["start_z"] = float(args.start_z)
        payload["step_y"] = float(args.step_y)
        payload["step_z"] = float(args.step_z)
        payload["min_step_y"] = float(args.min_step_y)
        payload["min_step_z"] = float(args.min_step_z)
        payload["max_iters"] = int(args.max_iters)
        payload["diagonal"] = not bool(args.no_diagonal)
        payload["continue_after_valid"] = bool(args.continue_after_valid)
    elif args.mode == "candidates":
        payload["coarse_step_mm"] = float(args.coarse_step_mm)
        payload["fine_radius_mm"] = float(args.fine_radius_mm)
        payload["fine_step_mm"] = float(args.fine_step_mm)
        payload["confirm_top_n"] = int(args.confirm_top_n)
    else:
        payload["square_size_mm"] = float(args.square_size_mm)
        payload["smart_initial_step_mm"] = float(args.smart_initial_step_mm)
        payload["smart_min_step_mm"] = float(args.smart_min_step_mm)
        payload["smart_max_iters"] = int(args.smart_max_iters)
        payload["beam_width"] = int(args.beam_width)
        payload["smart_diagonal_policy"] = str(args.smart_diagonal_policy)
        payload["smart_polish_step_mm"] = float(args.smart_polish_step_mm)
        payload["validation_grid_step_mm"] = float(args.validation_grid_step_mm)
        payload["outside_fallback_count"] = int(args.outside_fallback_count)
        payload["outside_fallback_max_rings"] = int(args.outside_fallback_max_rings)
        payload["outside_fallback_ring_step_mm"] = float(args.outside_fallback_ring_step_mm)
        payload["outside_fallback_edge_step_mm"] = float(args.outside_fallback_edge_step_mm)
    payload["radius_mm"] = None if args.radius_mm is None else float(args.radius_mm)
    payload["radius_step_mm"] = float(args.radius_step_mm)
    payload["top_k"] = int(args.top_k)
    payload["min_separation_mm"] = float(args.min_separation_mm)
    payload["min_y"] = min_y_mm
    payload["max_y"] = max_y_mm
    payload["min_z"] = min_z_mm
    payload["max_z"] = max_z_mm

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[sweep] wrote {args.output}")

    if best is not None:
        best_distance_mm = distance_from_seed_yz_mm(
            best,
            seed_y_mm=seed_y_mm,
            seed_z_mm=seed_z_mm,
        )
        print(
            "[sweep] best "
            f"status={best.status}, origin=({best.x_mm:.3f}, {best.y_mm:.3f}, {best.z_mm:.3f}), "
            f"worst={best.worst_joint_step_deg:.3f}, mean={best.mean_joint_step_deg:.6f}, "
            f"switches={best.config_switches}, bridge={best.bridge_like_segments}, "
            f"empty={best.ik_empty_row_count}, distance={best_distance_mm:.3f}mm, "
            f"time={best.timing_seconds:.3f}s"
        )
        print(
            "[sweep] suggested main.py constant "
            "TARGET_FRAME_A_ORIGIN_IN_FRAME2_MM = "
            f"({best.x_mm:.3f}, {best.y_mm:.3f}, {best.z_mm:.3f})"
        )
    _print_recommended_constants(
        ordered,
        seed_y_mm=seed_y_mm,
        seed_z_mm=seed_z_mm,
        top_k=int(args.top_k),
        min_separation_mm=float(args.min_separation_mm),
        preselected=tuple(recommended),
    )

    print_result_table(ordered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
