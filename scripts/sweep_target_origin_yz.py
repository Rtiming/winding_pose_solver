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
    EvalConfig,
    SweepEnvironment,
    parse_float_list,
    print_result_table,
    run_adaptive_sweep,
    run_grid_sweep,
)


def _build_environment() -> SweepEnvironment:
    return SweepEnvironment(
        validation_centerline_csv=VALIDATION_CENTERLINE_CSV,
        target_frame_rotation_xyz_deg=TARGET_FRAME_A_ROTATION_IN_FRAME2_XYZ_DEG,
        enable_custom_smoothing_and_pose_selection=ENABLE_CUSTOM_SMOOTHING_AND_POSE_SELECTION,
        robot_name=ROBOT_NAME,
        frame_name=FRAME_NAME,
        program_name=PROGRAM_NAME,
        ik_backend=IK_BACKEND,
        local_parallel_workers=1,
        local_parallel_min_batch_size=999999,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Parallel Y/Z sweep for TARGET_FRAME_A_ORIGIN_IN_FRAME2_MM under local six_axis_ik evaluation."
    )
    parser.add_argument("--mode", choices=("grid", "adaptive"), default="grid")
    parser.add_argument("--x", type=float, default=float(TARGET_FRAME_A_ORIGIN_IN_FRAME2_MM[0]))
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
    parser.add_argument("--min-y", type=float, default=None, help="Hard lower bound for origin Y.")
    parser.add_argument("--max-y", type=float, default=None, help="Hard upper bound for origin Y.")
    parser.add_argument("--min-z", type=float, default=None, help="Hard lower bound for origin Z.")
    parser.add_argument("--max-z", type=float, default=None, help="Hard upper bound for origin Z.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    environment = _build_environment()
    eval_config = EvalConfig(
        strategy=str(args.strategy),
        run_window_repair=not bool(args.skip_window_repair),
        run_inserted_repair=not bool(args.skip_inserted_repair),
    )

    start_ts = perf_counter()
    adaptive_trace: list[dict[str, float | int | str | bool]] = []
    y_values: tuple[float, ...] | None = None
    z_values: tuple[float, ...] | None = None
    if args.mode == "grid":
        all_results_by_key, y_values, z_values = run_grid_sweep(
            x_mm=float(args.x),
            y_values=parse_float_list(args.y_values),
            z_values=parse_float_list(args.z_values),
            environment=environment,
            eval_config=eval_config,
            workers=int(args.workers),
            prefix="[sweep]",
            min_y_mm=None if args.min_y is None else float(args.min_y),
            max_y_mm=None if args.max_y is None else float(args.max_y),
            min_z_mm=None if args.min_z is None else float(args.min_z),
            max_z_mm=None if args.max_z is None else float(args.max_z),
        )
        adaptive_trace = []
    else:
        all_results_by_key, adaptive_trace = run_adaptive_sweep(
            config=AdaptiveSweepConfig(
                x_mm=float(args.x),
                start_y_mm=float(args.start_y),
                start_z_mm=float(args.start_z),
                step_y_mm=float(args.step_y),
                step_z_mm=float(args.step_z),
                min_step_y_mm=float(args.min_step_y),
                min_step_z_mm=float(args.min_step_z),
                max_iters=max(1, int(args.max_iters)),
                include_diagonal=not bool(args.no_diagonal),
                min_y_mm=None if args.min_y is None else float(args.min_y),
                max_y_mm=None if args.max_y is None else float(args.max_y),
                min_z_mm=None if args.min_z is None else float(args.min_z),
                max_z_mm=None if args.max_z is None else float(args.max_z),
            ),
            environment=environment,
            eval_config=eval_config,
            workers=int(args.workers),
            prefix="[adaptive]",
        )

    ordered = tuple(sorted(all_results_by_key.values(), key=lambda item: item.rank_key()))
    best = ordered[0] if ordered else None
    elapsed = perf_counter() - start_ts

    payload: dict[str, object] = {
        "mode": str(args.mode),
        "x_mm": float(args.x),
        "workers": int(args.workers),
        "strategy": eval_config.strategy,
        "run_window_repair": eval_config.run_window_repair,
        "run_inserted_repair": eval_config.run_inserted_repair,
        "elapsed_seconds": elapsed,
        "evaluated_cases": len(ordered),
        "adaptive_trace": adaptive_trace,
        "best": asdict(best) if best is not None else None,
        "results": [asdict(item) for item in ordered],
    }
    if args.mode == "grid":
        payload["y_values"] = y_values if y_values is not None else parse_float_list(args.y_values)
        payload["z_values"] = z_values if z_values is not None else parse_float_list(args.z_values)
    else:
        payload["start_y"] = float(args.start_y)
        payload["start_z"] = float(args.start_z)
        payload["step_y"] = float(args.step_y)
        payload["step_z"] = float(args.step_z)
        payload["min_step_y"] = float(args.min_step_y)
        payload["min_step_z"] = float(args.min_step_z)
        payload["max_iters"] = int(args.max_iters)
        payload["diagonal"] = not bool(args.no_diagonal)
    payload["min_y"] = None if args.min_y is None else float(args.min_y)
    payload["max_y"] = None if args.max_y is None else float(args.max_y)
    payload["min_z"] = None if args.min_z is None else float(args.min_z)
    payload["max_z"] = None if args.max_z is None else float(args.max_z)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[sweep] wrote {args.output}")

    if best is not None:
        print(
            "[sweep] best "
            f"status={best.status}, origin=({best.x_mm:.3f}, {best.y_mm:.3f}, {best.z_mm:.3f}), "
            f"worst={best.worst_joint_step_deg:.3f}, mean={best.mean_joint_step_deg:.6f}, "
            f"switches={best.config_switches}, bridge={best.bridge_like_segments}, "
            f"empty={best.ik_empty_row_count}, time={best.timing_seconds:.3f}s"
        )

    print_result_table(ordered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
