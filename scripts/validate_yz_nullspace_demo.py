"""
Run a focused Frame-A Y/Z two-DOF validation and optionally import it to RoboDK.

Default focus is the 372 -> 373 neighborhood because it is the known place where
reachable targets can still produce undesirable physical detours.
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from main import APP_RUNTIME_SETTINGS, FRAME_NAME, ROBOT_NAME
from src.robodk_runtime.result_import import import_profile_result_to_robodk
from src.runtime.yz_nullspace_demo import (
    default_yz_nullspace_demo_run_id,
    run_yz_nullspace_demo,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate the project-level null-space idea: keep the focused target "
            "rows constrained, then use Frame-A Y/Z offsets as two free variables "
            "to improve IK continuity."
        )
    )
    parser.add_argument("--start", type=int, default=372, help="Focused pose-row start index.")
    parser.add_argument("--end", type=int, default=373, help="Focused pose-row end index.")
    parser.add_argument("--padding", type=int, default=8, help="Context rows on each side.")
    parser.add_argument(
        "--bridge-trigger-deg",
        type=float,
        default=None,
        help=(
            "Optional diagnostic override for bridge/problem-segment trigger. "
            "Use a lower value to stress-test whether Y/Z repair engages."
        ),
    )
    parser.add_argument("--run-id", default=None)
    parser.add_argument(
        "--local-workers",
        type=int,
        default=None,
        help="Override offline exact-profile worker count for this validation run.",
    )
    parser.add_argument(
        "--local-min-batch-size",
        type=int,
        default=None,
        help="Override minimum exact-profile batch size before multiprocessing engages.",
    )
    parser.add_argument(
        "--envelope-schedule",
        default=None,
        help="Comma-separated Frame-A Y/Z envelope schedule in mm, e.g. 6,12,24.",
    )
    parser.add_argument(
        "--step-schedule",
        default=None,
        help="Comma-separated Frame-A Y/Z step schedule in mm, e.g. 8,4,2.",
    )
    parser.add_argument(
        "--insertion-counts",
        default=None,
        help="Comma-separated inserted transition counts, e.g. 4,8,16.",
    )
    parser.add_argument(
        "--disable-joint-space-bridge",
        action="store_true",
        help="Disable the final joint-space bridge fallback for residual wrist flips.",
    )
    parser.add_argument(
        "--joint-space-bridge-max-insertions",
        type=int,
        default=None,
        help="Maximum joint-space bridge points inserted per residual segment.",
    )
    parser.add_argument(
        "--joint-space-bridge-max-tcp-deviation-mm",
        type=float,
        default=None,
        help="Maximum allowed TCP deviation before joint-space bridge fallback is rejected.",
    )
    parser.add_argument(
        "--joint-space-bridge-max-tcp-path-ratio",
        type=float,
        default=None,
        help="Maximum allowed TCP path/direct-distance ratio for joint-space bridge fallback.",
    )
    parser.add_argument(
        "--output-root",
        default=str(PROJECT_ROOT / "artifacts" / "diagnostics" / "yz_nullspace_demo"),
    )
    parser.add_argument(
        "--skip-refresh-csv",
        action="store_true",
        help="Use the existing data/tool_poses_frame2.csv instead of regenerating it.",
    )
    parser.add_argument(
        "--import-to-robodk",
        action="store_true",
        help="After offline validation, import the optimized path into the live RoboDK station.",
    )
    parser.add_argument("--robot", default=ROBOT_NAME)
    parser.add_argument("--frame", default=FRAME_NAME)
    parser.add_argument("--prefix", default="YZNS")
    parser.add_argument("--program-name", default=None)
    parser.add_argument("--no-program", action="store_true")
    parser.add_argument("--no-cartesian-markers", action="store_true")
    parser.add_argument("--keep-existing", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    run_id = args.run_id or default_yz_nullspace_demo_run_id()
    settings = _settings_with_overrides(args)
    artifacts = run_yz_nullspace_demo(
        settings,
        start_index=args.start,
        end_index=args.end,
        padding=args.padding,
        output_root=args.output_root,
        run_id=run_id,
        refresh_csv=not args.skip_refresh_csv,
        bridge_trigger_joint_delta_deg=args.bridge_trigger_deg,
    )

    print()
    print("=== Y/Z null-space validation ===")
    _print_result_line("baseline", artifacts.baseline_result)
    _print_result_line("optimized", artifacts.optimized_result)
    print(f"window: pose-row {artifacts.window_start_index}..{artifacts.window_end_index}")
    print(f"focus labels: {artifacts.focus_start_label}..{artifacts.focus_end_label}")
    print(f"artifacts: {artifacts.output_dir}")
    print(f"summary: {artifacts.summary_path}")
    print(f"optimized pose CSV: {artifacts.optimized_pose_csv}")
    print(f"optimized joint CSV: {artifacts.optimized_joint_path_csv}")

    if not args.import_to_robodk:
        print()
        print(
            "RoboDK import command: "
            f"python scripts/import_profile_result_to_robodk.py "
            f"--result {artifacts.optimized_result_path} "
            f"--robot \"{args.robot}\" --frame \"{args.frame}\" "
            f"--prefix {args.prefix} "
            f"--focus-start-label {artifacts.focus_start_label} "
            f"--focus-end-label {artifacts.focus_end_label}"
        )
        return 0

    print()
    print("Importing optimized path into RoboDK...")
    import_summary = import_profile_result_to_robodk(
        artifacts.optimized_result,
        robot_name=args.robot,
        frame_name=args.frame,
        program_name=args.program_name,
        prefix=args.prefix,
        clear_prefix=not args.keep_existing,
        create_program=not args.no_program,
        create_cartesian_markers=not args.no_cartesian_markers,
        focus_start_label=artifacts.focus_start_label,
        focus_end_label=artifacts.focus_end_label,
    )
    print(
        "RoboDK import complete: "
        f"program={import_summary.program_name}, "
        f"markers={import_summary.marker_count}, "
        f"program_targets={import_summary.program_target_count}, "
        f"frame={import_summary.frame_name}"
    )
    return 0


def _settings_with_overrides(args: argparse.Namespace):
    motion_settings = APP_RUNTIME_SETTINGS.motion_settings
    updates = {}
    if args.local_workers is not None:
        updates["local_parallel_workers"] = int(args.local_workers)
    if args.local_min_batch_size is not None:
        updates["local_parallel_min_batch_size"] = int(args.local_min_batch_size)
    if args.envelope_schedule:
        updates["frame_a_origin_yz_envelope_schedule_mm"] = _parse_float_tuple(
            args.envelope_schedule
        )
    if args.step_schedule:
        updates["frame_a_origin_yz_step_schedule_mm"] = _parse_float_tuple(
            args.step_schedule
        )
    if args.insertion_counts:
        updates["frame_a_origin_yz_insertion_counts"] = tuple(
            int(value) for value in _parse_float_tuple(args.insertion_counts)
        )
    if args.disable_joint_space_bridge:
        updates["enable_joint_space_bridge_repair"] = False
    if args.joint_space_bridge_max_insertions is not None:
        updates["joint_space_bridge_max_insertions_per_segment"] = int(
            args.joint_space_bridge_max_insertions
        )
    if args.joint_space_bridge_max_tcp_deviation_mm is not None:
        updates["joint_space_bridge_max_tcp_deviation_mm"] = float(
            args.joint_space_bridge_max_tcp_deviation_mm
        )
    if args.joint_space_bridge_max_tcp_path_ratio is not None:
        updates["joint_space_bridge_max_tcp_path_ratio"] = float(
            args.joint_space_bridge_max_tcp_path_ratio
        )
    if updates:
        motion_settings = replace(motion_settings, **updates)
        return replace(APP_RUNTIME_SETTINGS, motion_settings=motion_settings)
    return APP_RUNTIME_SETTINGS


def _parse_float_tuple(raw: str) -> tuple[float, ...]:
    values = tuple(float(part.strip()) for part in raw.split(",") if part.strip())
    if not values:
        raise ValueError(f"Expected at least one comma-separated value, got {raw!r}")
    return values


def _print_result_line(label: str, result) -> None:
    print(
        f"{label}: status={result.status}, "
        f"ik_empty={result.ik_empty_row_count}, "
        f"config_switches={result.config_switches}, "
        f"bridge_like={result.bridge_like_segments}, "
        f"worst_step={result.worst_joint_step_deg:.3f} deg, "
        f"selected={len(result.selected_path)}/{len(result.row_labels)}"
    )


if __name__ == "__main__":
    raise SystemExit(main())
