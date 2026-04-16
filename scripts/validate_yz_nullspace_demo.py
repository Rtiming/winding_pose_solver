"""
Run a focused Frame-A Y/Z two-DOF validation and optionally import it to RoboDK.

Default focus is the 372 -> 373 neighborhood because it is the known place where
reachable targets can still produce undesirable physical detours.
"""
from __future__ import annotations

import argparse
import sys
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
    artifacts = run_yz_nullspace_demo(
        APP_RUNTIME_SETTINGS,
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
