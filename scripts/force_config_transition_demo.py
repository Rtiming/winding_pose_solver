"""
Force a visible IK config-family transition and optionally import it to RoboDK.

This is a diagnostic viewer for the "configuration changes but target points stay
fixed" question.  It uses the same main.py runtime settings and target Frame-A
origin, then constrains IK candidates around a selected cut.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from main import APP_RUNTIME_SETTINGS, FRAME_NAME, ROBOT_NAME
from src.robodk_runtime.result_import import import_profile_result_to_robodk
from src.runtime.config_transition_demo import (
    default_config_transition_demo_run_id,
    parse_config_family,
    run_forced_config_transition_demo,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Force a visible IK config-family transition for RoboDK inspection."
    )
    parser.add_argument("--window-start", type=int, default=365)
    parser.add_argument("--window-end", type=int, default=381)
    parser.add_argument("--switch-after", type=int, default=372)
    parser.add_argument("--left-family", default="0,0,1")
    parser.add_argument("--right-family", default="0,0,0")
    parser.add_argument("--run-id", default=None)
    parser.add_argument(
        "--output-root",
        default=str(PROJECT_ROOT / "artifacts" / "diagnostics" / "config_transition_demo"),
    )
    parser.add_argument("--skip-refresh-csv", action="store_true")
    parser.add_argument("--import-to-robodk", action="store_true")
    parser.add_argument("--robot", default=ROBOT_NAME)
    parser.add_argument("--frame", default=FRAME_NAME)
    parser.add_argument("--prefix", default="CFGTRANS")
    parser.add_argument("--program-name", default=None)
    parser.add_argument("--keep-existing", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    run_id = args.run_id or default_config_transition_demo_run_id()
    left_family = parse_config_family(args.left_family)
    right_family = parse_config_family(args.right_family)
    artifacts = run_forced_config_transition_demo(
        APP_RUNTIME_SETTINGS,
        window_start_index=args.window_start,
        window_end_index=args.window_end,
        switch_after_index=args.switch_after,
        left_family=left_family,
        right_family=right_family,
        output_root=args.output_root,
        run_id=run_id,
        refresh_csv=not args.skip_refresh_csv,
    )

    print()
    print("=== Forced config transition demo ===")
    print(
        f"status={artifacts.result.status}, "
        f"selected={len(artifacts.result.selected_path)}/{len(artifacts.result.row_labels)}, "
        f"config_switches={artifacts.result.config_switches}, "
        f"bridge_like={artifacts.result.bridge_like_segments}, "
        f"worst_step={artifacts.result.worst_joint_step_deg:.3f} deg"
    )
    print(
        f"window={artifacts.window_start_index}..{artifacts.window_end_index}, "
        f"switch_after={artifacts.switch_after_index}, "
        f"family={artifacts.left_family}->{artifacts.right_family}"
    )
    print(f"artifacts: {artifacts.output_dir}")
    print(f"summary: {artifacts.summary_path}")
    print(f"result: {artifacts.result_path}")

    if not args.import_to_robodk:
        print()
        print(
            "RoboDK import command: "
            f"python scripts/import_profile_result_to_robodk.py "
            f"--result {artifacts.result_path} "
            f"--robot \"{args.robot}\" --frame \"{args.frame}\" "
            f"--prefix {args.prefix} "
            f"--focus-start-label {args.switch_after} "
            f"--focus-end-label {args.switch_after + 1}"
        )
        return 0

    summary = import_profile_result_to_robodk(
        artifacts.result,
        robot_name=args.robot,
        frame_name=args.frame,
        program_name=args.program_name,
        prefix=args.prefix,
        clear_prefix=not args.keep_existing,
        create_program=True,
        create_cartesian_markers=True,
        focus_start_label=str(args.switch_after),
        focus_end_label=str(args.switch_after + 1),
    )
    print(
        "RoboDK import complete: "
        f"program={summary.program_name}, "
        f"markers={summary.marker_count}, "
        f"program_targets={summary.program_target_count}, "
        f"prefix={summary.prefix}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
