from __future__ import annotations

import argparse
from pathlib import Path

from app_settings import build_app_runtime_settings
from src.app_runner import run_robodk_program_generation, run_visualization


# ---------------------------------------------------------------------------
# Main business parameters
# Keep the most frequently changed project inputs here.
# Detailed optimizer and RoboDK tuning lives in `app_settings.py`.
# ---------------------------------------------------------------------------

VALIDATION_CENTERLINE_CSV = Path("data/validation_centerline.csv")
TOOL_POSES_FRAME2_CSV = Path("data/tool_poses_frame2.csv")

TARGET_FRAME_A_ORIGIN_IN_FRAME2_MM = (1126.0, -400.0, 1200.0)
TARGET_FRAME_A_ROTATION_IN_FRAME2_XYZ_DEG = (-180.0, -14.0, -180.0)

ENABLE_CUSTOM_SMOOTHING_AND_POSE_SELECTION = True
ROBOT_NAME = "KUKA"
FRAME_NAME = "Frame 2"
PROGRAM_NAME = "Path_From_CSV"


APP_RUNTIME_SETTINGS = build_app_runtime_settings(
    validation_centerline_csv=VALIDATION_CENTERLINE_CSV,
    tool_poses_frame2_csv=TOOL_POSES_FRAME2_CSV,
    target_frame_origin_mm=TARGET_FRAME_A_ORIGIN_IN_FRAME2_MM,
    target_frame_rotation_xyz_deg=TARGET_FRAME_A_ROTATION_IN_FRAME2_XYZ_DEG,
    enable_custom_smoothing_and_pose_selection=ENABLE_CUSTOM_SMOOTHING_AND_POSE_SELECTION,
    robot_name=ROBOT_NAME,
    frame_name=FRAME_NAME,
    program_name=PROGRAM_NAME,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Solve the tool poses first, then generate a RoboDK program. "
            "Use --visualize to inspect only the centerline frames."
        )
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Visualize the centerline and process frames without generating a RoboDK program.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.visualize:
        label = "centerline visualization"
        action = run_visualization
    else:
        label = "RoboDK program generation"
        action = run_robodk_program_generation

    print(f"Running: {label}")
    try:
        action(APP_RUNTIME_SETTINGS)
    except KeyboardInterrupt:
        print("\nCancelled.")
        return 1
    except Exception as exc:
        print(f"Error: {exc}")
        return 1

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
