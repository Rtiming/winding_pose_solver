"""
Import a diagnostic row window into RoboDK as Cartesian targets for visual inspection.

Usage:
    python scripts/import_ik_window_to_robodk.py --start 395 --end 406 --padding 4

The created targets are named with an `IKDIAG_` prefix so they can be cleaned up and
recognized easily inside the station tree.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.core.geometry import _build_pose
from src.core.pose_csv import load_pose_rows


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Import a row window into RoboDK as targets.")
    parser.add_argument("--csv", default=str(PROJECT_ROOT / "data" / "tool_poses_frame2.csv"))
    parser.add_argument("--start", type=int, required=True)
    parser.add_argument("--end", type=int, required=True)
    parser.add_argument("--padding", type=int, default=4)
    parser.add_argument("--robot", default="KUKA")
    parser.add_argument("--frame", default="Frame 2")
    parser.add_argument("--prefix", default="IKDIAG")
    parser.add_argument("--clear-prefix", action="store_true", default=True)
    return parser.parse_args()


def _try_set_color(target, rgba: list[float]) -> None:
    if hasattr(target, "setColor"):
        try:
            target.setColor(rgba)
        except Exception:
            pass


def main() -> int:
    args = _parse_args()

    pose_rows = load_pose_rows(args.csv)
    if args.start < 0 or args.end >= len(pose_rows) or args.start > args.end:
        raise ValueError(
            f"Invalid window [{args.start}, {args.end}] for {len(pose_rows)} pose rows."
        )

    window_start = max(0, args.start - args.padding)
    window_end = min(len(pose_rows) - 1, args.end + args.padding)

    from robodk.robolink import Robolink, ITEM_TYPE_FRAME, ITEM_TYPE_ROBOT, ITEM_TYPE_TARGET
    from robodk.robomath import Mat

    rdk = Robolink()
    robot = rdk.Item(args.robot, ITEM_TYPE_ROBOT)
    frame = rdk.Item(args.frame, ITEM_TYPE_FRAME)
    if not robot.Valid():
        raise RuntimeError(f"Robot '{args.robot}' not found in RoboDK station.")
    if not frame.Valid():
        raise RuntimeError(f"Frame '{args.frame}' not found in RoboDK station.")

    if args.clear_prefix:
        try:
            target_names = rdk.ItemList(ITEM_TYPE_TARGET, True)
        except Exception:
            target_names = []
        for target_name in target_names:
            if not isinstance(target_name, str) or not target_name.startswith(args.prefix + "_"):
                continue
            target = rdk.Item(target_name, ITEM_TYPE_TARGET)
            if target.Valid():
                target.Delete()

    robot.setPoseFrame(frame)

    print(f"Importing rows {window_start}..{window_end} into RoboDK frame '{args.frame}'")
    for row_index in range(window_start, window_end + 1):
        row = pose_rows[row_index]
        focus = "FOCUS" if args.start <= row_index <= args.end else "CTX"
        target_name = f"{args.prefix}_{focus}_{row_index:03d}"

        existing = rdk.Item(target_name, ITEM_TYPE_TARGET)
        if existing.Valid():
            existing.Delete()

        target = rdk.AddTarget(target_name, frame, robot)
        target.setRobot(robot)
        target.setAsCartesianTarget()
        target.setPose(_build_pose(row, Mat))

        if args.start <= row_index <= args.end:
            _try_set_color(target, [1.0, 0.2, 0.2, 1.0])
        else:
            _try_set_color(target, [0.2, 0.8, 0.2, 1.0])

        print(f"  added {target_name}")

    print("Done. The diagnostic targets are visible under Frame 2 in RoboDK.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
