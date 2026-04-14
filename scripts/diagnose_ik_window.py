"""
Diagnose a contiguous row window by comparing live RoboDK IK and SixAxisIK.

Usage:
    python scripts/diagnose_ik_window.py --start 395 --end 406 --padding 4

Outputs:
  - Console table for the requested row window plus neighboring rows
  - CSV report under artifacts/diagnostics/
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.core.geometry import _build_pose
from src.ik_collection import _build_seed_joint_strategies, _collect_ik_candidates
from src.core.motion_settings import RoboDKMotionSettings
from src.path_optimizer import _build_optimizer_settings
from src.core.pose_csv import load_pose_rows
from src.core.robot_interface import RoboDKRobotInterface, SixAxisIKRobotInterface
from src.six_axis_ik.config import get_configured_frame_pose


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnose IK behavior around a row window.")
    parser.add_argument("--csv", default=str(PROJECT_ROOT / "data" / "tool_poses_frame2.csv"))
    parser.add_argument("--start", type=int, required=True)
    parser.add_argument("--end", type=int, required=True)
    parser.add_argument("--padding", type=int, default=4)
    parser.add_argument("--robot", default="KUKA")
    parser.add_argument("--frame", default="Frame 2")
    return parser.parse_args()


def _compute_wrist_reach(row: dict[str, float], frame_T: np.ndarray) -> float:
    x = float(row["x_mm"])
    y = float(row["y_mm"])
    z = float(row["z_mm"])
    r13 = float(row["r13"])
    r23 = float(row["r23"])
    r33 = float(row["r33"])

    pos_base = frame_T @ np.array([x, y, z, 1.0], dtype=float)
    flange_z_base = frame_T[:3, :3] @ np.array([r13, r23, r33], dtype=float)
    wrist = pos_base[:3] - (319.5 + 90.0) * flange_z_base
    wx, wy, wz = wrist
    radial = float(np.hypot(wx, wy))
    return float(np.hypot(radial - 150.0, wz - 443.5))


def _collect_counts(
    rows: list[dict[str, float]],
    *,
    robot_name: str,
    frame_name: str,
) -> list[dict[str, object]]:
    from robodk.robolink import Robolink
    from robodk.robomath import Mat

    settings = RoboDKMotionSettings()
    optimizer_settings = _build_optimizer_settings(6, settings)
    reach_frame_T = get_configured_frame_pose()

    rdk = Robolink()
    robot_item = rdk.Item(robot_name)
    frame_item = rdk.Item(frame_name)
    if not robot_item.Valid():
        raise RuntimeError(f"Robot '{robot_name}' not found in RoboDK station.")
    if not frame_item.Valid():
        raise RuntimeError(f"Frame '{frame_name}' not found in RoboDK station.")

    robot_item.setPoseFrame(frame_item)
    tool_pose = robot_item.PoseTool()
    reference_pose = robot_item.PoseFrame()

    robot_rdk = RoboDKRobotInterface(robot_item)
    robot_six = SixAxisIKRobotInterface(robodk_robot=robot_item)

    lower_raw, upper_raw, _ = robot_rdk.JointLimits()
    lower_limits = tuple(float(v) for v in lower_raw.list()[:6])
    upper_limits = tuple(float(v) for v in upper_raw.list()[:6])

    seeds_rdk = _build_seed_joint_strategies(
        robot=robot_rdk,
        lower_limits=lower_limits,
        upper_limits=upper_limits,
        joint_count=6,
    )
    seeds_six = _build_seed_joint_strategies(
        robot=robot_six,
        lower_limits=lower_limits,
        upper_limits=upper_limits,
        joint_count=6,
    )

    records: list[dict[str, object]] = []
    for row_index, row in enumerate(rows):
        pose = _build_pose(row, Mat)

        rdk_raw = robot_rdk.SolveIK_All(pose, tool_pose, reference_pose)
        six_raw = robot_six.SolveIK_All(pose, tool_pose, reference_pose)

        rdk_filtered = _collect_ik_candidates(
            robot_rdk,
            pose,
            tool_pose=tool_pose,
            reference_pose=reference_pose,
            lower_limits=lower_limits,
            upper_limits=upper_limits,
            seed_joints=seeds_rdk,
            joint_count=6,
            optimizer_settings=optimizer_settings,
            a1_lower_deg=settings.a1_min_deg,
            a1_upper_deg=settings.a1_max_deg,
            a2_max_deg=settings.a2_max_deg,
            joint_constraint_tolerance_deg=settings.joint_constraint_tolerance_deg,
        )
        six_filtered = _collect_ik_candidates(
            robot_six,
            pose,
            tool_pose=tool_pose,
            reference_pose=reference_pose,
            lower_limits=lower_limits,
            upper_limits=upper_limits,
            seed_joints=seeds_six,
            joint_count=6,
            optimizer_settings=optimizer_settings,
            a1_lower_deg=settings.a1_min_deg,
            a1_upper_deg=settings.a1_max_deg,
            a2_max_deg=settings.a2_max_deg,
            joint_constraint_tolerance_deg=settings.joint_constraint_tolerance_deg,
        )

        reach_mm = _compute_wrist_reach(row, reach_frame_T)
        excess_mm = reach_mm - 1690.0
        records.append(
            {
                "row_index": row_index,
                "row_label": int(row.get("index", row_index)),
                "x_mm": float(row["x_mm"]),
                "y_mm": float(row["y_mm"]),
                "z_mm": float(row["z_mm"]),
                "rdk_raw_count": len(rdk_raw),
                "rdk_filtered_count": len(rdk_filtered),
                "six_raw_count": len(six_raw),
                "six_filtered_count": len(six_filtered),
                "reach_mm": reach_mm,
                "reach_excess_mm": excess_mm,
            }
        )
    return records


def _print_report(records: list[dict[str, object]], *, focus_start: int, focus_end: int) -> None:
    print(
        "row  label   rdk_raw/rdk_ok  six_raw/six_ok   reach_mm  excess_mm  focus"
    )
    print(
        "---- ------  --------------  --------------  ---------  ---------  -----"
    )
    for record in records:
        row_index = int(record["row_index"])
        focus = "*" if focus_start <= row_index <= focus_end else ""
        print(
            f"{row_index:>4} "
            f"{int(record['row_label']):>6}  "
            f"{int(record['rdk_raw_count']):>3}/{int(record['rdk_filtered_count']):<3}           "
            f"{int(record['six_raw_count']):>3}/{int(record['six_filtered_count']):<3}        "
            f"{float(record['reach_mm']):>9.1f}  "
            f"{float(record['reach_excess_mm']):>9.1f}  "
            f"{focus}"
        )


def _write_csv_report(
    records: list[dict[str, object]],
    *,
    start: int,
    end: int,
    padding: int,
) -> Path:
    output_dir = PROJECT_ROOT / "artifacts" / "diagnostics"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"ik_window_{start}_{end}_pad{padding}.csv"
    fieldnames = list(records[0].keys()) if records else []
    with output_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)
    return output_path


def main() -> int:
    args = _parse_args()
    pose_rows = load_pose_rows(args.csv)
    if args.start < 0 or args.end >= len(pose_rows) or args.start > args.end:
        raise ValueError(
            f"Invalid window [{args.start}, {args.end}] for {len(pose_rows)} pose rows."
        )

    window_start = max(0, args.start - args.padding)
    window_end = min(len(pose_rows) - 1, args.end + args.padding)
    indexed_rows = pose_rows[window_start : window_end + 1]
    records = _collect_counts(indexed_rows, robot_name=args.robot, frame_name=args.frame)
    for offset, record in enumerate(records):
        record["row_index"] = window_start + offset
        record["row_label"] = int(indexed_rows[offset].get("index", window_start + offset))

    _print_report(records, focus_start=args.start, focus_end=args.end)
    csv_path = _write_csv_report(records, start=args.start, end=args.end, padding=args.padding)
    print(f"\nWrote report: {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
