"""
Per-point IK solvability comparison: RoboDK backend vs SixAxisIK backend.

Usage (requires RoboDK to be running with the station open):
    python scripts/compare_ik_backends.py

Output:
  - Per-row table showing which rows each backend fails to solve
  - Summary statistics
  - List of rows solvable by one backend but not the other
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.ik_collection import _build_pose, _collect_ik_candidates, _build_seed_joint_strategies
from src.path_optimizer import _build_optimizer_settings
from src.core.motion_settings import RoboDKMotionSettings
from src.core.pose_csv import load_pose_rows


def _test_robodk_backend(tool_pose, reference_pose, rows, settings):
    """Collect per-row IK candidate counts using the RoboDK backend.
    Requires a live RoboDK station."""
    from robodk.robolink import Robolink
    from robodk.robomath import Mat
    from src.core.robot_interface import RoboDKRobotInterface

    rdk = Robolink()
    robot_item = rdk.Item("KUKA")
    frame_item = rdk.Item("Frame 2")
    if not robot_item.Valid():
        raise RuntimeError("Robot 'KUKA' not found in RoboDK station.")
    if not frame_item.Valid():
        raise RuntimeError("Frame 'Frame 2' not found in RoboDK station.")

    ri = RoboDKRobotInterface(robot_item)
    opt_settings = _build_optimizer_settings(6, settings)
    lower_raw, upper_raw, _ = ri.JointLimits()
    lower = tuple(float(v) for v in lower_raw.list()[:6])
    upper = tuple(float(v) for v in upper_raw.list()[:6])
    seeds = _build_seed_joint_strategies(robot=ri, lower_limits=lower, upper_limits=upper, joint_count=6)

    tool_pose_mat = Mat(tool_pose.tolist())
    ref_pose_mat = Mat(reference_pose.tolist())

    counts = []
    t0 = time.perf_counter()
    for i, row in enumerate(rows):
        pose = _build_pose(row, Mat)
        cands = _collect_ik_candidates(
            ri, pose,
            tool_pose=tool_pose_mat, reference_pose=ref_pose_mat,
            lower_limits=lower, upper_limits=upper,
            seed_joints=seeds, joint_count=6,
            optimizer_settings=opt_settings,
            a1_lower_deg=settings.a1_min_deg, a1_upper_deg=settings.a1_max_deg,
            a2_max_deg=settings.a2_max_deg,
            joint_constraint_tolerance_deg=settings.joint_constraint_tolerance_deg,
        )
        counts.append(len(cands))
        if (i + 1) % 50 == 0:
            elapsed = time.perf_counter() - t0
            print(f"  RoboDK: {i+1}/{len(rows)} rows done ({elapsed:.1f}s)")

    return counts


def _test_sixaxisik_backend(tool_pose, reference_pose, rows, settings):
    """Collect per-row IK candidate counts using the SixAxisIK backend (no RoboDK needed)."""
    from robodk.robomath import Mat
    from src.core.robot_interface import SixAxisIKRobotInterface

    ri = SixAxisIKRobotInterface(robodk_robot=None)
    opt_settings = _build_optimizer_settings(6, settings)
    lower_raw, upper_raw, _ = ri.JointLimits()
    lower = tuple(float(v) for v in lower_raw.list()[:6])
    upper = tuple(float(v) for v in upper_raw.list()[:6])
    seeds = _build_seed_joint_strategies(robot=ri, lower_limits=lower, upper_limits=upper, joint_count=6)

    tool_pose_mat = Mat(tool_pose.tolist())
    ref_pose_mat = Mat(reference_pose.tolist())

    counts = []
    t0 = time.perf_counter()
    for i, row in enumerate(rows):
        pose = _build_pose(row, Mat)
        cands = _collect_ik_candidates(
            ri, pose,
            tool_pose=tool_pose_mat, reference_pose=ref_pose_mat,
            lower_limits=lower, upper_limits=upper,
            seed_joints=seeds, joint_count=6,
            optimizer_settings=opt_settings,
            a1_lower_deg=settings.a1_min_deg, a1_upper_deg=settings.a1_max_deg,
            a2_max_deg=settings.a2_max_deg,
            joint_constraint_tolerance_deg=settings.joint_constraint_tolerance_deg,
        )
        counts.append(len(cands))
        if (i + 1) % 50 == 0:
            elapsed = time.perf_counter() - t0
            print(f"  SixAxisIK: {i+1}/{len(rows)} rows done ({elapsed:.1f}s)")

    return counts


def _compute_wrist_reach(row, frame_T, tool_length_mm=319.5, wrist_to_flange_mm=90.0):
    """Compute wrist-center-to-J2-axis distance (arm reach metric)."""
    import numpy as np
    x = float(row.get('x_mm', 0))
    y = float(row.get('y_mm', 0))
    z = float(row.get('z_mm', 0))
    r13 = float(row.get('r13', 0))
    r23 = float(row.get('r23', 0))
    r33 = float(row.get('r33', 0))

    pos_base = frame_T @ np.array([x, y, z, 1.0])
    flange_z_frame = np.array([r13, r23, r33])
    flange_z_base = frame_T[:3, :3] @ flange_z_frame

    # Wrist center = TCP - (tool_length + wrist_to_flange) * flange_Z
    wrist = pos_base[:3] - (tool_length_mm + wrist_to_flange_mm) * flange_z_base
    Wx, Wy, Wz = wrist
    return float(((Wx**2 + Wy**2)**0.5 - 150.0)**2 + (Wz - 443.5)**2)**0.5


def main() -> int:
    import numpy as np
    from src.six_axis_ik.config import get_configured_tool_pose, get_configured_frame_pose

    settings = RoboDKMotionSettings()
    tool_T = get_configured_tool_pose()
    frame_T = get_configured_frame_pose()

    print(f"Loading pose rows from data/tool_poses_frame2.csv ...")
    rows = load_pose_rows(str(PROJECT_ROOT / "data" / "tool_poses_frame2.csv"))
    print(f"  {len(rows)} rows loaded.")

    # ── SixAxisIK backend (always available) ──────────────────────────────
    print("\n[1/2] Testing SixAxisIK backend ...")
    t0 = time.perf_counter()
    six_counts = _test_sixaxisik_backend(tool_T, frame_T, rows, settings)
    t_six = time.perf_counter() - t0
    six_empty = [i for i, c in enumerate(six_counts) if c == 0]
    print(f"  Done in {t_six:.1f}s  |  empty rows: {len(six_empty)}")

    # ── RoboDK backend (requires live station) ─────────────────────────────
    print("\n[2/2] Testing RoboDK backend (requires open RoboDK station) ...")
    try:
        t0 = time.perf_counter()
        rdk_counts = _test_robodk_backend(tool_T, frame_T, rows, settings)
        t_rdk = time.perf_counter() - t0
        rdk_empty = [i for i, c in enumerate(rdk_counts) if c == 0]
        print(f"  Done in {t_rdk:.1f}s  |  empty rows: {len(rdk_empty)}")
    except Exception as exc:
        print(f"  RoboDK backend not available: {exc}")
        rdk_counts = None
        rdk_empty = None

    # ── Comparison report ─────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("COMPARISON REPORT")
    print("=" * 70)

    print(f"\nSixAxisIK backend:  {len(rows) - len(six_empty)}/{len(rows)} rows solved "
          f"({len(six_empty)} empty)")
    if six_empty:
        print(f"  Empty rows: {six_empty[0]}–{six_empty[-1]} ({len(six_empty)} rows)")

    if rdk_counts is not None:
        print(f"RoboDK backend:     {len(rows) - len(rdk_empty)}/{len(rows)} rows solved "
              f"({len(rdk_empty)} empty)")
        if rdk_empty:
            print(f"  Empty rows: {rdk_empty[0]}–{rdk_empty[-1]} ({len(rdk_empty)} rows)")

        # Rows where backends disagree
        only_in_six = [i for i in six_empty if i not in rdk_empty]
        only_in_rdk = [i for i in rdk_empty if i not in six_empty]
        both_empty = [i for i in six_empty if i in rdk_empty]

        print(f"\nRows that ONLY SixAxisIK fails (RoboDK succeeds): {len(only_in_six)}")
        if only_in_six:
            print(f"  {only_in_six}")

        print(f"Rows that ONLY RoboDK fails (SixAxisIK succeeds): {len(only_in_rdk)}")
        if only_in_rdk:
            print(f"  Rows {only_in_rdk[0]}–{only_in_rdk[-1]} ({len(only_in_rdk)} rows)")

        print(f"Rows both backends fail: {len(both_empty)}")
        if both_empty:
            print(f"  Rows {both_empty[0]}–{both_empty[-1]} ({len(both_empty)} rows)")

    # ── Reach analysis for empty rows ─────────────────────────────────────
    print(f"\nGeometric reach analysis for SixAxisIK empty rows:")
    print(f"  (Arm reach limit = 810 + 880 = 1690 mm from J2 axis)")
    print(f"  {'Row':>5}  {'Reach mm':>10}  {'Excess mm':>10}  {'Status':>12}")
    for i in six_empty[:20]:
        reach = _compute_wrist_reach(rows[i], frame_T)
        excess = reach - 1690.0
        status = "UNREACHABLE" if reach > 1690.0 else "ok?"
        print(f"  {i:>5}  {reach:>10.1f}  {excess:>10.1f}  {status:>12}")
    if len(six_empty) > 20:
        print(f"  ... ({len(six_empty) - 20} more rows omitted)")

    print("\n" + "=" * 70)
    print(f"CONCLUSION:")
    if six_empty:
        max_reach = max(_compute_wrist_reach(rows[i], frame_T) for i in six_empty)
        min_reach = min(_compute_wrist_reach(rows[i], frame_T) for i in six_empty)
        print(f"  {len(six_empty)} rows are geometrically unreachable (rows {six_empty[0]}–{six_empty[-1]}).")
        print(f"  Wrist reach range: {min_reach:.0f}–{max_reach:.0f} mm (limit: 1690 mm).")
        print(f"  Maximum excess: {max_reach - 1690:.0f} mm — requires robot repositioning or shorter tool.")
    else:
        print("  All rows are solvable with the SixAxisIK backend.")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
