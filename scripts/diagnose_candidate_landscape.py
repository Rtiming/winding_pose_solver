"""
Offline IK candidate landscape diagnostic (no RoboDK needed).

Loads all pose rows, runs SixAxisIK on each, and produces:
  1. Per-row candidate counts and config_flags distribution
  2. Best corridor analysis: which single config family has the
     longest unbroken run with small joint steps
  3. Per-row minimum joint step to same-config next row
  4. Transition analysis: where config families break and why

Usage:
    python scripts/diagnose_candidate_landscape.py
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import math
from collections import Counter

import numpy as np


def _mat_from_dict(row: dict, mat_type) -> object:
    from src.core.geometry import _build_pose
    return _build_pose(row, mat_type)


def main() -> None:
    from src.core.robot_interface import SixAxisIKRobotInterface, _compute_kuka_config_flags
    from src.core.pose_csv import load_pose_rows
    from src.core.motion_settings import RoboDKMotionSettings
    from src.search.path_optimizer import _build_optimizer_settings
    from src.six_axis_ik.config import get_configured_tool_pose, get_configured_frame_pose

    try:
        from robodk.robomath import Mat
        mat_type = Mat
    except ImportError:
        # Fallback: use a minimal 4x4 matrix class
        from src.core.simple_mat import SimpleMat as mat_type  # type: ignore

    print(f"Loading pose rows ...")
    rows = list(load_pose_rows(str(PROJECT_ROOT / "data" / "tool_poses_frame2.csv")))
    print(f"  {len(rows)} rows loaded.\n")

    settings = RoboDKMotionSettings(
        a1_min_deg=-150.0,
        a1_max_deg=30.0,
        a2_max_deg=125.0,
        joint_constraint_tolerance_deg=1e-6,
    )

    robot = SixAxisIKRobotInterface(robodk_robot=None)
    opt_settings = _build_optimizer_settings(6, settings)
    lower_raw, upper_raw, _ = robot.JointLimits()
    lower = tuple(float(v) for v in lower_raw.list()[:6])
    upper = tuple(float(v) for v in upper_raw.list()[:6])
    model = robot._model

    tool_T_np = get_configured_tool_pose()
    frame_T_np = get_configured_frame_pose()

    # Build Mat wrappers
    tool_mat = mat_type(tool_T_np.tolist())
    frame_mat = mat_type(frame_T_np.tolist())

    print("Running SolveIK_All for all rows (SixAxisIK, no RoboDK) ...")

    per_row_candidates: list[list[tuple[float, ...]]] = []
    per_row_configs: list[list[tuple[int, int, int]]] = []

    A1_MIN = settings.a1_min_deg
    A1_MAX = settings.a1_max_deg
    A2_MAX = settings.a2_max_deg
    TOL = settings.joint_constraint_tolerance_deg

    for row_index, row in enumerate(rows):
        pose = _mat_from_dict(row, mat_type)
        all_sols = robot.SolveIK_All(pose, tool_mat, frame_mat)

        # Filter: joint limits + user constraints
        valid_joints = []
        valid_configs = []
        for sol in all_sols:
            j = tuple(float(v) for v in sol[:6])
            # Within joint limits
            if any(j[i] < lower[i] - TOL or j[i] > upper[i] + TOL for i in range(6)):
                continue
            # A1 constraint
            if not (A1_MIN - TOL <= j[0] <= A1_MAX + TOL):
                continue
            # A2 constraint
            if abs(j[1]) > A2_MAX + TOL:
                continue
            flags = _compute_kuka_config_flags(list(j), model)
            valid_joints.append(j)
            valid_configs.append(flags)

        per_row_candidates.append(valid_joints)
        per_row_configs.append(valid_configs)

        if (row_index + 1) % 50 == 0:
            print(f"  {row_index+1}/{len(rows)} rows processed ...")

    print(f"\n  Done. Total rows: {len(rows)}")

    # ── Summary ─────────────────────────────────────────────────────────────
    empty_rows = [i for i, cands in enumerate(per_row_candidates) if not cands]
    print(f"\nEmpty rows (no solution after filtering): {len(empty_rows)}")
    if empty_rows:
        print(f"  Rows: {empty_rows[:20]}{'...' if len(empty_rows) > 20 else ''}")

    # Config flag distribution across all solutions
    all_flags_flat = [f for row_flags in per_row_configs for f in row_flags]
    flag_counter = Counter(all_flags_flat)
    print(f"\nConfig flags distribution (all solutions):")
    for flags, count in sorted(flag_counter.items(), key=lambda x: -x[1]):
        print(f"  {flags}: {count} solutions")

    # Per-row dominant config
    dominant_config_per_row: list[tuple[int, int, int] | None] = []
    for row_flags in per_row_configs:
        if not row_flags:
            dominant_config_per_row.append(None)
        else:
            flag_counts = Counter(row_flags)
            dominant_config_per_row.append(max(flag_counts, key=flag_counts.get))

    # How many rows have each dominant config
    dominant_counter = Counter(f for f in dominant_config_per_row if f is not None)
    print(f"\nDominant config per row:")
    for flags, count in sorted(dominant_counter.items(), key=lambda x: -x[1]):
        print(f"  {flags}: {count}/{len(rows)} rows")

    # ── Corridor analysis ────────────────────────────────────────────────────
    # For each config family, find all rows that have at least one candidate
    # with that config, then find the longest run where consecutive rows have
    # same-config candidates with small joint steps.
    PREFERRED_STEP = 8.0  # deg - threshold for "smooth" step

    print(f"\nCorridor analysis (preferred_step_threshold={PREFERRED_STEP}°):")
    all_config_families = set(all_flags_flat)
    for family in sorted(all_config_families):
        # Per-row, pick the best candidate for this family
        family_candidates: list[tuple[float, ...] | None] = []
        for row_index in range(len(rows)):
            matching = [
                j for j, f in zip(per_row_candidates[row_index], per_row_configs[row_index])
                if f == family
            ]
            if not matching:
                family_candidates.append(None)
            else:
                # Pick the first one (sorted by solve_all order, which is by seed distance)
                family_candidates.append(matching[0])

        # Find longest run with continuous presence
        rows_with_family = sum(1 for c in family_candidates if c is not None)

        # Find longest run (consecutive rows all have this family)
        longest_run = 0
        current_run = 0
        run_start = 0
        best_run_start = 0
        for i, cand in enumerate(family_candidates):
            if cand is not None:
                current_run += 1
                if current_run > longest_run:
                    longest_run = current_run
                    best_run_start = run_start
            else:
                current_run = 0
                run_start = i + 1

        # Find max joint step within the best run
        max_step_in_best_run = 0.0
        big_steps_in_best_run = 0
        for i in range(best_run_start, best_run_start + longest_run - 1):
            if family_candidates[i] is not None and family_candidates[i + 1] is not None:
                step = max(abs(a - b) for a, b in zip(family_candidates[i], family_candidates[i + 1]))
                max_step_in_best_run = max(max_step_in_best_run, step)
                if step > PREFERRED_STEP:
                    big_steps_in_best_run += 1

        print(
            f"  {family}: present={rows_with_family}/{len(rows)} rows, "
            f"longest_consecutive_run={longest_run}, "
            f"max_step_in_run={max_step_in_best_run:.1f}°, "
            f"big_steps_in_run={big_steps_in_best_run}"
        )

    # ── Per-row minimum step analysis ────────────────────────────────────────
    # For each row i, and for each config family in row i,
    # find the minimum joint step to any same-config candidate in row i+1.
    print(f"\nPer-row minimum step to same-config next-row candidate:")
    print(f"  {'Row':>4}  {'Candidates':>10}  {'SameConf->next':>14}  {'Dominant':>12}  {'MinStep':>8}")

    total_breaks = 0
    break_rows = []
    for i in range(min(len(rows) - 1, 50)):  # Show first 50 rows
        row_cands = per_row_candidates[i]
        row_flags = per_row_configs[i]
        next_cands = per_row_candidates[i + 1]
        next_flags = per_row_configs[i + 1]

        if not row_cands or not next_cands:
            print(f"  {i:>4}  {'EMPTY':>10}")
            continue

        # For each (candidate, config) in row i, find min step to same-config in row i+1
        min_step = math.inf
        min_step_config = None
        for jcurr, fcurr in zip(row_cands, row_flags):
            for jnext, fnext in zip(next_cands, next_flags):
                if fcurr == fnext:
                    step = max(abs(a - b) for a, b in zip(jcurr, jnext))
                    if step < min_step:
                        min_step = step
                        min_step_config = fcurr

        dom_config = dominant_config_per_row[i]
        n_next_same = sum(1 for f in next_flags if f == dom_config)

        marker = " <<BREAK" if min_step > PREFERRED_STEP else ""
        if min_step > PREFERRED_STEP:
            total_breaks += 1
            break_rows.append(i)

        min_step_str = f"{min_step:.1f}" if math.isfinite(min_step) else "NONE"
        print(
            f"  {i:>4}  {len(row_cands):>10}  {n_next_same:>14}  {str(dom_config):>12}  {min_step_str:>8}°{marker}"
        )

    if len(rows) > 51:
        # Count remaining breaks
        remaining_breaks = 0
        for i in range(50, len(rows) - 1):
            row_cands = per_row_candidates[i]
            row_flags = per_row_configs[i]
            next_cands = per_row_candidates[i + 1]
            next_flags = per_row_configs[i + 1]
            if not row_cands or not next_cands:
                continue
            min_step = math.inf
            for jcurr, fcurr in zip(row_cands, row_flags):
                for jnext, fnext in zip(next_cands, next_flags):
                    if fcurr == fnext:
                        step = max(abs(a - b) for a, b in zip(jcurr, jnext))
                        min_step = min(min_step, step)
            if min_step > PREFERRED_STEP:
                total_breaks += 1
                break_rows.append(i)
        print(f"  ... ({len(rows) - 51} more rows not shown)")

    print(f"\nTotal transitions with min_step > {PREFERRED_STEP}°: {total_breaks}")
    if break_rows:
        print(f"Break rows: {break_rows}")

    # ── Deep-dive: first few break rows ─────────────────────────────────────
    print(f"\nDeep-dive: per-candidate detail for first 3 break rows:")
    for br in break_rows[:3]:
        print(f"\n  Row {br} candidates ({len(per_row_candidates[br])}):")
        for j, f in zip(per_row_candidates[br], per_row_configs[br]):
            print(f"    joints={[f'{v:.1f}' for v in j]}  config={f}")
        print(f"  Row {br+1} candidates ({len(per_row_candidates[br+1])}):")
        for j, f in zip(per_row_candidates[br+1], per_row_configs[br+1]):
            print(f"    joints={[f'{v:.1f}' for v in j]}  config={f}")

    # ── Full break row list ──────────────────────────────────────────────────
    print(f"\nAll {total_breaks} break rows (min_same-config step > {PREFERRED_STEP}°):")
    print(f"  {break_rows}")


if __name__ == "__main__":
    main()
