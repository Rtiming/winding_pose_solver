"""Scan Frame-A Y/Z offsets around suspected wrist/config transitions.

This is an offline SixAxisIK diagnostic.  It probes the Y/Z offset grid for
selected original CSV row pairs and reports:

- which IK config families are available per row
- whether the same family can connect the two rows under the offset-step limit
- how close any bridge/interpolated pose can get to the wrist singularity J5=0
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import math
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Sequence

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from main import APP_RUNTIME_SETTINGS
from src.core.geometry import _build_pose, _mean_abs_joint_delta
from src.core.motion_settings import build_motion_settings_from_dict, motion_settings_to_dict
from src.core.pose_csv import load_pose_rows
from src.robodk_runtime.eval_worker import open_offline_ik_station_context
from src.search.bridge_builder import _insert_interpolated_transition_rows
from src.search.ik_collection import _collect_ik_candidates, reset_ik_candidate_collection_cache
from src.search.path_optimizer import _build_optimizer_settings


@dataclass(frozen=True)
class _ScanState:
    dy_mm: float
    dz_mm: float
    joints: tuple[float, ...]
    config_flags: tuple[int, ...]
    branch_id: tuple[int, ...] | None
    singularity_penalty: float
    joint_limit_penalty: float


@dataclass(frozen=True)
class _FamilyAvailability:
    config_flags: tuple[int, ...]
    state_count: int
    offset_count: int
    min_abs_j5_deg: float
    min_abs_j5_offset_mm: tuple[float, float]
    min_abs_j5_joints_deg: tuple[float, ...]


@dataclass(frozen=True)
class _PairSummary:
    left_family: tuple[int, ...]
    right_family: tuple[int, ...]
    max_joint_delta_deg: float
    mean_joint_delta_deg: float
    left_offset_mm: tuple[float, float]
    right_offset_mm: tuple[float, float]
    offset_gap_mm: float
    left_joints_deg: tuple[float, ...]
    right_joints_deg: tuple[float, ...]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", default=str(PROJECT_ROOT / "data" / "tool_poses_frame2.csv"))
    parser.add_argument(
        "--segments",
        nargs="+",
        default=("53:54", "59:60", "372:373", "373:374"),
        help="Original CSV label pairs, e.g. 53:54 372:373.",
    )
    parser.add_argument("--max-offset-mm", type=float, default=36.0)
    parser.add_argument("--step-mm", type=float, default=4.0)
    parser.add_argument(
        "--max-offset-step-mm",
        type=float,
        default=4.0,
        help="Euclidean profile-step limit used for constrained pair reports.",
    )
    parser.add_argument("--bridge-samples", type=int, default=8)
    parser.add_argument("--top", type=int, default=5)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--run-id", default=None)
    parser.add_argument(
        "--output-root",
        default=str(PROJECT_ROOT / "artifacts" / "diagnostics" / "yz_transition_scan"),
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    run_id = args.run_id or "yz_transition_scan_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_root) / run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = tuple(load_pose_rows(args.csv))
    label_to_index = _build_label_to_index(rows)
    segments = tuple(_parse_segment(raw, label_to_index) for raw in args.segments)

    scan_grid = tuple(_build_offset_grid(args.max_offset_mm, args.step_mm))
    summaries = []
    print(
        "Y/Z transition scan: "
        f"grid={len(scan_grid)} offsets, max_abs={args.max_offset_mm:.1f} mm, "
        f"step={args.step_mm:.1f} mm, constrained_gap<={args.max_offset_step_mm:.1f} mm, "
        f"workers={max(1, int(args.workers))}"
    )

    worker_count = max(1, int(args.workers))
    if worker_count > 1 and len(segments) > 1:
        payloads = [
            {
                "rows": rows,
                "left_index": left_index,
                "right_index": right_index,
                "left_label": left_label,
                "right_label": right_label,
                "scan_grid": scan_grid,
                "max_offset_step_mm": float(args.max_offset_step_mm),
                "bridge_samples": max(0, int(args.bridge_samples)),
                "top": max(1, int(args.top)),
            }
            for left_index, right_index, left_label, right_label in segments
        ]
        with ProcessPoolExecutor(max_workers=min(worker_count, len(payloads))) as executor:
            results = list(executor.map(_scan_segment_worker, payloads))
        for log_text, segment_summary in results:
            print(log_text, end="")
            summaries.append(segment_summary)
    else:
        context = open_offline_ik_station_context(
            robot_name=APP_RUNTIME_SETTINGS.robot_name,
            frame_name=APP_RUNTIME_SETTINGS.frame_name,
        )
        motion_payload = motion_settings_to_dict(APP_RUNTIME_SETTINGS.motion_settings)
        motion_payload["ik_backend"] = "six_axis_ik"
        motion_settings = build_motion_settings_from_dict(motion_payload)
        optimizer_settings = _build_optimizer_settings(context.joint_count, motion_settings)
        reset_ik_candidate_collection_cache()

        for left_index, right_index, left_label, right_label in segments:
            print()
            print(f"=== Segment {left_label}->{right_label} (rows {left_index}->{right_index}) ===")
            segment_summary = _scan_segment(
                rows,
                left_index=left_index,
                right_index=right_index,
                left_label=left_label,
                right_label=right_label,
                scan_grid=scan_grid,
                max_offset_step_mm=float(args.max_offset_step_mm),
                bridge_samples=max(0, int(args.bridge_samples)),
                top=max(1, int(args.top)),
                context=context,
                optimizer_settings=optimizer_settings,
                motion_settings=motion_settings,
            )
            summaries.append(segment_summary)

    payload = {
        "run_id": run_id,
        "csv": str(args.csv),
        "max_offset_mm": float(args.max_offset_mm),
        "step_mm": float(args.step_mm),
        "max_offset_step_mm": float(args.max_offset_step_mm),
        "bridge_samples": int(args.bridge_samples),
        "segments": summaries,
    }
    output_path = output_dir / "scan_summary.json"
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print()
    print(f"Wrote scan summary: {output_path}")
    return 0


def _scan_segment_worker(payload: dict[str, object]) -> tuple[str, dict[str, object]]:
    context = open_offline_ik_station_context(
        robot_name=APP_RUNTIME_SETTINGS.robot_name,
        frame_name=APP_RUNTIME_SETTINGS.frame_name,
    )
    motion_payload = motion_settings_to_dict(APP_RUNTIME_SETTINGS.motion_settings)
    motion_payload["ik_backend"] = "six_axis_ik"
    motion_settings = build_motion_settings_from_dict(motion_payload)
    optimizer_settings = _build_optimizer_settings(context.joint_count, motion_settings)
    reset_ik_candidate_collection_cache()

    output = io.StringIO()
    left_label = str(payload["left_label"])
    right_label = str(payload["right_label"])
    left_index = int(payload["left_index"])
    right_index = int(payload["right_index"])
    with contextlib.redirect_stdout(output):
        print()
        print(f"=== Segment {left_label}->{right_label} (rows {left_index}->{right_index}) ===")
        segment_summary = _scan_segment(
            payload["rows"],  # type: ignore[arg-type]
            left_index=left_index,
            right_index=right_index,
            left_label=left_label,
            right_label=right_label,
            scan_grid=payload["scan_grid"],  # type: ignore[arg-type]
            max_offset_step_mm=float(payload["max_offset_step_mm"]),
            bridge_samples=int(payload["bridge_samples"]),
            top=int(payload["top"]),
            context=context,
            optimizer_settings=optimizer_settings,
            motion_settings=motion_settings,
        )
    return output.getvalue(), segment_summary


def _scan_segment(
    rows: Sequence[dict[str, float]],
    *,
    left_index: int,
    right_index: int,
    left_label: str,
    right_label: str,
    scan_grid: Sequence[tuple[float, float]],
    max_offset_step_mm: float,
    bridge_samples: int,
    top: int,
    context,
    optimizer_settings,
    motion_settings,
) -> dict[str, object]:
    left_states = _collect_grid_states(
        rows[left_index],
        scan_grid=scan_grid,
        context=context,
        optimizer_settings=optimizer_settings,
        motion_settings=motion_settings,
    )
    right_states = _collect_grid_states(
        rows[right_index],
        scan_grid=scan_grid,
        context=context,
        optimizer_settings=optimizer_settings,
        motion_settings=motion_settings,
    )

    left_family = _family_availability(left_states)
    right_family = _family_availability(right_states)
    _print_family_table("left", left_family)
    _print_family_table("right", right_family)

    same_family_unconstrained = _best_pairs_by_family(
        left_states,
        right_states,
        max_offset_step_mm=None,
        same_family_only=True,
    )[:top]
    same_family_constrained = _best_pairs_by_family(
        left_states,
        right_states,
        max_offset_step_mm=max_offset_step_mm,
        same_family_only=True,
    )[:top]
    cross_family_constrained = _best_pairs_by_family(
        left_states,
        right_states,
        max_offset_step_mm=max_offset_step_mm,
        same_family_only=False,
    )[:top]

    _print_pair_table("best same-family pairs, no offset-gap limit", same_family_unconstrained)
    _print_pair_table(
        f"best same-family pairs, offset gap <= {max_offset_step_mm:.1f} mm",
        same_family_constrained,
    )
    _print_pair_table(
        f"best cross-family pairs, offset gap <= {max_offset_step_mm:.1f} mm",
        cross_family_constrained,
    )

    bridge_summary = _scan_bridge_rows(
        rows,
        left_index=left_index,
        right_index=right_index,
        left_label=left_label,
        right_label=right_label,
        scan_grid=scan_grid,
        bridge_samples=bridge_samples,
        context=context,
        optimizer_settings=optimizer_settings,
        motion_settings=motion_settings,
    )
    if bridge_summary:
        print("Bridge/interpolation min |J5| by sample:")
        for entry in bridge_summary:
            best_any = entry["best_any"]
            best_same_pose_flip_pair = entry["best_same_pose_flip_pair"]
            pair_note = "none"
            if best_same_pose_flip_pair is not None:
                pair_note = (
                    f"gap={best_same_pose_flip_pair['max_joint_delta_deg']:.3f} deg "
                    f"families={tuple(best_same_pose_flip_pair['left_family'])}->"
                    f"{tuple(best_same_pose_flip_pair['right_family'])}"
                )
            print(
                f"  {entry['label']}: ratio={entry['ratio']:.3f}, "
                f"best |J5|={best_any['abs_j5_deg']:.3f} deg "
                f"family={tuple(best_any['family'])} offset={tuple(best_any['offset_mm'])}, "
                f"same-pose flip pair: {pair_note}"
            )

    return {
        "left_label": left_label,
        "right_label": right_label,
        "left_index": left_index,
        "right_index": right_index,
        "left_family_availability": [asdict(item) for item in left_family],
        "right_family_availability": [asdict(item) for item in right_family],
        "same_family_unconstrained": [asdict(item) for item in same_family_unconstrained],
        "same_family_constrained": [asdict(item) for item in same_family_constrained],
        "cross_family_constrained": [asdict(item) for item in cross_family_constrained],
        "bridge_samples": bridge_summary,
    }


def _collect_grid_states(
    row: dict[str, float],
    *,
    scan_grid: Sequence[tuple[float, float]],
    context,
    optimizer_settings,
    motion_settings,
) -> tuple[_ScanState, ...]:
    states: list[_ScanState] = []
    lower_limits = tuple(float(value) for value in context.lower_limits[: context.joint_count])
    upper_limits = tuple(float(value) for value in context.upper_limits[: context.joint_count])
    for dy_mm, dz_mm in scan_grid:
        adjusted_row = dict(row)
        adjusted_row["x_mm"] = float(row["x_mm"])
        adjusted_row["y_mm"] = float(row["y_mm"]) + float(dy_mm)
        adjusted_row["z_mm"] = float(row["z_mm"]) + float(dz_mm)
        pose = _build_pose(adjusted_row, context.mat_type)
        candidates = _collect_ik_candidates(
            context.robot,
            pose,
            tool_pose=context.tool_pose,
            reference_pose=context.reference_pose,
            lower_limits=lower_limits,
            upper_limits=upper_limits,
            seed_joints=(),
            joint_count=context.joint_count,
            optimizer_settings=optimizer_settings,
            a1_lower_deg=motion_settings.a1_min_deg,
            a1_upper_deg=motion_settings.a1_max_deg,
            a2_max_deg=motion_settings.a2_max_deg,
            joint_constraint_tolerance_deg=motion_settings.joint_constraint_tolerance_deg,
        )
        for candidate in candidates:
            states.append(
                _ScanState(
                    dy_mm=float(dy_mm),
                    dz_mm=float(dz_mm),
                    joints=tuple(float(value) for value in candidate.joints),
                    config_flags=tuple(int(value) for value in candidate.config_flags),
                    branch_id=candidate.branch_id,
                    singularity_penalty=float(candidate.singularity_penalty),
                    joint_limit_penalty=float(candidate.joint_limit_penalty),
                )
            )
    return tuple(states)


def _family_availability(states: Sequence[_ScanState]) -> tuple[_FamilyAvailability, ...]:
    grouped: dict[tuple[int, ...], list[_ScanState]] = {}
    for state in states:
        grouped.setdefault(state.config_flags, []).append(state)

    summaries: list[_FamilyAvailability] = []
    for family, family_states in grouped.items():
        best_j5_state = min(family_states, key=lambda state: abs(state.joints[4]))
        summaries.append(
            _FamilyAvailability(
                config_flags=family,
                state_count=len(family_states),
                offset_count=len({(state.dy_mm, state.dz_mm) for state in family_states}),
                min_abs_j5_deg=abs(best_j5_state.joints[4]),
                min_abs_j5_offset_mm=(best_j5_state.dy_mm, best_j5_state.dz_mm),
                min_abs_j5_joints_deg=best_j5_state.joints,
            )
        )
    summaries.sort(key=lambda item: (item.config_flags, item.min_abs_j5_deg))
    return tuple(summaries)


def _best_pairs_by_family(
    left_states: Sequence[_ScanState],
    right_states: Sequence[_ScanState],
    *,
    max_offset_step_mm: float | None,
    same_family_only: bool,
) -> tuple[_PairSummary, ...]:
    best_by_pair: dict[tuple[tuple[int, ...], tuple[int, ...]], _PairSummary] = {}
    right_by_family: dict[tuple[int, ...], list[_ScanState]] = {}
    for state in right_states:
        right_by_family.setdefault(state.config_flags, []).append(state)

    for left_state in left_states:
        if same_family_only:
            candidate_right_groups = ((left_state.config_flags, right_by_family.get(left_state.config_flags, ())),)
        else:
            candidate_right_groups = tuple(right_by_family.items())
        for right_family, right_group in candidate_right_groups:
            if not right_group:
                continue
            if same_family_only and right_family != left_state.config_flags:
                continue
            if not same_family_only and right_family == left_state.config_flags:
                continue
            for right_state in right_group:
                offset_gap = math.hypot(
                    right_state.dy_mm - left_state.dy_mm,
                    right_state.dz_mm - left_state.dz_mm,
                )
                if (
                    max_offset_step_mm is not None
                    and offset_gap > float(max_offset_step_mm) + 1e-9
                ):
                    continue
                max_delta = max(
                    abs(right_joint - left_joint)
                    for left_joint, right_joint in zip(left_state.joints, right_state.joints)
                )
                mean_delta = _mean_abs_joint_delta(left_state.joints, right_state.joints)
                pair_key = (left_state.config_flags, right_state.config_flags)
                summary = _PairSummary(
                    left_family=left_state.config_flags,
                    right_family=right_state.config_flags,
                    max_joint_delta_deg=float(max_delta),
                    mean_joint_delta_deg=float(mean_delta),
                    left_offset_mm=(left_state.dy_mm, left_state.dz_mm),
                    right_offset_mm=(right_state.dy_mm, right_state.dz_mm),
                    offset_gap_mm=float(offset_gap),
                    left_joints_deg=left_state.joints,
                    right_joints_deg=right_state.joints,
                )
                old = best_by_pair.get(pair_key)
                if old is None or _pair_sort_key(summary) < _pair_sort_key(old):
                    best_by_pair[pair_key] = summary

    return tuple(sorted(best_by_pair.values(), key=_pair_sort_key))


def _scan_bridge_rows(
    rows: Sequence[dict[str, float]],
    *,
    left_index: int,
    right_index: int,
    left_label: str,
    right_label: str,
    scan_grid: Sequence[tuple[float, float]],
    bridge_samples: int,
    context,
    optimizer_settings,
    motion_settings,
) -> list[dict[str, object]]:
    if bridge_samples <= 0:
        return []
    if right_index != left_index + 1:
        return []

    zero_profile = tuple((0.0, 0.0) for _ in rows)
    labels = tuple(_label_for_row(index, row) for index, row in enumerate(rows))
    inserted_flags = tuple(False for _ in rows)
    augmented_rows, _profile, augmented_labels, augmented_flags = _insert_interpolated_transition_rows(
        rows,
        zero_profile,
        labels,
        inserted_flags,
        segment_index=left_index,
        insertion_count=bridge_samples,
    )

    summaries: list[dict[str, object]] = []
    for local_insert_index in range(1, bridge_samples + 1):
        augmented_index = left_index + local_insert_index
        if not augmented_flags[augmented_index]:
            continue
        states = _collect_grid_states(
            augmented_rows[augmented_index],
            scan_grid=scan_grid,
            context=context,
            optimizer_settings=optimizer_settings,
            motion_settings=motion_settings,
        )
        if not states:
            continue
        best_any = min(states, key=lambda state: abs(state.joints[4]))
        best_same_pose_flip_pair = _best_same_pose_flip_pair(states)
        summaries.append(
            {
                "label": str(augmented_labels[augmented_index]),
                "ratio": float(local_insert_index / (bridge_samples + 1)),
                "best_any": {
                    "family": best_any.config_flags,
                    "abs_j5_deg": abs(best_any.joints[4]),
                    "offset_mm": (best_any.dy_mm, best_any.dz_mm),
                    "joints_deg": best_any.joints,
                },
                "best_same_pose_flip_pair": None
                if best_same_pose_flip_pair is None
                else asdict(best_same_pose_flip_pair),
            }
        )
    return summaries


def _best_same_pose_flip_pair(states: Sequence[_ScanState]) -> _PairSummary | None:
    by_offset_family: dict[tuple[float, float, tuple[int, ...]], list[_ScanState]] = {}
    for state in states:
        by_offset_family.setdefault((state.dy_mm, state.dz_mm, state.config_flags), []).append(state)

    best: _PairSummary | None = None
    families_by_offset: dict[tuple[float, float], set[tuple[int, ...]]] = {}
    for dy_mm, dz_mm, family in by_offset_family:
        families_by_offset.setdefault((dy_mm, dz_mm), set()).add(family)

    for offset, families in families_by_offset.items():
        sorted_families = sorted(families)
        for left_family in sorted_families:
            for right_family in sorted_families:
                if left_family >= right_family:
                    continue
                if len(left_family) >= 3 and len(right_family) >= 3:
                    same_nonflip = left_family[:2] == right_family[:2]
                    flip_changes = left_family[2] != right_family[2]
                    if not (same_nonflip and flip_changes):
                        continue
                for left_state in by_offset_family[(offset[0], offset[1], left_family)]:
                    for right_state in by_offset_family[(offset[0], offset[1], right_family)]:
                        max_delta = max(
                            abs(right_joint - left_joint)
                            for left_joint, right_joint in zip(left_state.joints, right_state.joints)
                        )
                        summary = _PairSummary(
                            left_family=left_family,
                            right_family=right_family,
                            max_joint_delta_deg=float(max_delta),
                            mean_joint_delta_deg=_mean_abs_joint_delta(
                                left_state.joints,
                                right_state.joints,
                            ),
                            left_offset_mm=offset,
                            right_offset_mm=offset,
                            offset_gap_mm=0.0,
                            left_joints_deg=left_state.joints,
                            right_joints_deg=right_state.joints,
                        )
                        if best is None or _pair_sort_key(summary) < _pair_sort_key(best):
                            best = summary
    return best


def _pair_sort_key(pair: _PairSummary) -> tuple[float, float, float]:
    return (
        float(pair.max_joint_delta_deg),
        float(pair.mean_joint_delta_deg),
        float(pair.offset_gap_mm),
    )


def _build_offset_grid(max_abs_offset_mm: float, step_mm: float) -> Iterable[tuple[float, float]]:
    if max_abs_offset_mm < 0.0:
        raise ValueError("--max-offset-mm must be non-negative.")
    if step_mm <= 0.0:
        raise ValueError("--step-mm must be positive.")

    values = set()
    steps = int(math.floor(max_abs_offset_mm / step_mm))
    for index in range(-steps, steps + 1):
        values.add(round(index * step_mm, 6))
    values.add(0.0)
    values.add(round(float(max_abs_offset_mm), 6))
    values.add(round(-float(max_abs_offset_mm), 6))
    sorted_values = tuple(sorted(values))
    for dy_mm in sorted_values:
        for dz_mm in sorted_values:
            yield (float(dy_mm), float(dz_mm))


def _build_label_to_index(rows: Sequence[dict[str, float]]) -> dict[str, int]:
    return {_label_for_row(index, row): index for index, row in enumerate(rows)}


def _label_for_row(index: int, row: dict[str, float]) -> str:
    if "index" in row:
        return str(int(float(row["index"])))
    if "source_row" in row:
        return str(int(float(row["source_row"])))
    return str(index)


def _parse_segment(raw: str, label_to_index: dict[str, int]) -> tuple[int, int, str, str]:
    if ":" not in raw:
        raise ValueError(f"Expected segment as left:right, got {raw!r}")
    left_label, right_label = (part.strip() for part in raw.split(":", 1))
    if left_label not in label_to_index:
        raise ValueError(f"Unknown left label {left_label!r}")
    if right_label not in label_to_index:
        raise ValueError(f"Unknown right label {right_label!r}")
    return label_to_index[left_label], label_to_index[right_label], left_label, right_label


def _print_family_table(name: str, summaries: Sequence[_FamilyAvailability]) -> None:
    print(f"{name} row family availability:")
    for item in summaries:
        print(
            f"  {item.config_flags}: states={item.state_count}, offsets={item.offset_count}, "
            f"min |J5|={item.min_abs_j5_deg:.3f} deg at {item.min_abs_j5_offset_mm}"
        )


def _print_pair_table(title: str, pairs: Sequence[_PairSummary]) -> None:
    print(title + ":")
    if not pairs:
        print("  none")
        return
    for pair in pairs:
        print(
            f"  {pair.left_family}->{pair.right_family}: "
            f"max={pair.max_joint_delta_deg:.3f} deg, "
            f"mean={pair.mean_joint_delta_deg:.3f} deg, "
            f"offset_gap={pair.offset_gap_mm:.3f} mm, "
            f"left={pair.left_offset_mm}, right={pair.right_offset_mm}, "
            f"J5={pair.left_joints_deg[4]:.3f}->{pair.right_joints_deg[4]:.3f}"
        )


if __name__ == "__main__":
    raise SystemExit(main())
