"""Scan compact Cartesian bridge poses for residual wrist/config flips.

The scan keeps the original endpoint joint values fixed, samples one inserted
Cartesian bridge pose near the endpoint segment, and asks whether that bridge
pose has both endpoint IK families so the path can switch through the bridge
without a large joint jump.
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import math
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Sequence

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from main import APP_RUNTIME_SETTINGS
from src.core.collab_models import ProfileEvaluationResult, load_json_file
from src.core.geometry import (
    _build_pose,
    _mean_abs_joint_delta,
    _multiply_rotation_matrices,
    _pose_row_from_rotation_translation,
    _pose_row_to_rotation_translation,
    _quaternion_to_rotation_matrix,
    _rotation_matrix_from_xyz_offset_deg,
    _rotation_matrix_to_quaternion,
    _slerp_quaternion,
)
from src.core.motion_settings import build_motion_settings_from_dict, motion_settings_to_dict
from src.robodk_runtime.eval_worker import open_offline_ik_station_context
from src.search.ik_collection import (
    _IK_DEDUP_DECIMALS,
    _append_candidate_if_unique,
    _collect_ik_candidates,
    reset_ik_candidate_collection_cache,
)
from src.search.path_optimizer import _build_optimizer_settings


@dataclass(frozen=True)
class _BridgeSpec:
    ratio: float
    translation_offset_mm: tuple[float, float, float]
    rotation_offset_deg: tuple[float, float, float]
    rotation_mode: str


_WORKER_CONTEXT = None
_WORKER_OPTIMIZER_SETTINGS = None
_WORKER_MOTION_SETTINGS = None


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--result",
        default=str(
            PROJECT_ROOT
            / "artifacts"
            / "diagnostics"
            / "yz_nullspace_demo"
            / "server_m400_no_joint_bridge_control_parallel32"
            / "optimized_result.json"
        ),
        help="Profile result with pose_rows and selected_path, usually the no-joint-bridge control.",
    )
    parser.add_argument(
        "--segments",
        nargs="+",
        default=("361:362", "375:376"),
        help="Row-label pairs to scan.",
    )
    parser.add_argument("--ratios", default="0.25,0.5,0.75")
    parser.add_argument("--translation-max-mm", type=float, default=0.0)
    parser.add_argument("--translation-step-mm", type=float, default=10.0)
    parser.add_argument(
        "--translation-axes",
        default="yz",
        choices=("none", "x", "y", "z", "xy", "xz", "yz", "xyz"),
    )
    parser.add_argument(
        "--rotation-values-deg",
        default="-90,-60,-30,0,30,60,90",
        help="Comma-separated values used in the selected rotation grid.",
    )
    parser.add_argument(
        "--rotation-grid",
        default="full",
        choices=("none", "single_axis", "xy", "xz", "yz", "full"),
    )
    parser.add_argument(
        "--rotation-modes",
        nargs="+",
        default=("local",),
        choices=("local", "frame"),
    )
    parser.add_argument("--max-detour-mm", type=float, default=30.0)
    parser.add_argument("--max-path-ratio", type=float, default=1000.0)
    parser.add_argument(
        "--candidate-mode",
        default="all_plus_seeded",
        choices=("all", "seeded", "all_plus_seeded"),
        help=(
            "IK candidates to test for the bridge pose. seeded uses the two endpoint "
            "joint states as seed-aware single-solution probes."
        ),
    )
    parser.add_argument(
        "--bridge-mode",
        default="family_pair",
        choices=("family_pair", "singular_phase", "both"),
        help=(
            "family_pair compares two same-pose IK branches directly. "
            "singular_phase projects the bridge candidate to A5=0 and allows "
            "A4/A6 phase redistribution at the singular pose."
        ),
    )
    parser.add_argument("--phase-step-limit-deg", type=float, default=20.0)
    parser.add_argument("--phase-max-insertions", type=int, default=24)
    parser.add_argument("--top", type=int, default=12)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--chunk-size", type=int, default=1000)
    parser.add_argument("--run-id", default=None)
    parser.add_argument(
        "--output-root",
        default=str(PROJECT_ROOT / "artifacts" / "diagnostics" / "cartesian_bridge_scan"),
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    run_id = args.run_id or "cartesian_bridge_scan_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_root) / run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    result = ProfileEvaluationResult.from_dict(load_json_file(args.result))
    if result.pose_rows is None:
        raise ValueError("The result must include pose_rows.")
    labels = tuple(str(label) for label in result.row_labels)
    specs = _build_bridge_specs(args)
    print(
        "Cartesian bridge scan: "
        f"segments={len(args.segments)}, specs={len(specs)}, "
        f"workers={max(1, int(args.workers))}, "
        f"max_detour={args.max_detour_mm:.1f} mm, "
        f"max_path_ratio={args.max_path_ratio:.1f}"
    )

    segment_summaries = []
    for raw_segment in args.segments:
        left_label, right_label = raw_segment.split(":", 1)
        left_index = labels.index(str(left_label))
        right_index = labels.index(str(right_label))
        if right_index <= left_index:
            raise ValueError(f"Invalid segment order: {raw_segment}")
        print()
        print(f"=== Segment {left_label}->{right_label} index {left_index}->{right_index} ===")
        summary = _scan_segment(
            result,
            left_index=left_index,
            right_index=right_index,
            specs=specs,
            max_detour_mm=float(args.max_detour_mm),
            max_path_ratio=float(args.max_path_ratio),
            top=max(1, int(args.top)),
            workers=max(1, int(args.workers)),
            chunk_size=max(1, int(args.chunk_size)),
            candidate_mode=str(args.candidate_mode),
            bridge_mode=str(args.bridge_mode),
            phase_step_limit_deg=float(args.phase_step_limit_deg),
            phase_max_insertions=max(0, int(args.phase_max_insertions)),
        )
        segment_summaries.append(summary)

    payload = {
        "run_id": run_id,
        "result": str(args.result),
        "segments": segment_summaries,
        "spec_count": len(specs),
        "max_detour_mm": float(args.max_detour_mm),
        "max_path_ratio": float(args.max_path_ratio),
        "candidate_mode": str(args.candidate_mode),
        "bridge_mode": str(args.bridge_mode),
        "phase_step_limit_deg": float(args.phase_step_limit_deg),
        "phase_max_insertions": max(0, int(args.phase_max_insertions)),
    }
    output_path = output_dir / "scan_summary.json"
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print()
    print(f"Wrote scan summary: {output_path}")
    return 0


def _scan_segment(
    result: ProfileEvaluationResult,
    *,
    left_index: int,
    right_index: int,
    specs: Sequence[_BridgeSpec],
    max_detour_mm: float,
    max_path_ratio: float,
    top: int,
    workers: int,
    chunk_size: int,
    candidate_mode: str,
    bridge_mode: str,
    phase_step_limit_deg: float,
    phase_max_insertions: int,
) -> dict[str, object]:
    segment_payload = _build_segment_payload(result, left_index, right_index)
    spec_chunks = [
        tuple(specs[start : start + chunk_size])
        for start in range(0, len(specs), chunk_size)
    ]

    all_results: list[dict[str, object]] = []
    skipped_by_detour = 0
    scanned_specs = 0
    if workers > 1 and len(spec_chunks) > 1:
        with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker) as executor:
            payloads = [
                {
                    "segment": segment_payload,
                    "specs": chunk,
                    "max_detour_mm": float(max_detour_mm),
                    "max_path_ratio": float(max_path_ratio),
                    "candidate_mode": str(candidate_mode),
                    "bridge_mode": str(bridge_mode),
                    "phase_step_limit_deg": float(phase_step_limit_deg),
                    "phase_max_insertions": int(phase_max_insertions),
                }
                for chunk in spec_chunks
            ]
            for chunk_result in executor.map(_scan_chunk_worker, payloads):
                all_results.extend(chunk_result["results"])
                skipped_by_detour += int(chunk_result["skipped_by_detour"])
                scanned_specs += int(chunk_result["scanned_specs"])
    else:
        _init_worker()
        for chunk in spec_chunks:
            chunk_result = _scan_chunk_worker(
                {
                    "segment": segment_payload,
                    "specs": chunk,
                    "max_detour_mm": float(max_detour_mm),
                    "max_path_ratio": float(max_path_ratio),
                    "candidate_mode": str(candidate_mode),
                    "bridge_mode": str(bridge_mode),
                    "phase_step_limit_deg": float(phase_step_limit_deg),
                    "phase_max_insertions": int(phase_max_insertions),
                }
            )
            all_results.extend(chunk_result["results"])
            skipped_by_detour += int(chunk_result["skipped_by_detour"])
            scanned_specs += int(chunk_result["scanned_specs"])

    all_results.sort(key=_result_sort_key)
    top_results = all_results[:top]
    print(
        f"scanned={scanned_specs}, skipped_by_detour={skipped_by_detour}, "
        f"bridge_candidates={len(all_results)}"
    )
    for rank, item in enumerate(top_results, start=1):
        print(
            f"  #{rank}: max={item['max_joint_delta_deg']:.3f} deg, "
            f"mean={item['mean_joint_delta_deg']:.3f} deg, "
            f"legs={tuple(round(float(value), 3) for value in item['leg_max_deltas_deg'])}, "
            f"strategy={item['strategy']}, "
            f"families={tuple(item['bridge_left_family'])}->{tuple(item['bridge_right_family'])}, "
            f"|J5|={item['bridge_abs_j5_deg']:.3f} deg, "
            f"ratio={item['path_ratio']:.3f}, dev={item['max_deviation_mm']:.3f} mm, "
            f"offset={tuple(item['translation_offset_mm'])}, "
            f"rot={tuple(item['rotation_offset_deg'])}, mode={item['rotation_mode']}"
        )

    return {
        "left_label": segment_payload["left_label"],
        "right_label": segment_payload["right_label"],
        "left_index": int(left_index),
        "right_index": int(right_index),
        "scanned_specs": int(scanned_specs),
        "skipped_by_detour": int(skipped_by_detour),
        "bridge_candidate_count": len(all_results),
        "top_results": top_results,
        "candidate_mode": str(candidate_mode),
        "bridge_mode": str(bridge_mode),
    }


def _init_worker() -> None:
    global _WORKER_CONTEXT, _WORKER_OPTIMIZER_SETTINGS, _WORKER_MOTION_SETTINGS
    if _WORKER_CONTEXT is not None:
        return
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        context = open_offline_ik_station_context(
            robot_name=APP_RUNTIME_SETTINGS.robot_name,
            frame_name=APP_RUNTIME_SETTINGS.frame_name,
        )
    motion_payload = motion_settings_to_dict(APP_RUNTIME_SETTINGS.motion_settings)
    motion_payload["ik_backend"] = "six_axis_ik"
    motion_settings = build_motion_settings_from_dict(motion_payload)
    optimizer_settings = _build_optimizer_settings(context.joint_count, motion_settings)
    reset_ik_candidate_collection_cache()
    _WORKER_CONTEXT = context
    _WORKER_OPTIMIZER_SETTINGS = optimizer_settings
    _WORKER_MOTION_SETTINGS = motion_settings


def _scan_chunk_worker(payload: dict[str, object]) -> dict[str, object]:
    if _WORKER_CONTEXT is None:
        _init_worker()
    assert _WORKER_CONTEXT is not None
    assert _WORKER_OPTIMIZER_SETTINGS is not None
    assert _WORKER_MOTION_SETTINGS is not None

    segment = payload["segment"]  # type: ignore[assignment]
    specs: Sequence[_BridgeSpec] = payload["specs"]  # type: ignore[assignment]
    max_detour_mm = float(payload["max_detour_mm"])
    max_path_ratio = float(payload["max_path_ratio"])
    candidate_mode = str(payload.get("candidate_mode", "all_plus_seeded"))
    bridge_mode = str(payload.get("bridge_mode", "family_pair"))
    phase_step_limit_deg = float(payload.get("phase_step_limit_deg", 20.0))
    phase_max_insertions = int(payload.get("phase_max_insertions", 24))

    results: list[dict[str, object]] = []
    skipped_by_detour = 0
    for spec in specs:
        bridge_row, detour_metrics = _build_bridge_pose_row(segment, spec)  # type: ignore[arg-type]
        bridge_row_rejected = not _passes_detour_filter(
            detour_metrics,
            max_detour_mm=max_detour_mm,
            max_path_ratio=max_path_ratio,
        )
        if bridge_row_rejected and bridge_mode == "family_pair":
            skipped_by_detour += 1
            continue
        candidates = _collect_bridge_candidates(
            bridge_row,
            segment=segment,  # type: ignore[arg-type]
            candidate_mode=candidate_mode,
        )
        if not candidates:
            continue
        if bridge_mode in ("family_pair", "both") and not bridge_row_rejected:
            bridge_result = _best_bridge_family_pair(segment, spec, candidates, detour_metrics)  # type: ignore[arg-type]
            if bridge_result is not None:
                results.append(bridge_result)
        elif bridge_mode in ("family_pair", "both") and bridge_row_rejected:
            skipped_by_detour += 1
        if bridge_mode in ("singular_phase", "both"):
            phase_result = _best_singular_phase_bridge(
                segment,  # type: ignore[arg-type]
                spec,
                candidates,
                phase_step_limit_deg=phase_step_limit_deg,
                phase_max_insertions=phase_max_insertions,
            )
            if phase_result is not None and _passes_detour_filter(
                phase_result,
                max_detour_mm=max_detour_mm,
                max_path_ratio=max_path_ratio,
            ):
                results.append(phase_result)
            elif phase_result is not None:
                skipped_by_detour += 1
    return {
        "results": results,
        "skipped_by_detour": skipped_by_detour,
        "scanned_specs": len(specs),
    }


def _collect_bridge_candidates(
    bridge_row: dict[str, float],
    *,
    segment: dict[str, object],
    candidate_mode: str,
):
    context = _WORKER_CONTEXT
    assert context is not None
    pose = _build_pose(bridge_row, context.mat_type)
    lower_limits = tuple(float(value) for value in context.lower_limits[: context.joint_count])
    upper_limits = tuple(float(value) for value in context.upper_limits[: context.joint_count])
    candidates = []
    if candidate_mode in ("all", "all_plus_seeded"):
        candidates.extend(
            _collect_ik_candidates(
                context.robot,
                pose,
                tool_pose=context.tool_pose,
                reference_pose=context.reference_pose,
                lower_limits=lower_limits,
                upper_limits=upper_limits,
                seed_joints=(),
                joint_count=context.joint_count,
                optimizer_settings=_WORKER_OPTIMIZER_SETTINGS,
                a1_lower_deg=_WORKER_MOTION_SETTINGS.a1_min_deg,
                a1_upper_deg=_WORKER_MOTION_SETTINGS.a1_max_deg,
                a2_max_deg=_WORKER_MOTION_SETTINGS.a2_max_deg,
                joint_constraint_tolerance_deg=_WORKER_MOTION_SETTINGS.joint_constraint_tolerance_deg,
            )
        )

    seen = {
        tuple(round(float(value), _IK_DEDUP_DECIMALS) for value in candidate.joints)
        for candidate in candidates
    }
    if candidate_mode in ("seeded", "all_plus_seeded"):
        solve_ik_seeded = getattr(context.robot, "SolveIKSeeded", context.robot.SolveIK)
        for seed in _bridge_seed_joints(segment):
            raw_solution = solve_ik_seeded(
                pose,
                list(seed),
                context.tool_pose,
                context.reference_pose,
            )
            _append_candidate_if_unique(
                candidates,
                seen,
                robot=context.robot,
                raw_joints=raw_solution,
                lower_limits=lower_limits,
                upper_limits=upper_limits,
                joint_count=context.joint_count,
                optimizer_settings=_WORKER_OPTIMIZER_SETTINGS,
                a1_lower_deg=_WORKER_MOTION_SETTINGS.a1_min_deg,
                a1_upper_deg=_WORKER_MOTION_SETTINGS.a1_max_deg,
                a2_max_deg=_WORKER_MOTION_SETTINGS.a2_max_deg,
                joint_constraint_tolerance_deg=_WORKER_MOTION_SETTINGS.joint_constraint_tolerance_deg,
            )

    candidates.sort(
        key=lambda candidate: (
            candidate.config_flags,
            candidate.joint_limit_penalty + candidate.singularity_penalty,
            candidate.joints,
        )
    )
    return tuple(candidates)


def _bridge_seed_joints(segment: dict[str, object]) -> tuple[tuple[float, ...], ...]:
    left_joints = tuple(float(value) for value in segment["left_joints"])  # type: ignore[arg-type]
    right_joints = tuple(float(value) for value in segment["right_joints"])  # type: ignore[arg-type]
    midpoint = tuple((left_value + right_value) * 0.5 for left_value, right_value in zip(left_joints, right_joints))
    seeds: list[tuple[float, ...]] = []
    seen: set[tuple[float, ...]] = set()
    for seed in (left_joints, right_joints, midpoint):
        key = tuple(round(float(value), _IK_DEDUP_DECIMALS) for value in seed)
        if key in seen:
            continue
        seen.add(key)
        seeds.append(seed)
    return tuple(seeds)


def _best_bridge_family_pair(
    segment: dict[str, object],
    spec: _BridgeSpec,
    candidates,
    detour_metrics: dict[str, float],
) -> dict[str, object] | None:
    left_joints = segment["left_joints"]  # type: ignore[assignment]
    right_joints = segment["right_joints"]  # type: ignore[assignment]
    left_family = tuple(segment["left_family"])  # type: ignore[arg-type]
    right_family = tuple(segment["right_family"])  # type: ignore[arg-type]
    best: dict[str, object] | None = None

    left_bridge_candidates = [candidate for candidate in candidates if candidate.config_flags == left_family]
    right_bridge_candidates = [candidate for candidate in candidates if candidate.config_flags == right_family]
    if not left_bridge_candidates or not right_bridge_candidates:
        return None

    for bridge_left in left_bridge_candidates:
        for bridge_right in right_bridge_candidates:
            leg_deltas = (
                _max_abs_joint_delta(left_joints, bridge_left.joints),  # type: ignore[arg-type]
                _max_abs_joint_delta(bridge_left.joints, bridge_right.joints),
                _max_abs_joint_delta(bridge_right.joints, right_joints),  # type: ignore[arg-type]
            )
            max_delta = max(leg_deltas)
            mean_delta = (
                _mean_abs_joint_delta(left_joints, bridge_left.joints)  # type: ignore[arg-type]
                + _mean_abs_joint_delta(bridge_left.joints, bridge_right.joints)
                + _mean_abs_joint_delta(bridge_right.joints, right_joints)  # type: ignore[arg-type]
            ) / 3.0
            abs_j5 = min(abs(float(bridge_left.joints[4])), abs(float(bridge_right.joints[4])))
            item = {
                "max_joint_delta_deg": float(max_delta),
                "mean_joint_delta_deg": float(mean_delta),
                "leg_max_deltas_deg": tuple(float(value) for value in leg_deltas),
                "bridge_abs_j5_deg": float(abs_j5),
                "strategy": "family_pair",
                "bridge_left_family": tuple(int(value) for value in bridge_left.config_flags),
                "bridge_right_family": tuple(int(value) for value in bridge_right.config_flags),
                "bridge_left_joints_deg": tuple(float(value) for value in bridge_left.joints),
                "bridge_right_joints_deg": tuple(float(value) for value in bridge_right.joints),
                "ratio": float(spec.ratio),
                "translation_offset_mm": tuple(float(value) for value in spec.translation_offset_mm),
                "rotation_offset_deg": tuple(float(value) for value in spec.rotation_offset_deg),
                "rotation_mode": spec.rotation_mode,
                **detour_metrics,
            }
            if best is None or _result_sort_key(item) < _result_sort_key(best):
                best = item
    return best


def _best_singular_phase_bridge(
    segment: dict[str, object],
    spec: _BridgeSpec,
    candidates,
    *,
    phase_step_limit_deg: float,
    phase_max_insertions: int,
) -> dict[str, object] | None:
    context = _WORKER_CONTEXT
    assert context is not None
    left_joints = segment["left_joints"]  # type: ignore[assignment]
    right_joints = segment["right_joints"]  # type: ignore[assignment]
    left_family = tuple(segment["left_family"])  # type: ignore[arg-type]
    right_family = tuple(segment["right_family"])  # type: ignore[arg-type]
    best: dict[str, object] | None = None

    left_bridge_candidates = [candidate for candidate in candidates if candidate.config_flags == left_family]
    right_bridge_candidates = [candidate for candidate in candidates if candidate.config_flags == right_family]
    if not left_bridge_candidates or not right_bridge_candidates:
        return None

    for bridge_left in left_bridge_candidates:
        for bridge_right in right_bridge_candidates:
            if _max_abs_joint_delta(bridge_left.joints[:3], bridge_right.joints[:3]) > 1e-5:
                continue

            base_singular_left = list(float(value) for value in bridge_left.joints)
            base_singular_right = list(float(value) for value in bridge_right.joints)
            base_singular_left[4] = 0.0
            base_singular_right[4] = 0.0
            lower_limits = tuple(float(value) for value in context.lower_limits[: context.joint_count])
            upper_limits = tuple(float(value) for value in context.upper_limits[: context.joint_count])
            for singular_left in _singular_wrist_turn_variants(
                base_singular_left,
                lower_limits=lower_limits,
                upper_limits=upper_limits,
            ):
                for singular_right in _singular_wrist_turn_variants(
                    base_singular_right,
                    lower_limits=lower_limits,
                    upper_limits=upper_limits,
                ):
                    if abs((singular_left[3] + singular_left[5]) - (singular_right[3] + singular_right[5])) > 1e-4:
                        continue

                    singular_pose = context.robot.PoseFromJointsInFrame(
                        singular_left,
                        context.tool_pose,
                        context.reference_pose,
                    )
                    singular_pose_row = _pose_row_from_pose_object(singular_pose)
                    mirror_pose = context.robot.PoseFromJointsInFrame(
                        singular_right,
                        context.tool_pose,
                        context.reference_pose,
                    )
                    mirror_pose_row = _pose_row_from_pose_object(mirror_pose)
                    if _point_distance(_translation(singular_pose_row), _translation(mirror_pose_row)) > 1e-3:
                        continue

                    detour_metrics = _detour_metrics(
                        segment["left_pose_row"],  # type: ignore[arg-type]
                        segment["right_pose_row"],  # type: ignore[arg-type]
                        singular_pose_row,
                    )
                    phase_delta = _max_abs_joint_delta(singular_left, singular_right)
                    phase_insertions = min(
                        max(0, int(math.ceil(phase_delta / max(phase_step_limit_deg, 1e-9))) - 1),
                        int(phase_max_insertions),
                    )
                    phase_leg_delta = phase_delta / float(phase_insertions + 1)
                    leg_deltas = (
                        _max_abs_joint_delta(left_joints, singular_left),  # type: ignore[arg-type]
                        float(phase_leg_delta),
                        _max_abs_joint_delta(singular_right, right_joints),  # type: ignore[arg-type]
                    )
                    max_delta = max(leg_deltas)
                    mean_delta = (
                        _mean_abs_joint_delta(left_joints, singular_left)  # type: ignore[arg-type]
                        + float(phase_leg_delta)
                        + _mean_abs_joint_delta(singular_right, right_joints)  # type: ignore[arg-type]
                    ) / 3.0
                    item = {
                        "max_joint_delta_deg": float(max_delta),
                        "mean_joint_delta_deg": float(mean_delta),
                        "leg_max_deltas_deg": tuple(float(value) for value in leg_deltas),
                        "bridge_abs_j5_deg": 0.0,
                        "strategy": "singular_phase",
                        "phase_insertions": int(phase_insertions),
                        "phase_raw_delta_deg": float(phase_delta),
                        "bridge_left_family": tuple(int(value) for value in bridge_left.config_flags),
                        "bridge_right_family": tuple(int(value) for value in bridge_right.config_flags),
                        "bridge_left_joints_deg": tuple(float(value) for value in singular_left),
                        "bridge_right_joints_deg": tuple(float(value) for value in singular_right),
                        "ratio": float(spec.ratio),
                        "translation_offset_mm": tuple(float(value) for value in spec.translation_offset_mm),
                        "rotation_offset_deg": tuple(float(value) for value in spec.rotation_offset_deg),
                        "rotation_mode": spec.rotation_mode,
                        **detour_metrics,
                    }
                    if best is None or _result_sort_key(item) < _result_sort_key(best):
                        best = item
    return best


def _singular_wrist_turn_variants(
    joints: Sequence[float],
    *,
    lower_limits: Sequence[float],
    upper_limits: Sequence[float],
) -> tuple[tuple[float, ...], ...]:
    offsets = (-720.0, -360.0, 0.0, 360.0, 720.0)
    variants: list[tuple[float, ...]] = []
    seen: set[tuple[float, ...]] = set()
    for joint4_offset in offsets:
        for joint6_offset in offsets:
            candidate = [float(value) for value in joints]
            candidate[3] += joint4_offset
            candidate[5] += joint6_offset
            if any(
                value < float(lower) - 1e-9 or value > float(upper) + 1e-9
                for value, lower, upper in zip(candidate, lower_limits, upper_limits)
            ):
                continue
            key = tuple(round(float(value), _IK_DEDUP_DECIMALS) for value in candidate)
            if key in seen:
                continue
            seen.add(key)
            variants.append(tuple(candidate))
    return tuple(variants)


def _result_sort_key(item: dict[str, object]) -> tuple[float, ...]:
    return (
        float(item["max_joint_delta_deg"]),
        float(item["mean_joint_delta_deg"]),
        float(item["bridge_abs_j5_deg"]),
        float(item["max_deviation_mm"]),
        float(item["path_ratio"]),
    )


def _passes_detour_filter(
    metrics: dict[str, object],
    *,
    max_detour_mm: float,
    max_path_ratio: float,
) -> bool:
    return (
        float(metrics["max_deviation_mm"]) <= float(max_detour_mm) + 1e-9
        and float(metrics["path_ratio"]) <= float(max_path_ratio) + 1e-9
    )


def _build_segment_payload(
    result: ProfileEvaluationResult,
    left_index: int,
    right_index: int,
) -> dict[str, object]:
    assert result.pose_rows is not None
    left_entry = result.selected_path[left_index]
    right_entry = result.selected_path[right_index]
    return {
        "left_label": str(result.row_labels[left_index]),
        "right_label": str(result.row_labels[right_index]),
        "left_pose_row": dict(result.pose_rows[left_index]),
        "right_pose_row": dict(result.pose_rows[right_index]),
        "left_joints": tuple(float(value) for value in left_entry.joints),
        "right_joints": tuple(float(value) for value in right_entry.joints),
        "left_family": tuple(int(value) for value in left_entry.config_flags),
        "right_family": tuple(int(value) for value in right_entry.config_flags),
    }


def _build_bridge_pose_row(
    segment: dict[str, object],
    spec: _BridgeSpec,
) -> tuple[dict[str, float], dict[str, float]]:
    left_row = segment["left_pose_row"]  # type: ignore[assignment]
    right_row = segment["right_pose_row"]  # type: ignore[assignment]
    left_rotation, left_translation = _pose_row_to_rotation_translation(left_row)
    right_rotation, right_translation = _pose_row_to_rotation_translation(right_row)
    base_rotation = _quaternion_to_rotation_matrix(
        _slerp_quaternion(
            _rotation_matrix_to_quaternion(left_rotation),
            _rotation_matrix_to_quaternion(right_rotation),
            spec.ratio,
        )
    )
    delta_rotation = _rotation_matrix_from_xyz_offset_deg(spec.rotation_offset_deg)
    if spec.rotation_mode == "frame":
        bridge_rotation = _multiply_rotation_matrices(delta_rotation, base_rotation)
    else:
        bridge_rotation = _multiply_rotation_matrices(base_rotation, delta_rotation)

    bridge_translation = tuple(
        float(left_value + (right_value - left_value) * spec.ratio + offset_value)
        for left_value, right_value, offset_value in zip(
            left_translation,
            right_translation,
            spec.translation_offset_mm,
        )
    )
    bridge_row = _pose_row_from_rotation_translation(bridge_rotation, bridge_translation)
    detour_metrics = _detour_metrics(left_row, right_row, bridge_row)
    return bridge_row, detour_metrics


def _detour_metrics(
    left_row: dict[str, float],
    right_row: dict[str, float],
    bridge_row: dict[str, float],
) -> dict[str, float]:
    left = _translation(left_row)
    bridge = _translation(bridge_row)
    right = _translation(right_row)
    direct = _point_distance(left, right)
    path = _point_distance(left, bridge) + _point_distance(bridge, right)
    return {
        "direct_distance_mm": float(direct),
        "path_distance_mm": float(path),
        "path_ratio": float(path / max(direct, 1e-9)),
        "max_deviation_mm": float(_point_to_segment_distance(bridge, left, right)),
    }


def _build_bridge_specs(args: argparse.Namespace) -> tuple[_BridgeSpec, ...]:
    ratios = _parse_float_tuple(args.ratios)
    translation_values = _symmetric_values(
        max_abs=float(args.translation_max_mm),
        step=float(args.translation_step_mm),
    )
    rotation_values = _parse_float_tuple(args.rotation_values_deg)
    translation_offsets = _translation_offsets(translation_values, args.translation_axes)
    rotation_offsets = _rotation_offsets(rotation_values, args.rotation_grid)
    specs = []
    for ratio in ratios:
        for translation_offset in translation_offsets:
            for rotation_mode in args.rotation_modes:
                for rotation_offset in rotation_offsets:
                    specs.append(
                        _BridgeSpec(
                            ratio=float(ratio),
                            translation_offset_mm=tuple(float(value) for value in translation_offset),
                            rotation_offset_deg=tuple(float(value) for value in rotation_offset),
                            rotation_mode=str(rotation_mode),
                        )
                    )
    return tuple(specs)


def _translation_offsets(
    values: Sequence[float],
    axes: str,
) -> tuple[tuple[float, float, float], ...]:
    enabled = set() if axes == "none" else set(axes)
    offsets = []
    for x_value in values if "x" in enabled else (0.0,):
        for y_value in values if "y" in enabled else (0.0,):
            for z_value in values if "z" in enabled else (0.0,):
                offsets.append((float(x_value), float(y_value), float(z_value)))
    return tuple(offsets)


def _rotation_offsets(
    values: Sequence[float],
    grid: str,
) -> tuple[tuple[float, float, float], ...]:
    if grid == "none":
        return ((0.0, 0.0, 0.0),)
    axes = {
        "single_axis": ("single_axis",),
        "xy": ("x", "y"),
        "xz": ("x", "z"),
        "yz": ("y", "z"),
        "full": ("x", "y", "z"),
    }[grid]
    offsets: set[tuple[float, float, float]] = set()
    if axes == ("single_axis",):
        offsets.add((0.0, 0.0, 0.0))
        for value in values:
            offsets.add((float(value), 0.0, 0.0))
            offsets.add((0.0, float(value), 0.0))
            offsets.add((0.0, 0.0, float(value)))
    else:
        for x_value in values if "x" in axes else (0.0,):
            for y_value in values if "y" in axes else (0.0,):
                for z_value in values if "z" in axes else (0.0,):
                    offsets.add((float(x_value), float(y_value), float(z_value)))
    return tuple(sorted(offsets))


def _symmetric_values(*, max_abs: float, step: float) -> tuple[float, ...]:
    if max_abs <= 0.0:
        return (0.0,)
    count = int(math.floor(max_abs / max(step, 1e-9)))
    values = {0.0}
    for index in range(1, count + 1):
        value = round(index * step, 9)
        values.add(value)
        values.add(-value)
    values.add(float(max_abs))
    values.add(-float(max_abs))
    return tuple(sorted(values))


def _parse_float_tuple(raw: str) -> tuple[float, ...]:
    values = tuple(float(part.strip()) for part in raw.split(",") if part.strip())
    if not values:
        raise ValueError(f"Expected at least one comma-separated value, got {raw!r}")
    return values


def _translation(row: dict[str, float]) -> tuple[float, float, float]:
    return (float(row["x_mm"]), float(row["y_mm"]), float(row["z_mm"]))


def _pose_row_from_pose_object(pose) -> dict[str, float]:
    return {
        "x_mm": float(pose[0, 3]),
        "y_mm": float(pose[1, 3]),
        "z_mm": float(pose[2, 3]),
        "r11": float(pose[0, 0]),
        "r12": float(pose[0, 1]),
        "r13": float(pose[0, 2]),
        "r21": float(pose[1, 0]),
        "r22": float(pose[1, 1]),
        "r23": float(pose[1, 2]),
        "r31": float(pose[2, 0]),
        "r32": float(pose[2, 1]),
        "r33": float(pose[2, 2]),
    }


def _point_distance(first: Sequence[float], second: Sequence[float]) -> float:
    return math.sqrt(sum((float(b) - float(a)) ** 2 for a, b in zip(first, second)))


def _point_to_segment_distance(
    point: Sequence[float],
    start: Sequence[float],
    end: Sequence[float],
) -> float:
    direction = tuple(float(b) - float(a) for a, b in zip(start, end))
    length_sq = sum(value * value for value in direction)
    if length_sq <= 1e-12:
        return _point_distance(point, start)
    ratio = sum((float(p) - float(a)) * d for p, a, d in zip(point, start, direction)) / length_sq
    ratio = max(0.0, min(1.0, ratio))
    projection = tuple(float(a) + ratio * d for a, d in zip(start, direction))
    return _point_distance(point, projection)


def _max_abs_joint_delta(first: Sequence[float], second: Sequence[float]) -> float:
    return max((abs(float(b) - float(a)) for a, b in zip(first, second)), default=0.0)


if __name__ == "__main__":
    raise SystemExit(main())
