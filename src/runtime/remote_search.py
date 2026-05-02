from __future__ import annotations

import argparse
import math
from collections import Counter
from typing import Iterable

from src.core.collab_models import (
    EvaluationBatchRequest,
    EvaluationBatchResult,
    ProfileEvaluationRequest,
    ProfileEvaluationResult,
    RemoteSearchRequest,
    RemoteSearchSummary,
    load_json_file,
    write_json_file,
)


def _result_policy_priority(result: ProfileEvaluationResult) -> float:
    metadata = dict(getattr(result, "metadata", {}) or {})
    branch_policy = str(metadata.get("branch_policy", ""))
    if branch_policy == "require_lower":
        return 0.0
    if branch_policy == "prefer_lower":
        return 1.0
    if branch_policy == "unguided":
        return 2.0
    if str(metadata.get("strategy", "")) == "nominal":
        return 3.0
    return 4.0


def _result_sort_key(result: ProfileEvaluationResult) -> tuple[object, ...]:
    return (
        float(result.invalid_row_count),
        float(result.ik_empty_row_count),
        float(result.bridge_like_segments),
        float(getattr(result, "big_circle_step_count", 0)),
        float(getattr(result, "posture_stress_score", 0.0)),
        float(result.worst_joint_step_deg),
        float(result.mean_joint_step_deg),
        float(result.config_switches),
        _result_policy_priority(result),
        float(result.total_cost),
        str(result.request_id),
        float(result.timing_seconds),
    )


def _clone_request(
    base_request: ProfileEvaluationRequest,
    *,
    request_id: str,
    profile: tuple[tuple[float, float], ...],
    metadata: dict[str, object],
    motion_settings_updates: dict[str, object] | None = None,
) -> ProfileEvaluationRequest:
    # Batch candidates only evaluate profiles quickly — repairs are expensive and should
    # only run on the final best candidate during program generation.
    motion_settings = dict(base_request.motion_settings)
    # Let the outer evaluation batch own process-level parallelism; avoid nested pools
    # inside each candidate process.
    motion_settings["local_parallel_workers"] = 1
    motion_settings["local_parallel_min_batch_size"] = 999999
    motion_settings.update(motion_settings_updates or {})
    return ProfileEvaluationRequest(
        request_id=request_id,
        robot_name=base_request.robot_name,
        frame_name=base_request.frame_name,
        motion_settings=motion_settings,
        reference_pose_rows=tuple(dict(row) for row in base_request.reference_pose_rows),
        frame_a_origin_yz_profile_mm=profile,
        row_labels=tuple(base_request.row_labels),
        inserted_flags=tuple(base_request.inserted_flags),
        strategy="exact_profile",
        start_joints=base_request.start_joints,
        run_window_repair=False,
        run_inserted_repair=False,
        include_pose_rows_in_result=False,
        create_program=False,
        program_name=base_request.program_name,
        optimized_csv_path=base_request.optimized_csv_path,
        metadata=metadata,
    )


def _max_abs_offset_mm(base_request: ProfileEvaluationRequest) -> float:
    motion_settings = base_request.motion_settings
    envelope_values = tuple(
        float(value)
        for value in motion_settings.get("frame_a_origin_yz_envelope_schedule_mm", (6.0,))
    )
    return max((value for value in envelope_values if value >= 0.0), default=0.0)


def _clamp_profile(
    profile: tuple[tuple[float, float], ...],
    *,
    max_abs_offset_mm: float,
) -> tuple[tuple[float, float], ...]:
    return tuple(
        (
            round(max(-max_abs_offset_mm, min(max_abs_offset_mm, float(dy_mm))), 6),
            round(max(-max_abs_offset_mm, min(max_abs_offset_mm, float(dz_mm))), 6),
        )
        for dy_mm, dz_mm in profile
    )


def _profile_continuity_step_limit(
    motion_settings: dict[str, object],
) -> float:
    raw_steps = motion_settings.get("frame_a_origin_yz_step_schedule_mm", (2.0, 1.0))
    try:
        step_values = tuple(raw_steps)
    except TypeError:
        step_values = (raw_steps,)

    positive_steps: list[float] = []
    for value in step_values:
        try:
            step_mm = float(value)
        except (TypeError, ValueError):
            continue
        if step_mm > 0.0:
            positive_steps.append(step_mm)
    positive_steps.sort()

    reference_step_mm = (
        positive_steps[min(1, len(positive_steps) - 1)]
        if positive_steps
        else 1.0
    )
    return max(2.0, 2.0 * reference_step_mm)


def _profile_is_continuous_enough(
    profile: tuple[tuple[float, float], ...],
    *,
    max_step_mm: float,
) -> bool:
    if len(profile) <= 1:
        return True
    return all(
        math.hypot(
            float(current_offset[0]) - float(previous_offset[0]),
            float(current_offset[1]) - float(previous_offset[1]),
        )
        <= max_step_mm + 1e-9
        for previous_offset, current_offset in zip(profile, profile[1:])
    )


def _finalize_candidate_profile(
    profile: tuple[tuple[float, float], ...],
    *,
    base_profile: tuple[tuple[float, float], ...],
    max_abs_offset_mm: float,
    lock_endpoints: bool,
    max_profile_step_mm: float,
) -> tuple[tuple[float, float], ...] | None:
    if len(profile) != len(base_profile):
        return None

    finalized_profile = list(_clamp_profile(profile, max_abs_offset_mm=max_abs_offset_mm))
    if lock_endpoints and len(finalized_profile) == len(base_profile) and finalized_profile:
        finalized_profile[0] = (
            round(float(base_profile[0][0]), 6),
            round(float(base_profile[0][1]), 6),
        )
        finalized_profile[-1] = (
            round(float(base_profile[-1][0]), 6),
            round(float(base_profile[-1][1]), 6),
        )

    finalized = tuple(finalized_profile)
    if not _profile_is_continuous_enough(
        finalized,
        max_step_mm=max_profile_step_mm,
    ):
        return None
    return finalized


def _candidate_sort_key(candidate: ProfileEvaluationRequest) -> tuple[object, ...]:
    strategy = str(candidate.metadata.get("strategy", ""))
    strategy_priority = {
        "nominal": 0.0,
        "branch_policy": 0.5,
        "active_set_window": 0.8,
        "active_set_ramp": 0.9,
        "local_window": 1.0,
        "local_corridor_asym": 2.0,
        "local_basis": 3.0,
    }.get(strategy, 9.0)
    return (
        strategy_priority,
        _candidate_policy_priority(candidate),
        abs(float(candidate.metadata.get("dy_mm", 0.0)))
        + abs(float(candidate.metadata.get("dz_mm", 0.0))),
        abs(float(candidate.metadata.get("left_dy_mm", 0.0)))
        + abs(float(candidate.metadata.get("left_dz_mm", 0.0)))
        + abs(float(candidate.metadata.get("right_dy_mm", 0.0)))
        + abs(float(candidate.metadata.get("right_dz_mm", 0.0))),
        candidate.request_id,
    )


def _candidate_policy_priority(candidate: ProfileEvaluationRequest) -> float:
    branch_policy = str(candidate.metadata.get("branch_policy", ""))
    if branch_policy == "require_lower":
        return 0.0
    if branch_policy == "prefer_lower":
        return 1.0
    if branch_policy == "unguided":
        return 2.0
    return 3.0


def _candidate_group_key(candidate: ProfileEvaluationRequest) -> str:
    strategy = str(candidate.metadata.get("strategy", ""))
    if strategy in {"active_set_window", "active_set_ramp"}:
        return strategy
    if strategy != "local_window":
        return strategy or "other"

    dy_mm = float(candidate.metadata.get("dy_mm", 0.0))
    dz_mm = float(candidate.metadata.get("dz_mm", 0.0))
    if abs(dy_mm) > 1e-9 and abs(dz_mm) <= 1e-9:
        return "local_window_y"
    if abs(dy_mm) <= 1e-9 and abs(dz_mm) > 1e-9:
        return "local_window_z"
    return "local_window_diag"


def _choose_focus_segments(
    request: RemoteSearchRequest,
) -> tuple[int, ...]:
    baseline = request.baseline_result
    row_labels = list(request.base_request.row_labels)
    label_to_index = {label: index for index, label in enumerate(row_labels)}

    focus_segments: list[int] = []
    if baseline is not None and baseline.failing_segments:
        focus_segments.extend(segment.segment_index for segment in baseline.failing_segments[:6])

    if baseline is not None and baseline.ik_empty_rows:
        for label in baseline.ik_empty_rows[:6]:
            index = label_to_index.get(label)
            if index is None:
                continue
            focus_segments.append(max(0, min(len(row_labels) - 2, index - 1)))

    if baseline is not None and baseline.selected_path:
        focus_segments.extend(
            _largest_joint_step_segments(
                baseline.selected_path,
                max_segments=6,
            )
        )

    for preferred_pair in ("381", "382", "380"):
        index = label_to_index.get(preferred_pair)
        if index is not None:
            focus_segments.append(max(0, min(len(row_labels) - 2, index)))

    if not focus_segments and len(row_labels) >= 2:
        focus_segments.append(max(0, len(row_labels) // 2 - 1))

    unique_segments: list[int] = []
    seen: set[int] = set()
    for segment in focus_segments:
        if segment in seen:
            continue
        seen.add(segment)
        unique_segments.append(segment)
    return tuple(unique_segments[:6])


def _largest_joint_step_segments(
    selected_path: Iterable[object],
    *,
    max_segments: int,
) -> tuple[int, ...]:
    path = tuple(selected_path)
    if len(path) < 2 or max_segments <= 0:
        return ()

    ranked_segments: list[tuple[float, int]] = []
    for segment_index, (previous_entry, current_entry) in enumerate(zip(path, path[1:])):
        previous_joints = tuple(float(value) for value in getattr(previous_entry, "joints", ()))
        current_joints = tuple(float(value) for value in getattr(current_entry, "joints", ()))
        if not previous_joints or not current_joints:
            continue
        max_delta = max(
            (
                abs(current - previous)
                for previous, current in zip(previous_joints, current_joints)
            ),
            default=0.0,
        )
        ranked_segments.append((float(max_delta), int(segment_index)))

    ranked_segments.sort(key=lambda item: (-item[0], item[1]))
    return tuple(segment_index for _max_delta, segment_index in ranked_segments[:max_segments])


def _build_nominal_candidate(request: RemoteSearchRequest) -> ProfileEvaluationRequest:
    row_count = len(request.base_request.reference_pose_rows)
    return _clone_request(
        request.base_request,
        request_id=f"round{request.round_index}_nominal",
        profile=tuple((0.0, 0.0) for _ in range(row_count)),
        metadata={
            "strategy": "nominal",
            "round_index": int(request.round_index),
        },
    )


def _build_initial_policy_candidates(
    request: RemoteSearchRequest,
) -> list[ProfileEvaluationRequest]:
    row_count = len(request.base_request.reference_pose_rows)
    zero_profile = tuple((0.0, 0.0) for _ in range(row_count))
    candidates = [_build_nominal_candidate(request)]
    policy_specs: tuple[tuple[str, dict[str, object]], ...] = (
        (
            "prefer_lower",
            {
                "preferred_lower_config_flag": 1,
                "lower_config_preference_weight": 250.0,
                "use_guided_config_path": False,
            },
        ),
        (
            "require_lower",
            {
                "require_lower_config_flag": 1,
                "use_guided_config_path": False,
            },
        ),
        (
            "unguided",
            {
                "use_guided_config_path": False,
            },
        ),
    )
    for policy_name, motion_updates in policy_specs:
        candidates.append(
            _clone_request(
                request.base_request,
                request_id=f"round{request.round_index}_{policy_name}",
                profile=zero_profile,
                metadata={
                    "strategy": "branch_policy",
                    "branch_policy": policy_name,
                    "round_index": int(request.round_index),
                },
                motion_settings_updates=motion_updates,
            )
        )
    return candidates


def _apply_local_window_delta(
    profile: tuple[tuple[float, float], ...],
    *,
    segment_index: int,
    window_radius: int,
    dy_mm: float,
    dz_mm: float,
) -> tuple[tuple[float, float], ...]:
    updated_profile = [list(offset) for offset in profile]
    center = min(len(updated_profile) - 1, segment_index + 1)
    for row_index in range(
        max(0, center - window_radius),
        min(len(updated_profile), center + window_radius + 1),
    ):
        distance = abs(row_index - center)
        weight = max(0.0, 1.0 - distance / max(1, window_radius + 1))
        updated_profile[row_index][0] += dy_mm * weight
        updated_profile[row_index][1] += dz_mm * weight
    return tuple((round(offset[0], 6), round(offset[1], 6)) for offset in updated_profile)


def _apply_asymmetric_corridor_delta(
    profile: tuple[tuple[float, float], ...],
    *,
    segment_index: int,
    window_radius: int,
    left_dy_mm: float,
    left_dz_mm: float,
    right_dy_mm: float,
    right_dz_mm: float,
) -> tuple[tuple[float, float], ...]:
    updated_profile = [list(offset) for offset in profile]
    center = min(len(updated_profile) - 1, segment_index + 1)
    corridor_start = max(0, center - window_radius)
    corridor_end = min(len(updated_profile) - 1, center + window_radius)
    for row_index in range(corridor_start, corridor_end + 1):
        if row_index <= center:
            distance = center - row_index
            side_radius = max(1, center - corridor_start + 1)
            weight = max(0.0, 1.0 - distance / side_radius)
            updated_profile[row_index][0] += left_dy_mm * weight
            updated_profile[row_index][1] += left_dz_mm * weight
        else:
            distance = row_index - center
            side_radius = max(1, corridor_end - center + 1)
            weight = max(0.0, 1.0 - distance / side_radius)
            updated_profile[row_index][0] += right_dy_mm * weight
            updated_profile[row_index][1] += right_dz_mm * weight
    return tuple((round(offset[0], 6), round(offset[1], 6)) for offset in updated_profile)


def _apply_two_node_basis_delta(
    profile: tuple[tuple[float, float], ...],
    *,
    segment_index: int,
    window_radius: int,
    start_dy_mm: float,
    start_dz_mm: float,
    end_dy_mm: float,
    end_dz_mm: float,
) -> tuple[tuple[float, float], ...]:
    updated_profile = [list(offset) for offset in profile]
    center = min(len(updated_profile) - 1, segment_index + 1)
    corridor_start = max(0, center - window_radius)
    corridor_end = min(len(updated_profile) - 1, center + window_radius)
    corridor_length = max(1, corridor_end - corridor_start)
    for row_index in range(corridor_start, corridor_end + 1):
        alpha = (row_index - corridor_start) / corridor_length
        updated_profile[row_index][0] += ((1.0 - alpha) * start_dy_mm) + (alpha * end_dy_mm)
        updated_profile[row_index][1] += ((1.0 - alpha) * start_dz_mm) + (alpha * end_dz_mm)
    return tuple((round(offset[0], 6), round(offset[1], 6)) for offset in updated_profile)


def _active_set_cluster_bounds(
    profile: tuple[tuple[float, float], ...],
    *,
    segment_indices: tuple[int, ...],
    window_radius: int,
) -> tuple[int, int]:
    if not segment_indices:
        return (0, max(0, len(profile) - 1))
    centers = [min(len(profile) - 1, max(0, int(segment) + 1)) for segment in segment_indices]
    return (
        max(0, min(centers) - max(0, int(window_radius))),
        min(len(profile) - 1, max(centers) + max(0, int(window_radius))),
    )


def _segment_label(
    request: RemoteSearchRequest,
    segment_index: int,
) -> str:
    row_labels = request.base_request.row_labels
    left_index = max(0, min(len(row_labels) - 1, int(segment_index)))
    right_index = max(0, min(len(row_labels) - 1, int(segment_index) + 1))
    return f"{row_labels[left_index]}->{row_labels[right_index]}"


def _active_segment_label(
    request: RemoteSearchRequest,
    segment_indices: tuple[int, ...],
) -> str:
    row_labels = request.base_request.row_labels
    left_index = max(0, min(len(row_labels) - 1, min(segment_indices)))
    right_index = max(0, min(len(row_labels) - 1, max(segment_indices) + 1))
    return f"{row_labels[left_index]}->{row_labels[right_index]}"


def _append_profile_candidate(
    candidates: list[ProfileEvaluationRequest],
    *,
    request: RemoteSearchRequest,
    raw_profile: tuple[tuple[float, float], ...],
    base_profile: tuple[tuple[float, float], ...],
    max_abs_offset_mm: float,
    lock_endpoints: bool,
    max_profile_step_mm: float,
    seen_profiles: set[tuple[tuple[float, float], ...]],
    request_id_prefix: str,
    ordinal: int,
    metadata: dict[str, object],
) -> int:
    profile = _finalize_candidate_profile(
        raw_profile,
        base_profile=base_profile,
        max_abs_offset_mm=max_abs_offset_mm,
        lock_endpoints=lock_endpoints,
        max_profile_step_mm=max_profile_step_mm,
    )
    if profile is None or profile in seen_profiles:
        return ordinal

    seen_profiles.add(profile)
    next_ordinal = ordinal + 1
    candidates.append(
        _clone_request(
            request.base_request,
            request_id=f"round{request.round_index}_{request_id_prefix}_{next_ordinal:03d}",
            profile=profile,
            metadata=metadata,
        )
    )
    return next_ordinal


def _apply_active_set_window_delta(
    profile: tuple[tuple[float, float], ...],
    *,
    segment_indices: tuple[int, ...],
    window_radius: int,
    dy_mm: float,
    dz_mm: float,
) -> tuple[tuple[float, float], ...]:
    updated_profile = [list(offset) for offset in profile]
    row_weights = [0.0 for _ in updated_profile]
    for segment_index in segment_indices:
        center = min(len(updated_profile) - 1, max(0, int(segment_index) + 1))
        for row_index in range(
            max(0, center - window_radius),
            min(len(updated_profile), center + window_radius + 1),
        ):
            distance = abs(row_index - center)
            weight = max(0.0, 1.0 - distance / max(1, window_radius + 1))
            row_weights[row_index] = max(row_weights[row_index], weight)

    for row_index, weight in enumerate(row_weights):
        if weight <= 0.0:
            continue
        updated_profile[row_index][0] += dy_mm * weight
        updated_profile[row_index][1] += dz_mm * weight
    return tuple((round(offset[0], 6), round(offset[1], 6)) for offset in updated_profile)


def _apply_active_set_ramp_delta(
    profile: tuple[tuple[float, float], ...],
    *,
    segment_indices: tuple[int, ...],
    window_radius: int,
    start_dy_mm: float,
    start_dz_mm: float,
    end_dy_mm: float,
    end_dz_mm: float,
) -> tuple[tuple[float, float], ...]:
    updated_profile = [list(offset) for offset in profile]
    cluster_start, cluster_end = _active_set_cluster_bounds(
        profile,
        segment_indices=segment_indices,
        window_radius=window_radius,
    )
    cluster_span = max(1, cluster_end - cluster_start)
    for row_index in range(cluster_start, cluster_end + 1):
        alpha = (row_index - cluster_start) / cluster_span
        taper = math.sin(math.pi * alpha)
        dy_mm = ((1.0 - alpha) * start_dy_mm) + (alpha * end_dy_mm)
        dz_mm = ((1.0 - alpha) * start_dz_mm) + (alpha * end_dz_mm)
        updated_profile[row_index][0] += taper * dy_mm
        updated_profile[row_index][1] += taper * dz_mm
    return tuple((round(offset[0], 6), round(offset[1], 6)) for offset in updated_profile)


def _build_local_candidates(request: RemoteSearchRequest) -> list[ProfileEvaluationRequest]:
    motion_settings = request.base_request.motion_settings
    baseline_result = request.baseline_result
    base_profile = (
        baseline_result.frame_a_origin_yz_profile_mm
        if baseline_result is not None
        else request.base_request.frame_a_origin_yz_profile_mm
    )
    if not base_profile:
        base_profile = tuple((0.0, 0.0) for _ in request.base_request.reference_pose_rows)
    base_profile = tuple(
        (float(dy_mm), float(dz_mm))
        for dy_mm, dz_mm in base_profile
    )

    max_abs_offset_mm = _max_abs_offset_mm(request.base_request)
    max_profile_step_mm = _profile_continuity_step_limit(motion_settings)
    lock_endpoints = bool(
        motion_settings.get("lock_frame_a_origin_yz_profile_endpoints", True)
    )
    focus_segments = _choose_focus_segments(request)
    step_values = tuple(
        float(value)
        for value in motion_settings.get("frame_a_origin_yz_step_schedule_mm", (2.0, 1.0))
    )
    window_radius = int(motion_settings.get("frame_a_origin_yz_window_radius", 8))
    candidates: list[ProfileEvaluationRequest] = []
    seen_profiles: set[tuple[tuple[float, float], ...]] = set()
    ordinal = 0
    active_segments = tuple(int(segment) for segment in focus_segments[:6])

    if len(active_segments) >= 2:
        active_label = _active_segment_label(request, active_segments)
        for step_mm in step_values[:2]:
            for dy_mm, dz_mm in (
                (step_mm, 0.0),
                (-step_mm, 0.0),
                (0.0, step_mm),
                (0.0, -step_mm),
                (step_mm, step_mm),
                (step_mm, -step_mm),
                (-step_mm, step_mm),
                (-step_mm, -step_mm),
            ):
                profile = _apply_active_set_window_delta(
                    base_profile,
                    segment_indices=active_segments,
                    window_radius=window_radius,
                    dy_mm=dy_mm,
                    dz_mm=dz_mm,
                )
                ordinal = _append_profile_candidate(
                    candidates,
                    request=request,
                    raw_profile=profile,
                    base_profile=base_profile,
                    max_abs_offset_mm=max_abs_offset_mm,
                    lock_endpoints=lock_endpoints,
                    max_profile_step_mm=max_profile_step_mm,
                    seen_profiles=seen_profiles,
                    request_id_prefix="active",
                    ordinal=ordinal,
                    metadata={
                        "strategy": "active_set_window",
                        "segment_indices": active_segments,
                        "active_segment_count": len(active_segments),
                        "segment_label": active_label,
                        "window_radius": window_radius,
                        "dy_mm": dy_mm,
                        "dz_mm": dz_mm,
                        "step_mm": step_mm,
                    },
                )

            for start_dy_mm, start_dz_mm, end_dy_mm, end_dz_mm in (
                (step_mm, 0.0, -step_mm, 0.0),
                (-step_mm, 0.0, step_mm, 0.0),
                (0.0, step_mm, 0.0, -step_mm),
                (0.0, -step_mm, 0.0, step_mm),
                (step_mm, 0.0, 0.0, 0.0),
                (-step_mm, 0.0, 0.0, 0.0),
                (0.0, step_mm, 0.0, 0.0),
                (0.0, -step_mm, 0.0, 0.0),
                (0.0, 0.0, step_mm, 0.0),
                (0.0, 0.0, -step_mm, 0.0),
                (0.0, 0.0, 0.0, step_mm),
                (0.0, 0.0, 0.0, -step_mm),
            ):
                profile = _apply_active_set_ramp_delta(
                    base_profile,
                    segment_indices=active_segments,
                    window_radius=window_radius,
                    start_dy_mm=start_dy_mm,
                    start_dz_mm=start_dz_mm,
                    end_dy_mm=end_dy_mm,
                    end_dz_mm=end_dz_mm,
                )
                ordinal = _append_profile_candidate(
                    candidates,
                    request=request,
                    raw_profile=profile,
                    base_profile=base_profile,
                    max_abs_offset_mm=max_abs_offset_mm,
                    lock_endpoints=lock_endpoints,
                    max_profile_step_mm=max_profile_step_mm,
                    seen_profiles=seen_profiles,
                    request_id_prefix="active",
                    ordinal=ordinal,
                    metadata={
                        "strategy": "active_set_ramp",
                        "segment_indices": active_segments,
                        "active_segment_count": len(active_segments),
                        "segment_label": active_label,
                        "window_radius": window_radius,
                        "left_dy_mm": start_dy_mm,
                        "left_dz_mm": start_dz_mm,
                        "right_dy_mm": end_dy_mm,
                        "right_dz_mm": end_dz_mm,
                        "step_mm": step_mm,
                    },
                )

    for segment_index in focus_segments:
        for step_mm in step_values[:2]:
            for dy_mm, dz_mm in (
                (step_mm, 0.0),
                (-step_mm, 0.0),
                (0.0, step_mm),
                (0.0, -step_mm),
                (step_mm, step_mm),
                (step_mm, -step_mm),
                (-step_mm, step_mm),
                (-step_mm, -step_mm),
            ):
                profile = _apply_local_window_delta(
                    base_profile,
                    segment_index=segment_index,
                    window_radius=window_radius,
                    dy_mm=dy_mm,
                    dz_mm=dz_mm,
                )
                ordinal = _append_profile_candidate(
                    candidates,
                    request=request,
                    raw_profile=profile,
                    base_profile=base_profile,
                    max_abs_offset_mm=max_abs_offset_mm,
                    lock_endpoints=lock_endpoints,
                    max_profile_step_mm=max_profile_step_mm,
                    seen_profiles=seen_profiles,
                    request_id_prefix="local",
                    ordinal=ordinal,
                    metadata={
                        "strategy": "local_window",
                        "segment_index": segment_index,
                        "segment_label": _segment_label(request, segment_index),
                        "window_radius": window_radius,
                        "dy_mm": dy_mm,
                        "dz_mm": dz_mm,
                        "step_mm": step_mm,
                    },
                )

            for left_dy_mm, left_dz_mm, right_dy_mm, right_dz_mm in (
                (step_mm, 0.0, 0.0, 0.0),
                (-step_mm, 0.0, 0.0, 0.0),
                (0.0, step_mm, 0.0, 0.0),
                (0.0, -step_mm, 0.0, 0.0),
                (0.0, 0.0, step_mm, 0.0),
                (0.0, 0.0, -step_mm, 0.0),
                (0.0, 0.0, 0.0, step_mm),
                (0.0, 0.0, 0.0, -step_mm),
                (step_mm, 0.0, -step_mm, 0.0),
                (-step_mm, 0.0, step_mm, 0.0),
                (0.0, step_mm, 0.0, -step_mm),
                (0.0, -step_mm, 0.0, step_mm),
            ):
                profile = _apply_asymmetric_corridor_delta(
                    base_profile,
                    segment_index=segment_index,
                    window_radius=window_radius,
                    left_dy_mm=left_dy_mm,
                    left_dz_mm=left_dz_mm,
                    right_dy_mm=right_dy_mm,
                    right_dz_mm=right_dz_mm,
                )
                ordinal = _append_profile_candidate(
                    candidates,
                    request=request,
                    raw_profile=profile,
                    base_profile=base_profile,
                    max_abs_offset_mm=max_abs_offset_mm,
                    lock_endpoints=lock_endpoints,
                    max_profile_step_mm=max_profile_step_mm,
                    seen_profiles=seen_profiles,
                    request_id_prefix="corridor",
                    ordinal=ordinal,
                    metadata={
                        "strategy": "local_corridor_asym",
                        "segment_index": segment_index,
                        "segment_label": _segment_label(request, segment_index),
                        "window_radius": window_radius,
                        "left_dy_mm": left_dy_mm,
                        "left_dz_mm": left_dz_mm,
                        "right_dy_mm": right_dy_mm,
                        "right_dz_mm": right_dz_mm,
                        "step_mm": step_mm,
                    },
                )

            for start_dy_mm, start_dz_mm, end_dy_mm, end_dz_mm in (
                (step_mm, 0.0, 0.0, 0.0),
                (-step_mm, 0.0, 0.0, 0.0),
                (0.0, step_mm, 0.0, 0.0),
                (0.0, -step_mm, 0.0, 0.0),
                (0.0, 0.0, step_mm, 0.0),
                (0.0, 0.0, -step_mm, 0.0),
                (0.0, 0.0, 0.0, step_mm),
                (0.0, 0.0, 0.0, -step_mm),
                (step_mm, 0.0, -step_mm, 0.0),
                (0.0, step_mm, 0.0, -step_mm),
            ):
                profile = _apply_two_node_basis_delta(
                    base_profile,
                    segment_index=segment_index,
                    window_radius=window_radius,
                    start_dy_mm=start_dy_mm,
                    start_dz_mm=start_dz_mm,
                    end_dy_mm=end_dy_mm,
                    end_dz_mm=end_dz_mm,
                )
                ordinal = _append_profile_candidate(
                    candidates,
                    request=request,
                    raw_profile=profile,
                    base_profile=base_profile,
                    max_abs_offset_mm=max_abs_offset_mm,
                    lock_endpoints=lock_endpoints,
                    max_profile_step_mm=max_profile_step_mm,
                    seen_profiles=seen_profiles,
                    request_id_prefix="basis",
                    ordinal=ordinal,
                    metadata={
                        "strategy": "local_basis",
                        "segment_index": segment_index,
                        "segment_label": _segment_label(request, segment_index),
                        "window_radius": window_radius,
                        "left_dy_mm": start_dy_mm,
                        "left_dz_mm": start_dz_mm,
                        "right_dy_mm": end_dy_mm,
                        "right_dz_mm": end_dz_mm,
                        "step_mm": step_mm,
                    },
                )
    return candidates


def _limit_candidates_with_diversity(
    candidates: list[ProfileEvaluationRequest],
    *,
    candidate_limit: int,
    prefer_local_candidates: bool,
) -> tuple[ProfileEvaluationRequest, ...]:
    sorted_candidates = sorted(candidates, key=_candidate_sort_key)
    if not sorted_candidates:
        return ()

    if not prefer_local_candidates:
        return tuple(sorted_candidates[: max(1, candidate_limit)])

    strategy_order = (
        "nominal",
        "branch_policy",
        "active_set_window",
        "active_set_ramp",
        "local_window_y",
        "local_window_z",
        "local_corridor_asym",
        "local_basis",
        "local_window_diag",
    )
    grouped_candidates: dict[str, list[ProfileEvaluationRequest]] = {
        strategy: [] for strategy in {
            "nominal",
            "branch_policy",
            "active_set_window",
            "active_set_ramp",
            "local_window_y",
            "local_window_z",
            "local_window_diag",
            "local_corridor_asym",
            "local_basis",
        }
    }
    extra_candidates: list[ProfileEvaluationRequest] = []
    for candidate in sorted_candidates:
        strategy = _candidate_group_key(candidate)
        if strategy in grouped_candidates:
            grouped_candidates[strategy].append(candidate)
        else:
            extra_candidates.append(candidate)

    selected: list[ProfileEvaluationRequest] = []
    selected_ids: set[str] = set()
    strategy_positions = {
        strategy: 0 for strategy in grouped_candidates
    }
    while len(selected) < max(1, candidate_limit):
        added_any = False
        for strategy in strategy_order:
            strategy_candidates = grouped_candidates[strategy]
            strategy_index = strategy_positions[strategy]
            if strategy_index >= len(strategy_candidates):
                continue
            candidate = strategy_candidates[strategy_index]
            strategy_positions[strategy] = strategy_index + 1
            if candidate.request_id in selected_ids:
                continue
            selected.append(candidate)
            selected_ids.add(candidate.request_id)
            added_any = True
            if len(selected) >= max(1, candidate_limit):
                break
        if not added_any:
            break

    if len(selected) < max(1, candidate_limit):
        for candidate in sorted_candidates:
            if candidate.request_id in selected_ids:
                continue
            selected.append(candidate)
            selected_ids.add(candidate.request_id)
            if len(selected) >= max(1, candidate_limit):
                break

    return tuple(selected)


def propose_candidates(request: RemoteSearchRequest) -> EvaluationBatchRequest:
    prefer_local_candidates = bool(request.metadata.get("prefer_local_candidates"))
    candidates: list[ProfileEvaluationRequest] = []
    if request.baseline_result is None:
        candidates.extend(_build_initial_policy_candidates(request))
    else:
        candidates.extend(_build_local_candidates(request))

    local_first = prefer_local_candidates or request.round_index >= 2
    limited = _limit_candidates_with_diversity(
        candidates,
        candidate_limit=max(1, request.candidate_limit),
        prefer_local_candidates=local_first,
    )
    return EvaluationBatchRequest(evaluations=limited)


def summarize_results(results: Iterable[ProfileEvaluationResult]) -> RemoteSearchSummary:
    result_list = list(results)
    sorted_results = sorted(result_list, key=_result_sort_key)
    failing_segments = Counter()
    ik_empty_rows = Counter()
    for result in result_list:
        for segment in result.failing_segments:
            failing_segments[f"{segment.left_label}->{segment.right_label}"] += 1
        for label in result.ik_empty_rows:
            ik_empty_rows[label] += 1

    if not sorted_results:
        return RemoteSearchSummary(
            best_request_id=None,
            result_count=0,
            sorted_request_ids=(),
            failing_segment_counts={},
            ik_empty_row_counts={},
            conclusion="No evaluation results were provided.",
        )

    best_result = sorted_results[0]
    target_reachable = (
        int(best_result.invalid_row_count) == 0
        and int(best_result.ik_empty_row_count) == 0
        and bool(best_result.selected_path)
    )
    if target_reachable and (
        int(best_result.bridge_like_segments) > 0
        or int(getattr(best_result, "big_circle_step_count", 0)) > 0
        or int(best_result.config_switches) > 0
        or best_result.failing_segments
    ):
        conclusion = (
            f"Found a target-reachable candidate with continuity warnings: "
            f"{best_result.request_id} "
            f"(config_switches={best_result.config_switches}, "
            f"bridge_like_segments={best_result.bridge_like_segments}, "
            f"big_circle_step_count={int(getattr(best_result, 'big_circle_step_count', 0))}, "
            f"worst_joint_step={best_result.worst_joint_step_deg:.3f} deg)."
        )
    elif target_reachable:
        conclusion = (
            f"Found a target-reachable clean candidate: {best_result.request_id} "
            f"(worst_joint_step={best_result.worst_joint_step_deg:.3f} deg)."
        )
    else:
        conclusion = (
            f"No target-reachable candidate yet; best request is {best_result.request_id} with "
            f"ik_empty_rows={best_result.ik_empty_row_count}, "
            f"config_switches={best_result.config_switches}, "
            f"bridge_like_segments={best_result.bridge_like_segments}, "
            f"big_circle_step_count={int(getattr(best_result, 'big_circle_step_count', 0))}."
        )

    notes = []
    if failing_segments:
        hot_segment, hot_count = failing_segments.most_common(1)[0]
        notes.append(f"Most frequent failing segment: {hot_segment} ({hot_count} occurrences).")
    if ik_empty_rows:
        hot_row, hot_count = ik_empty_rows.most_common(1)[0]
        notes.append(f"Most frequent IK-empty row: {hot_row} ({hot_count} occurrences).")

    return RemoteSearchSummary(
        best_request_id=best_result.request_id,
        result_count=len(result_list),
        sorted_request_ids=tuple(result.request_id for result in sorted_results),
        failing_segment_counts=dict(failing_segments),
        ik_empty_row_counts=dict(ik_empty_rows),
        conclusion=conclusion,
        notes=tuple(notes),
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate remote profile candidates or summarize evaluated batches."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    propose_parser = subparsers.add_parser("propose", help="Generate candidate profile evaluations.")
    propose_parser.add_argument("--request", required=True, help="Path to a RemoteSearchRequest JSON file.")
    propose_parser.add_argument("--candidates", required=True, help="Path to write the candidate batch JSON.")

    summarize_parser = subparsers.add_parser("summarize", help="Summarize evaluated candidate results.")
    summarize_parser.add_argument("--results", required=True, help="Path to an EvaluationBatchResult JSON file.")
    summarize_parser.add_argument("--summary", required=True, help="Path to write the summary JSON.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    try:
        if args.command == "propose":
            remote_request = RemoteSearchRequest.from_dict(load_json_file(args.request))
            batch = propose_candidates(remote_request)
            output_path = write_json_file(args.candidates, batch.to_dict())
            print(f"Wrote candidate batch: {output_path}")
            return 0

        results_payload = load_json_file(args.results)
        batch_result = EvaluationBatchResult.from_dict(results_payload)
        summary = summarize_results(batch_result.results)
        output_path = write_json_file(args.summary, summary.to_dict())
        print(summary.conclusion)
        for note in summary.notes:
            print(note)
        print(f"Wrote summary: {output_path}")
        return 0
    except Exception as exc:
        print(f"Error: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
