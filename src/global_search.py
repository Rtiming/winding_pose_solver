from __future__ import annotations

import math
from typing import Sequence

from src.geometry import _build_pose, _trim_joint_vector
from src.ik_collection import (
    _build_seed_joint_strategies,
    _collect_ik_candidates,
)
from src.path_optimizer import _optimize_joint_path, _summarize_selected_path
from src.types import _IKCandidate, _IKLayer, _PathOptimizerSettings, _PathSearchResult


def _search_best_exact_pose_path(
    pose_rows: Sequence[dict[str, float]],
    *,
    robot,
    mat_type,
    move_type: str,
    start_joints: tuple[float, ...],
    tool_pose,
    reference_pose,
    joint_count: int,
    motion_settings,
    optimizer_settings: _PathOptimizerSettings,
    a1_lower_deg: float,
    a1_upper_deg: float,
    a2_max_deg: float,
    joint_constraint_tolerance_deg: float,
) -> _PathSearchResult:
    reference_pose_rows = tuple(dict(row) for row in pose_rows)
    row_labels = _extract_row_labels(reference_pose_rows)
    inserted_flags = tuple(False for _ in reference_pose_rows)

    lower_limits_raw, upper_limits_raw, _ = robot.JointLimits()
    lower_limits = _trim_joint_vector(lower_limits_raw.list(), joint_count)
    upper_limits = _trim_joint_vector(upper_limits_raw.list(), joint_count)
    search_seed_joints = _build_seed_joint_strategies(
        robot=robot,
        lower_limits=lower_limits,
        upper_limits=upper_limits,
        joint_count=joint_count,
    )

    preview_indices = _build_preview_indices(reference_pose_rows)
    full_cache: dict[tuple[tuple[float, float], ...], _PathSearchResult] = {}
    preview_cache: dict[tuple[float, float], _PathSearchResult] = {}

    def evaluate_profile(
        frame_a_origin_yz_profile_mm: Sequence[tuple[float, float]],
    ) -> _PathSearchResult:
        cache_key = _profile_cache_key(frame_a_origin_yz_profile_mm)
        cached_result = full_cache.get(cache_key)
        if cached_result is not None:
            return cached_result

        result = _evaluate_frame_a_origin_profile(
            reference_pose_rows,
            frame_a_origin_yz_profile_mm=frame_a_origin_yz_profile_mm,
            row_labels=row_labels,
            inserted_flags=inserted_flags,
            robot=robot,
            mat_type=mat_type,
            move_type=move_type,
            start_joints=start_joints,
            tool_pose=tool_pose,
            reference_pose=reference_pose,
            joint_count=joint_count,
            optimizer_settings=optimizer_settings,
            a1_lower_deg=a1_lower_deg,
            a1_upper_deg=a1_upper_deg,
            a2_max_deg=a2_max_deg,
            joint_constraint_tolerance_deg=joint_constraint_tolerance_deg,
            seed_joints=search_seed_joints,
            lower_limits=lower_limits,
            upper_limits=upper_limits,
            bridge_trigger_joint_delta_deg=motion_settings.bridge_trigger_joint_delta_deg,
        )
        full_cache[cache_key] = result
        return result

    def evaluate_uniform_preview(dy_mm: float, dz_mm: float) -> _PathSearchResult:
        cache_key = (round(float(dy_mm), 6), round(float(dz_mm), 6))
        cached_result = preview_cache.get(cache_key)
        if cached_result is not None:
            return cached_result

        preview_reference_rows = tuple(reference_pose_rows[index] for index in preview_indices)
        preview_labels = tuple(row_labels[index] for index in preview_indices)
        preview_flags = tuple(inserted_flags[index] for index in preview_indices)
        preview_profile = tuple((float(dy_mm), float(dz_mm)) for _ in preview_reference_rows)
        result = _evaluate_frame_a_origin_profile(
            preview_reference_rows,
            frame_a_origin_yz_profile_mm=preview_profile,
            row_labels=preview_labels,
            inserted_flags=preview_flags,
            robot=robot,
            mat_type=mat_type,
            move_type=move_type,
            start_joints=start_joints,
            tool_pose=tool_pose,
            reference_pose=reference_pose,
            joint_count=joint_count,
            optimizer_settings=optimizer_settings,
            a1_lower_deg=a1_lower_deg,
            a1_upper_deg=a1_upper_deg,
            a2_max_deg=a2_max_deg,
            joint_constraint_tolerance_deg=joint_constraint_tolerance_deg,
            seed_joints=search_seed_joints,
            lower_limits=lower_limits,
            upper_limits=upper_limits,
            bridge_trigger_joint_delta_deg=motion_settings.bridge_trigger_joint_delta_deg,
        )
        preview_cache[cache_key] = result
        return result

    zero_profile = tuple((0.0, 0.0) for _ in reference_pose_rows)
    best_result = evaluate_profile(zero_profile)
    row0_candidates = len(best_result.ik_layers[0].candidates) if best_result.ik_layers else 0
    print(
        "Baseline fixed-orientation profile [dy=0, dz=0] mm: "
        f"row0_candidates={row0_candidates}, "
        f"ik_empty_rows={best_result.ik_empty_row_count}, "
        f"config_switches={best_result.config_switches}, "
        f"worst_joint_step={best_result.worst_joint_step_deg:.3f} deg."
    )
    if _path_search_goal_is_satisfied(best_result, optimizer_settings=optimizer_settings):
        return best_result

    for envelope_mm in motion_settings.frame_a_origin_yz_envelope_schedule_mm:
        for step_mm in motion_settings.frame_a_origin_yz_step_schedule_mm:
            if step_mm <= 0.0 or envelope_mm < 0.0 or step_mm > envelope_mm + 1e-9:
                continue

            preview_candidates = [
                evaluate_uniform_preview(dy_mm, dz_mm)
                for dy_mm, dz_mm in _iter_uniform_profile_offsets(
                    max_abs_offset_mm=envelope_mm,
                    step_mm=step_mm,
                )
            ]
            preview_candidates.sort(key=_path_search_sort_key)

            for preview_result in preview_candidates[: min(8, len(preview_candidates))]:
                dy_mm, dz_mm = preview_result.frame_a_origin_yz_profile_mm[0]
                profile = tuple((dy_mm, dz_mm) for _ in reference_pose_rows)
                candidate_result = evaluate_profile(profile)
                if _path_search_sort_key(candidate_result) < _path_search_sort_key(best_result):
                    best_result = candidate_result
                    print(
                        "Accepted uniform Frame-2 profile candidate: "
                        f"dy={dy_mm:.3f} mm, dz={dz_mm:.3f} mm, "
                        f"ik_empty_rows={candidate_result.ik_empty_row_count}, "
                        f"config_switches={candidate_result.config_switches}, "
                        f"bridge_like_segments={candidate_result.bridge_like_segments}, "
                        f"worst_joint_step={candidate_result.worst_joint_step_deg:.3f} deg."
                    )
                    if _path_search_goal_is_satisfied(
                        best_result,
                        optimizer_settings=optimizer_settings,
                    ):
                        return best_result

    return best_result


def _evaluate_frame_a_origin_profile(
    reference_pose_rows: Sequence[dict[str, float]],
    *,
    frame_a_origin_yz_profile_mm: Sequence[tuple[float, float]],
    row_labels: Sequence[str],
    inserted_flags: Sequence[bool],
    robot,
    mat_type,
    move_type: str,
    start_joints: tuple[float, ...],
    tool_pose,
    reference_pose,
    joint_count: int,
    optimizer_settings: _PathOptimizerSettings,
    a1_lower_deg: float,
    a1_upper_deg: float,
    a2_max_deg: float,
    joint_constraint_tolerance_deg: float,
    seed_joints: Sequence[tuple[float, ...]],
    lower_limits: Sequence[float],
    upper_limits: Sequence[float],
    bridge_trigger_joint_delta_deg: float,
) -> _PathSearchResult:
    adjusted_pose_rows = _apply_frame_a_origin_yz_profile(
        reference_pose_rows,
        frame_a_origin_yz_profile_mm=frame_a_origin_yz_profile_mm,
    )
    ik_layers, ik_empty_row_count = _build_ik_layers_with_diagnostics(
        adjusted_pose_rows,
        robot=robot,
        mat_type=mat_type,
        tool_pose=tool_pose,
        reference_pose=reference_pose,
        joint_count=joint_count,
        optimizer_settings=optimizer_settings,
        a1_lower_deg=a1_lower_deg,
        a1_upper_deg=a1_upper_deg,
        a2_max_deg=a2_max_deg,
        joint_constraint_tolerance_deg=joint_constraint_tolerance_deg,
        seed_joints=seed_joints,
        lower_limits=lower_limits,
        upper_limits=upper_limits,
    )

    selected_path: tuple[_IKCandidate, ...] = ()
    total_cost = math.inf
    if ik_empty_row_count == 0 and ik_layers:
        try:
            selected_path_list, total_cost = _optimize_joint_path(
                ik_layers,
                robot=robot,
                move_type=move_type,
                start_joints=start_joints,
                optimizer_settings=optimizer_settings,
            )
            selected_path = tuple(selected_path_list)
        except RuntimeError:
            selected_path = ()
            total_cost = math.inf

    if selected_path:
        (
            config_switches,
            bridge_like_segments,
            worst_joint_step_deg,
            mean_joint_step_deg,
        ) = _summarize_selected_path(
            selected_path,
            bridge_trigger_joint_delta_deg=bridge_trigger_joint_delta_deg,
        )
    else:
        config_switches = len(adjusted_pose_rows)
        bridge_like_segments = len(adjusted_pose_rows)
        worst_joint_step_deg = math.inf
        mean_joint_step_deg = math.inf

    (
        offset_step_jitter_mm,
        offset_jerk_mm,
        max_abs_offset_mm,
        total_abs_offset_mm,
    ) = _summarize_profile_metrics(frame_a_origin_yz_profile_mm)

    return _PathSearchResult(
        reference_pose_rows=tuple(dict(row) for row in reference_pose_rows),
        pose_rows=adjusted_pose_rows,
        ik_layers=ik_layers,
        selected_path=selected_path,
        total_cost=total_cost,
        frame_a_origin_yz_profile_mm=tuple(
            (float(dy_mm), float(dz_mm)) for dy_mm, dz_mm in frame_a_origin_yz_profile_mm
        ),
        row_labels=tuple(str(label) for label in row_labels),
        inserted_flags=tuple(bool(flag) for flag in inserted_flags),
        invalid_row_count=0,
        ik_empty_row_count=ik_empty_row_count,
        config_switches=config_switches,
        bridge_like_segments=bridge_like_segments,
        worst_joint_step_deg=worst_joint_step_deg,
        mean_joint_step_deg=mean_joint_step_deg,
        offset_step_jitter_mm=offset_step_jitter_mm,
        offset_jerk_mm=offset_jerk_mm,
        max_abs_offset_mm=max_abs_offset_mm,
        total_abs_offset_mm=total_abs_offset_mm,
    )


def _apply_frame_a_origin_yz_profile(
    reference_pose_rows: Sequence[dict[str, float]],
    *,
    frame_a_origin_yz_profile_mm: Sequence[tuple[float, float]],
) -> tuple[dict[str, float], ...]:
    adjusted_rows: list[dict[str, float]] = []
    for reference_row, (dy_mm, dz_mm) in zip(reference_pose_rows, frame_a_origin_yz_profile_mm):
        adjusted_row = dict(reference_row)
        adjusted_row["x_mm"] = float(reference_row["x_mm"])
        adjusted_row["y_mm"] = float(reference_row["y_mm"]) + float(dy_mm)
        adjusted_row["z_mm"] = float(reference_row["z_mm"]) + float(dz_mm)
        adjusted_rows.append(adjusted_row)
    return tuple(adjusted_rows)


def _build_ik_layers_with_diagnostics(
    pose_rows: Sequence[dict[str, float]],
    *,
    robot,
    mat_type,
    tool_pose,
    reference_pose,
    joint_count: int,
    optimizer_settings: _PathOptimizerSettings,
    a1_lower_deg: float,
    a1_upper_deg: float,
    a2_max_deg: float,
    joint_constraint_tolerance_deg: float,
    seed_joints: Sequence[tuple[float, ...]],
    lower_limits: Sequence[float],
    upper_limits: Sequence[float],
) -> tuple[tuple[_IKLayer, ...], int]:
    ik_layers: list[_IKLayer] = []
    ik_empty_row_count = 0
    for pose_row in pose_rows:
        pose = _build_pose(pose_row, mat_type)
        candidates = _collect_ik_candidates(
            robot,
            pose,
            tool_pose=tool_pose,
            reference_pose=reference_pose,
            lower_limits=tuple(lower_limits),
            upper_limits=tuple(upper_limits),
            seed_joints=seed_joints,
            joint_count=joint_count,
            optimizer_settings=optimizer_settings,
            a1_lower_deg=a1_lower_deg,
            a1_upper_deg=a1_upper_deg,
            a2_max_deg=a2_max_deg,
            joint_constraint_tolerance_deg=joint_constraint_tolerance_deg,
        )
        if not candidates:
            ik_empty_row_count += 1
        ik_layers.append(_IKLayer(pose=pose, candidates=tuple(candidates)))
    return tuple(ik_layers), ik_empty_row_count


def _path_search_sort_key(result: _PathSearchResult) -> tuple[float, ...]:
    return (
        float(result.invalid_row_count),
        float(result.ik_empty_row_count),
        float(result.config_switches),
        float(result.bridge_like_segments),
        float(result.worst_joint_step_deg),
        float(result.mean_joint_step_deg),
        float(result.offset_step_jitter_mm),
        float(result.offset_jerk_mm),
        float(result.max_abs_offset_mm),
        float(result.total_abs_offset_mm),
        float(result.total_cost),
    )


def _path_search_goal_is_satisfied(
    result: _PathSearchResult,
    *,
    optimizer_settings: _PathOptimizerSettings,
) -> bool:
    if result.invalid_row_count != 0 or result.ik_empty_row_count != 0:
        return False
    if result.config_switches != 0 or result.bridge_like_segments != 0:
        return False
    preferred_limit = max(optimizer_settings.preferred_joint_step_deg, default=0.0)
    return result.worst_joint_step_deg <= preferred_limit + 1e-9


def _iter_uniform_profile_offsets(
    *,
    max_abs_offset_mm: float,
    step_mm: float,
) -> tuple[tuple[float, float], ...]:
    if step_mm <= 0.0 or max_abs_offset_mm < 0.0:
        return ()

    shell_limit = int(math.floor(max_abs_offset_mm / step_mm + 1e-9))
    offsets: list[tuple[float, float]] = []
    for dy_index in range(-shell_limit, shell_limit + 1):
        for dz_index in range(-shell_limit, shell_limit + 1):
            offset = (dy_index * step_mm, dz_index * step_mm)
            if any(abs(value) > max_abs_offset_mm + 1e-9 for value in offset):
                continue
            offsets.append(offset)

    offsets.sort(
        key=lambda offset: (
            0.0 if abs(offset[0]) <= 1e-9 and abs(offset[1]) <= 1e-9 else 1.0,
            abs(offset[0]) + abs(offset[1]),
            max(abs(offset[0]), abs(offset[1])),
            offset,
        )
    )
    return tuple(offsets)


def _build_preview_indices(
    pose_rows: Sequence[dict[str, float]],
    *,
    target_count: int = 24,
) -> tuple[int, ...]:
    if len(pose_rows) <= target_count:
        return tuple(range(len(pose_rows)))

    step = max(1, math.ceil((len(pose_rows) - 1) / max(1, target_count - 1)))
    indices = list(range(0, len(pose_rows), step))
    if indices[-1] != len(pose_rows) - 1:
        indices.append(len(pose_rows) - 1)
    return tuple(indices)


def _extract_row_labels(pose_rows: Sequence[dict[str, float]]) -> tuple[str, ...]:
    labels: list[str] = []
    for row_index, pose_row in enumerate(pose_rows):
        if "index" in pose_row:
            labels.append(str(int(float(pose_row["index"]))))
        elif "source_row" in pose_row:
            labels.append(str(int(float(pose_row["source_row"]))))
        else:
            labels.append(str(row_index))
    return tuple(labels)


def _profile_cache_key(
    frame_a_origin_yz_profile_mm: Sequence[tuple[float, float]],
) -> tuple[tuple[float, float], ...]:
    return tuple(
        (round(float(dy_mm), 6), round(float(dz_mm), 6))
        for dy_mm, dz_mm in frame_a_origin_yz_profile_mm
    )


def _summarize_profile_metrics(
    frame_a_origin_yz_profile_mm: Sequence[tuple[float, float]],
) -> tuple[float, float, float, float]:
    if not frame_a_origin_yz_profile_mm:
        return 0.0, 0.0, 0.0, 0.0

    magnitudes = [
        math.hypot(float(dy_mm), float(dz_mm))
        for dy_mm, dz_mm in frame_a_origin_yz_profile_mm
    ]
    first_differences = [
        math.hypot(
            float(current[0]) - float(previous[0]),
            float(current[1]) - float(previous[1]),
        )
        for previous, current in zip(
            frame_a_origin_yz_profile_mm,
            frame_a_origin_yz_profile_mm[1:],
        )
    ]
    second_differences = [
        math.hypot(
            float(next_delta[0]) - float(previous_delta[0]),
            float(next_delta[1]) - float(previous_delta[1]),
        )
        for previous_delta, next_delta in zip(
            zip(first_differences, [0.0] * len(first_differences)),
            zip(first_differences[1:], [0.0] * max(0, len(first_differences) - 1)),
        )
    ]

    offset_step_jitter_mm = (
        sum(first_differences) / len(first_differences) if first_differences else 0.0
    )
    if len(frame_a_origin_yz_profile_mm) >= 3:
        jerk_terms = []
        for previous_value, current_value, next_value in zip(
            frame_a_origin_yz_profile_mm,
            frame_a_origin_yz_profile_mm[1:],
            frame_a_origin_yz_profile_mm[2:],
        ):
            jerk_terms.append(
                math.hypot(
                    float(next_value[0]) - 2.0 * float(current_value[0]) + float(previous_value[0]),
                    float(next_value[1]) - 2.0 * float(current_value[1]) + float(previous_value[1]),
                )
            )
        offset_jerk_mm = sum(jerk_terms) / len(jerk_terms)
    else:
        offset_jerk_mm = 0.0

    return (
        offset_step_jitter_mm,
        offset_jerk_mm,
        max(magnitudes, default=0.0),
        sum(magnitudes),
    )
