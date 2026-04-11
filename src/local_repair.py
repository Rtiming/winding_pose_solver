from __future__ import annotations

from typing import Sequence

from src.types import (
    _IKCandidate,
    _IKLayer,
    _PathOptimizerSettings,
)
from src.geometry import (
    _clip_seed_to_limits,
    _rotation_matrix_from_xyz_offset_deg,
    _pose_row_to_rotation_translation,
    _pose_row_from_rotation_translation,
    _multiply_rotation_matrices,
)
from src.path_optimizer import (
    _summarize_selected_path,
    _optimize_joint_path,
    _selected_path_quality_key,
    _path_is_clean_enough_for_program_generation,
)
from src.ik_collection import (
    _IK_DEDUP_DECIMALS,
    _append_candidate_if_unique,
    _build_seed_joint_strategies,
    _append_seed_if_unique,
    _build_ik_layers,
)
def _refine_path_near_wrist_singularity(
    ik_layers: Sequence[_IKLayer],
    selected_path: Sequence[_IKCandidate],
    *,
    total_cost: float,
    start_joints: tuple[float, ...],
    robot,
    tool_pose,
    reference_pose,
    joint_count: int,
    motion_settings,
    optimizer_settings: _PathOptimizerSettings,
    lower_limits: Sequence[float],
    upper_limits: Sequence[float],
    a1_lower_deg: float,
    a1_upper_deg: float,
    a2_max_deg: float,
    joint_constraint_tolerance_deg: float,
) -> tuple[tuple[_IKLayer, ...], tuple[_IKCandidate, ...], float]:
    """在腕奇异窗口附近做一次 targeted IK refinement。"""

    if (
        not motion_settings.enable_wrist_singularity_refinement
        or joint_count < 6
        or not ik_layers
        or len(ik_layers) != len(selected_path)
    ):
        return tuple(ik_layers), tuple(selected_path), total_cost

    refine_indices = _collect_wrist_refinement_indices(selected_path, motion_settings)
    if not refine_indices:
        return tuple(ik_layers), tuple(selected_path), total_cost

    refine_index_set = set(refine_indices)
    augmented_layers: list[_IKLayer] = []
    new_candidate_count = 0
    lower_limits_tuple = tuple(lower_limits)
    upper_limits_tuple = tuple(upper_limits)

    for layer_index, layer in enumerate(ik_layers):
        if layer_index not in refine_index_set:
            augmented_layers.append(layer)
            continue

        candidates = list(layer.candidates)
        seen = {
            tuple(round(value, _IK_DEDUP_DECIMALS) for value in candidate.joints)
            for candidate in candidates
        }
        seed_strategies = _build_wrist_refinement_seed_strategies(
            selected_path,
            layer_index,
            motion_settings=motion_settings,
            lower_limits=lower_limits_tuple,
            upper_limits=upper_limits_tuple,
        )
        before_count = len(candidates)
        for seed in seed_strategies:
            raw_solution = robot.SolveIK(layer.pose, list(seed), tool_pose, reference_pose)
            _append_candidate_if_unique(
                candidates,
                seen,
                robot=robot,
                raw_joints=raw_solution,
                lower_limits=lower_limits_tuple,
                upper_limits=upper_limits_tuple,
                joint_count=joint_count,
                optimizer_settings=optimizer_settings,
                a1_lower_deg=a1_lower_deg,
                a1_upper_deg=a1_upper_deg,
                a2_max_deg=a2_max_deg,
                joint_constraint_tolerance_deg=joint_constraint_tolerance_deg,
            )

        candidates.sort(
            key=lambda candidate: (
                candidate.config_flags,
                candidate.joint_limit_penalty + candidate.singularity_penalty,
                candidate.joints,
            )
        )
        augmented_layers.append(_IKLayer(pose=layer.pose, candidates=tuple(candidates)))
        new_candidate_count += len(candidates) - before_count

    if new_candidate_count <= 0:
        return tuple(ik_layers), tuple(selected_path), total_cost

    refined_path_list, refined_cost = _optimize_joint_path(
        augmented_layers,
        robot=robot,
        move_type=motion_settings.move_type,
        start_joints=start_joints,
        optimizer_settings=optimizer_settings,
    )
    refined_path = tuple(refined_path_list)

    previous_quality = _selected_path_quality_key(
        selected_path,
        total_cost=total_cost,
        bridge_trigger_joint_delta_deg=motion_settings.bridge_trigger_joint_delta_deg,
    )
    refined_quality = _selected_path_quality_key(
        refined_path,
        total_cost=refined_cost,
        bridge_trigger_joint_delta_deg=motion_settings.bridge_trigger_joint_delta_deg,
    )
    if refined_quality >= previous_quality:
        return tuple(ik_layers), tuple(selected_path), total_cost

    print(
        "Accepted wrist singularity refinement: "
        f"window_count={len(refine_indices)}, added_candidates={new_candidate_count}, "
        f"path_cost={refined_cost:.3f}."
    )
    return tuple(augmented_layers), refined_path, refined_cost


def _collect_wrist_refinement_indices(
    selected_path: Sequence[_IKCandidate],
    motion_settings,
) -> tuple[int, ...]:
    """找出需要补做腕部 refinement 的路径窗口。"""

    if not selected_path:
        return ()

    refine_indices: set[int] = set()
    radius = max(0, motion_settings.wrist_refinement_window_radius)
    for index, candidate in enumerate(selected_path):
        if len(candidate.joints) < 6:
            continue

        should_refine = abs(candidate.joints[4]) < motion_settings.wrist_refinement_a5_threshold_deg
        if not should_refine:
            for neighbor_index in (index - 1, index + 1):
                if neighbor_index < 0 or neighbor_index >= len(selected_path):
                    continue
                neighbor_candidate = selected_path[neighbor_index]
                if len(neighbor_candidate.joints) < 6:
                    continue
                large_wrist_step = max(
                    abs(candidate.joints[3] - neighbor_candidate.joints[3]),
                    abs(candidate.joints[5] - neighbor_candidate.joints[5]),
                )
                if (
                    large_wrist_step > motion_settings.wrist_refinement_large_wrist_step_deg
                    or candidate.config_flags != neighbor_candidate.config_flags
                ):
                    should_refine = True
                    break

        if not should_refine:
            continue

        for expand_index in range(index - radius, index + radius + 1):
            if 0 <= expand_index < len(selected_path):
                refine_indices.add(expand_index)

    return tuple(sorted(refine_indices))


def _build_wrist_refinement_seed_strategies(
    selected_path: Sequence[_IKCandidate],
    target_index: int,
    *,
    motion_settings,
    lower_limits: Sequence[float],
    upper_limits: Sequence[float],
) -> tuple[tuple[float, ...], ...]:
    """构造用于奇异窗口二次求解的 seed 集。"""

    seen: set[tuple[float, ...]] = set()
    seeds: list[tuple[float, ...]] = []
    radius = max(1, motion_settings.wrist_refinement_window_radius + 1)
    reference_a6 = _estimate_reference_a6_for_wrist_refinement(
        selected_path,
        target_index,
        a5_threshold_deg=motion_settings.wrist_refinement_a5_threshold_deg,
    )
    a6_targets = [reference_a6, 0.0]
    for phase_offset_deg in motion_settings.wrist_refinement_phase_offsets_deg:
        a6_targets.append(reference_a6 - phase_offset_deg)
        a6_targets.append(reference_a6 + phase_offset_deg)

    for neighbor_index in range(
        max(0, target_index - radius),
        min(len(selected_path), target_index + radius + 1),
    ):
        base_seed = selected_path[neighbor_index].joints
        _append_seed_if_unique(seeds, seen, _clip_seed_to_limits(base_seed, lower_limits, upper_limits))
        if len(base_seed) < 6:
            continue

        for target_a6 in a6_targets:
            compensation_deg = base_seed[5] - target_a6
            variant = list(base_seed)
            variant[3] = variant[3] + compensation_deg
            variant[5] = target_a6
            _append_seed_if_unique(
                seeds,
                seen,
                _clip_seed_to_limits(variant, lower_limits, upper_limits),
            )

        for phase_offset_deg in motion_settings.wrist_refinement_phase_offsets_deg:
            for direction in (-1.0, 1.0):
                variant = list(base_seed)
                phase_shift = direction * phase_offset_deg
                variant[3] = variant[3] + phase_shift
                variant[5] = variant[5] - phase_shift
                _append_seed_if_unique(
                    seeds,
                    seen,
                    _clip_seed_to_limits(variant, lower_limits, upper_limits),
                )

    return tuple(seeds)


def _estimate_reference_a6_for_wrist_refinement(
    selected_path: Sequence[_IKCandidate],
    target_index: int,
    *,
    a5_threshold_deg: float,
) -> float:
    """估计当前奇异窗口希望维持的 A6 相位参考值。"""

    candidate_values: list[float] = []
    for search_radius in range(0, len(selected_path)):
        left_index = target_index - search_radius
        right_index = target_index + search_radius
        for index in (left_index, right_index):
            if index < 0 or index >= len(selected_path):
                continue
            candidate = selected_path[index]
            if len(candidate.joints) < 6:
                continue
            if abs(candidate.joints[4]) >= a5_threshold_deg:
                candidate_values.append(candidate.joints[5])
        if candidate_values:
            break

    if candidate_values:
        return sum(candidate_values) / len(candidate_values)
    return selected_path[target_index].joints[5]


def _redistribute_orientation_in_solution_rich_window(
    base_pose_rows: Sequence[dict[str, float]],
    ik_layers: Sequence[_IKLayer],
    selected_path: Sequence[_IKCandidate],
    *,
    total_cost: float,
    start_joints: tuple[float, ...],
    robot,
    mat_type,
    tool_pose,
    reference_pose,
    joint_count: int,
    motion_settings,
    optimizer_settings: _PathOptimizerSettings,
    lower_limits: Sequence[float],
    upper_limits: Sequence[float],
    a1_lower_deg: float,
    a1_upper_deg: float,
    a2_max_deg: float,
    joint_constraint_tolerance_deg: float,
    base_rotation_offset_deg: Sequence[float],
) -> tuple[tuple[_IKLayer, ...], tuple[_IKCandidate, ...], float]:
    """在"解比较多"的窗口里搜索纯姿态重分配方案。"""

    if (
        not motion_settings.enable_solution_rich_orientation_redistribution
        or len(selected_path) <= 1
    ):
        return tuple(ik_layers), tuple(selected_path), total_cost

    problem_segments = _collect_problem_segments(
        selected_path,
        bridge_trigger_joint_delta_deg=motion_settings.bridge_trigger_joint_delta_deg,
    )
    if not problem_segments:
        return tuple(ik_layers), tuple(selected_path), total_cost

    candidate_windows = _build_solution_rich_window_candidates(
        ik_layers,
        problem_segment_index=problem_segments[0][0],
        window_radius=motion_settings.solution_rich_orientation_window_radius,
    )
    candidate_offsets = _build_solution_rich_orientation_offsets(
        base_rotation_offset_deg,
        offset_deg=motion_settings.solution_rich_orientation_offset_deg,
    )
    if not candidate_windows or not candidate_offsets:
        return tuple(ik_layers), tuple(selected_path), total_cost

    lower_limits_tuple = tuple(float(value) for value in lower_limits[:joint_count])
    upper_limits_tuple = tuple(float(value) for value in upper_limits[:joint_count])
    search_seed_joints = _build_seed_joint_strategies(
        robot=robot,
        lower_limits=lower_limits_tuple,
        upper_limits=upper_limits_tuple,
        joint_count=joint_count,
    )
    quick_baseline_quality = _orientation_redistribution_candidate_sort_key(
        ik_layers,
        selected_path,
        total_cost=total_cost,
        window_start=-1,
        window_end=-1,
        robot=robot,
        tool_pose=tool_pose,
        reference_pose=reference_pose,
        motion_settings=motion_settings,
        optimizer_settings=optimizer_settings,
        lower_limits=lower_limits,
        upper_limits=upper_limits,
        a1_lower_deg=a1_lower_deg,
        a1_upper_deg=a1_upper_deg,
        evaluate_fixed_position_bridge=False,
    )
    candidate_results: list[
        tuple[
            tuple[float, ...],
            tuple[_IKLayer, ...],
            tuple[_IKCandidate, ...],
            float,
            tuple[int, int],
            tuple[float, ...],
        ]
    ] = []

    for window_start, window_end in candidate_windows:
        for rotation_offset_deg in candidate_offsets:
            candidate_pose_rows = _apply_orientation_redistribution_window(
                base_pose_rows,
                window_start=window_start,
                window_end=window_end,
                rotation_offset_deg=rotation_offset_deg,
            )
            try:
                reused_prefix_layers = tuple(ik_layers[:window_start])
                rebuilt_suffix_layers = tuple(
                    _build_ik_layers(
                        candidate_pose_rows[window_start:],
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
                        seed_joints_override=search_seed_joints,
                        lower_limits_override=lower_limits_tuple,
                        upper_limits_override=upper_limits_tuple,
                        log_summary=False,
                    )
                )
                candidate_layers = reused_prefix_layers + rebuilt_suffix_layers
                candidate_path_list, candidate_cost = _optimize_joint_path(
                    candidate_layers,
                    robot=robot,
                    move_type=motion_settings.move_type,
                    start_joints=start_joints,
                    optimizer_settings=optimizer_settings,
                )
            except RuntimeError:
                continue

            candidate_path = tuple(candidate_path_list)
            candidate_quality = _orientation_redistribution_candidate_sort_key(
                candidate_layers,
                candidate_path,
                total_cost=candidate_cost,
                window_start=window_start,
                window_end=window_end,
                robot=robot,
                tool_pose=tool_pose,
                reference_pose=reference_pose,
                motion_settings=motion_settings,
                optimizer_settings=optimizer_settings,
                lower_limits=lower_limits,
                upper_limits=upper_limits,
                a1_lower_deg=a1_lower_deg,
                a1_upper_deg=a1_upper_deg,
                evaluate_fixed_position_bridge=False,
            )
            if candidate_quality >= quick_baseline_quality:
                continue

            candidate_results.append(
                (
                    candidate_quality,
                    candidate_layers,
                    candidate_path,
                    candidate_cost,
                    (window_start, window_end),
                    rotation_offset_deg,
                )
            )

    if not candidate_results:
        return tuple(ik_layers), tuple(selected_path), total_cost

    candidate_results.sort(key=lambda item: item[0])
    (
        quick_best_quality,
        quick_best_layers,
        quick_best_path,
        quick_best_cost,
        quick_best_window,
        quick_best_offset,
    ) = candidate_results[0]
    if quick_best_quality >= quick_baseline_quality:
        return tuple(ik_layers), tuple(selected_path), total_cost
    best_result = (
        quick_best_layers,
        quick_best_path,
        quick_best_cost,
        quick_best_window,
        quick_best_offset,
    )

    refined_layers, refined_path, refined_cost, refined_window, refined_offset = best_result
    print(
        "Accepted solution-rich orientation redistribution: "
        f"segment={problem_segments[0][0]}->{problem_segments[0][0] + 1}, "
        f"window={refined_window[0]}->{refined_window[1]}, "
        f"offset xyz(deg)={[round(value, 3) for value in refined_offset]}, "
        f"path_cost={refined_cost:.3f}."
    )
    return refined_layers, refined_path, refined_cost


def _orientation_redistribution_candidate_sort_key(
    ik_layers: Sequence[_IKLayer],
    selected_path: Sequence[_IKCandidate],
    *,
    total_cost: float,
    window_start: int,
    window_end: int,
    robot,
    tool_pose,
    reference_pose,
    motion_settings,
    optimizer_settings: _PathOptimizerSettings,
    lower_limits: Sequence[float],
    upper_limits: Sequence[float],
    a1_lower_deg: float,
    a1_upper_deg: float,
    evaluate_fixed_position_bridge: bool,
) -> tuple[float, ...]:
    """给"解丰富窗口姿态重分配"候选打分。"""

    problem_segments = _collect_problem_segments(
        selected_path,
        bridge_trigger_joint_delta_deg=motion_settings.bridge_trigger_joint_delta_deg,
    )
    switch_segments = [
        segment_index
        for segment_index, config_changed, _max_joint_delta, _mean_joint_delta in problem_segments
        if config_changed
    ]
    switch_in_window = any(
        window_start <= segment_index + 1 <= window_end for segment_index in switch_segments
    )

    fixed_position_feasible = False
    if evaluate_fixed_position_bridge:
        candidate_segments_to_check = switch_segments or [
            segment_index
            for segment_index, _config_changed, _max_joint_delta, _mean_joint_delta in problem_segments
        ]
        for segment_index in candidate_segments_to_check:
            if segment_index < 0 or segment_index + 1 >= len(ik_layers):
                continue
            if _position_locked_bridge_is_feasible_for_segment(
                ik_layers,
                selected_path,
                segment_index=segment_index,
                robot=robot,
                tool_pose=tool_pose,
                reference_pose=reference_pose,
                motion_settings=motion_settings,
                optimizer_settings=optimizer_settings,
                lower_limits=lower_limits,
                upper_limits=upper_limits,
                a1_lower_deg=a1_lower_deg,
                a1_upper_deg=a1_upper_deg,
            ):
                fixed_position_feasible = True
                break

    (
        config_switches,
        bridge_like_segments,
        worst_joint_step_deg,
        mean_joint_step_deg,
    ) = _summarize_selected_path(
        selected_path,
        bridge_trigger_joint_delta_deg=motion_settings.bridge_trigger_joint_delta_deg,
    )
    return (
        0.0 if fixed_position_feasible else 1.0,
        0.0 if switch_in_window else 1.0,
        float(bridge_like_segments),
        float(config_switches),
        float(worst_joint_step_deg),
        float(mean_joint_step_deg),
        float(total_cost),
    )


def _position_locked_bridge_is_feasible_for_segment(
    ik_layers: Sequence[_IKLayer],
    selected_path: Sequence[_IKCandidate],
    *,
    segment_index: int,
    robot,
    tool_pose,
    reference_pose,
    motion_settings,
    optimizer_settings: _PathOptimizerSettings,
    lower_limits: Sequence[float],
    upper_limits: Sequence[float],
    a1_lower_deg: float,
    a1_upper_deg: float,
) -> bool:
    """快速判断某个相邻段是否存在固定位置桥接解。"""

    from src.bridge_builder import _build_position_locked_bridge_segment
    try:
        _build_position_locked_bridge_segment(
            segment_index=segment_index,
            target_index_width=3,
            current_target_name="TMP",
            previous_pose=ik_layers[segment_index].pose,
            current_pose=ik_layers[segment_index + 1].pose,
            previous_candidate=selected_path[segment_index],
            current_candidate=selected_path[segment_index + 1],
            robot=robot,
            tool_pose=tool_pose,
            reference_pose=reference_pose,
            motion_settings=motion_settings,
            optimizer_settings=optimizer_settings,
            lower_limits=lower_limits,
            upper_limits=upper_limits,
            a1_lower_deg=a1_lower_deg,
            a1_upper_deg=a1_upper_deg,
        )
    except RuntimeError:
        return False
    return True


def _collect_problem_segments(
    selected_path: Sequence[_IKCandidate],
    *,
    bridge_trigger_joint_delta_deg: float,
) -> tuple[tuple[int, bool, float, float], ...]:
    """找出当前 exact path 里最值得进一步处理的坏段。"""

    from src.geometry import _mean_abs_joint_delta
    segments: list[tuple[int, bool, float, float]] = []
    for segment_index, (previous_candidate, current_candidate) in enumerate(
        zip(selected_path, selected_path[1:])
    ):
        joint_deltas = [
            abs(current - previous)
            for previous, current in zip(previous_candidate.joints, current_candidate.joints)
        ]
        max_joint_delta = max(joint_deltas, default=0.0)
        mean_joint_delta = _mean_abs_joint_delta(
            previous_candidate.joints,
            current_candidate.joints,
        )
        config_changed = previous_candidate.config_flags != current_candidate.config_flags
        if config_changed or max_joint_delta > bridge_trigger_joint_delta_deg:
            segments.append(
                (
                    segment_index,
                    config_changed,
                    max_joint_delta,
                    mean_joint_delta,
                )
            )

    segments.sort(
        key=lambda item: (
            -int(item[1]),
            -float(item[2]),
            -float(item[3]),
        )
    )
    return tuple(segments)


def _build_solution_rich_window_candidates(
    ik_layers: Sequence[_IKLayer],
    *,
    problem_segment_index: int,
    window_radius: int,
) -> tuple[tuple[int, int], ...]:
    """围绕坏段构造几个"解丰富窗口"候选。"""

    if not ik_layers:
        return ()

    start_lower_bound = max(0, problem_segment_index - max(0, window_radius))
    end_upper_bound = min(len(ik_layers) - 1, problem_segment_index + 1 + max(0, window_radius))
    left_indices = [
        index
        for index in range(start_lower_bound, problem_segment_index + 1)
        if _layer_is_solution_rich(ik_layers[index])
    ]
    right_indices = [
        index
        for index in range(problem_segment_index + 1, end_upper_bound + 1)
        if _layer_is_solution_rich(ik_layers[index])
    ]
    if not left_indices:
        left_indices = [start_lower_bound, problem_segment_index]
    if not right_indices:
        right_indices = [problem_segment_index + 1, end_upper_bound]

    candidate_windows = [
        (left_indices[0], right_indices[-1]),
        (left_indices[len(left_indices) // 2], right_indices[len(right_indices) // 2]),
        (left_indices[-1], right_indices[0]),
    ]

    unique_windows: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    for window_start, window_end in candidate_windows:
        normalized_window = (min(window_start, window_end), max(window_start, window_end))
        if normalized_window[0] == normalized_window[1]:
            continue
        if normalized_window in seen:
            continue
        seen.add(normalized_window)
        unique_windows.append(normalized_window)
    return tuple(unique_windows)


def _layer_is_solution_rich(layer: _IKLayer) -> bool:
    """判断某一层是否适合作为"提前重分配姿态"的窗口点。"""

    config_groups = {candidate.config_flags for candidate in layer.candidates}
    return len(layer.candidates) >= 4 or len(config_groups) >= 2


def _build_solution_rich_orientation_offsets(
    base_rotation_offset_deg: Sequence[float],
    *,
    offset_deg: float,
) -> tuple[tuple[float, float, float], ...]:
    """根据当前整路径姿态偏置方向，构造局部姿态重分配的候选偏置。"""

    if offset_deg <= 0.0:
        return ()

    axis_signs = []
    for value in base_rotation_offset_deg[:3]:
        if value > 1e-9:
            axis_signs.append(1.0)
        elif value < -1e-9:
            axis_signs.append(-1.0)
        else:
            axis_signs.append(1.0)

    sx, sy, sz = axis_signs
    raw_offsets = (
        (sx * offset_deg, 0.0, 0.0),
        (0.0, sy * offset_deg, 0.0),
        (0.0, 0.0, sz * offset_deg),
        (sx * offset_deg, sy * offset_deg, 0.0),
        (sx * offset_deg, 0.0, sz * offset_deg),
        (0.0, sy * offset_deg, sz * offset_deg),
        (sx * offset_deg, sy * offset_deg, sz * offset_deg),
    )

    offsets: list[tuple[float, float, float]] = []
    seen: set[tuple[float, float, float]] = set()
    for rotation_offset_deg in raw_offsets:
        normalized_offset = tuple(float(value) for value in rotation_offset_deg)
        if not any(abs(value) > 1e-9 for value in normalized_offset):
            continue
        if normalized_offset in seen:
            continue
        seen.add(normalized_offset)
        offsets.append(normalized_offset)
    return tuple(offsets)


def _apply_orientation_redistribution_window(
    base_pose_rows: Sequence[dict[str, float]],
    *,
    window_start: int,
    window_end: int,
    rotation_offset_deg: Sequence[float],
) -> tuple[dict[str, float], ...]:
    """对一段窗口后的所有路径点做"位置锁定、姿态渐变"的重分配。"""

    if window_end < window_start:
        return tuple(dict(row) for row in base_pose_rows)

    redistributed_rows: list[dict[str, float]] = []
    transition_span = max(1, window_end - window_start + 1)
    for row_index, pose_row in enumerate(base_pose_rows):
        if row_index < window_start:
            redistributed_rows.append(dict(pose_row))
            continue

        if row_index > window_end:
            interpolation_ratio = 1.0
        else:
            interpolation_ratio = (row_index - window_start + 1) / transition_span

        local_rotation_offset_deg = tuple(
            float(value) * interpolation_ratio for value in rotation_offset_deg
        )
        local_rotation = _rotation_matrix_from_xyz_offset_deg(local_rotation_offset_deg)
        base_rotation, base_translation = _pose_row_to_rotation_translation(pose_row)
        redistributed_rows.append(
            _pose_row_from_rotation_translation(
                _multiply_rotation_matrices(local_rotation, base_rotation),
                base_translation,
            )
        )
    return tuple(redistributed_rows)
