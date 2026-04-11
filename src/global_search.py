from __future__ import annotations

import math
from typing import Sequence

from src.types import (
    _IKCandidate,
    _IKLayer,
    _PathOptimizerSettings,
    _PathSearchResult,
)
from src.geometry import (
    _trim_joint_vector,
    _clip_seed_to_limits,
    _pose_row_to_rotation_translation,
    _pose_row_from_rotation_translation,
    _rotation_matrix_from_xyz_offset_deg,
    _multiply_rotation_matrices,
    _multiply_rotation_vector,
    _subtract_vectors,
    _add_vectors,
)
from src.path_optimizer import (
    _summarize_selected_path,
    _optimize_joint_path,
)
from src.ik_collection import (
    _IK_DEDUP_DECIMALS,
    _build_ik_layers,
    _build_seed_joint_strategies,
)


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
    """搜索"整条路径统一刚体旋转"后的最优 exact-pose 解。"""

    cache: dict[tuple[float, float, float], _PathSearchResult | None] = {}
    preview_cache: dict[tuple[float, float, float], _PathSearchResult | None] = {}
    preview_pose_rows = _build_global_pose_search_preview_rows(pose_rows)
    lower_limits_raw, upper_limits_raw, _ = robot.JointLimits()
    lower_limits = _trim_joint_vector(lower_limits_raw.list(), joint_count)
    upper_limits = _trim_joint_vector(upper_limits_raw.list(), joint_count)
    search_seed_joints = _build_seed_joint_strategies(
        robot=robot,
        lower_limits=lower_limits,
        upper_limits=upper_limits,
        joint_count=joint_count,
    )
    preview_seed_joints = _build_preview_seed_joint_strategies(
        robot=robot,
        lower_limits=lower_limits,
        upper_limits=upper_limits,
        joint_count=joint_count,
    )

    from dataclasses import replace
    preview_optimizer_settings = replace(
        optimizer_settings,
        enable_joint_continuity_constraint=False,
    )

    def evaluate(rotation_offset_deg: tuple[float, float, float]) -> _PathSearchResult | None:
        cache_key = tuple(round(value, 6) for value in rotation_offset_deg)
        if cache_key in cache:
            return cache[cache_key]

        candidate_pose_rows = tuple(
            _apply_global_pose_rotation_to_pose_row(row, rotation_offset_deg, motion_settings)
            for row in pose_rows
        )
        try:
            ik_layers = tuple(
                _build_ik_layers(
                    candidate_pose_rows,
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
                    lower_limits_override=lower_limits,
                    upper_limits_override=upper_limits,
                    log_summary=False,
                )
            )
            selected_path_list, total_cost = _optimize_joint_path(
                ik_layers,
                robot=robot,
                move_type=move_type,
                start_joints=start_joints,
                optimizer_settings=optimizer_settings,
            )
        except RuntimeError:
            cache[cache_key] = None
            return None

        selected_path = tuple(selected_path_list)
        config_switches, bridge_like_segments, worst_joint_step_deg, mean_joint_step_deg = (
            _summarize_selected_path(
                selected_path,
                bridge_trigger_joint_delta_deg=motion_settings.bridge_trigger_joint_delta_deg,
            )
        )
        result = _PathSearchResult(
            pose_rows=candidate_pose_rows,
            ik_layers=ik_layers,
            selected_path=selected_path,
            total_cost=total_cost,
            rotation_offset_deg=rotation_offset_deg,
            config_switches=config_switches,
            bridge_like_segments=bridge_like_segments,
            worst_joint_step_deg=worst_joint_step_deg,
            mean_joint_step_deg=mean_joint_step_deg,
        )
        cache[cache_key] = result
        return result

    def evaluate_preview(rotation_offset_deg: tuple[float, float, float]) -> _PathSearchResult | None:
        cache_key = tuple(round(value, 6) for value in rotation_offset_deg)
        if cache_key in preview_cache:
            return preview_cache[cache_key]

        candidate_preview_rows = tuple(
            _apply_global_pose_rotation_to_pose_row(row, rotation_offset_deg, motion_settings)
            for row in preview_pose_rows
        )
        try:
            preview_layers = tuple(
                _build_ik_layers(
                    candidate_preview_rows,
                    robot=robot,
                    mat_type=mat_type,
                    tool_pose=tool_pose,
                    reference_pose=reference_pose,
                    joint_count=joint_count,
                    optimizer_settings=preview_optimizer_settings,
                    a1_lower_deg=a1_lower_deg,
                    a1_upper_deg=a1_upper_deg,
                    a2_max_deg=a2_max_deg,
                    joint_constraint_tolerance_deg=joint_constraint_tolerance_deg,
                    seed_joints_override=preview_seed_joints,
                    lower_limits_override=lower_limits,
                    upper_limits_override=upper_limits,
                    log_summary=False,
                )
            )
            preview_path_list, preview_cost = _optimize_joint_path(
                preview_layers,
                robot=robot,
                move_type=move_type,
                start_joints=start_joints,
                optimizer_settings=preview_optimizer_settings,
            )
        except RuntimeError:
            preview_cache[cache_key] = None
            return None

        preview_path = tuple(preview_path_list)
        config_switches, bridge_like_segments, worst_joint_step_deg, mean_joint_step_deg = (
            _summarize_selected_path(
                preview_path,
                bridge_trigger_joint_delta_deg=motion_settings.bridge_trigger_joint_delta_deg,
            )
        )
        preview_result = _PathSearchResult(
            pose_rows=candidate_preview_rows,
            ik_layers=preview_layers,
            selected_path=preview_path,
            total_cost=preview_cost,
            rotation_offset_deg=rotation_offset_deg,
            config_switches=config_switches,
            bridge_like_segments=bridge_like_segments,
            worst_joint_step_deg=worst_joint_step_deg,
            mean_joint_step_deg=mean_joint_step_deg,
        )
        preview_cache[cache_key] = preview_result
        return preview_result

    search_enabled = (
        motion_settings.enable_global_pose_search
        and motion_settings.global_pose_search_origin_mm is not None
        and bool(motion_settings.global_pose_search_step_schedule_deg)
    )
    # Always fully evaluate the baseline (0,0,0) offset.
    # A preview failure at (0,0,0) can happen when the sparse 24-row sample
    # happens to land on or near the wrist singularity and the IK layer has no
    # candidates with the lightweight preview seeds.  In that case the old code
    # skipped the baseline entirely and fell back to the 180-degree family-seed
    # search, which moves the IK branch-switch away from the near-singularity
    # region (where wrist refinement can fix it) into an unrelated part of the
    # path (where it cannot be fixed).  Evaluating baseline unconditionally
    # avoids that detour.
    baseline_result = evaluate((0.0, 0.0, 0.0))

    if not search_enabled:
        if baseline_result is None:
            raise RuntimeError("The original pose rows do not yield any globally feasible path.")
        return baseline_result

    if best_result := baseline_result:
        if _path_search_goal_is_satisfied(best_result, optimizer_settings=optimizer_settings):
            return best_result

    best_result = baseline_result
    search_center = (0.0, 0.0, 0.0)
    max_offset_deg = motion_settings.global_pose_search_max_offset_deg
    best_offset = list(best_result.rotation_offset_deg) if best_result is not None else [0.0, 0.0, 0.0]

    for step_deg in motion_settings.global_pose_search_step_schedule_deg:
        if step_deg <= 0.0:
            continue

        if best_result is None:
            seed_candidates: list[_PathSearchResult] = []
            for offset_shell in _iter_global_pose_search_shell_offsets(
                step_deg,
                max_offset_deg=max_offset_deg,
            ):
                preview_candidates: list[_PathSearchResult] = []
                for candidate_offset in offset_shell:
                    preview_result = evaluate_preview(candidate_offset)
                    if preview_result is None:
                        continue
                    preview_candidates.append(preview_result)

                if not preview_candidates:
                    continue

                preview_candidates.sort(key=_path_search_sort_key)
                for preview_result in preview_candidates[: min(6, len(preview_candidates))]:
                    candidate_result = evaluate(preview_result.rotation_offset_deg)
                    if candidate_result is None:
                        continue
                    seed_candidates.append(candidate_result)

                if seed_candidates:
                    break

            # If no seeds found within the normal search shell, try 180-degree family-seed
            # offsets as a last resort.  These represent physically different IK families
            # (different approach to the workspace) that may make the path feasible.
            if not seed_candidates:
                family_preview_candidates: list[_PathSearchResult] = []
                for family_offset in _iter_global_pose_search_family_seed_offsets():
                    preview_result = evaluate_preview(family_offset)
                    if preview_result is None:
                        continue
                    family_preview_candidates.append(preview_result)

                family_preview_candidates.sort(key=_path_search_sort_key)
                for preview_result in family_preview_candidates[: min(6, len(family_preview_candidates))]:
                    candidate_result = evaluate(preview_result.rotation_offset_deg)
                    if candidate_result is None:
                        continue
                    seed_candidates.append(candidate_result)

            if not seed_candidates:
                continue

            best_result = min(seed_candidates, key=_path_search_sort_key)
            best_offset = list(best_result.rotation_offset_deg)
            # Keep the search center at origin so the subsequent gradient refinement
            # explores all ±max_offset_deg offsets from (0,0,0), not from the family seed.
            # This allows finding the closest feasible offset to the original orientation.
            search_center = (0.0, 0.0, 0.0)
            print(
                "Global pose search seeded offset "
                f"{[round(value, 3) for value in best_result.rotation_offset_deg]} deg: "
                f"bridge_like_segments={best_result.bridge_like_segments}, "
                f"config_switches={best_result.config_switches}, "
                f"worst_joint_step={best_result.worst_joint_step_deg:.3f} deg."
            )
            if _path_search_goal_is_satisfied(best_result, optimizer_settings=optimizer_settings):
                return best_result

        improved = True
        while improved:
            improved = False
            for axis_index in range(3):
                for direction in (-1.0, 1.0):
                    candidate_offset = list(best_offset)
                    candidate_offset[axis_index] += direction * step_deg
                    if (
                        abs(candidate_offset[axis_index] - search_center[axis_index])
                        > max_offset_deg + 1e-9
                    ):
                        continue

                    candidate_result = evaluate(tuple(candidate_offset))
                    if candidate_result is None:
                        continue
                    if _is_path_search_result_better(candidate_result, best_result):
                        best_result = candidate_result
                        best_offset = list(candidate_result.rotation_offset_deg)
                        improved = True
                        print(
                            "Global pose search accepted offset "
                            f"{[round(value, 3) for value in candidate_result.rotation_offset_deg]} deg: "
                            f"bridge_like_segments={candidate_result.bridge_like_segments}, "
                            f"config_switches={candidate_result.config_switches}, "
                            f"worst_joint_step={candidate_result.worst_joint_step_deg:.3f} deg."
                        )

        if best_result is not None and _path_search_goal_is_satisfied(
            best_result,
            optimizer_settings=optimizer_settings,
        ):
            return best_result

    if best_result is None:
        raise RuntimeError(
            "No globally feasible path was found, even after searching rigid orientation "
            "offsets for the whole path."
        )
    return best_result


def _is_path_search_result_better(
    candidate_result: _PathSearchResult,
    reference_result: _PathSearchResult,
) -> bool:
    """比较两组整路径姿态偏置搜索结果的优先级。"""

    return _path_search_sort_key(candidate_result) < _path_search_sort_key(reference_result)


def _path_search_sort_key(result: _PathSearchResult) -> tuple[float, ...]:
    """把搜索结果压成稳定的排序键。"""

    total_offset_deg = sum(abs(value) for value in result.rotation_offset_deg)
    return (
        float(result.bridge_like_segments),
        float(result.config_switches),
        float(result.worst_joint_step_deg),
        float(result.mean_joint_step_deg),
        float(result.total_cost),
        float(total_offset_deg),
    )


def _iter_global_pose_search_shell_offsets(
    step_deg: float,
    *,
    max_offset_deg: float,
) -> tuple[tuple[tuple[float, float, float], ...], ...]:
    """在当前 coarse 步长下枚举一圈全路径统一姿态偏置候选。"""

    if step_deg <= 0.0 or max_offset_deg <= 0.0:
        return ()

    max_shell = int(math.floor(max_offset_deg / step_deg + 1e-9))
    offset_shells: list[tuple[tuple[float, float, float], ...]] = []
    for shell_index in range(1, max_shell + 1):
        shell_offsets: list[tuple[float, float, float]] = []
        for x_index in range(-shell_index, shell_index + 1):
            for y_index in range(-shell_index, shell_index + 1):
                for z_index in range(-shell_index, shell_index + 1):
                    if max(abs(x_index), abs(y_index), abs(z_index)) != shell_index:
                        continue
                    offset = (
                        x_index * step_deg,
                        y_index * step_deg,
                        z_index * step_deg,
                    )
                    if any(abs(value) > max_offset_deg + 1e-9 for value in offset):
                        continue
                    shell_offsets.append(offset)

        shell_offsets.sort(
            key=lambda offset: (
                -sum(1 for value in offset if abs(value) > 1e-9),
                sum(abs(value) for value in offset),
                offset,
            )
        )
        if shell_offsets:
            offset_shells.append(tuple(shell_offsets))
    return tuple(offset_shells)


def _build_global_pose_search_preview_rows(
    pose_rows: Sequence[dict[str, float]],
    *,
    target_count: int = 24,
) -> tuple[dict[str, float], ...]:
    """为全局位姿起始搜索构造一个稀疏预览路径。"""

    if len(pose_rows) <= target_count:
        return tuple(dict(row) for row in pose_rows)

    step = max(1, math.ceil((len(pose_rows) - 1) / max(1, target_count - 1)))
    indices = list(range(0, len(pose_rows), step))
    if indices[-1] != len(pose_rows) - 1:
        indices.append(len(pose_rows) - 1)
    return tuple(dict(pose_rows[index]) for index in indices)


def _iter_global_pose_search_family_seed_offsets() -> tuple[tuple[float, float, float], ...]:
    """枚举一组 180 度翻转的姿态族中心。"""

    family_offsets: list[tuple[float, float, float]] = []
    for x_offset in (0.0, -180.0):
        for y_offset in (0.0, -180.0):
            for z_offset in (0.0, -180.0):
                offset = (x_offset, y_offset, z_offset)
                if not any(abs(value) > 1e-9 for value in offset):
                    continue
                family_offsets.append(offset)

    family_offsets.sort(
        key=lambda offset: (
            -sum(1 for value in offset if abs(value) > 1e-9),
            sum(abs(value) for value in offset),
            offset,
        )
    )
    return tuple(family_offsets)


def _build_preview_seed_joint_strategies(
    *,
    robot,
    lower_limits: tuple[float, ...],
    upper_limits: tuple[float, ...],
    joint_count: int,
) -> tuple[tuple[float, ...], ...]:
    """为全局位姿 preview 准备一组更轻量的 seed。"""

    seeds: list[tuple[float, ...]] = []

    current_joints = _trim_joint_vector(robot.Joints().list(), joint_count)
    if current_joints:
        seeds.append(current_joints)

    home_joints = _trim_joint_vector(robot.JointsHome().list(), joint_count)
    if home_joints:
        seeds.append(home_joints)

    midpoint = tuple((lower + upper) * 0.5 for lower, upper in zip(lower_limits, upper_limits))
    seeds.append(midpoint)
    seeds.append(_clip_seed_to_limits((0.0,) * joint_count, lower_limits, upper_limits))

    if joint_count >= 6:
        for joint5 in (-60.0, 60.0):
            wrist_seed = list(midpoint)
            wrist_seed[4] = joint5
            wrist_seed[5] = 0.0
            seeds.append(_clip_seed_to_limits(wrist_seed, lower_limits, upper_limits))

    deduped_seeds: list[tuple[float, ...]] = []
    seen: set[tuple[float, ...]] = set()
    for seed in seeds:
        dedup_key = tuple(round(value, _IK_DEDUP_DECIMALS) for value in seed)
        if dedup_key in seen:
            continue
        seen.add(dedup_key)
        deduped_seeds.append(seed)

    return tuple(deduped_seeds)


def _path_search_goal_is_satisfied(
    result: _PathSearchResult,
    *,
    optimizer_settings: _PathOptimizerSettings,
) -> bool:
    """判断当前 exact path 是否已经好到可以停止继续搜整条刚体偏置。"""

    if result.bridge_like_segments != 0 or result.config_switches != 0:
        return False
    preferred_limit = max(optimizer_settings.preferred_joint_step_deg, default=0.0)
    return result.worst_joint_step_deg <= preferred_limit + 1e-9


def _apply_global_pose_rotation_to_pose_row(
    pose_row: dict[str, float],
    rotation_offset_deg: Sequence[float],
    motion_settings,
) -> dict[str, float]:
    """把单个 pose row 绕统一锚点做刚体旋转。"""

    if (
        motion_settings.global_pose_search_origin_mm is None
        or not any(abs(value) > 1e-9 for value in rotation_offset_deg)
    ):
        return dict(pose_row)

    base_rotation, base_translation = _pose_row_to_rotation_translation(pose_row)
    search_rotation = _rotation_matrix_from_xyz_offset_deg(rotation_offset_deg)
    origin = motion_settings.global_pose_search_origin_mm

    rotated_rotation = _multiply_rotation_matrices(search_rotation, base_rotation)
    translated_from_origin = _subtract_vectors(base_translation, origin)
    rotated_translation = _add_vectors(
        _multiply_rotation_vector(search_rotation, translated_from_origin),
        origin,
    )
    return _pose_row_from_rotation_translation(rotated_rotation, rotated_translation)
