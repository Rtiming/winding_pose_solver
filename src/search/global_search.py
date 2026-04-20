from __future__ import annotations

import math
from dataclasses import replace
from typing import Sequence

from src.core.geometry import _build_pose, _trim_joint_vector
from src.search.ik_collection import (
    _build_seed_joint_strategies,
    _collect_ik_candidates,
)
from src.search.path_optimizer import _optimize_joint_path, _summarize_selected_path
from src.core.types import _IKCandidate, _IKLayer, _PathOptimizerSettings, _PathSearchResult


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
    if getattr(robot, "ik_seed_invariant", False):
        search_seed_joints: tuple[tuple[float, ...], ...] = ()
    else:
        search_seed_joints = _build_seed_joint_strategies(
            robot=robot,
            lower_limits=lower_limits,
            upper_limits=upper_limits,
            joint_count=joint_count,
        )

    zero_profile = tuple((0.0, 0.0) for _ in reference_pose_rows)
    best_result = _evaluate_frame_a_origin_profile(
        reference_pose_rows,
        frame_a_origin_yz_profile_mm=zero_profile,
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
        lock_profile_endpoints=bool(
            getattr(motion_settings, "lock_frame_a_origin_yz_profile_endpoints", True)
        ),
    )
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
    reused_ik_layers: Sequence[_IKLayer] | None = None,
    recompute_row_indices: Sequence[int] | None = None,
    lock_profile_endpoints: bool = False,
) -> _PathSearchResult:
    frame_a_origin_yz_profile_mm = _normalize_frame_a_origin_yz_profile(
        reference_pose_rows,
        frame_a_origin_yz_profile_mm,
        lock_profile_endpoints=lock_profile_endpoints,
    )
    if (
        recompute_row_indices is not None
        and len(reference_pose_rows) >= 2
    ):
        recompute_set = {int(index) for index in recompute_row_indices}
        terminal_index = len(reference_pose_rows) - 1
        if (
            lock_profile_endpoints
            or (
                _reference_path_has_terminal_start_copy(reference_pose_rows)
                and (0 in recompute_set or terminal_index in recompute_set)
            )
        ):
            recompute_set.add(0)
            recompute_set.add(terminal_index)
        recompute_row_indices = tuple(sorted(recompute_set))
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
        reused_ik_layers=reused_ik_layers,
        recompute_row_indices=recompute_row_indices,
    )
    return _finalize_frame_a_origin_profile_result(
        reference_pose_rows=reference_pose_rows,
        adjusted_pose_rows=adjusted_pose_rows,
        ik_layers=ik_layers,
        frame_a_origin_yz_profile_mm=frame_a_origin_yz_profile_mm,
        row_labels=row_labels,
        inserted_flags=inserted_flags,
        robot=robot,
        move_type=move_type,
        start_joints=start_joints,
        optimizer_settings=optimizer_settings,
        bridge_trigger_joint_delta_deg=bridge_trigger_joint_delta_deg,
    )


def _finalize_frame_a_origin_profile_result(
    *,
    reference_pose_rows: Sequence[dict[str, float]],
    adjusted_pose_rows: Sequence[dict[str, float]],
    ik_layers: Sequence[_IKLayer],
    frame_a_origin_yz_profile_mm: Sequence[tuple[float, float]],
    row_labels: Sequence[str],
    inserted_flags: Sequence[bool],
    robot,
    move_type: str,
    start_joints: tuple[float, ...],
    optimizer_settings: _PathOptimizerSettings,
    bridge_trigger_joint_delta_deg: float,
) -> _PathSearchResult:
    ik_layers_tuple = tuple(ik_layers)
    ik_empty_row_count = sum(1 for layer in ik_layers_tuple if not layer.candidates)

    selected_path: tuple[_IKCandidate, ...] = ()
    total_cost = math.inf
    if ik_empty_row_count == 0 and ik_layers_tuple:
        try:
            selected_path_list, total_cost = _optimize_joint_path(
                ik_layers_tuple,
                robot=robot,
                move_type=move_type,
                start_joints=start_joints,
                optimizer_settings=optimizer_settings,
                require_terminal_match_start=_reference_path_has_terminal_start_copy(
                    reference_pose_rows
                ),
            )
            selected_path = tuple(selected_path_list)
        except RuntimeError:
            if bool(getattr(optimizer_settings, "enable_joint_continuity_constraint", False)):
                # Keep a diagnostic path when the hard per-step constraint is too
                # tight. Official delivery is still blocked later by continuity
                # gates, but users can inspect the actual bad segments.
                try:
                    fallback_optimizer_settings = replace(
                        optimizer_settings,
                        enable_joint_continuity_constraint=False,
                    )
                    selected_path_list, total_cost = _optimize_joint_path(
                        ik_layers_tuple,
                        robot=robot,
                        move_type=move_type,
                        start_joints=start_joints,
                        optimizer_settings=fallback_optimizer_settings,
                        require_terminal_match_start=_reference_path_has_terminal_start_copy(
                            reference_pose_rows
                        ),
                    )
                    selected_path = tuple(selected_path_list)
                except RuntimeError:
                    selected_path = ()
                    total_cost = math.inf
            else:
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
        pose_rows=tuple(adjusted_pose_rows),
        ik_layers=ik_layers_tuple,
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


def _normalize_frame_a_origin_yz_profile(
    reference_pose_rows: Sequence[dict[str, float]],
    frame_a_origin_yz_profile_mm: Sequence[tuple[float, float]],
    *,
    lock_profile_endpoints: bool = False,
) -> tuple[tuple[float, float], ...]:
    profile = tuple(
        (float(dy_mm), float(dz_mm))
        for dy_mm, dz_mm in frame_a_origin_yz_profile_mm
    )
    if not profile:
        return profile

    terminal_index = len(profile) - 1
    has_terminal_start_copy = (
        len(profile) >= 2 and _reference_path_has_terminal_start_copy(reference_pose_rows)
    )
    if lock_profile_endpoints:
        locked_profile = list(profile)
        locked_profile[0] = (0.0, 0.0)
        if len(locked_profile) >= 2:
            locked_profile[terminal_index] = (0.0, 0.0)
        return tuple(locked_profile)

    if has_terminal_start_copy:
        profile = (*profile[:-1], profile[0])
    return profile


def _close_terminal_profile_if_needed(
    reference_pose_rows: Sequence[dict[str, float]],
    frame_a_origin_yz_profile_mm: Sequence[tuple[float, float]],
) -> tuple[tuple[float, float], ...]:
    return _normalize_frame_a_origin_yz_profile(
        reference_pose_rows,
        frame_a_origin_yz_profile_mm,
        lock_profile_endpoints=False,
    )


def _reference_path_has_terminal_start_copy(
    reference_pose_rows: Sequence[dict[str, float]],
) -> bool:
    if len(reference_pose_rows) < 2:
        return False
    first_row = reference_pose_rows[0]
    terminal_row = reference_pose_rows[-1]
    if "source_row" not in first_row or "source_row" not in terminal_row:
        return False
    if int(float(first_row["source_row"])) != int(float(terminal_row["source_row"])):
        return False
    return _pose_rows_match(first_row, terminal_row)


def _pose_rows_match(
    first_row: dict[str, float],
    second_row: dict[str, float],
    *,
    tolerance: float = 1e-6,
) -> bool:
    for column in (
        "x_mm",
        "y_mm",
        "z_mm",
        "r11",
        "r12",
        "r13",
        "r21",
        "r22",
        "r23",
        "r31",
        "r32",
        "r33",
    ):
        if column not in first_row or column not in second_row:
            return False
        if abs(float(first_row[column]) - float(second_row[column])) > tolerance:
            return False
    return True


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
    reused_ik_layers: Sequence[_IKLayer] | None = None,
    recompute_row_indices: Sequence[int] | None = None,
) -> tuple[tuple[_IKLayer, ...], int]:
    ik_layers: list[_IKLayer] = []
    ik_empty_row_count = 0
    lower_limits_tuple = tuple(float(value) for value in lower_limits)
    upper_limits_tuple = tuple(float(value) for value in upper_limits)
    recompute_row_set = (
        set(int(index) for index in recompute_row_indices)
        if recompute_row_indices is not None
        else None
    )
    can_reuse_layers = (
        reused_ik_layers is not None and len(reused_ik_layers) == len(pose_rows)
    )
    for pose_row in pose_rows:
        row_index = len(ik_layers)
        if (
            can_reuse_layers
            and recompute_row_set is not None
            and row_index not in recompute_row_set
        ):
            reused_layer = reused_ik_layers[row_index]
            if not reused_layer.candidates:
                ik_empty_row_count += 1
            ik_layers.append(reused_layer)
            continue

        pose = _build_pose(pose_row, mat_type)
        candidates = _collect_ik_candidates(
            robot,
            pose,
            tool_pose=tool_pose,
            reference_pose=reference_pose,
            lower_limits=lower_limits_tuple,
            upper_limits=upper_limits_tuple,
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
