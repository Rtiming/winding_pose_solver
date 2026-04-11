from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

from src.bridge_builder import _insert_interpolated_transition_rows
from src.geometry import _build_pose, _mean_abs_joint_delta
from src.global_search import _evaluate_frame_a_origin_profile, _path_search_sort_key
from src.ik_collection import _collect_ik_candidates
from src.path_optimizer import _candidate_transition_penalty, _joint_transition_penalty
from src.types import _IKCandidate, _PathOptimizerSettings, _PathSearchResult


@dataclass(frozen=True)
class _WindowState:
    dy_mm: float
    dz_mm: float
    candidate: _IKCandidate


def _refine_path_with_frame_a_origin_profile(
    search_result: _PathSearchResult,
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
    lower_limits: Sequence[float],
    upper_limits: Sequence[float],
    a1_lower_deg: float,
    a1_upper_deg: float,
    a2_max_deg: float,
    joint_constraint_tolerance_deg: float,
) -> _PathSearchResult:
    best_result = search_result
    if not best_result.selected_path:
        return best_result

    lower_limits_tuple = tuple(float(value) for value in lower_limits[:joint_count])
    upper_limits_tuple = tuple(float(value) for value in upper_limits[:joint_count])

    for pass_index in range(max(0, motion_settings.frame_a_origin_yz_max_passes)):
        problem_segments = _collect_problem_segments(
            best_result.selected_path,
            bridge_trigger_joint_delta_deg=motion_settings.bridge_trigger_joint_delta_deg,
        )
        if not problem_segments:
            break

        accepted_result: _PathSearchResult | None = None
        accepted_metadata: tuple[int, int, int, float, float] | None = None
        for segment_index, _config_changed, _max_joint_delta, _mean_joint_delta in problem_segments[:5]:
            window_start = max(0, segment_index - motion_settings.frame_a_origin_yz_window_radius)
            window_end = min(
                len(best_result.reference_pose_rows) - 1,
                segment_index + 1 + motion_settings.frame_a_origin_yz_window_radius,
            )

            for envelope_mm in motion_settings.frame_a_origin_yz_envelope_schedule_mm:
                for step_mm in motion_settings.frame_a_origin_yz_step_schedule_mm:
                    candidate_profile = _solve_window_profile_dp(
                        best_result,
                        window_start=window_start,
                        window_end=window_end,
                        max_abs_offset_mm=envelope_mm,
                        step_mm=step_mm,
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
                        lower_limits=lower_limits_tuple,
                        upper_limits=upper_limits_tuple,
                        start_joints=start_joints,
                    )
                    if candidate_profile is None:
                        continue

                    candidate_result = _evaluate_frame_a_origin_profile(
                        best_result.reference_pose_rows,
                        frame_a_origin_yz_profile_mm=candidate_profile,
                        row_labels=best_result.row_labels,
                        inserted_flags=best_result.inserted_flags,
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
                        seed_joints=_build_local_seed_joints(best_result, window_start, window_end),
                        lower_limits=lower_limits_tuple,
                        upper_limits=upper_limits_tuple,
                        bridge_trigger_joint_delta_deg=motion_settings.bridge_trigger_joint_delta_deg,
                    )
                    if _path_search_sort_key(candidate_result) < _path_search_sort_key(best_result):
                        if (
                            accepted_result is None
                            or _path_search_sort_key(candidate_result)
                            < _path_search_sort_key(accepted_result)
                        ):
                            accepted_result = candidate_result
                            accepted_metadata = (
                                segment_index,
                                window_start,
                                window_end,
                                float(envelope_mm),
                                float(step_mm),
                            )

        if accepted_result is None:
            break
        assert accepted_metadata is not None
        segment_index, window_start, window_end, envelope_mm, step_mm = accepted_metadata
        print(
            "Accepted Frame-2 Y/Z window refinement: "
            f"pass={pass_index + 1}, "
            f"segment={best_result.row_labels[segment_index]}->{best_result.row_labels[segment_index + 1]}, "
            f"window={best_result.row_labels[window_start]}->{best_result.row_labels[window_end]}, "
            f"envelope={envelope_mm:.1f} mm, step={step_mm:.1f} mm, "
            f"bridge_like_segments={accepted_result.bridge_like_segments}, "
            f"worst_joint_step={accepted_result.worst_joint_step_deg:.3f} deg."
        )
        best_result = accepted_result

    return best_result


def _attempt_inserted_transition_repair(
    search_result: _PathSearchResult,
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
    lower_limits: Sequence[float],
    upper_limits: Sequence[float],
    a1_lower_deg: float,
    a1_upper_deg: float,
    a2_max_deg: float,
    joint_constraint_tolerance_deg: float,
) -> _PathSearchResult | None:
    if not search_result.selected_path:
        return None

    lower_limits_tuple = tuple(float(value) for value in lower_limits[:joint_count])
    upper_limits_tuple = tuple(float(value) for value in upper_limits[:joint_count])
    problem_segments = _collect_problem_segments(
        search_result.selected_path,
        bridge_trigger_joint_delta_deg=motion_settings.bridge_trigger_joint_delta_deg,
    )
    if not problem_segments:
        return search_result

    best_clean_result: _PathSearchResult | None = None
    for segment_index, _config_changed, _max_joint_delta, _mean_joint_delta in problem_segments[:3]:
        for insertion_count in motion_settings.frame_a_origin_yz_insertion_counts:
            (
                augmented_reference_rows,
                augmented_profile,
                augmented_labels,
                augmented_flags,
            ) = _insert_interpolated_transition_rows(
                search_result.reference_pose_rows,
                search_result.frame_a_origin_yz_profile_mm,
                search_result.row_labels,
                search_result.inserted_flags,
                segment_index=segment_index,
                insertion_count=insertion_count,
            )
            candidate_result = _evaluate_frame_a_origin_profile(
                augmented_reference_rows,
                frame_a_origin_yz_profile_mm=augmented_profile,
                row_labels=augmented_labels,
                inserted_flags=augmented_flags,
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
                seed_joints=_build_local_seed_joints(search_result, segment_index, segment_index + 1),
                lower_limits=lower_limits_tuple,
                upper_limits=upper_limits_tuple,
                bridge_trigger_joint_delta_deg=motion_settings.bridge_trigger_joint_delta_deg,
            )
            residual_problems = _collect_problem_segments(
                candidate_result.selected_path,
                bridge_trigger_joint_delta_deg=motion_settings.bridge_trigger_joint_delta_deg,
            )
            if residual_problems:
                continue
            if best_clean_result is None or _path_search_sort_key(candidate_result) < _path_search_sort_key(best_clean_result):
                best_clean_result = candidate_result
                print(
                    "Accepted inserted transition samples: "
                    f"segment={search_result.row_labels[segment_index]}->{search_result.row_labels[segment_index + 1]}, "
                    f"inserted_points={insertion_count}, "
                    f"waypoint_count={len(candidate_result.pose_rows)}, "
                    f"worst_joint_step={candidate_result.worst_joint_step_deg:.3f} deg."
                )

    return best_clean_result


def _solve_window_profile_dp(
    search_result: _PathSearchResult,
    *,
    window_start: int,
    window_end: int,
    max_abs_offset_mm: float,
    step_mm: float,
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
    lower_limits: Sequence[float],
    upper_limits: Sequence[float],
    start_joints: Sequence[float],
) -> tuple[tuple[float, float], ...] | None:
    if step_mm <= 0.0 or max_abs_offset_mm < 0.0 or window_end < window_start:
        return None

    offset_step_limit_mm = max(1.0, step_mm * 2.0)
    window_layers: list[tuple[_WindowState, ...]] = []
    dedup_rounding = 6

    for row_index in range(window_start, window_end + 1):
        reference_row = search_result.reference_pose_rows[row_index]
        current_offset = search_result.frame_a_origin_yz_profile_mm[row_index]
        row_states: list[_WindowState] = []
        seen: set[tuple[float, float, tuple[float, ...]]] = set()

        for dy_mm, dz_mm in _build_window_offset_states(
            current_offset,
            max_abs_offset_mm=max_abs_offset_mm,
            step_mm=step_mm,
        ):
            adjusted_row = dict(reference_row)
            adjusted_row["x_mm"] = float(reference_row["x_mm"])
            adjusted_row["y_mm"] = float(reference_row["y_mm"]) + dy_mm
            adjusted_row["z_mm"] = float(reference_row["z_mm"]) + dz_mm
            pose = _build_pose(adjusted_row, mat_type)
            candidates = _collect_ik_candidates(
                robot,
                pose,
                tool_pose=tool_pose,
                reference_pose=reference_pose,
                lower_limits=tuple(lower_limits),
                upper_limits=tuple(upper_limits),
                seed_joints=_build_local_seed_joints(search_result, window_start, window_end),
                joint_count=joint_count,
                optimizer_settings=optimizer_settings,
                a1_lower_deg=a1_lower_deg,
                a1_upper_deg=a1_upper_deg,
                a2_max_deg=a2_max_deg,
                joint_constraint_tolerance_deg=joint_constraint_tolerance_deg,
            )
            for candidate in candidates[:6]:
                dedup_key = (
                    round(dy_mm, dedup_rounding),
                    round(dz_mm, dedup_rounding),
                    tuple(round(value, dedup_rounding) for value in candidate.joints),
                )
                if dedup_key in seen:
                    continue
                seen.add(dedup_key)
                row_states.append(_WindowState(dy_mm=dy_mm, dz_mm=dz_mm, candidate=candidate))

        if not row_states:
            return None

        row_states.sort(
            key=lambda state: (
                state.candidate.config_flags,
                state.candidate.joint_limit_penalty + state.candidate.singularity_penalty,
                abs(state.dy_mm) + abs(state.dz_mm),
                state.candidate.joints,
            )
        )
        window_layers.append(tuple(row_states))

    previous_costs: list[float] = []
    backpointers: list[list[int]] = []

    left_boundary_offset = (
        search_result.frame_a_origin_yz_profile_mm[window_start - 1]
        if window_start > 0
        else (0.0, 0.0)
    )
    right_boundary_offset = (
        search_result.frame_a_origin_yz_profile_mm[window_end + 1]
        if window_end + 1 < len(search_result.frame_a_origin_yz_profile_mm)
        else (0.0, 0.0)
    )
    left_boundary_candidates = (
        search_result.ik_layers[window_start - 1].candidates
        if window_start > 0 and window_start - 1 < len(search_result.ik_layers)
        else ()
    )
    right_boundary_candidates = (
        search_result.ik_layers[window_end + 1].candidates
        if window_end + 1 < len(search_result.ik_layers)
        else ()
    )

    for state in window_layers[0]:
        if not _passes_offset_step_constraint(
            left_boundary_offset,
            (state.dy_mm, state.dz_mm),
            limit_mm=offset_step_limit_mm,
        ):
            previous_costs.append(math.inf)
            continue
        if left_boundary_candidates:
            transition_cost = min(
                _candidate_transition_penalty(
                    boundary_candidate,
                    state.candidate,
                    optimizer_settings,
                )
                for boundary_candidate in left_boundary_candidates
            )
        else:
            transition_cost = optimizer_settings.start_transition_weight * _joint_transition_penalty(
                start_joints,
                state.candidate.joints,
                optimizer_settings,
            )
        previous_costs.append(
            transition_cost
            + _window_state_node_cost(state)
            + _offset_transition_penalty(left_boundary_offset, (state.dy_mm, state.dz_mm))
        )

    for layer_index in range(1, len(window_layers)):
        current_layer = window_layers[layer_index]
        previous_layer = window_layers[layer_index - 1]
        current_costs = [math.inf] * len(current_layer)
        current_backpointers = [-1] * len(current_layer)

        for current_index, current_state in enumerate(current_layer):
            best_cost = math.inf
            best_previous_index = -1
            for previous_index, previous_state in enumerate(previous_layer):
                if not math.isfinite(previous_costs[previous_index]):
                    continue
                if not _passes_offset_step_constraint(
                    (previous_state.dy_mm, previous_state.dz_mm),
                    (current_state.dy_mm, current_state.dz_mm),
                    limit_mm=offset_step_limit_mm,
                ):
                    continue
                transition_cost = _candidate_transition_penalty(
                    previous_state.candidate,
                    current_state.candidate,
                    optimizer_settings,
                )
                transition_cost += _offset_transition_penalty(
                    (previous_state.dy_mm, previous_state.dz_mm),
                    (current_state.dy_mm, current_state.dz_mm),
                )
                total_cost = (
                    previous_costs[previous_index]
                    + transition_cost
                    + _window_state_node_cost(current_state)
                )
                if total_cost < best_cost:
                    best_cost = total_cost
                    best_previous_index = previous_index

            current_costs[current_index] = best_cost
            current_backpointers[current_index] = best_previous_index

        if not any(math.isfinite(cost) for cost in current_costs):
            return None

        previous_costs = current_costs
        backpointers.append(current_backpointers)

    best_total_cost = math.inf
    best_last_index = -1
    for state_index, state in enumerate(window_layers[-1]):
        if not math.isfinite(previous_costs[state_index]):
            continue
        if not _passes_offset_step_constraint(
            (state.dy_mm, state.dz_mm),
            right_boundary_offset,
            limit_mm=offset_step_limit_mm,
        ):
            continue
        transition_cost = _offset_transition_penalty(
            (state.dy_mm, state.dz_mm),
            right_boundary_offset,
        )
        if right_boundary_candidates:
            transition_cost += min(
                _candidate_transition_penalty(
                    state.candidate,
                    boundary_candidate,
                    optimizer_settings,
                )
                for boundary_candidate in right_boundary_candidates
            )
        total_cost = previous_costs[state_index] + transition_cost
        if total_cost < best_total_cost:
            best_total_cost = total_cost
            best_last_index = state_index

    if best_last_index < 0:
        return None

    chosen_states = [window_layers[-1][best_last_index]]
    for layer_index in range(len(window_layers) - 2, -1, -1):
        best_last_index = backpointers[layer_index][best_last_index]
        chosen_states.append(window_layers[layer_index][best_last_index])
    chosen_states.reverse()

    updated_profile = list(search_result.frame_a_origin_yz_profile_mm)
    for row_index, state in zip(range(window_start, window_end + 1), chosen_states):
        updated_profile[row_index] = (float(state.dy_mm), float(state.dz_mm))
    return tuple(updated_profile)


def _collect_problem_segments(
    selected_path: Sequence[_IKCandidate],
    *,
    bridge_trigger_joint_delta_deg: float,
) -> tuple[tuple[int, bool, float, float], ...]:
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
            segments.append((segment_index, config_changed, max_joint_delta, mean_joint_delta))

    segments.sort(
        key=lambda item: (
            -int(item[1]),
            -float(item[2]),
            -float(item[3]),
        )
    )
    return tuple(segments)


def _format_focus_segment_report(search_result: _PathSearchResult) -> str:
    lines = ["Focused segment report:"]
    for start_label, end_label in (("372", "373"), ("380", "381")):
        lines.extend(_describe_focus_pair(search_result, start_label, end_label))
    return "\n".join(lines)


def _format_failure_diagnostics(
    search_result: _PathSearchResult,
    *,
    bridge_trigger_joint_delta_deg: float,
) -> str:
    lines = [
        "No strictly acceptable continuous path was found under the fixed Frame-A orientation "
        "and the allowed per-point Frame-2 Y/Z origin optimization.",
        (
            "Summary: "
            f"ik_empty_rows={search_result.ik_empty_row_count}, "
            f"config_switches={search_result.config_switches}, "
            f"bridge_like_segments={search_result.bridge_like_segments}, "
            f"worst_joint_step={search_result.worst_joint_step_deg:.3f} deg."
        ),
    ]

    if search_result.ik_empty_row_count > 0:
        empty_rows = [
            search_result.row_labels[index]
            for index, layer in enumerate(search_result.ik_layers)
            if not layer.candidates
        ]
        lines.append("IK-empty rows: " + ", ".join(empty_rows[:12]))

    problem_segments = _collect_problem_segments(
        search_result.selected_path,
        bridge_trigger_joint_delta_deg=bridge_trigger_joint_delta_deg,
    )
    if problem_segments:
        lines.append("Failing segments:")
        for segment_index, config_changed, max_joint_delta, mean_joint_delta in problem_segments[:8]:
            left_label = search_result.row_labels[segment_index]
            right_label = search_result.row_labels[segment_index + 1]
            left_profile = search_result.frame_a_origin_yz_profile_mm[segment_index]
            right_profile = search_result.frame_a_origin_yz_profile_mm[segment_index + 1]
            left_layer = search_result.ik_layers[segment_index]
            right_layer = search_result.ik_layers[segment_index + 1]
            left_families = len({candidate.config_flags for candidate in left_layer.candidates})
            right_families = len({candidate.config_flags for candidate in right_layer.candidates})
            lines.append(
                f"  {left_label}->{right_label}: "
                f"config_changed={config_changed}, "
                f"max_joint_delta={max_joint_delta:.3f} deg, "
                f"mean_joint_delta={mean_joint_delta:.3f} deg, "
                f"profile=({left_profile[0]:.3f}, {left_profile[1]:.3f}) -> "
                f"({right_profile[0]:.3f}, {right_profile[1]:.3f}) mm, "
                f"candidate_families={left_families}->{right_families}, "
                f"candidate_counts={len(left_layer.candidates)}->{len(right_layer.candidates)}"
            )

    lines.append(_format_focus_segment_report(search_result))
    return "\n".join(lines)


def _build_local_seed_joints(
    search_result: _PathSearchResult,
    window_start: int,
    window_end: int,
) -> tuple[tuple[float, ...], ...]:
    seeds: list[tuple[float, ...]] = []
    seen: set[tuple[float, ...]] = set()

    def append_seed(candidate: _IKCandidate | None) -> None:
        if candidate is None:
            return
        seed = tuple(float(value) for value in candidate.joints)
        if seed in seen:
            return
        seen.add(seed)
        seeds.append(seed)

    for row_index in range(max(0, window_start - 1), min(len(search_result.selected_path), window_end + 2)):
        append_seed(search_result.selected_path[row_index])

    for layer in search_result.ik_layers[max(0, window_start - 1) : min(len(search_result.ik_layers), window_end + 2)]:
        for candidate in layer.candidates[:4]:
            append_seed(candidate)

    return tuple(seeds)


def _build_window_offset_states(
    current_offset: tuple[float, float],
    *,
    max_abs_offset_mm: float,
    step_mm: float,
) -> tuple[tuple[float, float], ...]:
    dy0, dz0 = (float(current_offset[0]), float(current_offset[1]))
    multipliers = (-3, -2, -1, 0, 1, 2, 3)
    states: list[tuple[float, float]] = []
    seen: set[tuple[float, float]] = set()

    def append_state(dy_mm: float, dz_mm: float) -> None:
        clipped_state = (
            round(max(-max_abs_offset_mm, min(max_abs_offset_mm, dy_mm)), 6),
            round(max(-max_abs_offset_mm, min(max_abs_offset_mm, dz_mm)), 6),
        )
        if clipped_state in seen:
            return
        seen.add(clipped_state)
        states.append(clipped_state)

    append_state(dy0, dz0)
    for dy_multiplier in multipliers:
        for dz_multiplier in multipliers:
            append_state(dy0 + dy_multiplier * step_mm, dz0 + dz_multiplier * step_mm)

    for dy_edge in (-max_abs_offset_mm, 0.0, max_abs_offset_mm):
        for dz_edge in (-max_abs_offset_mm, 0.0, max_abs_offset_mm):
            append_state(dy_edge, dz_edge)

    states.sort(key=lambda value: (abs(value[0] - dy0) + abs(value[1] - dz0), abs(value[0]) + abs(value[1]), value))
    return tuple(states)


def _window_state_node_cost(state: _WindowState) -> float:
    return (
        0.22 * (state.candidate.joint_limit_penalty + state.candidate.singularity_penalty)
        + 0.20 * math.hypot(state.dy_mm, state.dz_mm)
    )


def _offset_transition_penalty(
    previous_offset: tuple[float, float],
    current_offset: tuple[float, float],
) -> float:
    return 6.0 * math.hypot(
        float(current_offset[0]) - float(previous_offset[0]),
        float(current_offset[1]) - float(previous_offset[1]),
    )


def _passes_offset_step_constraint(
    previous_offset: tuple[float, float],
    current_offset: tuple[float, float],
    *,
    limit_mm: float,
) -> bool:
    return (
        abs(float(current_offset[0]) - float(previous_offset[0])) <= limit_mm + 1e-9
        and abs(float(current_offset[1]) - float(previous_offset[1])) <= limit_mm + 1e-9
    )


def _describe_focus_pair(
    search_result: _PathSearchResult,
    start_label: str,
    end_label: str,
) -> list[str]:
    labels = list(search_result.row_labels)
    if start_label not in labels or end_label not in labels:
        return [f"  {start_label}->{end_label}: labels not present in current path."]

    start_index = labels.index(start_label)
    end_index = labels.index(end_label)
    if end_index <= start_index:
        return [f"  {start_label}->{end_label}: label order is not monotonic in current path."]

    lines: list[str] = []
    if end_index == start_index + 1:
        lines.append(_format_single_segment_line(search_result, start_index))
        return lines

    lines.append(
        f"  {start_label}->{end_label}: {end_index - start_index - 1} inserted transition sample(s) in between."
    )
    for segment_index in range(start_index, end_index):
        lines.append(_format_single_segment_line(search_result, segment_index))
    return lines


def _format_single_segment_line(
    search_result: _PathSearchResult,
    segment_index: int,
) -> str:
    left_label = search_result.row_labels[segment_index]
    right_label = search_result.row_labels[segment_index + 1]
    if segment_index + 1 >= len(search_result.selected_path):
        return f"    {left_label}->{right_label}: no selected joint path available."

    previous_candidate = search_result.selected_path[segment_index]
    current_candidate = search_result.selected_path[segment_index + 1]
    joint_deltas = [
        abs(current - previous)
        for previous, current in zip(previous_candidate.joints, current_candidate.joints)
    ]
    max_joint_delta = max(joint_deltas, default=0.0)
    previous_profile = search_result.frame_a_origin_yz_profile_mm[segment_index]
    current_profile = search_result.frame_a_origin_yz_profile_mm[segment_index + 1]
    return (
        f"    {left_label}->{right_label}: "
        f"config={previous_candidate.config_flags}->{current_candidate.config_flags}, "
        f"max_joint_delta={max_joint_delta:.3f} deg, "
        f"profile=({previous_profile[0]:.3f}, {previous_profile[1]:.3f}) -> "
        f"({current_profile[0]:.3f}, {current_profile[1]:.3f}) mm"
    )
