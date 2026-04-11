from __future__ import annotations

import math
from typing import Sequence

from src.types import (
    _BridgeCandidate,
    _IKCandidate,
    _IKLayer,
    _PathOptimizerSettings,
    _ProgramWaypoint,
)
from src.geometry import (
    _normalize_step_limits,
    _interpolate_joints,
    _pose_rotation_distance_deg,
    _pose_translation_distance_mm,
    _translation_from_pose,
    _rotation_from_pose,
    _build_pose_from_rotation_translation,
    _extract_joint_tuple,
    _is_within_joint_limits,
    _passes_user_joint_constraints,
    _mean_abs_joint_delta,
)
from src.path_optimizer import (
    _joint_limit_penalty,
    _singularity_penalty,
    _joint_transition_penalty,
    _candidate_transition_penalty,
    _evaluate_move_l_transition,
    _passes_step_limit,
)
from src.ik_collection import (
    _IK_DEDUP_DECIMALS,
    _CONFIG_FLAG_COUNT,
    _append_seed_if_unique,
)

# 姿态桥接层的候选数量上限。
_BRIDGE_LAYER_CANDIDATE_LIMIT = 18

# "固定法兰位置"的姿态桥接段参数。
_POSITION_LOCK_BRIDGE_MIN_SEGMENTS = 6
_POSITION_LOCK_BRIDGE_ORIENTATION_STEP_DEG = 1.0


def _needs_pose_bridge(
    previous_candidate: _IKCandidate,
    current_candidate: _IKCandidate,
    motion_settings,
) -> bool:
    """判断相邻两点之间是否需要插入姿态重构桥接点。"""

    if not motion_settings.enable_pose_bridge:
        return False

    joint_deltas = [
        abs(current - previous)
        for previous, current in zip(previous_candidate.joints, current_candidate.joints)
    ]
    if joint_deltas and max(joint_deltas) > motion_settings.bridge_trigger_joint_delta_deg:
        return True

    return previous_candidate.config_flags != current_candidate.config_flags


def _build_position_locked_bridge_segment(
    *,
    segment_index: int,
    target_index_width: int,
    current_target_name: str,
    previous_pose,
    current_pose,
    previous_candidate: _IKCandidate,
    current_candidate: _IKCandidate,
    robot,
    tool_pose,
    reference_pose,
    motion_settings,
    optimizer_settings: _PathOptimizerSettings,
    lower_limits: Sequence[float],
    upper_limits: Sequence[float],
    a1_lower_deg: float,
    a1_upper_deg: float,
) -> tuple[list[_ProgramWaypoint], _ProgramWaypoint]:
    """构造关节空间插值桥接段（MoveJ 语义）。

    原来的实现试图保持法兰 TCP 位置不变，但这要求在腕奇异（A5≈0°）附近求解
    IK 并通过 MoveL 移动，两者在奇异点都容易失败。

    新实现直接在关节空间线性插值，用 MoveJ 逐步过渡。
    优点：
      - 不调用 IK，完全避免奇异点数值问题；
      - MoveJ 对关节空间的插值不经过笛卡尔奇异点；
      - 对于腕部配置切换（A4/A6 跳变 ≈160°），纯腕关节旋转对 TCP 位置影响极小
        （在 A5≈0° 奇异附近，A4/A6 轴与 TCP 位移轴几乎垂直）。
    """
    return _build_joint_interpolation_bridge_segment(
        segment_index=segment_index,
        target_index_width=target_index_width,
        current_target_name=current_target_name,
        current_pose=current_pose,
        previous_candidate=previous_candidate,
        current_candidate=current_candidate,
        robot=robot,
        tool_pose=tool_pose,
        reference_pose=reference_pose,
        motion_settings=motion_settings,
    )


def _build_joint_interpolation_bridge_segment(
    *,
    segment_index: int,
    target_index_width: int,
    current_target_name: str,
    current_pose,
    previous_candidate: _IKCandidate,
    current_candidate: _IKCandidate,
    robot,
    tool_pose,
    reference_pose,
    motion_settings,
) -> tuple[list[_ProgramWaypoint], _ProgramWaypoint]:
    """在关节空间线性插值，生成 MoveJ 桥接路点序列。

    不调用 IK，直接在 prev_joints 和 curr_joints 之间按 bridge_step_deg 细分。
    在腕奇异（A5≈0°）附近，MoveJ 完全避免了笛卡尔奇异问题；
    TCP 位置的微小偏差（纯腕旋转对 TCP 位移影响极小）在配置切换桥接段中可接受。
    """
    bridge_step_limits = _normalize_step_limits(
        motion_settings.bridge_step_deg,
        len(previous_candidate.joints),
    )
    segment_count = max(
        _POSITION_LOCK_BRIDGE_MIN_SEGMENTS,
        12 if previous_candidate.config_flags != current_candidate.config_flags else 0,
        math.ceil(
            max(
                abs(current - previous) / limit
                for previous, current, limit in zip(
                    previous_candidate.joints,
                    current_candidate.joints,
                    bridge_step_limits,
                )
            )
        ),
    )

    bridge_waypoints: list[_ProgramWaypoint] = []
    for step in range(1, segment_count):
        ratio = step / segment_count
        interp_joints = _interpolate_joints(
            previous_candidate.joints,
            current_candidate.joints,
            ratio,
        )
        interp_pose = robot.SolveFK(list(interp_joints), tool_pose, reference_pose)
        bridge_waypoints.append(
            _ProgramWaypoint(
                name=f"P_{segment_index:0{target_index_width}d}_BR_{step:02d}",
                pose=interp_pose,
                joints=interp_joints,
                move_type="MoveJ",
                is_bridge=True,
            )
        )

    current_waypoint = _ProgramWaypoint(
        name=current_target_name,
        pose=current_pose,
        joints=current_candidate.joints,
        move_type="MoveL",
        is_bridge=False,
    )
    return bridge_waypoints, current_waypoint


def _build_position_locked_bridge_segment_for_anchor(
    *,
    segment_index: int,
    target_index_width: int,
    current_target_name: str,
    previous_pose,
    current_pose,
    previous_candidate: _IKCandidate,
    current_candidate: _IKCandidate,
    robot,
    tool_pose,
    reference_pose,
    motion_settings,
    optimizer_settings: _PathOptimizerSettings,
    lower_limits: Sequence[float],
    upper_limits: Sequence[float],
    a1_lower_deg: float,
    a1_upper_deg: float,
    lock_to_current_position: bool,
) -> tuple[list[_ProgramWaypoint], _ProgramWaypoint]:
    """按指定锚点位置构造固定位置桥接段。"""

    bridge_step_limits = _normalize_step_limits(
        motion_settings.bridge_step_deg,
        len(previous_candidate.joints),
    )
    joint_segment_count = max(
        2,
        math.ceil(
            max(
                abs(current - previous) / limit
                for previous, current, limit in zip(
                    previous_candidate.joints,
                    current_candidate.joints,
                    bridge_step_limits,
                )
            )
        ),
    )
    orientation_delta_deg = _pose_rotation_distance_deg(previous_pose, current_pose)
    orientation_segment_count = max(
        1,
        math.ceil(orientation_delta_deg / _POSITION_LOCK_BRIDGE_ORIENTATION_STEP_DEG),
    )
    segment_count = max(
        joint_segment_count,
        orientation_segment_count,
        _POSITION_LOCK_BRIDGE_MIN_SEGMENTS,
    )
    if previous_candidate.config_flags != current_candidate.config_flags:
        segment_count = max(segment_count, 12)

    locked_translation = _translation_from_pose(
        current_pose if lock_to_current_position else previous_pose
    )
    ratio_indices = range(0, segment_count) if lock_to_current_position else range(1, segment_count + 1)
    rotation_search_deg = _build_position_locked_rotation_search_levels(
        motion_settings.bridge_rotation_search_deg,
        previous_candidate,
        current_candidate,
    )

    bridge_layers: list[tuple[_BridgeCandidate, ...]] = []
    for ratio_index in ratio_indices:
        interpolation_ratio = ratio_index / segment_count
        desired_joints = _interpolate_joints(
            previous_candidate.joints,
            current_candidate.joints,
            interpolation_ratio,
        )
        desired_pose = robot.SolveFK(list(desired_joints), tool_pose, reference_pose)
        base_pose = _build_pose_from_rotation_translation(
            desired_pose,
            _rotation_from_pose(desired_pose),
            locked_translation,
        )
        bridge_layers.append(
            _collect_position_locked_bridge_candidates(
                base_pose=base_pose,
                desired_joints=desired_joints,
                previous_candidate=previous_candidate,
                current_candidate=current_candidate,
                robot=robot,
                tool_pose=tool_pose,
                reference_pose=reference_pose,
                motion_settings=motion_settings,
                rotation_search_deg=rotation_search_deg,
                optimizer_settings=optimizer_settings,
                lower_limits=lower_limits,
                upper_limits=upper_limits,
                a1_lower_deg=a1_lower_deg,
                a1_upper_deg=a1_upper_deg,
            )
        )

    selected_bridge_states, final_reached_joints = _optimize_position_locked_bridge_layers(
        bridge_layers,
        final_pose=current_pose,
        final_preferred_joints=current_candidate.joints,
        start_pose=previous_pose,
        start_joints=previous_candidate.joints,
        robot=robot,
        optimizer_settings=optimizer_settings,
        bridge_step_limits=bridge_step_limits,
    )

    bridge_waypoints = [
        _ProgramWaypoint(
            name=f"P_{segment_index:0{target_index_width}d}_BR_{bridge_index:02d}",
            pose=bridge_state.pose,
            joints=bridge_state.joints,
            # MoveJ instead of MoveL: joint-space motion crosses the wrist singularity
            # safely without the Cartesian planner encountering a degenerate Jacobian.
            # TCP position is only approximately fixed per step, but wrist-only rotations
            # produce negligible TCP deviation for the small bridge_step_limits used here.
            move_type="MoveJ",
            is_bridge=True,
        )
        for bridge_index, bridge_state in enumerate(selected_bridge_states, start=1)
    ]
    current_waypoint = _ProgramWaypoint(
        name=current_target_name,
        pose=current_pose,
        joints=final_reached_joints,
        move_type="MoveL",
        is_bridge=False,
    )
    return bridge_waypoints, current_waypoint


def _collect_position_locked_bridge_candidates(
    *,
    base_pose,
    desired_joints: tuple[float, ...],
    previous_candidate: _IKCandidate,
    current_candidate: _IKCandidate,
    robot,
    tool_pose,
    reference_pose,
    motion_settings,
    rotation_search_deg: Sequence[float],
    optimizer_settings: _PathOptimizerSettings,
    lower_limits: Sequence[float],
    upper_limits: Sequence[float],
    a1_lower_deg: float,
    a1_upper_deg: float,
) -> tuple[_BridgeCandidate, ...]:
    """为单个"固定位置桥接层"收集姿态候选。"""

    candidates: list[_BridgeCandidate] = []
    seen_joint_keys: set[tuple[float, ...]] = set()
    seed_joints_options = _build_position_locked_seed_strategies(
        desired_joints=desired_joints,
        previous_candidate=previous_candidate,
        current_candidate=current_candidate,
        motion_settings=motion_settings,
        lower_limits=lower_limits,
        upper_limits=upper_limits,
    )

    for pose_candidate in _iter_position_locked_pose_candidates(
        base_pose,
        rotation_search_deg=rotation_search_deg,
        previous_candidate=previous_candidate,
        current_candidate=current_candidate,
        motion_settings=motion_settings,
    ):
        for joint_candidate in _iter_joint_solutions_for_pose_with_seeds(
            robot,
            pose_candidate,
            seed_joints_options,
            tool_pose,
            reference_pose,
        ):
            if not _is_within_joint_limits(joint_candidate, lower_limits, upper_limits):
                continue
            if not _passes_user_joint_constraints(
                joint_candidate,
                a1_lower_deg=a1_lower_deg,
                a1_upper_deg=a1_upper_deg,
                a2_max_deg=motion_settings.a2_max_deg,
                tolerance_deg=motion_settings.joint_constraint_tolerance_deg,
            ):
                continue

            dedup_key = tuple(round(value, _IK_DEDUP_DECIMALS) for value in joint_candidate)
            if dedup_key in seen_joint_keys:
                continue
            seen_joint_keys.add(dedup_key)
            candidates.append(
                _build_position_locked_bridge_candidate(
                    robot=robot,
                    candidate_pose=pose_candidate,
                    candidate_joints=joint_candidate,
                    base_pose=base_pose,
                    desired_joints=desired_joints,
                    optimizer_settings=optimizer_settings,
                    lower_limits=lower_limits,
                    upper_limits=upper_limits,
                )
            )

    if not candidates:
        raise RuntimeError("No fixed-position bridge candidates remain after applying constraints.")

    candidates.sort(
        key=lambda candidate: (
            candidate.node_cost,
            candidate.config_flags,
            candidate.joints,
        )
    )
    return tuple(candidates[:_BRIDGE_LAYER_CANDIDATE_LIMIT])


def _build_position_locked_seed_strategies(
    *,
    desired_joints: Sequence[float],
    previous_candidate: _IKCandidate,
    current_candidate: _IKCandidate,
    motion_settings,
    lower_limits: Sequence[float],
    upper_limits: Sequence[float],
) -> tuple[tuple[float, ...], ...]:
    """为固定位置桥接构造更偏向腕奇异重构的 seed 集。"""

    seeds: list[tuple[float, ...]] = []
    seen: set[tuple[float, ...]] = set()
    base_seeds = (
        tuple(float(value) for value in desired_joints),
        previous_candidate.joints,
        current_candidate.joints,
    )
    phase_offsets = tuple(
        sorted(
            {
                15.0,
                30.0,
                45.0,
                60.0,
                90.0,
                *(
                    abs(float(value))
                    for value in motion_settings.wrist_refinement_phase_offsets_deg
                ),
            }
        )
    )
    reference_a6_values = [
        base_seed[5]
        for base_seed in base_seeds
        if len(base_seed) >= 6
    ]
    reference_a6_values.append(0.0)
    preferred_a6 = sum(reference_a6_values) / max(1, len(reference_a6_values))

    from src.geometry import _clip_seed_to_limits
    for base_seed in base_seeds:
        clipped_seed = _clip_seed_to_limits(base_seed, lower_limits, upper_limits)
        _append_seed_if_unique(seeds, seen, clipped_seed)
        if len(base_seed) < 6:
            continue

        for target_a6 in (
            preferred_a6,
            previous_candidate.joints[5],
            current_candidate.joints[5],
            0.0,
        ):
            compensation_deg = base_seed[5] - target_a6
            variant = list(base_seed)
            variant[3] = variant[3] + compensation_deg
            variant[5] = target_a6
            _append_seed_if_unique(
                seeds,
                seen,
                _clip_seed_to_limits(variant, lower_limits, upper_limits),
            )

        for phase_offset_deg in phase_offsets:
            for direction in (-1.0, 1.0):
                phase_shift = direction * phase_offset_deg
                variant = list(base_seed)
                variant[3] = variant[3] + phase_shift
                variant[5] = variant[5] - phase_shift
                _append_seed_if_unique(
                    seeds,
                    seen,
                    _clip_seed_to_limits(variant, lower_limits, upper_limits),
                )

    return tuple(seeds)


def _build_position_locked_rotation_search_levels(
    base_levels_deg: Sequence[float],
    previous_candidate: _IKCandidate,
    current_candidate: _IKCandidate,
) -> tuple[float, ...]:
    """为固定位置桥接构造更激进的姿态搜索角度集。"""

    search_levels = {abs(float(value)) for value in base_levels_deg}
    joint_deltas = [
        abs(current - previous)
        for previous, current in zip(previous_candidate.joints, current_candidate.joints)
    ]
    max_joint_delta = max(joint_deltas, default=0.0)
    near_wrist_singularity = (
        len(previous_candidate.joints) >= 5
        and len(current_candidate.joints) >= 5
        and min(abs(previous_candidate.joints[4]), abs(current_candidate.joints[4])) <= 15.0
    )
    if previous_candidate.config_flags != current_candidate.config_flags or max_joint_delta >= 150.0:
        search_levels.update({1.0, 2.0, 5.0, 10.0, 15.0, 20.0, 40.0, 60.0, 90.0})
    elif max_joint_delta >= 90.0:
        search_levels.update({1.0, 2.0, 5.0, 10.0, 15.0, 20.0, 30.0, 45.0})
    else:
        search_levels.update({1.0, 2.0, 5.0, 10.0})
    if near_wrist_singularity:
        search_levels.update({3.0, 7.5, 12.0, 25.0, 35.0})

    return tuple(sorted(search_levels))


def _iter_position_locked_pose_candidates(
    base_pose,
    *,
    rotation_search_deg: Sequence[float],
    previous_candidate: _IKCandidate,
    current_candidate: _IKCandidate,
    motion_settings,
):
    """为固定位置桥接生成更丰富的纯姿态候选。"""

    from robodk.robomath import rotx, roty, rotz

    yielded: set[tuple[float, ...]] = set()

    def yield_if_new(pose_candidate) -> None:
        signature = tuple(
            round(float(pose_candidate[row, column]), 6)
            for row in range(4)
            for column in range(4)
        )
        if signature in yielded:
            return
        yielded.add(signature)
        nonlocal_generated.append(pose_candidate)

    nonlocal_generated: list[object] = []
    for pose_candidate in _iter_bridge_pose_candidates(
        base_pose,
        (0.0,),
        rotation_search_deg,
    ):
        yield_if_new(pose_candidate)

    compound_levels_deg = tuple(
        level_deg
        for level_deg in sorted({abs(float(value)) for value in rotation_search_deg if value != 0.0})
        if level_deg <= 30.0
    )
    if (
        len(previous_candidate.joints) >= 5
        and len(current_candidate.joints) >= 5
        and min(abs(previous_candidate.joints[4]), abs(current_candidate.joints[4]))
        <= motion_settings.wrist_refinement_a5_threshold_deg
    ):
        for angle_deg in compound_levels_deg:
            angle_rad = math.radians(angle_deg)
            for first_sign in (-1.0, 1.0):
                for second_sign in (-1.0, 1.0):
                    yield_if_new(base_pose * rotx(first_sign * angle_rad) * rotz(second_sign * angle_rad))
                    yield_if_new(base_pose * roty(first_sign * angle_rad) * rotz(second_sign * angle_rad))
                    yield_if_new(base_pose * rotx(first_sign * angle_rad) * roty(second_sign * angle_rad))

    for pose_candidate in nonlocal_generated:
        yield pose_candidate


def _build_position_locked_bridge_candidate(
    *,
    robot,
    candidate_pose,
    candidate_joints: tuple[float, ...],
    base_pose,
    desired_joints: Sequence[float],
    optimizer_settings: _PathOptimizerSettings,
    lower_limits: Sequence[float],
    upper_limits: Sequence[float],
) -> _BridgeCandidate:
    """构造固定位置桥接层的一个候选状态。"""

    config_values = robot.JointsConfig(list(candidate_joints)).list()
    config_flags = tuple(int(round(value)) for value in config_values[:_CONFIG_FLAG_COUNT])
    joint_limit_penalty_val = _joint_limit_penalty(
        candidate_joints,
        lower_limits,
        upper_limits,
        optimizer_settings,
    )
    singularity_penalty_val = _singularity_penalty(robot, candidate_joints, optimizer_settings)
    return _BridgeCandidate(
        pose=candidate_pose,
        joints=candidate_joints,
        config_flags=config_flags,
        node_cost=(
            80.0 * _pose_rotation_distance_deg(base_pose, candidate_pose)
            + 0.05 * joint_limit_penalty_val
            + 0.05 * singularity_penalty_val
            + 0.03 * _joint_transition_penalty(
                desired_joints,
                candidate_joints,
                optimizer_settings,
            )
        ),
    )


def _optimize_position_locked_bridge_layers(
    bridge_layers: Sequence[Sequence[_BridgeCandidate]],
    *,
    final_pose,
    final_preferred_joints: tuple[float, ...],
    start_pose,
    start_joints: tuple[float, ...],
    robot,
    optimizer_settings: _PathOptimizerSettings,
    bridge_step_limits: Sequence[float],
) -> tuple[list[_BridgeCandidate], tuple[float, ...]]:
    """对固定位置桥接层做 DP，转移可行性由关节步长限制决定（MoveJ 语义）。

    原来用 MoveL_Test 做可行性判断，但笛卡尔直线运动会在腕奇异（A5≈0°）附近
    失败（Jacobian 奇异 → RoboDK 返回错误）。
    改用关节步长检查：只要相邻桥接帧的关节变化量在 bridge_step_limits 之内，
    就视为可达，对应 RoboDK 中的 MoveJ 指令。
    """

    if not bridge_layers:
        return [], start_joints

    start_state = _build_runtime_bridge_state(robot, start_pose, start_joints)
    previous_states: list[_BridgeCandidate | None] = [start_state]
    previous_costs = [0.0]
    backpointers: list[list[int]] = []
    state_layers: list[list[_BridgeCandidate | None]] = []

    for layer in bridge_layers:
        current_costs = [math.inf] * len(layer)
        current_backpointers = [-1] * len(layer)
        current_states: list[_BridgeCandidate | None] = [None] * len(layer)

        for current_index, current_candidate in enumerate(layer):
            best_cost = math.inf
            best_previous_index = -1

            for previous_index, previous_state in enumerate(previous_states):
                if previous_state is None or not math.isfinite(previous_costs[previous_index]):
                    continue

                # Joint-step feasibility: each axis change must be within bridge_step_limits.
                # This replaces MoveL_Test which fails across the wrist singularity (A5=0°).
                if not _passes_step_limit(
                    previous_state.joints,
                    current_candidate.joints,
                    bridge_step_limits,
                ):
                    continue

                total_cost = (
                    previous_costs[previous_index]
                    + current_candidate.node_cost
                    + _position_locked_bridge_transition_cost(
                        previous_state,
                        current_candidate,
                        preferred_target_joints=current_candidate.joints,
                        optimizer_settings=optimizer_settings,
                    )
                )
                if total_cost < best_cost:
                    best_cost = total_cost
                    best_previous_index = previous_index

            current_costs[current_index] = best_cost
            current_backpointers[current_index] = best_previous_index
            # Use the candidate itself as the reached state (MoveJ goes to exact joints).
            current_states[current_index] = current_candidate if math.isfinite(best_cost) else None

        if not any(math.isfinite(cost) for cost in current_costs):
            raise RuntimeError("No feasible fixed-position posture bridge could be found.")

        previous_states = current_states
        previous_costs = current_costs
        backpointers.append(current_backpointers)
        state_layers.append(current_states)

    # For MoveJ, the final step (bridge end → next winding point) is always reachable
    # because MoveJ goes directly to the specified joint angles.
    # Pick the last bridge state that minimises the transition cost to final_preferred_joints.
    best_total_cost = math.inf
    best_last_index = -1
    final_state = _build_runtime_bridge_state(robot, final_pose, final_preferred_joints)
    for previous_index, previous_state in enumerate(previous_states):
        if previous_state is None or not math.isfinite(previous_costs[previous_index]):
            continue

        total_cost = (
            previous_costs[previous_index]
            + _position_locked_bridge_transition_cost(
                previous_state,
                final_state,
                preferred_target_joints=final_preferred_joints,
                optimizer_settings=optimizer_settings,
            )
        )
        if total_cost < best_total_cost:
            best_total_cost = total_cost
            best_last_index = previous_index

    if best_last_index < 0:
        raise RuntimeError("The fixed-position bridge cannot connect back to the next path point.")

    selected_states = [state_layers[-1][best_last_index]]
    for layer_index in range(len(state_layers) - 1, 0, -1):
        best_last_index = backpointers[layer_index][best_last_index]
        selected_states.append(state_layers[layer_index - 1][best_last_index])

    selected_states.reverse()
    return [state for state in selected_states if state is not None], final_preferred_joints


def _build_runtime_bridge_state(robot, pose, joints: tuple[float, ...]) -> _BridgeCandidate:
    """把 MoveL_Test 的实际到达关节封装成运行态桥接状态。"""

    config_values = robot.JointsConfig(list(joints)).list()
    config_flags = tuple(int(round(value)) for value in config_values[:_CONFIG_FLAG_COUNT])
    return _BridgeCandidate(
        pose=pose,
        joints=joints,
        config_flags=config_flags,
        node_cost=0.0,
    )


def _position_locked_bridge_transition_cost(
    previous_state: _BridgeCandidate,
    current_state: _BridgeCandidate,
    *,
    preferred_target_joints: Sequence[float],
    optimizer_settings: _PathOptimizerSettings,
) -> float:
    """固定位置桥接段的转移代价。"""

    return (
        0.15
        * _candidate_transition_penalty(
            previous_state,
            current_state,
            optimizer_settings,
        )
        + 12.0 * _pose_rotation_distance_deg(previous_state.pose, current_state.pose)
        + 0.08
        * _joint_transition_penalty(
            preferred_target_joints,
            current_state.joints,
            optimizer_settings,
        )
    )


def _build_bridge_waypoints(
    *,
    segment_index: int,
    target_index_width: int,
    previous_pose,
    current_pose,
    previous_candidate: _IKCandidate,
    current_candidate: _IKCandidate,
    robot,
    tool_pose,
    reference_pose,
    motion_settings,
    optimizer_settings: _PathOptimizerSettings,
    lower_limits: Sequence[float],
    upper_limits: Sequence[float],
    a1_lower_deg: float,
    a1_upper_deg: float,
) -> list[_ProgramWaypoint]:
    """在一个坏段中插入桥接点，尽量把法兰位姿改动压小。"""

    bridge_step_limits = _normalize_step_limits(
        motion_settings.bridge_step_deg,
        len(previous_candidate.joints),
    )
    segment_count = max(
        2,
        math.ceil(
            max(
                abs(current - previous) / limit
                for previous, current, limit in zip(
                    previous_candidate.joints,
                    current_candidate.joints,
                    bridge_step_limits,
                )
            )
        ),
    )

    bridge_layers: list[tuple[_BridgeCandidate, ...]] = []
    for bridge_index in range(1, segment_count):
        interpolation_ratio = bridge_index / segment_count
        desired_joints = _interpolate_joints(
            previous_candidate.joints,
            current_candidate.joints,
            interpolation_ratio,
        )
        bridge_layers.append(
            _collect_bridge_candidates_for_layer(
                previous_pose=previous_pose,
                current_pose=current_pose,
                previous_candidate=previous_candidate,
                current_candidate=current_candidate,
                desired_joints=desired_joints,
                interpolation_ratio=interpolation_ratio,
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
        )

    selected_bridge_path = _optimize_bridge_candidate_layers(
        bridge_layers,
        previous_pose=previous_pose,
        current_pose=current_pose,
        previous_candidate=previous_candidate,
        current_candidate=current_candidate,
        optimizer_settings=optimizer_settings,
        bridge_step_limits=bridge_step_limits,
    )

    return [
        _ProgramWaypoint(
            name=f"P_{segment_index:0{target_index_width}d}_BR_{bridge_index:02d}",
            pose=bridge_candidate.pose,
            joints=bridge_candidate.joints,
            move_type="MoveJ",
            is_bridge=True,
        )
        for bridge_index, bridge_candidate in enumerate(selected_bridge_path, start=1)
    ]


def _solve_bridge_waypoint(
    *,
    anchor_pose,
    desired_joints: tuple[float, ...],
    previous_bridge_joints: tuple[float, ...],
    robot,
    tool_pose,
    reference_pose,
    motion_settings,
    optimizer_settings: _PathOptimizerSettings,
    lower_limits: Sequence[float],
    upper_limits: Sequence[float],
    a1_lower_deg: float,
    a1_upper_deg: float,
    bridge_step_limits: Sequence[float],
) -> tuple[object, tuple[float, ...]]:
    """为单个桥接采样点求一组更稳的过渡位姿/关节。"""

    fallback_pose = robot.SolveFK(list(desired_joints), tool_pose, reference_pose)
    best_pose = fallback_pose
    best_joints = desired_joints
    best_cost = _bridge_candidate_cost(
        anchor_pose=anchor_pose,
        candidate_pose=fallback_pose,
        desired_joints=desired_joints,
        candidate_joints=desired_joints,
        previous_bridge_joints=previous_bridge_joints,
        optimizer_settings=optimizer_settings,
    )

    seen_joint_keys: set[tuple[float, ...]] = set()
    for pose_candidate in _iter_bridge_pose_candidates(
        anchor_pose,
        motion_settings.bridge_translation_search_mm,
        motion_settings.bridge_rotation_search_deg,
    ):
        for joint_candidate in _iter_joint_solutions_for_pose(
            robot,
            pose_candidate,
            desired_joints,
            tool_pose,
            reference_pose,
        ):
            if not _is_within_joint_limits(joint_candidate, lower_limits, upper_limits):
                continue
            if not _passes_user_joint_constraints(
                joint_candidate,
                a1_lower_deg=a1_lower_deg,
                a1_upper_deg=a1_upper_deg,
                a2_max_deg=motion_settings.a2_max_deg,
                tolerance_deg=motion_settings.joint_constraint_tolerance_deg,
            ):
                continue
            if not _passes_step_limit(previous_bridge_joints, joint_candidate, bridge_step_limits):
                continue

            dedup_key = tuple(round(value, _IK_DEDUP_DECIMALS) for value in joint_candidate)
            if dedup_key in seen_joint_keys:
                continue
            seen_joint_keys.add(dedup_key)

            actual_pose = robot.SolveFK(list(joint_candidate), tool_pose, reference_pose)
            candidate_cost = _bridge_candidate_cost(
                anchor_pose=anchor_pose,
                candidate_pose=actual_pose,
                desired_joints=desired_joints,
                candidate_joints=joint_candidate,
                previous_bridge_joints=previous_bridge_joints,
                optimizer_settings=optimizer_settings,
            )
            if candidate_cost < best_cost:
                best_cost = candidate_cost
                best_pose = actual_pose
                best_joints = joint_candidate

    return best_pose, best_joints


def _iter_bridge_pose_candidates(
    anchor_pose,
    translation_search_mm: Sequence[float],
    rotation_search_deg: Sequence[float],
):
    """围绕锚点位姿生成一组小扰动候选。"""

    from robodk.robomath import rotx, roty, rotz, transl

    yield anchor_pose

    translation_levels = sorted({abs(float(value)) for value in translation_search_mm if value != 0.0})
    rotation_levels_deg = sorted({abs(float(value)) for value in rotation_search_deg if value != 0.0})

    for distance_mm in translation_levels:
        for sign in (-1.0, 1.0):
            yield anchor_pose * transl(sign * distance_mm, 0.0, 0.0)
            yield anchor_pose * transl(0.0, sign * distance_mm, 0.0)
            yield anchor_pose * transl(0.0, 0.0, sign * distance_mm)

    for angle_deg in rotation_levels_deg:
        angle_rad = math.radians(angle_deg)
        for sign in (-1.0, 1.0):
            yield anchor_pose * rotx(sign * angle_rad)
            yield anchor_pose * roty(sign * angle_rad)
            yield anchor_pose * rotz(sign * angle_rad)


def _iter_joint_solutions_for_pose(
    robot,
    pose_candidate,
    desired_joints: Sequence[float],
    tool_pose,
    reference_pose,
):
    """为候选位姿枚举一组关节解。"""

    yielded: set[tuple[float, ...]] = set()

    seed_solution = _extract_joint_tuple(
        robot.SolveIK(pose_candidate, list(desired_joints), tool_pose, reference_pose),
        len(desired_joints),
    )
    if seed_solution:
        dedup_key = tuple(round(value, _IK_DEDUP_DECIMALS) for value in seed_solution)
        yielded.add(dedup_key)
        yield seed_solution

    for raw_solution in robot.SolveIK_All(pose_candidate, tool_pose, reference_pose):
        joint_solution = _extract_joint_tuple(raw_solution, len(desired_joints))
        if not joint_solution:
            continue
        dedup_key = tuple(round(value, _IK_DEDUP_DECIMALS) for value in joint_solution)
        if dedup_key in yielded:
            continue
        yielded.add(dedup_key)
        yield joint_solution


def _bridge_candidate_cost(
    *,
    anchor_pose,
    candidate_pose,
    desired_joints: Sequence[float],
    candidate_joints: Sequence[float],
    previous_bridge_joints: Sequence[float],
    optimizer_settings: _PathOptimizerSettings,
) -> float:
    """桥接候选评分：先保位姿，再保关节平滑。"""

    translation_error_mm = _pose_translation_distance_mm(anchor_pose, candidate_pose)
    rotation_error_deg = _pose_rotation_distance_deg(anchor_pose, candidate_pose)
    desired_joint_cost = _mean_abs_joint_delta(desired_joints, candidate_joints)
    step_joint_cost = _mean_abs_joint_delta(previous_bridge_joints, candidate_joints)

    return (
        30.0 * translation_error_mm
        + 12.0 * rotation_error_deg
        + 0.25 * desired_joint_cost
        + 0.10 * step_joint_cost
        + 0.005 * _joint_transition_penalty(
            desired_joints,
            candidate_joints,
            optimizer_settings,
        )
    )


def _collect_bridge_candidates_for_layer(
    *,
    previous_pose,
    current_pose,
    previous_candidate: _IKCandidate,
    current_candidate: _IKCandidate,
    desired_joints: tuple[float, ...],
    interpolation_ratio: float,
    robot,
    tool_pose,
    reference_pose,
    motion_settings,
    optimizer_settings: _PathOptimizerSettings,
    lower_limits: Sequence[float],
    upper_limits: Sequence[float],
    a1_lower_deg: float,
    a1_upper_deg: float,
) -> tuple[_BridgeCandidate, ...]:
    """为单个桥接层先收集候选，后续再统一交给桥接 DP。"""

    desired_pose = robot.SolveFK(list(desired_joints), tool_pose, reference_pose)
    fallback_candidate = _build_bridge_candidate_state(
        robot=robot,
        candidate_pose=desired_pose,
        candidate_joints=desired_joints,
        desired_joints=desired_joints,
        previous_pose=previous_pose,
        current_pose=current_pose,
        interpolation_ratio=interpolation_ratio,
        optimizer_settings=optimizer_settings,
        lower_limits=lower_limits,
        upper_limits=upper_limits,
    )

    other_candidates: list[_BridgeCandidate] = []
    seen_joint_keys = {
        tuple(round(value, _IK_DEDUP_DECIMALS) for value in fallback_candidate.joints)
    }
    seed_joints_options = (
        desired_joints,
        previous_candidate.joints,
        current_candidate.joints,
    )

    for anchor_pose in _iter_bridge_anchor_poses(
        previous_pose,
        current_pose,
        desired_pose,
    ):
        for pose_candidate in _iter_bridge_pose_candidates(
            anchor_pose,
            motion_settings.bridge_translation_search_mm,
            motion_settings.bridge_rotation_search_deg,
        ):
            for joint_candidate in _iter_joint_solutions_for_pose_with_seeds(
                robot,
                pose_candidate,
                seed_joints_options,
                tool_pose,
                reference_pose,
            ):
                if not _is_within_joint_limits(joint_candidate, lower_limits, upper_limits):
                    continue
                if not _passes_user_joint_constraints(
                    joint_candidate,
                    a1_lower_deg=a1_lower_deg,
                    a1_upper_deg=a1_upper_deg,
                    a2_max_deg=motion_settings.a2_max_deg,
                    tolerance_deg=motion_settings.joint_constraint_tolerance_deg,
                ):
                    continue

                dedup_key = tuple(round(value, _IK_DEDUP_DECIMALS) for value in joint_candidate)
                if dedup_key in seen_joint_keys:
                    continue
                seen_joint_keys.add(dedup_key)

                actual_pose = robot.SolveFK(list(joint_candidate), tool_pose, reference_pose)
                other_candidates.append(
                    _build_bridge_candidate_state(
                        robot=robot,
                        candidate_pose=actual_pose,
                        candidate_joints=joint_candidate,
                        desired_joints=desired_joints,
                        previous_pose=previous_pose,
                        current_pose=current_pose,
                        interpolation_ratio=interpolation_ratio,
                        optimizer_settings=optimizer_settings,
                        lower_limits=lower_limits,
                        upper_limits=upper_limits,
                    )
                )

    other_candidates.sort(
        key=lambda candidate: (
            candidate.node_cost,
            candidate.config_flags,
            candidate.joints,
        )
    )
    limited_candidates = [fallback_candidate]
    limited_candidates.extend(
        other_candidates[: max(0, _BRIDGE_LAYER_CANDIDATE_LIMIT - 1)]
    )
    return tuple(limited_candidates)


def _build_bridge_candidate_state(
    *,
    robot,
    candidate_pose,
    candidate_joints: tuple[float, ...],
    desired_joints: Sequence[float],
    previous_pose,
    current_pose,
    interpolation_ratio: float,
    optimizer_settings: _PathOptimizerSettings,
    lower_limits: Sequence[float],
    upper_limits: Sequence[float],
) -> _BridgeCandidate:
    """把桥接候选封装成统一结构，便于桥接 DP 使用。"""

    config_values = robot.JointsConfig(list(candidate_joints)).list()
    config_flags = tuple(int(round(value)) for value in config_values[:_CONFIG_FLAG_COUNT])
    joint_limit_penalty_val = _joint_limit_penalty(
        candidate_joints,
        lower_limits,
        upper_limits,
        optimizer_settings,
    )
    singularity_penalty_val = _singularity_penalty(robot, candidate_joints, optimizer_settings)
    return _BridgeCandidate(
        pose=candidate_pose,
        joints=candidate_joints,
        config_flags=config_flags,
        node_cost=_bridge_layer_candidate_cost(
            previous_pose=previous_pose,
            current_pose=current_pose,
            candidate_pose=candidate_pose,
            desired_joints=desired_joints,
            candidate_joints=candidate_joints,
            interpolation_ratio=interpolation_ratio,
            optimizer_settings=optimizer_settings,
            joint_limit_penalty=joint_limit_penalty_val,
            singularity_penalty=singularity_penalty_val,
        ),
    )


def _iter_bridge_anchor_poses(previous_pose, current_pose, desired_pose):
    """同时围绕前点、后点和关节插值 FK 位姿做桥接搜索。"""

    yielded: set[tuple[float, ...]] = set()
    for pose in (previous_pose, current_pose, desired_pose):
        signature = tuple(round(float(pose[row, column]), 6) for row in range(4) for column in range(4))
        if signature in yielded:
            continue
        yielded.add(signature)
        yield pose


def _iter_joint_solutions_for_pose_with_seeds(
    robot,
    pose_candidate,
    seed_joints_options: Sequence[Sequence[float]],
    tool_pose,
    reference_pose,
):
    """针对桥接层的同一个位姿候选，用多组 seed 尽量诱导多个 IK 分支。"""

    yielded: set[tuple[float, ...]] = set()
    joint_count = len(seed_joints_options[0]) if seed_joints_options else 0

    for seed_joints in seed_joints_options:
        joint_solution = _extract_joint_tuple(
            robot.SolveIK(pose_candidate, list(seed_joints), tool_pose, reference_pose),
            joint_count,
        )
        if not joint_solution:
            continue
        dedup_key = tuple(round(value, _IK_DEDUP_DECIMALS) for value in joint_solution)
        if dedup_key in yielded:
            continue
        yielded.add(dedup_key)
        yield joint_solution

    for raw_solution in robot.SolveIK_All(pose_candidate, tool_pose, reference_pose):
        joint_solution = _extract_joint_tuple(raw_solution, joint_count)
        if not joint_solution:
            continue
        dedup_key = tuple(round(value, _IK_DEDUP_DECIMALS) for value in joint_solution)
        if dedup_key in yielded:
            continue
        yielded.add(dedup_key)
        yield joint_solution


def _bridge_layer_candidate_cost(
    *,
    previous_pose,
    current_pose,
    candidate_pose,
    desired_joints: Sequence[float],
    candidate_joints: Sequence[float],
    interpolation_ratio: float,
    optimizer_settings: _PathOptimizerSettings,
    joint_limit_penalty: float,
    singularity_penalty: float,
) -> float:
    """桥接层节点代价：优先尽量保持法兰位姿不变，再兼顾关节平滑。"""

    previous_translation_error_mm = _pose_translation_distance_mm(previous_pose, candidate_pose)
    previous_rotation_error_deg = _pose_rotation_distance_deg(previous_pose, candidate_pose)
    current_translation_error_mm = _pose_translation_distance_mm(current_pose, candidate_pose)
    current_rotation_error_deg = _pose_rotation_distance_deg(current_pose, candidate_pose)
    desired_joint_cost = _mean_abs_joint_delta(desired_joints, candidate_joints)

    return (
        40.0 * min(previous_translation_error_mm, current_translation_error_mm)
        + 15.0 * min(previous_rotation_error_deg, current_rotation_error_deg)
        + 12.0
        * (
            (1.0 - interpolation_ratio) * previous_translation_error_mm
            + interpolation_ratio * current_translation_error_mm
        )
        + 4.0
        * (
            (1.0 - interpolation_ratio) * previous_rotation_error_deg
            + interpolation_ratio * current_rotation_error_deg
        )
        + 0.20 * desired_joint_cost
        + 0.006 * _joint_transition_penalty(
            desired_joints,
            candidate_joints,
            optimizer_settings,
        )
        + 0.04 * joint_limit_penalty
        + 0.04 * singularity_penalty
    )


def _optimize_bridge_candidate_layers(
    bridge_layers: Sequence[Sequence[_BridgeCandidate]],
    *,
    previous_pose,
    current_pose,
    previous_candidate: _IKCandidate,
    current_candidate: _IKCandidate,
    optimizer_settings: _PathOptimizerSettings,
    bridge_step_limits: Sequence[float],
) -> list[_BridgeCandidate]:
    """对整段桥接层做一次局部 DP，而不是逐点贪心。"""

    if not bridge_layers:
        return []

    start_state = _BridgeCandidate(
        pose=previous_pose,
        joints=previous_candidate.joints,
        config_flags=previous_candidate.config_flags,
        node_cost=0.0,
    )
    end_state = _BridgeCandidate(
        pose=current_pose,
        joints=current_candidate.joints,
        config_flags=current_candidate.config_flags,
        node_cost=0.0,
    )

    previous_layer: Sequence[_BridgeCandidate] = (start_state,)
    previous_costs = [0.0]
    backpointers: list[list[int]] = []

    for layer in bridge_layers:
        current_costs = [math.inf] * len(layer)
        current_backpointers = [-1] * len(layer)

        for current_index, current_state in enumerate(layer):
            best_cost = math.inf
            best_previous_index = -1

            for previous_index, previous_state in enumerate(previous_layer):
                if not _passes_step_limit(
                    previous_state.joints,
                    current_state.joints,
                    bridge_step_limits,
                ):
                    continue

                total_cost = (
                    previous_costs[previous_index]
                    + _bridge_layer_transition_cost(
                        previous_state,
                        current_state,
                        optimizer_settings,
                    )
                    + current_state.node_cost
                )
                if total_cost < best_cost:
                    best_cost = total_cost
                    best_previous_index = previous_index

            current_costs[current_index] = best_cost
            current_backpointers[current_index] = best_previous_index

        if not any(math.isfinite(cost) for cost in current_costs):
            raise RuntimeError("No feasible bridge candidate sequence could be found.")

        previous_layer = layer
        previous_costs = current_costs
        backpointers.append(current_backpointers)

    best_total_cost = math.inf
    best_last_index = -1
    for previous_index, previous_state in enumerate(previous_layer):
        if not _passes_step_limit(
            previous_state.joints,
            end_state.joints,
            bridge_step_limits,
        ):
            continue

        total_cost = previous_costs[previous_index] + _bridge_layer_transition_cost(
            previous_state,
            end_state,
            optimizer_settings,
        )
        if total_cost < best_total_cost:
            best_total_cost = total_cost
            best_last_index = previous_index

    if best_last_index < 0:
        raise RuntimeError("Bridge candidates could not connect back to the final path point.")

    selected_path = [bridge_layers[-1][best_last_index]]
    for layer_index in range(len(bridge_layers) - 1, 0, -1):
        best_last_index = backpointers[layer_index][best_last_index]
        selected_path.append(bridge_layers[layer_index - 1][best_last_index])

    selected_path.reverse()
    return selected_path


def _bridge_layer_transition_cost(
    previous_state: _BridgeCandidate,
    current_state: _BridgeCandidate,
    optimizer_settings: _PathOptimizerSettings,
) -> float:
    """桥接层转移代价：同时压关节变化和法兰位姿变化。"""

    translation_delta_mm = _pose_translation_distance_mm(previous_state.pose, current_state.pose)
    rotation_delta_deg = _pose_rotation_distance_deg(previous_state.pose, current_state.pose)
    return (
        0.18
        * _candidate_transition_penalty(
            previous_state,
            current_state,
            optimizer_settings,
        )
        + 28.0 * translation_delta_mm
        + 10.0 * rotation_delta_deg
    )
