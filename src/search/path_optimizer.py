from __future__ import annotations

import math
from typing import Sequence

from src.core.types import (
    _IKCandidate,
    _IKLayer,
    _PathOptimizerSettings,
)
from src.core.geometry import (
    _normalize_step_limits,
    _pose_translation_distance_mm,
    _pose_rotation_distance_deg,
    _trim_joint_vector,
    _mean_abs_joint_delta,
)


# 按 joint tuple 缓存 RoboDK 的派生量，避免在全局搜索与局部重分配时反复查询同一批结果。
_ROBOT_CONFIG_FLAGS_CACHE: dict[tuple[int, tuple[float, ...]], tuple[int, ...]] = {}
_ROBOT_SINGULARITY_PENALTY_CACHE: dict[
    tuple[int, tuple[float, ...], "_PathOptimizerSettings"],
    float,
] = {}


def _build_optimizer_settings(
    joint_count: int,
    motion_settings,
) -> _PathOptimizerSettings:
    """根据机器人轴数构造 DP 代价权重。"""

    base_weights = (1.0, 1.0, 1.1, 1.4, 2.0, 2.7)
    preferred_step_defaults = (5.0, 5.0, 5.0, 25.0, 25.0, 25.0)
    if joint_count <= len(base_weights):
        weights = base_weights[:joint_count]
    else:
        weights = base_weights + (base_weights[-1],) * (joint_count - len(base_weights))

    hard_step_limits = _normalize_step_limits(motion_settings.max_joint_step_deg, joint_count)
    if joint_count <= len(preferred_step_defaults):
        preferred_step_limits = preferred_step_defaults[:joint_count]
    else:
        preferred_step_limits = preferred_step_defaults + (
            preferred_step_defaults[-1],
        ) * (joint_count - len(preferred_step_defaults))

    # "优选连续性阈值"不能宽于真正的硬连续性阈值，否则走廊评分会鼓励一条实际上不可走的边。
    preferred_step_limits = tuple(
        min(hard_limit, preferred_limit)
        for hard_limit, preferred_limit in zip(hard_step_limits, preferred_step_limits)
    )

    return _PathOptimizerSettings(
        joint_delta_weights=weights,
        enable_joint_continuity_constraint=motion_settings.enable_joint_continuity_constraint,
        max_joint_step_deg=hard_step_limits,
        preferred_joint_step_deg=preferred_step_limits,
        wrist_phase_lock_threshold_deg=motion_settings.wrist_phase_lock_threshold_deg,
        rear_switch_penalty=float(getattr(motion_settings, "rear_switch_penalty", 2000.0)),
        lower_switch_penalty=float(getattr(motion_settings, "lower_switch_penalty", 2000.0)),
        flip_switch_penalty=float(getattr(motion_settings, "flip_switch_penalty", 2000.0)),
    )


def _summarize_selected_path(
    selected_path: Sequence[_IKCandidate],
    *,
    bridge_trigger_joint_delta_deg: float,
    config_switch_min_joint_delta_deg: float = 5.0,
) -> tuple[int, int, float, float]:
    """统计一条已选 joint path 的关键连续性指标。

    Config switches where max_joint_delta < config_switch_min_joint_delta_deg
    are treated as benign (e.g. a wrist-flip bit that crosses zero while J5 is
    near the wrist singularity).  Only meaningful config transitions — those
    accompanied by a joint movement of at least config_switch_min_joint_delta_deg
    — are counted against the path quality score.
    """

    if len(selected_path) <= 1:
        return 0, 0, 0.0, 0.0

    config_switches = 0
    bridge_like_segments = 0
    worst_joint_step_deg = 0.0
    mean_joint_step_sum = 0.0

    for previous_candidate, current_candidate in zip(selected_path, selected_path[1:]):
        joint_deltas = [
            abs(current - previous)
            for previous, current in zip(previous_candidate.joints, current_candidate.joints)
        ]
        max_joint_delta = max(joint_deltas, default=0.0)
        worst_joint_step_deg = max(worst_joint_step_deg, max_joint_delta)
        mean_joint_step_sum += _mean_abs_joint_delta(
            previous_candidate.joints,
            current_candidate.joints,
        )

        config_changed = previous_candidate.config_flags != current_candidate.config_flags
        # Only count config switches that move joints by a meaningful amount.
        # Near-zero changes (e.g. wrist-flip at J5≈0°) are benign and must not
        # be treated as bridge-like segments or invalidate the path.
        meaningful_switch = config_changed and max_joint_delta >= config_switch_min_joint_delta_deg
        if meaningful_switch:
            config_switches += 1
        if meaningful_switch or max_joint_delta > bridge_trigger_joint_delta_deg:
            bridge_like_segments += 1

    mean_joint_step_deg = mean_joint_step_sum / max(1, len(selected_path) - 1)
    return config_switches, bridge_like_segments, worst_joint_step_deg, mean_joint_step_deg


def _path_is_clean_enough_for_program_generation(
    selected_path: Sequence[_IKCandidate],
    *,
    bridge_trigger_joint_delta_deg: float,
    preferred_joint_step_deg: Sequence[float],
) -> bool:
    """判断 exact path 是否已经足够稳定，可以直接进入程序落地阶段。"""

    if not selected_path:
        return False

    from src.search.local_repair import _collect_problem_segments
    problem_segments = _collect_problem_segments(
        selected_path,
        bridge_trigger_joint_delta_deg=bridge_trigger_joint_delta_deg,
    )
    if problem_segments:
        return False

    _config_switches, _bridge_like_segments, worst_joint_step_deg, _mean_joint_step_deg = (
        _summarize_selected_path(
            selected_path,
            bridge_trigger_joint_delta_deg=bridge_trigger_joint_delta_deg,
        )
    )
    preferred_limit = max(preferred_joint_step_deg, default=0.0)
    return worst_joint_step_deg <= preferred_limit + 1e-9


def _optimize_joint_path(
    ik_layers: Sequence[_IKLayer],
    *,
    robot,
    move_type: str,
    start_joints: tuple[float, ...],
    optimizer_settings: _PathOptimizerSettings,
) -> tuple[list[_IKCandidate], float]:
    """使用动态规划在整条路径上选出一条全局代价最小的关节序列。"""

    if not ik_layers:
        return [], 0.0

    corridor_scores = _compute_candidate_corridor_scores(ik_layers, optimizer_settings)
    guided_config_path = _build_guided_config_path(
        ik_layers,
        start_joints=start_joints,
        optimizer_settings=optimizer_settings,
    )
    if (
        guided_config_path is not None
        and not _guided_config_path_is_feasible(
            ik_layers,
            guided_config_path=guided_config_path,
            optimizer_settings=optimizer_settings,
        )
    ):
        guided_config_path = None

    # 第 0 层的代价 = 从当前机器人关节走到该候选的转移代价 + 该候选自己的节点代价。
    previous_costs = []
    for candidate_index, candidate in enumerate(ik_layers[0].candidates):
        if guided_config_path is not None and candidate.config_flags != guided_config_path[0]:
            previous_costs.append(math.inf)
            continue
        previous_costs.append(
            _candidate_node_cost(
                candidate,
                corridor_score=corridor_scores[0][candidate_index],
                optimizer_settings=optimizer_settings,
            )
            + optimizer_settings.start_transition_weight
            * _joint_transition_penalty(start_joints, candidate.joints, optimizer_settings)
        )
    backpointers: list[list[int]] = []

    previous_layer = ik_layers[0]
    for layer_index in range(1, len(ik_layers)):
        current_layer = ik_layers[layer_index]

        # MoveL 的可行性跟"前一个候选关节"强相关，因此这里先缓存：
        # 从上一层每个候选出发，走线性移动到当前笛卡尔位姿是否可达。
        move_l_cache: list[tuple[float, tuple[float, ...] | None]] | None = None
        if move_type == "MoveL":
            move_l_cache = [
                _evaluate_move_l_transition(
                    robot,
                    start_joints=candidate.joints,
                    target_pose=current_layer.pose,
                    joint_count=len(candidate.joints),
                    optimizer_settings=optimizer_settings,
                )
                for candidate in previous_layer.candidates
            ]

        current_costs = [math.inf] * len(current_layer.candidates)
        current_backpointers = [-1] * len(current_layer.candidates)

        for current_index, current_candidate in enumerate(current_layer.candidates):
            if (
                guided_config_path is not None
                and current_candidate.config_flags != guided_config_path[layer_index]
            ):
                continue
            node_cost = _candidate_node_cost(
                current_candidate,
                corridor_score=corridor_scores[layer_index][current_index],
                optimizer_settings=optimizer_settings,
            )
            best_cost = math.inf
            best_previous_index = -1

            for previous_index, previous_candidate in enumerate(previous_layer.candidates):
                if not math.isfinite(previous_costs[previous_index]):
                    continue
                if (
                    guided_config_path is not None
                    and previous_candidate.config_flags != guided_config_path[layer_index - 1]
                ):
                    continue
                if not _passes_joint_continuity_constraint(
                    previous_candidate.joints,
                    current_candidate.joints,
                    optimizer_settings,
                ):
                    continue

                transition_cost = _candidate_transition_penalty(
                    previous_candidate,
                    current_candidate,
                    optimizer_settings,
                )

                if move_l_cache is not None:
                    linear_penalty, reached_joints = move_l_cache[previous_index]
                    if not math.isfinite(linear_penalty):
                        continue
                    transition_cost += linear_penalty

                    # 即便 RoboDK 线性插补能到达目标点，线性到达时的实际末端关节也可能
                    # 跟当前候选关节有偏差，因此这里再给一个分支不一致惩罚。
                    if reached_joints is not None:
                        transition_cost += optimizer_settings.move_l_branch_mismatch_weight * (
                            _joint_transition_penalty(
                                reached_joints,
                                current_candidate.joints,
                                optimizer_settings,
                            )
                        )

                total_cost = previous_costs[previous_index] + transition_cost + node_cost
                if total_cost < best_cost:
                    best_cost = total_cost
                    best_previous_index = previous_index

            current_costs[current_index] = best_cost
            current_backpointers[current_index] = best_previous_index

        if not any(math.isfinite(cost) for cost in current_costs):
            message = (
                f"No globally feasible {move_type} sequence could be found for target index "
                f"{layer_index}."
            )
            if guided_config_path is not None:
                message += " The guided config-family path is too restrictive."
            if optimizer_settings.enable_joint_continuity_constraint:
                message += (
                    " The joint continuity constraint may be too strict for the current path."
                )
            raise RuntimeError(message)

        backpointers.append(current_backpointers)
        previous_layer = current_layer
        previous_costs = current_costs

    end_index = min(range(len(previous_costs)), key=previous_costs.__getitem__)
    total_cost = previous_costs[end_index]
    selected_path = [ik_layers[-1].candidates[end_index]]

    # 通过回溯指针得到整条最优路径。
    for layer_index in range(len(ik_layers) - 2, -1, -1):
        end_index = backpointers[layer_index][end_index]
        selected_path.append(ik_layers[layer_index].candidates[end_index])

    selected_path.reverse()
    return selected_path, total_cost


def _build_guided_config_path(
    ik_layers: Sequence[_IKLayer],
    *,
    start_joints: Sequence[float],
    optimizer_settings: _PathOptimizerSettings,
) -> tuple[tuple[int, ...], ...] | None:
    """先在 `config_flags` 层面求一条"位形族路径"。

    候选层 DP 的问题是：同一个 config family 内部通常还有很多具体关节候选，
    如果直接在细粒度层面优化，局部节点代价可能会把整条路径带离本来可连续的族。

    所以这里先做一层更粗的规划：
    - 只决定每一层优先属于哪个 family；
    - 以"少切族"为最高优先级；
    - 如果必须切，就把切换放在跨族连接代价最小的层。
    """

    if not ik_layers:
        return ()

    candidate_groups_by_layer = [
        _group_candidates_by_config(layer.candidates)
        for layer in ik_layers
    ]

    previous_costs: dict[tuple[int, ...], float] = {}
    for config_flags, candidates in candidate_groups_by_layer[0].items():
        previous_costs[config_flags] = min(
            optimizer_settings.start_transition_weight
            * _joint_transition_penalty(start_joints, candidate.joints, optimizer_settings)
            for candidate in candidates
        )

    backpointers: list[dict[tuple[int, ...], tuple[int, ...]]] = []
    for layer_index in range(1, len(candidate_groups_by_layer)):
        previous_groups = candidate_groups_by_layer[layer_index - 1]
        current_groups = candidate_groups_by_layer[layer_index]
        current_costs: dict[tuple[int, ...], float] = {}
        current_backpointers: dict[tuple[int, ...], tuple[int, ...]] = {}

        for current_flags, current_candidates in current_groups.items():
            best_cost = math.inf
            best_previous_flags: tuple[int, ...] | None = None
            for previous_flags, previous_candidates in previous_groups.items():
                if previous_flags not in previous_costs:
                    continue

                transition_cost = _best_config_group_transition_cost(
                    previous_candidates,
                    current_candidates,
                    optimizer_settings,
                )
                if not math.isfinite(transition_cost):
                    continue

                total_cost = previous_costs[previous_flags] + transition_cost
                if previous_flags != current_flags:
                    total_cost += optimizer_settings.family_switch_penalty
                else:
                    total_cost -= optimizer_settings.same_config_stay_bonus

                if total_cost < best_cost:
                    best_cost = total_cost
                    best_previous_flags = previous_flags

            if best_previous_flags is not None:
                current_costs[current_flags] = best_cost
                current_backpointers[current_flags] = best_previous_flags

        if not current_costs:
            return None

        previous_costs = current_costs
        backpointers.append(current_backpointers)

    end_flags = min(previous_costs, key=previous_costs.__getitem__)
    guided_flags = [end_flags]
    for layer_index in range(len(backpointers) - 1, -1, -1):
        end_flags = backpointers[layer_index][end_flags]
        guided_flags.append(end_flags)
    guided_flags.reverse()
    return tuple(guided_flags)


def _group_candidates_by_config(
    candidates: Sequence[_IKCandidate],
) -> dict[tuple[int, ...], tuple[_IKCandidate, ...]]:
    """按 `config_flags` 把同一层候选分组。"""

    grouped: dict[tuple[int, ...], list[_IKCandidate]] = {}
    for candidate in candidates:
        grouped.setdefault(candidate.config_flags, []).append(candidate)
    return {config_flags: tuple(group) for config_flags, group in grouped.items()}


def _best_config_group_transition_cost(
    previous_candidates: Sequence[_IKCandidate],
    current_candidates: Sequence[_IKCandidate],
    optimizer_settings: _PathOptimizerSettings,
) -> float:
    """估计两个 config-family 组之间的最优连接代价。"""

    best_cost = math.inf
    for previous_candidate in previous_candidates:
        for current_candidate in current_candidates:
            if not _passes_joint_continuity_constraint(
                previous_candidate.joints,
                current_candidate.joints,
                optimizer_settings,
            ):
                continue
            best_cost = min(
                best_cost,
                _joint_transition_penalty(
                    previous_candidate.joints,
                    current_candidate.joints,
                    optimizer_settings,
                ),
            )
    return best_cost


def _guided_config_path_is_feasible(
    ik_layers: Sequence[_IKLayer],
    *,
    guided_config_path: Sequence[tuple[int, ...]],
    optimizer_settings: _PathOptimizerSettings,
) -> bool:
    """检查 family guidance 是否真能落到一条具体候选链上。"""

    if len(guided_config_path) != len(ik_layers):
        return False

    reachable_indices = [
        index
        for index, candidate in enumerate(ik_layers[0].candidates)
        if candidate.config_flags == guided_config_path[0]
    ]
    if not reachable_indices:
        return False

    for layer_index in range(1, len(ik_layers)):
        current_reachable: list[int] = []
        for current_index, current_candidate in enumerate(ik_layers[layer_index].candidates):
            if current_candidate.config_flags != guided_config_path[layer_index]:
                continue
            for previous_index in reachable_indices:
                previous_candidate = ik_layers[layer_index - 1].candidates[previous_index]
                if _passes_joint_continuity_constraint(
                    previous_candidate.joints,
                    current_candidate.joints,
                    optimizer_settings,
                ):
                    current_reachable.append(current_index)
                    break
        if not current_reachable:
            return False
        reachable_indices = current_reachable

    return True


def _evaluate_move_l_transition(
    robot,
    *,
    start_joints: tuple[float, ...],
    target_pose,
    joint_count: int,
    optimizer_settings: _PathOptimizerSettings,
) -> tuple[float, tuple[float, ...] | None]:
    """评估从某个起始关节出发执行 MoveL 是否可行。"""

    status = robot.MoveL_Test(list(start_joints), target_pose)
    if status != 0:
        return optimizer_settings.move_l_unreachable_penalty, None

    reached_joints = _trim_joint_vector(robot.Joints().list(), joint_count)
    return 0.0, reached_joints


def _candidate_transition_penalty(
    previous_candidate: _IKCandidate,
    current_candidate: _IKCandidate,
    optimizer_settings: _PathOptimizerSettings,
) -> float:
    """计算两个候选之间的转移代价。"""

    cost = _joint_transition_penalty(
        previous_candidate.joints,
        current_candidate.joints,
        optimizer_settings,
    )

    previous_flags = previous_candidate.config_flags
    current_flags = current_candidate.config_flags
    if len(previous_flags) >= 1 and len(current_flags) >= 1 and previous_flags[0] != current_flags[0]:
        cost += optimizer_settings.rear_switch_penalty
    if len(previous_flags) >= 2 and len(current_flags) >= 2 and previous_flags[1] != current_flags[1]:
        cost += optimizer_settings.lower_switch_penalty
    if len(previous_flags) >= 3 and len(current_flags) >= 3 and previous_flags[2] != current_flags[2]:
        cost += optimizer_settings.flip_switch_penalty

    # J5 正负号改变通常对应 wrist flip，额外惩罚。
    if len(previous_candidate.joints) >= 5:
        if previous_candidate.joints[4] * current_candidate.joints[4] < 0.0:
            cost += optimizer_settings.wrist_flip_sign_penalty

    # 大幅转 J6 往往是"能到，但没必要"的多圈旋转，单独抑制。
    if len(previous_candidate.joints) >= 6:
        joint6_delta = abs(current_candidate.joints[5] - previous_candidate.joints[5])
        if joint6_delta > optimizer_settings.joint6_spin_threshold_deg:
            cost += (
                joint6_delta - optimizer_settings.joint6_spin_threshold_deg
            ) * optimizer_settings.joint6_spin_penalty_per_deg

        # 参考 FINA11.src：在 A5 接近 0 的腕奇异区，优先锁住 A6 的相位连续性。
        # 这样即便必须穿过奇异附近，也尽量让 A4 去连续变化，而不是让 A6 突然翻转。
        min_abs_a5_deg = math.inf
        if len(previous_candidate.joints) >= 5:
            min_abs_a5_deg = min(
                abs(previous_candidate.joints[4]),
                abs(current_candidate.joints[4]),
            )
        if min_abs_a5_deg < optimizer_settings.wrist_phase_lock_threshold_deg:
            normalized = (
                optimizer_settings.wrist_phase_lock_threshold_deg - min_abs_a5_deg
            ) / optimizer_settings.wrist_phase_lock_threshold_deg
            cost += (
                optimizer_settings.wrist_phase_lock_penalty_per_deg
                * normalized
                * joint6_delta
            )

    # 如果相邻两点本来就能用同一配置标志并且小步平滑连接，则给一个奖励，
    # 让 DP 更愿意待在这条"自然连续"的 exact-pose 走廊里。
    if previous_flags == current_flags:
        cost -= optimizer_settings.same_config_stay_bonus
        if _passes_preferred_continuity(
            previous_candidate.joints,
            current_candidate.joints,
            optimizer_settings,
        ):
            cost -= optimizer_settings.preferred_transition_bonus

    return cost


def _passes_joint_continuity_constraint(
    previous_joints: Sequence[float],
    current_joints: Sequence[float],
    optimizer_settings: _PathOptimizerSettings,
) -> bool:
    """检查相邻两点之间是否满足连续性硬约束。"""

    if not optimizer_settings.enable_joint_continuity_constraint:
        return True

    for delta_deg, limit_deg in zip(
        (abs(current - previous) for previous, current in zip(previous_joints, current_joints)),
        optimizer_settings.max_joint_step_deg,
    ):
        if delta_deg > limit_deg:
            return False

    return True


def _passes_preferred_continuity(
    previous_joints: Sequence[float],
    current_joints: Sequence[float],
    optimizer_settings: _PathOptimizerSettings,
) -> bool:
    """检查相邻两点是否满足"优选连续性"。

    这不是硬约束，失败也不代表不可用。
    它只用于识别那些能长距离保持同一位形族的 exact-pose 候选走廊。
    """

    for delta_deg, limit_deg in zip(
        (abs(current - previous) for previous, current in zip(previous_joints, current_joints)),
        optimizer_settings.preferred_joint_step_deg,
    ):
        if delta_deg > limit_deg:
            return False
    return True


def _candidate_node_cost(
    candidate: _IKCandidate,
    *,
    corridor_score: float,
    optimizer_settings: _PathOptimizerSettings,
) -> float:
    """计算单个候选节点的总代价。

    节点代价由两部分组成：
    1. 近限位 / 近奇异惩罚，但做缩放，避免局部代价压倒整条路径连续性；
    2. 连续位形走廊奖励，鼓励 DP 留在能跨越更多层的平滑分支里。
    """

    raw_penalty = candidate.joint_limit_penalty + candidate.singularity_penalty
    corridor_bonus = min(optimizer_settings.corridor_bonus_cap, corridor_score) * (
        optimizer_settings.corridor_bonus_per_step
    )
    return optimizer_settings.node_penalty_scale * raw_penalty - corridor_bonus


def _compute_candidate_corridor_scores(
    ik_layers: Sequence[_IKLayer],
    optimizer_settings: _PathOptimizerSettings,
) -> list[list[float]]:
    """估计每个候选点处在"连续位形走廊"中的强度。

    做法：
    1. 只考虑"同一 config_flags 且满足优选连续性"的边；
    2. 分别向前、向后做最长链长度 DP；
    3. 把前后长度合并成节点走廊分数。

    这个分数越高，说明该候选越像一条长距离稳定分支上的中间点。
    """

    if not ik_layers:
        return []

    forward_lengths: list[list[int]] = [
        [1] * len(layer.candidates) for layer in ik_layers
    ]
    backward_lengths: list[list[int]] = [
        [1] * len(layer.candidates) for layer in ik_layers
    ]

    for layer_index in range(len(ik_layers) - 2, -1, -1):
        current_layer = ik_layers[layer_index]
        next_layer = ik_layers[layer_index + 1]
        for current_index, current_candidate in enumerate(current_layer.candidates):
            best_reach = 0
            for next_index, next_candidate in enumerate(next_layer.candidates):
                if current_candidate.config_flags != next_candidate.config_flags:
                    continue
                if not _passes_preferred_continuity(
                    current_candidate.joints,
                    next_candidate.joints,
                    optimizer_settings,
                ):
                    continue
                best_reach = max(best_reach, forward_lengths[layer_index + 1][next_index])
            forward_lengths[layer_index][current_index] = 1 + best_reach

    for layer_index in range(1, len(ik_layers)):
        previous_layer = ik_layers[layer_index - 1]
        current_layer = ik_layers[layer_index]
        for current_index, current_candidate in enumerate(current_layer.candidates):
            best_reach = 0
            for previous_index, previous_candidate in enumerate(previous_layer.candidates):
                if previous_candidate.config_flags != current_candidate.config_flags:
                    continue
                if not _passes_preferred_continuity(
                    previous_candidate.joints,
                    current_candidate.joints,
                    optimizer_settings,
                ):
                    continue
                best_reach = max(best_reach, backward_lengths[layer_index - 1][previous_index])
            backward_lengths[layer_index][current_index] = 1 + best_reach

    corridor_scores: list[list[float]] = []
    for layer_index, layer in enumerate(ik_layers):
        layer_scores = []
        for candidate_index, _candidate in enumerate(layer.candidates):
            forward_length = forward_lengths[layer_index][candidate_index]
            backward_length = backward_lengths[layer_index][candidate_index]
            layer_scores.append(float(forward_length + backward_length - 2))
        corridor_scores.append(layer_scores)

    return corridor_scores


def _joint_transition_penalty(
    previous_joints: Sequence[float],
    current_joints: Sequence[float],
    optimizer_settings: _PathOptimizerSettings,
) -> float:
    """计算关节变化代价。

    这里同时考虑：
    1. 绝对变化量之和；
    2. 大变化的平方惩罚；
    3. 单步最大跳变额外惩罚。
    """

    deltas = [current - previous for previous, current in zip(previous_joints, current_joints)]
    abs_cost = sum(
        weight * abs(delta)
        for weight, delta in zip(optimizer_settings.joint_delta_weights, deltas)
    )
    squared_cost = sum(
        weight * delta * delta
        for weight, delta in zip(optimizer_settings.joint_delta_weights, deltas)
    )
    cost = (
        optimizer_settings.abs_delta_weight * abs_cost
        + optimizer_settings.squared_delta_weight * squared_cost
    )

    if deltas:
        max_delta = max(abs(delta) for delta in deltas)
        if max_delta > optimizer_settings.large_jump_threshold_deg:
            excess = max_delta - optimizer_settings.large_jump_threshold_deg
            cost += optimizer_settings.large_jump_penalty_weight * excess * excess

    return cost


def _joint_limit_penalty(
    joints: Sequence[float],
    lower_limits: Sequence[float],
    upper_limits: Sequence[float],
    optimizer_settings: _PathOptimizerSettings,
) -> float:
    """计算关节接近机器人自身限位时的惩罚。"""

    penalty = 0.0
    for joint, lower, upper in zip(joints, lower_limits, upper_limits):
        span = upper - lower
        if span <= 0.0:
            continue

        margin = min(joint - lower, upper - joint)
        margin_ratio = margin / span
        if margin_ratio < optimizer_settings.joint_limit_margin_ratio:
            normalized = (
                optimizer_settings.joint_limit_margin_ratio - margin_ratio
            ) / optimizer_settings.joint_limit_margin_ratio
            penalty += optimizer_settings.joint_limit_penalty_weight * normalized * normalized
    return penalty


def _singularity_penalty(
    robot,
    joints: Sequence[float],
    optimizer_settings: _PathOptimizerSettings,
) -> float:
    """估算奇异附近惩罚。

    这里不依赖完整雅可比矩阵，而是用两个简单但稳定的近似指标：
    1. J5 接近 0 度时，腕部奇异风险升高；
    2. 肩-肘-腕三点几乎共线时，手臂奇异风险升高。
    """

    joints_key = tuple(float(value) for value in joints)
    cache_key = (id(robot), joints_key, optimizer_settings)
    cached_penalty = _ROBOT_SINGULARITY_PENALTY_CACHE.get(cache_key)
    if cached_penalty is not None:
        return cached_penalty

    penalty = 0.0

    if len(joints) >= 5:
        wrist_measure = abs(math.sin(math.radians(joints[4])))
        threshold = math.sin(math.radians(optimizer_settings.wrist_singularity_threshold_deg))
        if wrist_measure < threshold:
            normalized = (threshold - wrist_measure) / threshold
            penalty += (
                optimizer_settings.wrist_singularity_penalty_weight * normalized * normalized
            )

    from src.core.geometry import _translation_from_pose, _normalized_cross_measure, _subtract_vectors
    joint_poses = robot.JointPoses(list(joints))
    if len(joint_poses) >= 4:
        shoulder = _translation_from_pose(joint_poses[1])
        elbow = _translation_from_pose(joint_poses[2])
        wrist = _translation_from_pose(joint_poses[3])
        arm_measure = _normalized_cross_measure(
            _subtract_vectors(elbow, shoulder),
            _subtract_vectors(wrist, elbow),
        )
        if arm_measure < optimizer_settings.arm_singularity_threshold:
            normalized = (
                optimizer_settings.arm_singularity_threshold - arm_measure
            ) / optimizer_settings.arm_singularity_threshold
            penalty += optimizer_settings.arm_singularity_penalty_weight * normalized * normalized

    _ROBOT_SINGULARITY_PENALTY_CACHE[cache_key] = penalty
    return penalty


def _passes_step_limit(
    previous_joints: Sequence[float],
    current_joints: Sequence[float],
    step_limits: Sequence[float],
) -> bool:
    """检查一步关节变化是否落在给定阈值内。"""

    return all(
        abs(current - previous) <= limit
        for previous, current, limit in zip(previous_joints, current_joints, step_limits)
    )


def _selected_path_quality_key(
    selected_path: Sequence[_IKCandidate],
    *,
    total_cost: float,
    bridge_trigger_joint_delta_deg: float,
) -> tuple[float, ...]:
    """把最终路径压成一个可比较的质量排序键。"""

    (
        config_switches,
        bridge_like_segments,
        worst_joint_step_deg,
        mean_joint_step_deg,
    ) = _summarize_selected_path(
        selected_path,
        bridge_trigger_joint_delta_deg=bridge_trigger_joint_delta_deg,
    )
    return (
        float(bridge_like_segments),
        float(config_switches),
        float(worst_joint_step_deg),
        float(mean_joint_step_deg),
        float(total_cost),
    )
