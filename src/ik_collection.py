from __future__ import annotations

from typing import Sequence

from src.types import (
    _IKCandidate,
    _IKLayer,
    _PathOptimizerSettings,
)
from src.geometry import (
    _build_pose,
    _trim_joint_vector,
    _extract_joint_tuple,
    _clip_seed_to_limits,
    _is_within_joint_limits,
    _passes_user_joint_constraints,
)
from src.path_optimizer import (
    _ROBOT_CONFIG_FLAGS_CACHE,
    _joint_limit_penalty,
    _singularity_penalty,
)

# RoboDK 的 JointsConfig() 前 3 个标志位分别表示：
# 1. rear/front
# 2. lower/upper
# 3. flip/non-flip
_CONFIG_FLAG_COUNT = 3

# 对 IK 解去重时使用的角度保留小数位数。
_IK_DEDUP_DECIMALS = 6


def _build_ik_layers(
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
    seed_joints_override: Sequence[tuple[float, ...]] | None = None,
    lower_limits_override: Sequence[float] | None = None,
    upper_limits_override: Sequence[float] | None = None,
    log_summary: bool = True,
) -> list[_IKLayer]:
    """为整条路径逐点收集候选 IK 解。"""

    if lower_limits_override is None or upper_limits_override is None:
        lower_limits_raw, upper_limits_raw, _ = robot.JointLimits()
        lower_limits = _trim_joint_vector(lower_limits_raw.list(), joint_count)
        upper_limits = _trim_joint_vector(upper_limits_raw.list(), joint_count)
    else:
        lower_limits = tuple(float(value) for value in lower_limits_override[:joint_count])
        upper_limits = tuple(float(value) for value in upper_limits_override[:joint_count])

    # 通过不同 seed 诱导 RoboDK 返回不同分支附近的解。
    if seed_joints_override is None:
        seed_joints = _build_seed_joint_strategies(
            robot=robot,
            lower_limits=lower_limits,
            upper_limits=upper_limits,
            joint_count=joint_count,
        )
    else:
        seed_joints = tuple(seed_joints_override)

    ik_layers: list[_IKLayer] = []
    total_candidates = 0
    for row_index, row in enumerate(pose_rows):
        pose = _build_pose(row, mat_type)
        candidates = _collect_ik_candidates(
            robot,
            pose,
            tool_pose=tool_pose,
            reference_pose=reference_pose,
            lower_limits=lower_limits,
            upper_limits=upper_limits,
            seed_joints=seed_joints,
            joint_count=joint_count,
            optimizer_settings=optimizer_settings,
            a1_lower_deg=a1_lower_deg,
            a1_upper_deg=a1_upper_deg,
            a2_max_deg=a2_max_deg,
            joint_constraint_tolerance_deg=joint_constraint_tolerance_deg,
        )
        if not candidates:
            raise RuntimeError(
                f"No IK candidates remain for CSV pose row {row_index} after applying hard "
                f"constraints: A1 in [{a1_lower_deg:.1f}, {a1_upper_deg:.1f}] deg, "
                f"A2 < {a2_max_deg:.1f} deg."
            )

        ik_layers.append(_IKLayer(pose=pose, candidates=tuple(candidates)))
        total_candidates += len(candidates)

    if log_summary:
        print(
            f"Collected {total_candidates} IK candidate(s) across {len(ik_layers)} target pose(s)."
        )
    return ik_layers


def _collect_ik_candidates(
    robot,
    pose,
    *,
    tool_pose,
    reference_pose,
    lower_limits: tuple[float, ...],
    upper_limits: tuple[float, ...],
    seed_joints: Sequence[tuple[float, ...]],
    joint_count: int,
    optimizer_settings: _PathOptimizerSettings,
    a1_lower_deg: float,
    a1_upper_deg: float,
    a2_max_deg: float,
    joint_constraint_tolerance_deg: float,
) -> list[_IKCandidate]:
    """收集一个路径点的所有候选 IK 解。

    这里同时使用两类入口：
    1. SolveIK_All：让 RoboDK 直接枚举当前能给出的所有支路；
    2. SolveIK + 多 seed：进一步诱导 RoboDK 靠近不同局部支路求解。
    """

    candidates: list[_IKCandidate] = []
    seen: set[tuple[float, ...]] = set()

    all_solutions = robot.SolveIK_All(pose, tool_pose, reference_pose)
    for raw_solution in all_solutions:
        _append_candidate_if_unique(
            candidates,
            seen,
            robot=robot,
            raw_joints=raw_solution,
            lower_limits=lower_limits,
            upper_limits=upper_limits,
            joint_count=joint_count,
            optimizer_settings=optimizer_settings,
            a1_lower_deg=a1_lower_deg,
            a1_upper_deg=a1_upper_deg,
            a2_max_deg=a2_max_deg,
            joint_constraint_tolerance_deg=joint_constraint_tolerance_deg,
        )

    for seed in seed_joints:
        raw_solution = robot.SolveIK(pose, list(seed), tool_pose, reference_pose)
        _append_candidate_if_unique(
            candidates,
            seen,
            robot=robot,
            raw_joints=raw_solution,
            lower_limits=lower_limits,
            upper_limits=upper_limits,
            joint_count=joint_count,
            optimizer_settings=optimizer_settings,
            a1_lower_deg=a1_lower_deg,
            a1_upper_deg=a1_upper_deg,
            a2_max_deg=a2_max_deg,
            joint_constraint_tolerance_deg=joint_constraint_tolerance_deg,
        )

    # 先按配置标志、再按节点代价排序，方便调试时观察候选。
    candidates.sort(
        key=lambda candidate: (
            candidate.config_flags,
            candidate.joint_limit_penalty + candidate.singularity_penalty,
            candidate.joints,
        )
    )
    return candidates


def _append_candidate_if_unique(
    candidates: list[_IKCandidate],
    seen: set[tuple[float, ...]],
    *,
    robot,
    raw_joints,
    lower_limits: tuple[float, ...],
    upper_limits: tuple[float, ...],
    joint_count: int,
    optimizer_settings: _PathOptimizerSettings,
    a1_lower_deg: float,
    a1_upper_deg: float,
    a2_max_deg: float,
    joint_constraint_tolerance_deg: float,
) -> None:
    """把一个 IK 解加入候选集合。

    会按以下顺序过滤：
    1. 空解 / 非法解；
    2. 超出机器人自身关节限位；
    3. 不满足用户指定的 A1 / A2 硬约束；
    4. 与已有候选重复。
    """

    joints = _extract_joint_tuple(raw_joints, joint_count)
    if not joints:
        return

    if not _is_within_joint_limits(joints, lower_limits, upper_limits):
        return

    if not _passes_user_joint_constraints(
        joints,
        a1_lower_deg=a1_lower_deg,
        a1_upper_deg=a1_upper_deg,
        a2_max_deg=a2_max_deg,
        tolerance_deg=joint_constraint_tolerance_deg,
    ):
        return

    dedup_key = tuple(round(value, _IK_DEDUP_DECIMALS) for value in joints)
    if dedup_key in seen:
        return
    seen.add(dedup_key)

    config_flags = _candidate_config_flags(robot, joints)
    candidates.append(
        _IKCandidate(
            joints=joints,
            config_flags=config_flags,
            joint_limit_penalty=_joint_limit_penalty(
                joints,
                lower_limits,
                upper_limits,
                optimizer_settings,
            ),
            singularity_penalty=_singularity_penalty(robot, joints, optimizer_settings),
        )
    )


def _candidate_config_flags(robot, joints: tuple[float, ...]) -> tuple[int, ...]:
    """缓存 RoboDK `JointsConfig()` 结果，减少重复查询。"""

    cache_key = (id(robot), joints)
    cached_flags = _ROBOT_CONFIG_FLAGS_CACHE.get(cache_key)
    if cached_flags is not None:
        return cached_flags

    config_values = robot.JointsConfig(list(joints)).list()
    config_flags = tuple(int(round(value)) for value in config_values[:_CONFIG_FLAG_COUNT])
    _ROBOT_CONFIG_FLAGS_CACHE[cache_key] = config_flags
    return config_flags


def _build_seed_joint_strategies(
    *,
    robot,
    lower_limits: tuple[float, ...],
    upper_limits: tuple[float, ...],
    joint_count: int,
) -> tuple[tuple[float, ...], ...]:
    """构造一组 seed，用来诱导 SolveIK 返回不同支路附近的解。"""

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

    # 在关节空间中再撒几组均匀点，增加命中不同支路的概率。
    for ratio in (0.15, 0.50, 0.85):
        seeds.append(
            tuple(
                lower + (upper - lower) * ratio
                for lower, upper in zip(lower_limits, upper_limits)
            )
        )

    # 对 6 轴机器人，额外人为扰动 J5 / J6，尽量诱导出 wrist flip 等不同分支。
    if joint_count >= 6:
        for joint5 in (-90.0, 90.0):
            for joint6 in (-180.0, 0.0, 180.0):
                wrist_seed = list(midpoint)
                wrist_seed[4] = joint5
                wrist_seed[5] = joint6
                seeds.append(_clip_seed_to_limits(wrist_seed, lower_limits, upper_limits))

    # 去重，避免重复种子造成无意义的 IK 调用。
    unique_seeds: list[tuple[float, ...]] = []
    seen: set[tuple[float, ...]] = set()
    for seed in seeds:
        rounded = tuple(round(value, _IK_DEDUP_DECIMALS) for value in seed)
        if rounded in seen:
            continue
        seen.add(rounded)
        unique_seeds.append(seed)
    return tuple(unique_seeds)


def _append_seed_if_unique(
    seeds: list[tuple[float, ...]],
    seen: set[tuple[float, ...]],
    seed: Sequence[float],
) -> None:
    """把 seed 去重后加入列表。"""

    normalized_seed = tuple(float(value) for value in seed)
    dedup_key = tuple(round(value, _IK_DEDUP_DECIMALS) for value in normalized_seed)
    if dedup_key in seen:
        return
    seen.add(dedup_key)
    seeds.append(normalized_seed)
