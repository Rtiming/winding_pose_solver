from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class _PathOptimizerSettings:
    """动态规划路径优化时使用的代价权重。"""

    # 各关节的基础权重。后面的大轴通常更容易导致大幅姿态变化，因此权重略高。
    joint_delta_weights: tuple[float, ...]

    # 关节变化的 L1 / L2 混合代价。
    abs_delta_weight: float = 1.0
    squared_delta_weight: float = 0.015

    # 配置切换惩罚。
    rear_switch_penalty: float = 40.0
    lower_switch_penalty: float = 55.0
    flip_switch_penalty: float = 140.0

    # 额外抑制腕部翻转、J6 大幅旋转和大跳变。
    wrist_flip_sign_penalty: float = 100.0
    joint6_spin_threshold_deg: float = 120.0
    joint6_spin_penalty_per_deg: float = 1.5
    large_jump_threshold_deg: float = 35.0
    large_jump_penalty_weight: float = 0.3

    # 逼近关节限位时的惩罚。
    joint_limit_margin_ratio: float = 0.10
    joint_limit_penalty_weight: float = 180.0

    # 奇异附近惩罚。
    wrist_singularity_threshold_deg: float = 12.0
    wrist_singularity_penalty_weight: float = 110.0
    arm_singularity_threshold: float = 0.12
    arm_singularity_penalty_weight: float = 90.0

    # MoveL 额外代价。
    move_l_branch_mismatch_weight: float = 0.6
    move_l_unreachable_penalty: float = math.inf

    # 点与点之间的连续性硬约束。
    enable_joint_continuity_constraint: bool = True
    max_joint_step_deg: tuple[float, ...] = (5.0, 5.0, 5.0, 60.0, 45.0, 90.0)

    # 第一个点与机器人当前姿态的关系只作为轻量参考，避免当前站点姿态把整条路径"带歪"。
    start_transition_weight: float = 0.20

    # 候选点自身的"近限位 / 近奇异"代价仍保留，但只占一部分权重，
    # 以免局部节点惩罚过大，反而逼着 DP 提前离开整条更平顺的位形走廊。
    node_penalty_scale: float = 0.22

    # "优选连续性"不是硬约束，而是用来识别整条路径中那些能长距离维持同一位形族的候选走廊。
    preferred_joint_step_deg: tuple[float, ...] = (5.0, 5.0, 5.0, 25.0, 25.0, 25.0)
    preferred_transition_bonus: float = 12.0
    same_config_stay_bonus: float = 8.0
    corridor_bonus_per_step: float = 3.0
    corridor_bonus_cap: float = 20.0

    # 先在 config_flags 层面做一遍"位形族路径规划"时使用的切换惩罚。
    # 这个值刻意比普通转移代价大很多，目的是：
    # 1. 先问"能不能整段不切族"；
    # 2. 如果不能，再问"在哪个层切族最划算"。
    family_switch_penalty: float = 4000.0

    # 参考 FINA11.src：当 A5 接近 0 时，优先保持 A6 相位连续，
    # 尽量不要在腕奇异附近把姿态突变甩给 A6。
    wrist_phase_lock_threshold_deg: float = 12.0
    wrist_phase_lock_penalty_per_deg: float = 6.0


@dataclass(frozen=True)
class _IKCandidate:
    """某一个路径点的一组候选 IK 解。"""

    joints: tuple[float, ...]
    config_flags: tuple[int, ...]
    joint_limit_penalty: float
    singularity_penalty: float


@dataclass(frozen=True)
class _IKLayer:
    """路径中的一个离散点。

    pose:
        该点的笛卡尔位姿。
    candidates:
        该点所有可行的关节候选解。
    """

    pose: object
    candidates: tuple[_IKCandidate, ...]


@dataclass(frozen=True)
class _ProgramWaypoint:
    """最终写入 RoboDK 程序的离散路点。

    这里既包括原始路径点，也包括自动插入的桥接点。
    """

    name: str
    pose: object
    joints: tuple[float, ...]
    move_type: str
    is_bridge: bool


@dataclass(frozen=True)
class _BridgeCandidate:
    """桥接层中的一个候选状态。"""

    pose: object
    joints: tuple[float, ...]
    config_flags: tuple[int, ...]
    node_cost: float


@dataclass(frozen=True)
class _PathSearchResult:
    """一次完整 Frame-A 原点 Y/Z 轮廓搜索的结果。"""

    reference_pose_rows: tuple[dict[str, float], ...]
    pose_rows: tuple[dict[str, float], ...]
    ik_layers: tuple[_IKLayer, ...]
    selected_path: tuple[_IKCandidate, ...]
    total_cost: float
    frame_a_origin_yz_profile_mm: tuple[tuple[float, float], ...]
    row_labels: tuple[str, ...]
    inserted_flags: tuple[bool, ...]
    invalid_row_count: int
    ik_empty_row_count: int
    config_switches: int
    bridge_like_segments: int
    worst_joint_step_deg: float
    mean_joint_step_deg: float
    offset_step_jitter_mm: float
    offset_jerk_mm: float
    max_abs_offset_mm: float
    total_abs_offset_mm: float
