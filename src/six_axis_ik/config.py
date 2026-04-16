"""
项目系统配置文件。

约定：
1. `main.py` 只放“这一轮要跑什么”的用户输入，
   比如 FK 关节角、IK 目标位姿、模式开关。
2. 这个文件只放“机器人本体和系统级配置”，
   比如 RoboDK 连接、Tool、Frame、硬限位、本地 POE 参数。
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np

from .kinematics import JOINT_COUNT, RobotModel, make_pose_zyx

README_HINT = (
    "提示：main.py 只改运行输入；机器人参数和详细说明见 README.md 与 six_axis_ik/config.py。"
    " See README.md for details."
)

# ---------------------------------------------------------------------------
# 默认后端
# ---------------------------------------------------------------------------
# 现在默认以后端本地模型为主，RoboDK 只负责校验。
DEFAULT_BACKEND = "local"

# 目标位姿默认解释为“参考坐标系下的 TCP 位姿”。
DEFAULT_TARGET_SPACE = "frame"

# ---------------------------------------------------------------------------
# RoboDK 连接设置
# ---------------------------------------------------------------------------
ROBODK_HOST = "localhost"
ROBODK_PORT: Optional[int] = None
ROBODK_ROBOT_NAME = "KUKA"

# ---------------------------------------------------------------------------
# 机器人公共配置
# ---------------------------------------------------------------------------
# 下面这些值已经和 2026-03-29 当时的 RoboDK 实机模型对齐。
CONFIGURED_TOOL_POSE_XYZ_RZYX_MM_DEG = (0.0, 0.0, 319.5, 0.0, 0.0, 0.0)
CONFIGURED_FRAME_POSE_XYZ_RZYX_MM_DEG = (451.49, -78.17, -20.0, 35.0, 0.0, 0.0)
CONFIGURED_JOINT_LIMITS_DEG = [
    (-185.0, 185.0),
    (-95.0, 155.0),
    (-210.0, 88.0),
    (-200.0, 200.0),
    (-200.0, 200.0),
    (-358.0, 358.0),
]

# ---------------------------------------------------------------------------
# 应用层 IK 过滤范围
# ---------------------------------------------------------------------------
# 这组范围是“解筛选范围”，不是机器人硬限位。
# 本地 IK / RoboDK IK 都会先算，再按这里过滤出你想保留的解。
IK_FILTER_JOINT_RANGES_DEG = [
    (-185.0, 185.0),
    (-95.0, 155.0),
    (-210.0, 88.0),
    (-200.0, 200.0),
    (-200.0, 200.0),
    (-358.0, 358.0),
]

# 是否保留 RoboDK `SolveIK_All` 返回的额外列。
KEEP_EXTRA_ROBODK_COLUMNS = True

# ---------------------------------------------------------------------------
# 本地 POE 模型参数
# ---------------------------------------------------------------------------
# 这是当前本地模型的核心。
#
# 含义说明：
# 1. `LOCAL_JOINT_AXIS_DIRECTIONS_BASE`
#    零位姿下，每个关节轴在机器人基坐标系里的方向。
# 2. `LOCAL_JOINT_AXIS_POINTS_BASE_MM`
#    零位姿下，每个关节轴线上任意一点，单位 mm。
# 3. `LOCAL_JOINT_SENSES`
#    命令角和右手定则的关系。+1 表示同向，-1 表示反向。
# 4. `LOCAL_HOME_FLANGE_MATRIX`
#    全零关节角时，法兰在机器人基坐标系下的 4x4 位姿矩阵。
#
# 这组值已经通过 RoboDK link pose + FK 离线校准过，当前本地 FK 可以与 RoboDK 对齐。
LOCAL_JOINT_AXIS_DIRECTIONS_BASE = [
    (0.0, 0.0, 1.0),
    (0.0, 1.0, 0.0),
    (0.0, 1.0, 0.0),
    (1.0, 0.0, 0.0),
    (0.0, 1.0, 0.0),
    (1.0, 0.0, 0.0),
]

LOCAL_JOINT_AXIS_POINTS_BASE_MM = [
    (0.0, 0.0, 0.0),
    (150.0, 0.0, 443.5),
    (150.0, 0.0, 1253.5),
    (1007.0, 0.0, 1453.5),
    (1007.0, 0.0, 1453.5),
    (1007.0, 0.0, 1453.5),
]

LOCAL_JOINT_SENSES = [-1.0, 1.0, 1.0, -1.0, 1.0, -1.0]

LOCAL_HOME_FLANGE_MATRIX = [
    [0.0, 0.0, 1.0, 1097.0],
    [0.0, 1.0, 0.0, 0.0],
    [-1.0, 0.0, 0.0, 1453.5],
    [0.0, 0.0, 0.0, 1.0],
]

# ---------------------------------------------------------------------------
# 本地数值 IK 与多解枚举
# ---------------------------------------------------------------------------
# 姿态误差在最小二乘中的权重。数值越大，姿态误差越“贵”。
NUMERIC_ROTATION_WEIGHT_MM = 200.0

# 单次完整 IK 的最大迭代次数。
NUMERIC_MAX_NFEV = 250

# 腕心位置阶段的 q2/q3 初值种子。
# 这组种子不是最终解，而是“给 1~3 轴分支探路”的。
LOCAL_ARM_Q2Q3_SEEDS_DEG = [
    (-40.0, -120.0),
    (40.0, -120.0),
    (-40.0, 40.0),
    (80.0, -40.0),
]

# 完整 6 轴 IK 时，给腕部 4~6 轴的初值种子。
# 这里故意同时放了正反 wrist flip 的组合。
LOCAL_WRIST_SEED_TRIPLETS_DEG = [
    (0.0, -90.0, 0.0),
    (0.0, 90.0, 0.0),
    (-180.0, -90.0, 180.0),
    (-180.0, 90.0, 180.0),
]

# 下面这些阈值决定了“本地解算器什么时候认为一组解是有效的”。
LOCAL_ARM_POSITION_TOLERANCE_MM = 1e-4
LOCAL_IK_POSITION_TOLERANCE_MM = 1e-4
LOCAL_IK_ORIENTATION_TOLERANCE_DEG = 1e-5
LOCAL_IK_PERIODIC_DEDUP_TOLERANCE_DEG = 1e-2


def _validate_joint_ranges(ranges: Sequence[Tuple[float, float]]) -> None:
    """校验关节范围配置。"""
    if len(ranges) != JOINT_COUNT:
        raise ValueError(f"Expected {JOINT_COUNT} joint ranges, got {len(ranges)}.")

    for index, (lower_deg, upper_deg) in enumerate(ranges, start=1):
        if lower_deg > upper_deg:
            raise ValueError(f"Joint {index} lower limit must be <= upper limit.")


def _ranges_to_arrays(ranges: Sequence[Tuple[float, float]]) -> Tuple[np.ndarray, np.ndarray]:
    """把 `[(low, high), ...]` 转成两个 numpy 向量。"""
    _validate_joint_ranges(ranges)
    lower = np.array([pair[0] for pair in ranges], dtype=float)
    upper = np.array([pair[1] for pair in ranges], dtype=float)
    return lower, upper


def get_configured_tool_pose() -> np.ndarray:
    """返回配置中的 Tool 位姿矩阵。"""
    return make_pose_zyx(*CONFIGURED_TOOL_POSE_XYZ_RZYX_MM_DEG)


def get_configured_frame_pose() -> np.ndarray:
    """返回配置中的 Frame 位姿矩阵。"""
    return make_pose_zyx(*CONFIGURED_FRAME_POSE_XYZ_RZYX_MM_DEG)


def get_configured_lower_limits_deg() -> np.ndarray:
    """返回机器人硬限位下限。"""
    lower, _ = _ranges_to_arrays(CONFIGURED_JOINT_LIMITS_DEG)
    return lower


def get_configured_upper_limits_deg() -> np.ndarray:
    """返回机器人硬限位上限。"""
    _, upper = _ranges_to_arrays(CONFIGURED_JOINT_LIMITS_DEG)
    return upper


def get_filter_lower_limits_deg() -> np.ndarray:
    """返回应用层解过滤下限。"""
    lower, _ = _ranges_to_arrays(IK_FILTER_JOINT_RANGES_DEG)
    return lower


def get_filter_upper_limits_deg() -> np.ndarray:
    """返回应用层解过滤上限。"""
    _, upper = _ranges_to_arrays(IK_FILTER_JOINT_RANGES_DEG)
    return upper


def get_default_seed_joints() -> np.ndarray:
    """返回默认的排序种子 / 初始关节角。"""
    return np.zeros(JOINT_COUNT, dtype=float)


def build_local_robot_model() -> RobotModel:
    """用配置文件里的 POE 参数构造本地机器人模型。"""
    return RobotModel(
        joint_axis_directions_base=np.array(LOCAL_JOINT_AXIS_DIRECTIONS_BASE, dtype=float),
        joint_axis_points_base_mm=np.array(LOCAL_JOINT_AXIS_POINTS_BASE_MM, dtype=float),
        joint_senses=np.array(LOCAL_JOINT_SENSES, dtype=float),
        home_flange_T=np.array(LOCAL_HOME_FLANGE_MATRIX, dtype=float),
        joint_min_deg=get_configured_lower_limits_deg(),
        joint_max_deg=get_configured_upper_limits_deg(),
        tool_T=get_configured_tool_pose(),
        frame_T=get_configured_frame_pose(),
    )
