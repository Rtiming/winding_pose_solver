"""
机器人 IK 后端抽象层。

提供两个实现：
  - RoboDKRobotInterface：直接委托给 RoboDK robot 对象（原有行为不变）
  - SixAxisIKRobotInterface：用内置 SixAxisIK 本地求解器替代 RoboDK IK

两者均以 RoboDK robot 的鸭子类型 API 为接口，因此 ik_collection.py、
path_optimizer.py、global_search.py 等模块无需改动函数签名。
"""

from __future__ import annotations

import math
from typing import Any, Sequence


# ---------------------------------------------------------------------------
# 辅助类型：模拟 RoboDK 返回带 .list() 的对象
# ---------------------------------------------------------------------------


class _ListWrapper:
    """把 Python 列表包装成带 `.list()` 方法的对象，仿照 RoboDK 返回类型。"""

    __slots__ = ("_values",)

    def __init__(self, values: Sequence[float]) -> None:
        self._values: list[float] = [float(v) for v in values]

    def list(self) -> list[float]:
        return list(self._values)

    def __iter__(self):
        return iter(self._values)

    def __len__(self) -> int:
        return len(self._values)

    def __getitem__(self, index):
        return self._values[index]


# ---------------------------------------------------------------------------
# RoboDK 包装器（保留原有行为）
# ---------------------------------------------------------------------------


class RoboDKRobotInterface:
    """把 RoboDK robot 对象包装成接口对象，保留全部原有行为。

    其他所有属性访问（setJoints、Render 等）会透明地委托给底层 robot 对象。
    """

    ik_seed_invariant = False

    def __init__(self, robot: Any) -> None:
        self._robot = robot

    def __getattr__(self, name: str) -> Any:
        return getattr(self._robot, name)

    # 明确代理 IK 相关方法（避免 __getattr__ 覆盖潜在子类覆写）

    def JointLimits(self):  # type: ignore[return]
        return self._robot.JointLimits()

    def Joints(self):  # type: ignore[return]
        return self._robot.Joints()

    def JointsHome(self):  # type: ignore[return]
        return self._robot.JointsHome()

    def SolveIK_All(self, pose: Any, tool_pose: Any, reference_pose: Any) -> list:
        return self._robot.SolveIK_All(pose, tool_pose, reference_pose)

    def SolveIK(self, pose: Any, seed: list[float], tool_pose: Any, reference_pose: Any) -> Any:
        return self._robot.SolveIK(pose, seed, tool_pose, reference_pose)

    def JointsConfig(self, joints_list: list[float]) -> Any:
        return self._robot.JointsConfig(joints_list)

    def JointPoses(self, joints_list: list[float]) -> list:
        return self._robot.JointPoses(joints_list)


# ---------------------------------------------------------------------------
# SixAxisIK 包装器
# ---------------------------------------------------------------------------


class SixAxisIKRobotInterface:
    """基于内置 SixAxisIK 求解器的 IK 后端，鸭子类型 API 与 RoboDK robot 一致。

    仍依赖 robodk_robot（可选）来读取当前关节状态；
    所有 IK、配置标志、奇异性惩罚均由本地求解器计算。
    """

    ik_seed_invariant = True

    def __init__(self, robodk_robot: Any = None) -> None:
        """
        Parameters
        ----------
        robodk_robot : optional
            RoboDK robot 对象，用于读取当前关节角（seeding）。
            若为 None，则 Joints() 返回零位。
        """
        from src.six_axis_ik.interface import SixAxisIKSolver
        from src.six_axis_ik import config as six_axis_config

        self._robodk_robot = robodk_robot
        self._solver = SixAxisIKSolver.from_config(ik_backend="pure_analytic")
        self._model = self._solver.robot_model
        self._config = six_axis_config
        # Cache last SolveIK_All result to avoid re-running analytic solver for each seed call
        self._cached_all_solutions: list[list[float]] = []
        self._cached_pose_key: tuple | None = None

    # ------------------------------------------------------------------
    # 关节限位
    # ------------------------------------------------------------------

    def JointLimits(self):
        """返回 (lower_wrapper, upper_wrapper, None)，与 RoboDK 接口一致。"""
        lower = list(self._model.joint_min_deg)
        upper = list(self._model.joint_max_deg)
        return _ListWrapper(lower), _ListWrapper(upper), None

    # ------------------------------------------------------------------
    # 当前关节 / 零位关节
    # ------------------------------------------------------------------

    def Joints(self):
        """若有 RoboDK robot，从中读取当前关节；否则返回零位。"""
        if self._robodk_robot is not None:
            try:
                return self._robodk_robot.Joints()
            except Exception:
                pass
        return _ListWrapper([0.0] * 6)

    def JointsHome(self):
        """返回零位（SixAxisIK 的 home 配置）。"""
        return _ListWrapper([0.0] * 6)

    # ------------------------------------------------------------------
    # IK 求解
    # ------------------------------------------------------------------

    def _pose_cache_key(self, pose: Any) -> tuple:
        """Build a hashable cache key from a pose Mat."""
        try:
            rows = pose.rows
            return tuple(rows[r][c] for r in range(4) for c in range(4))
        except Exception:
            return id(pose)

    def SolveIK_All(self, pose: Any, tool_pose: Any, reference_pose: Any) -> list[list[float]]:
        """枚举所有 IK 分支，返回各分支关节角列表。

        利用 SixAxisIK 的多解枚举（腕心候选 × 腕部种子 × 周期扩展），
        覆盖比 RoboDK 随机种子更完整的解空间。
        结果被缓存，供后续 SolveIK 调用（同一 pose）直接复用，避免重复计算。
        """
        key = self._pose_cache_key(pose)
        self._cached_pose_key = key
        result = self._solver.solve_ik_all_joint_vectors(
            pose,
            target_space="frame",
            tool_pose=tool_pose,
            reference_pose=reference_pose,
        )
        # filtered_solutions 已经通过关节限位过滤，直接返回
        self._cached_all_solutions = [sol.tolist() for sol in result]
        return list(self._cached_all_solutions)

    def SolveIK(self, pose: Any, seed: list[float], tool_pose: Any, reference_pose: Any) -> list[float]:
        """带种子的单次 IK。

        对 SixAxisIK 而言，解析求解已在 SolveIK_All 中完成；
        如果缓存命中（同一 pose），直接从缓存中选取距 seed 最近的解，
        不再重复运行解析求解器。
        """
        key = self._pose_cache_key(pose)
        if key == self._cached_pose_key:
            # Cache hit: return empty if SolveIK_All already found no solutions,
            # otherwise pick the candidate closest to the seed joint vector.
            if not self._cached_all_solutions:
                return []
            seed_arr = seed if isinstance(seed, list) else list(seed)
            best = min(
                self._cached_all_solutions,
                key=lambda s: sum((a - b) ** 2 for a, b in zip(s, seed_arr)),
            )
            return best

        # Cache miss — run full IK and update cache
        result = self._solver.solve_ik(
            pose,
            target_space="frame",
            seed_joints_deg=seed,
            tool_pose=tool_pose,
            reference_pose=reference_pose,
        )
        if result.success and result.preferred_solution is not None:
            return result.preferred_solution.joints_deg.tolist()
        return []

    # ------------------------------------------------------------------
    # 配置标志（代替 RoboDK JointsConfig）
    # ------------------------------------------------------------------

    def JointsConfig(self, joints_list: list[float]) -> _ListWrapper:
        """根据关节角计算 KUKA 风格配置标志（3 bit）。

        bit0: Overhead/Rear  — 腕心 X < J2 轴 X（150 mm）则为 1
        bit1: Elbow          — J3 < 0 为 1（肘部向下）
        bit2: Wrist Flip     — J5 < 0 为 1
        """
        flags = _compute_kuka_config_flags(joints_list, self._model)
        return _ListWrapper(list(flags))

    # ------------------------------------------------------------------
    # 关节位姿（代替 RoboDK JointPoses，用于奇异性惩罚）
    # ------------------------------------------------------------------

    def JointPoses(self, joints_list: list[float]) -> list:
        """返回 [None, shoulder_T, elbow_T, wrist_T]，与 _singularity_penalty 期望一致。

        每个元素是 4x4 numpy 矩阵，可用 pose[row, col] 访问。
        """
        return _compute_link_poses(joints_list, self._model)

    # ------------------------------------------------------------------
    # 非 IK 操作：优先委托给 RoboDK robot；离线模式下特殊处理
    # ------------------------------------------------------------------

    def setJoints(self, joints_list) -> None:
        """设置关节角。有 RoboDK robot 则委托；离线模式下忽略。"""
        if self._robodk_robot is not None:
            self._robodk_robot.setJoints(joints_list)
        # 离线模式：no-op

    def __getattr__(self, name: str) -> Any:
        if self._robodk_robot is not None:
            return getattr(self._robodk_robot, name)
        raise AttributeError(
            f"SixAxisIKRobotInterface has no attribute '{name}' "
            "and no underlying RoboDK robot to delegate to."
        )


# ---------------------------------------------------------------------------
# 工厂函数
# ---------------------------------------------------------------------------


def build_robot_interface(
    ik_backend: str,
    robodk_robot: Any,
) -> RoboDKRobotInterface | SixAxisIKRobotInterface:
    """根据后端名称构造对应的机器人接口。

    Parameters
    ----------
    ik_backend : str
        "robodk" 或 "six_axis_ik"
    robodk_robot : Any
        RoboDK robot 对象（总是需要，因为程序生成仍依赖 RoboDK）
    """
    if ik_backend == "six_axis_ik":
        print(f"[IK backend] Using SixAxisIK local solver.")
        return SixAxisIKRobotInterface(robodk_robot=robodk_robot)
    if ik_backend == "robodk":
        return RoboDKRobotInterface(robodk_robot)
    raise ValueError(
        f"Unknown IK backend '{ik_backend}'. Valid values: 'robodk', 'six_axis_ik'."
    )


# ---------------------------------------------------------------------------
# 几何辅助：KUKA 配置标志计算
# ---------------------------------------------------------------------------


def _compute_kuka_config_flags(
    joints_list: Sequence[float],
    model: Any,
) -> tuple[int, int, int]:
    """从关节角计算 KUKA 机器人配置标志（不依赖 RoboDK）。

    返回 (bit0, bit1, bit2)，与 RoboDK JointsConfig().list()[:3] 对应：
      bit0 = Rear/Overhead: 腕心在 J2 轴（X=150mm）前方则 0，后方则 1
      bit1 = Elbow Up/Down: J3 >= 0 为 0（肘部向上），J3 < 0 为 1
      bit2 = Wrist Flip:    J5 >= 0 为 0（非翻转），J5 < 0 为 1
    """
    joints = list(joints_list)
    j3 = joints[2] if len(joints) > 2 else 0.0
    j5 = joints[4] if len(joints) > 4 else 0.0

    # 腕心位置用于判断 Overhead/Rear
    try:
        import numpy as np
        wrist_center = model.fk_wrist_center_in_robot_base(joints)
        # J2 轴在 X=150mm 处；腕心 X < 150 mm 认为 Overhead/Rear
        bit0 = 1 if float(wrist_center[0]) < 150.0 else 0
    except Exception:
        bit0 = 0

    bit1 = 1 if j3 < 0.0 else 0
    bit2 = 1 if j5 < 0.0 else 0
    return (bit0, bit1, bit2)


# ---------------------------------------------------------------------------
# 几何辅助：关节链接位姿计算（用于奇异性惩罚）
# ---------------------------------------------------------------------------


def _compute_link_poses(
    joints_list: Sequence[float],
    model: Any,
) -> list:
    """计算用于奇异性惩罚的关键连杆位姿。

    返回 [None, shoulder_T, elbow_T, wrist_T]：
      - shoulder_T：J1 旋转后，J2 轴的 4x4 位姿（shoulder 位置）
      - elbow_T：  J1+J2 旋转后，J3 轴的 4x4 位姿（elbow 位置）
      - wrist_T：  腕心的 4x4 位姿（前 3 轴确定的 position）

    每个矩阵是 4x4 numpy array，支持 pose[row, col] 访问，
    与 _translation_from_pose 中的 pose_matrix[0,3] 等调用兼容。
    """
    import numpy as np

    joints = list(joints_list)

    # J2 轴在零位姿下的坐标
    j2_point_home = model.joint_axis_points_base_mm[1]  # (150, 0, 443.5)
    # J3 轴在零位姿下的坐标
    j3_point_home = model.joint_axis_points_base_mm[2]  # (150, 0, 1253.5)

    # 前 1 个关节（J1）的累积变换
    T1 = model.fk_partial(joints, n_joints=1)
    # 在 T1 坐标系下，J2 轴点在 base 中的位置
    shoulder_pos = (T1 @ np.array([j2_point_home[0], j2_point_home[1], j2_point_home[2], 1.0]))[:3]
    shoulder_T = np.eye(4, dtype=float)
    shoulder_T[:3, 3] = shoulder_pos

    # 前 2 个关节（J1+J2）的累积变换
    T2 = model.fk_partial(joints, n_joints=2)
    # 在 T2 坐标系下，J3 轴点在 base 中的位置
    # J3 轴点在 J2 局部坐标中的偏移：j3_point_home - j2_point_home（仅近似，因为零位重合）
    # 精确做法：J3 点在零位下绝对坐标 j3_point_home，经 T2 逆变换可找到本地坐标，但实际上：
    # T_fk_partial 已经把 j2_point 用作旋转轴了，故这里直接用 home 坐标再旋转
    elbow_pos = (T2 @ np.array([j3_point_home[0], j3_point_home[1], j3_point_home[2], 1.0]))[:3]
    elbow_T = np.eye(4, dtype=float)
    elbow_T[:3, 3] = elbow_pos

    # 腕心位置
    wrist_center = model.fk_wrist_center_in_robot_base(joints)
    wrist_T = np.eye(4, dtype=float)
    wrist_T[:3, 3] = wrist_center

    # 返回列表索引：[0]=None（基座）, [1]=肩, [2]=肘, [3]=腕心
    return [None, shoulder_T, elbow_T, wrist_T]
