"""
六轴机械臂本地运动学核心模块。

当前主模型使用 POE（Product Of Exponentials，螺旋轴指数积）形式，
不再依赖“猜测出来的标准 DH 参数”。
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from itertools import product
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as R

JOINT_COUNT = 6
POSE_VECTOR_SIZE = 6
ARM_JOINT_COUNT = 3


def joint_distance_deg(q1_deg: Iterable[float], q2_deg: Iterable[float]) -> float:
    """计算两组关节角之间的周期距离，单位度。"""
    q1 = as_joint_vector(q1_deg, "q1_deg")
    q2 = as_joint_vector(q2_deg, "q2_deg")

    wrapped_diff = np.abs(q1 - q2)
    wrapped_diff = np.minimum(wrapped_diff, np.abs(wrapped_diff - 360.0))
    wrapped_diff = np.minimum(wrapped_diff, np.abs(wrapped_diff + 360.0))
    return float(np.linalg.norm(wrapped_diff))


def as_joint_vector(q_deg: Iterable[float], name: str = "q_deg") -> np.ndarray:
    """把输入整理成长度固定为 6 的关节角向量。"""
    q = np.asarray(list(q_deg), dtype=float).reshape(-1)
    if q.size != JOINT_COUNT:
        raise ValueError(f"{name} must contain {JOINT_COUNT} values, got {q.size}.")
    return q


def as_joint_matrix(values: Iterable[Iterable[float]], name: str) -> np.ndarray:
    """把输入整理成 `6 x 3` 的矩阵。"""
    matrix = np.asarray(list(values), dtype=float)
    if matrix.shape != (JOINT_COUNT, 3):
        raise ValueError(f"{name} must have shape (6, 3), got {matrix.shape}.")
    return matrix


def as_transform(T: np.ndarray, name: str = "T") -> np.ndarray:
    """校验输入是否为 4x4 齐次变换矩阵。"""
    arr = np.asarray(T, dtype=float)
    if arr.shape != (4, 4):
        raise ValueError(f"{name} must be a 4x4 homogeneous transform, got shape {arr.shape}.")
    return arr


def normalize_angle_deg(angle_deg: float) -> float:
    """把角度规范到 `(-180, 180]`。"""
    wrapped = ((float(angle_deg) + 180.0) % 360.0) - 180.0
    if wrapped <= -180.0:
        wrapped += 360.0
    return wrapped


def standard_dh(theta_deg: float, d_mm: float, a_mm: float, alpha_deg: float) -> np.ndarray:
    """
    兼容保留：生成标准 DH 单节变换矩阵。

    当前主流程已经不用 DH 建模，但保留这个工具，方便以后做对比实验。
    """
    th = np.deg2rad(theta_deg)
    al = np.deg2rad(alpha_deg)

    ct, st = np.cos(th), np.sin(th)
    ca, sa = np.cos(al), np.sin(al)

    return np.array(
        [
            [ct, -st * ca, st * sa, a_mm * ct],
            [st, ct * ca, -ct * sa, a_mm * st],
            [0.0, sa, ca, d_mm],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=float,
    )


def skew_symmetric(vector_xyz: Sequence[float]) -> np.ndarray:
    """把 3 维向量转换成反对称矩阵 `[w]x`。"""
    x_value, y_value, z_value = np.asarray(vector_xyz, dtype=float)
    return np.array(
        [
            [0.0, -z_value, y_value],
            [z_value, 0.0, -x_value],
            [-y_value, x_value, 0.0],
        ],
        dtype=float,
    )


def revolute_twist_transform(
    axis_direction_xyz: Sequence[float],
    axis_point_xyz_mm: Sequence[float],
    angle_deg: float,
) -> np.ndarray:
    """计算一个旋转关节的 POE 指数映射。"""
    axis_direction = np.asarray(axis_direction_xyz, dtype=float)
    axis_point = np.asarray(axis_point_xyz_mm, dtype=float)

    angle_rad = np.deg2rad(angle_deg)
    axis_skew = skew_symmetric(axis_direction)
    identity = np.eye(3, dtype=float)

    rotation_matrix = (
        identity
        + np.sin(angle_rad) * axis_skew
        + (1.0 - np.cos(angle_rad)) * (axis_skew @ axis_skew)
    )

    linear_velocity = -np.cross(axis_direction, axis_point)
    translation_matrix = (
        identity * angle_rad
        + (1.0 - np.cos(angle_rad)) * axis_skew
        + (angle_rad - np.sin(angle_rad)) * (axis_skew @ axis_skew)
    )
    translation_mm = translation_matrix @ linear_velocity

    transform = np.eye(4, dtype=float)
    transform[:3, :3] = rotation_matrix
    transform[:3, 3] = translation_mm
    return transform


def make_pose_zyx(
    x_mm: float,
    y_mm: float,
    z_mm: float,
    rz_deg: float,
    ry_deg: float,
    rx_deg: float,
) -> np.ndarray:
    """根据位置和 ZYX 欧拉角构造 4x4 齐次变换矩阵。"""
    T = np.eye(4, dtype=float)
    T[:3, :3] = R.from_euler("ZYX", [rz_deg, ry_deg, rx_deg], degrees=True).as_matrix()
    T[:3, 3] = [x_mm, y_mm, z_mm]
    return T


def invert_transform(T: np.ndarray) -> np.ndarray:
    """求刚体变换矩阵的逆。"""
    T = as_transform(T)
    rotation_matrix = T[:3, :3]
    position = T[:3, 3]

    inverse = np.eye(4, dtype=float)
    inverse[:3, :3] = rotation_matrix.T
    inverse[:3, 3] = -rotation_matrix.T @ position
    return inverse


def pose_to_xyz_zyx(T: np.ndarray) -> np.ndarray:
    """把变换矩阵转成 `[X, Y, Z, Rz, Ry, Rx]`。"""
    T = as_transform(T)
    xyz = T[:3, 3]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        euler_zyx_deg = R.from_matrix(T[:3, :3]).as_euler("ZYX", degrees=True)

    return np.array(
        [xyz[0], xyz[1], xyz[2], euler_zyx_deg[0], euler_zyx_deg[1], euler_zyx_deg[2]],
        dtype=float,
    )


def pose_error_vector(
    current_T: np.ndarray,
    target_T: np.ndarray,
    rotation_weight_mm: float,
) -> np.ndarray:
    """计算数值 IK 使用的 6 维残差向量。"""
    current_T = as_transform(current_T, "current_T")
    target_T = as_transform(target_T, "target_T")

    position_error_mm = target_T[:3, 3] - current_T[:3, 3]
    rotation_error = target_T[:3, :3] @ current_T[:3, :3].T
    rotation_error_weighted = R.from_matrix(rotation_error).as_rotvec() * rotation_weight_mm

    return np.concatenate([position_error_mm, rotation_error_weighted])


def rotation_error_deg(current_T: np.ndarray, target_T: np.ndarray) -> float:
    """计算两个姿态之间的最小旋转误差角，单位度。"""
    current_T = as_transform(current_T, "current_T")
    target_T = as_transform(target_T, "target_T")

    rotation_error = target_T[:3, :3] @ current_T[:3, :3].T
    return float(np.rad2deg(np.linalg.norm(R.from_matrix(rotation_error).as_rotvec())))


def joints_within_limits(
    q_deg: Sequence[float],
    lower_deg: Sequence[float],
    upper_deg: Sequence[float],
    atol: float = 1e-9,
) -> bool:
    """判断一组关节角是否落在指定区间内。"""
    q = as_joint_vector(q_deg, "q_deg")
    lower = as_joint_vector(lower_deg, "lower_deg")
    upper = as_joint_vector(upper_deg, "upper_deg")
    return bool(np.all(q >= lower - atol) and np.all(q <= upper + atol))


def deduplicate_joint_vectors_periodic(
    joint_vectors_deg: Sequence[Sequence[float]],
    tolerance_deg: float,
) -> List[np.ndarray]:
    """按 360 度周期对解向量去重。"""
    unique_vectors: List[np.ndarray] = []
    for candidate in joint_vectors_deg:
        candidate_vector = as_joint_vector(candidate, "candidate")
        if any(joint_distance_deg(candidate_vector, saved) <= tolerance_deg for saved in unique_vectors):
            continue
        unique_vectors.append(candidate_vector)
    return unique_vectors


@dataclass
class IKResult:
    """单次数值 IK 求解结果。"""

    q_deg: np.ndarray
    success: bool
    message: str
    nfev: int
    cost: float
    residual: np.ndarray
    target_flange_base_T: np.ndarray


@dataclass
class LocalIKSolution:
    """本地 `ik-all` 的一组最终解。"""

    index: int
    branch_index: int
    joints_deg: np.ndarray
    seed_joints_deg: np.ndarray
    turn_offsets: np.ndarray
    within_robot_limits: bool
    within_filter_limits: bool
    distance_to_seed_deg: Optional[float]
    residual_norm: float
    position_error_mm: float
    orientation_error_deg: float


@dataclass
class LocalIKSolutionSet:
    """本地多解 IK 的结构化结果。"""

    target_pose: np.ndarray
    target_space: str
    seed_joints_deg: Optional[np.ndarray]
    robot_lower_limits_deg: np.ndarray
    robot_upper_limits_deg: np.ndarray
    filter_lower_limits_deg: np.ndarray
    filter_upper_limits_deg: np.ndarray
    arm_candidate_solutions_deg: List[np.ndarray]
    branch_solutions: List[LocalIKSolution]
    all_solutions: List[LocalIKSolution]
    filtered_solutions: List[LocalIKSolution]


@dataclass
class RobotModel:
    """
    六轴机械臂本地 POE 模型。

    核心输入包括：
    - `joint_axis_directions_base`
      零位姿下，每个关节轴在机器人基坐标系中的方向
    - `joint_axis_points_base_mm`
      零位姿下，每个关节轴线上任意一点
    - `joint_senses`
      关节方向修正。`+1` 表示命令角与右手定则一致，`-1` 表示相反
    - `home_flange_T`
      全零关节角时，法兰在机器人基坐标系下的位姿
    """

    joint_axis_directions_base: np.ndarray
    joint_axis_points_base_mm: np.ndarray
    joint_senses: np.ndarray
    home_flange_T: np.ndarray
    joint_min_deg: np.ndarray
    joint_max_deg: np.ndarray
    tool_T: np.ndarray
    frame_T: np.ndarray
    tool_T_inv: np.ndarray = field(init=False, repr=False)
    frame_T_inv: np.ndarray = field(init=False, repr=False)
    home_wrist_center_base_mm: np.ndarray = field(init=False, repr=False)
    wrist_center_to_flange_in_flange_mm: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """构造完成后统一做参数校验和缓存。"""
        self.joint_axis_directions_base = as_joint_matrix(
            self.joint_axis_directions_base, "joint_axis_directions_base"
        )
        self.joint_axis_points_base_mm = as_joint_matrix(
            self.joint_axis_points_base_mm, "joint_axis_points_base_mm"
        )
        self.joint_senses = as_joint_vector(self.joint_senses, "joint_senses")
        self.home_flange_T = as_transform(self.home_flange_T, "home_flange_T")
        self.joint_min_deg = as_joint_vector(self.joint_min_deg, "joint_min_deg")
        self.joint_max_deg = as_joint_vector(self.joint_max_deg, "joint_max_deg")
        self.tool_T = as_transform(self.tool_T, "tool_T")
        self.frame_T = as_transform(self.frame_T, "frame_T")

        if np.any(self.joint_min_deg > self.joint_max_deg):
            raise ValueError("Each joint minimum must be <= joint maximum.")

        axis_norms = np.linalg.norm(self.joint_axis_directions_base, axis=1)
        if np.any(axis_norms <= 1e-12):
            raise ValueError("Each joint axis direction must be non-zero.")
        self.joint_axis_directions_base = self.joint_axis_directions_base / axis_norms[:, None]

        if np.any(np.abs(np.abs(self.joint_senses) - 1.0) > 1e-9):
            raise ValueError("Each joint sense must be either +1 or -1.")

        self.tool_T_inv = invert_transform(self.tool_T)
        self.frame_T_inv = invert_transform(self.frame_T)

        # 对当前这台六轴机械臂来说，4/5/6 轴在零位姿下共用同一个腕心点。
        self.home_wrist_center_base_mm = self.joint_axis_points_base_mm[3].copy()
        self.wrist_center_to_flange_in_flange_mm = (
            self.home_flange_T[:3, :3].T
            @ (self.home_flange_T[:3, 3] - self.home_wrist_center_base_mm)
        )

    def clip_joints(self, q_deg: Iterable[float]) -> np.ndarray:
        """把关节角裁剪到机器人硬限位范围内。"""
        q = as_joint_vector(q_deg)
        return np.clip(q, self.joint_min_deg, self.joint_max_deg)

    def within_joint_limits(self, q_deg: Iterable[float], atol: float = 1e-9) -> bool:
        """判断关节角是否位于机器人硬限位范围内。"""
        return joints_within_limits(q_deg, self.joint_min_deg, self.joint_max_deg, atol=atol)

    def fk_flange(self, q_deg: Iterable[float]) -> np.ndarray:
        """
        计算法兰相对机器人基坐标系的位姿。

        POE 公式为：
            T(q) = exp(S1 * theta1) * ... * exp(S6 * theta6) * M
        """
        q = as_joint_vector(q_deg)

        transform = np.eye(4, dtype=float)
        for index in range(JOINT_COUNT):
            effective_angle_deg = q[index] * self.joint_senses[index]
            transform = transform @ revolute_twist_transform(
                self.joint_axis_directions_base[index],
                self.joint_axis_points_base_mm[index],
                effective_angle_deg,
            )
        return transform @ self.home_flange_T

    def fk_tcp_in_robot_base(self, q_deg: Iterable[float]) -> np.ndarray:
        """计算 TCP 相对机器人基坐标系的位姿。"""
        return self.fk_flange(q_deg) @ self.tool_T

    def fk_tcp_in_frame(self, q_deg: Iterable[float]) -> np.ndarray:
        """计算 TCP 相对当前参考坐标系的位姿。"""
        T_base_tcp = self.fk_tcp_in_robot_base(q_deg)
        return self.frame_T_inv @ T_base_tcp

    def flange_target_from_frame_tcp_target(self, target_frame_tcp_T: np.ndarray) -> np.ndarray:
        """把参考坐标系下的目标 TCP 位姿转成基坐标系下的目标法兰位姿。"""
        target_frame_tcp_T = as_transform(target_frame_tcp_T, "target_frame_tcp_T")
        return self.frame_T @ target_frame_tcp_T @ self.tool_T_inv

    def wrist_center_from_flange_pose(self, flange_base_T: np.ndarray) -> np.ndarray:
        """从法兰位姿反推出腕心位置。"""
        flange_base_T = as_transform(flange_base_T, "flange_base_T")
        return flange_base_T[:3, 3] - flange_base_T[:3, :3] @ self.wrist_center_to_flange_in_flange_mm

    def fk_wrist_center_in_robot_base(self, q_deg: Iterable[float]) -> np.ndarray:
        """计算腕心相对机器人基坐标系的位置。"""
        flange_base_T = self.fk_flange(q_deg)
        return self.wrist_center_from_flange_pose(flange_base_T)

    def fk_partial(self, q_deg: Iterable[float], n_joints: int) -> np.ndarray:
        """计算前 n_joints 个关节的 POE 累积变换，不含 home 法兰矩阵。

        返回的是 4x4 矩阵，表示在施加前 n_joints 个关节后的坐标系变换。
        用于计算奇异性惩罚中的肩部/肘部/腕心位置。
        """
        q = as_joint_vector(q_deg)
        count = max(0, min(n_joints, JOINT_COUNT))
        transform = np.eye(4, dtype=float)
        for index in range(count):
            effective_angle_deg = q[index] * self.joint_senses[index]
            transform = transform @ revolute_twist_transform(
                self.joint_axis_directions_base[index],
                self.joint_axis_points_base_mm[index],
                effective_angle_deg,
            )
        return transform

    def _solve_position_only_ik_for_arm(
        self,
        q1_deg: float,
        target_wrist_center_mm: np.ndarray,
        q2q3_seed_deg: Sequence[float],
        max_nfev: int,
    ) -> Tuple[np.ndarray, float]:
        """兼容保留：转交给独立数值求解器。"""
        from .numeric_solver import NumericIKSolver

        return NumericIKSolver(self)._solve_position_only_ik_for_arm(
            q1_deg=q1_deg,
            target_wrist_center_mm=target_wrist_center_mm,
            q2q3_seed_deg=q2q3_seed_deg,
            max_nfev=max_nfev,
        )

    def solve_arm_position_candidates(
        self,
        target_flange_base_T: np.ndarray,
        q2q3_seed_pairs_deg: Sequence[Sequence[float]],
        position_tolerance_mm: float,
        dedup_tolerance_deg: float,
        max_nfev: int,
    ) -> List[np.ndarray]:
        """兼容保留：转交给独立数值求解器。"""
        from .numeric_solver import NumericIKSolver

        return NumericIKSolver(self).solve_arm_position_candidates(
            target_flange_base_T=target_flange_base_T,
            q2q3_seed_pairs_deg=q2q3_seed_pairs_deg,
            position_tolerance_mm=position_tolerance_mm,
            dedup_tolerance_deg=dedup_tolerance_deg,
            max_nfev=max_nfev,
        )

    def _ik_numeric_on_flange_target(
        self,
        target_flange_base_T: np.ndarray,
        q0_deg: Sequence[float],
        rotation_weight_mm: float,
        max_nfev: int,
    ) -> IKResult:
        """兼容保留：转交给独立数值求解器。"""
        from .numeric_solver import NumericIKSolver

        return NumericIKSolver(self)._solve_numeric_on_flange_target(
            target_flange_base_T=target_flange_base_T,
            q0_deg=q0_deg,
            rotation_weight_mm=rotation_weight_mm,
            max_nfev=max_nfev,
        )

    def ik_numeric(
        self,
        target_frame_tcp_T: np.ndarray,
        q0_deg: Optional[Iterable[float]] = None,
        rotation_weight_mm: float = 200.0,
        max_nfev: int = 200,
        raise_on_fail: bool = True,
    ) -> IKResult:
        """兼容保留：转交给独立数值求解器。"""
        from .numeric_solver import NumericIKSolver

        return NumericIKSolver(self).solve_ik(
            target_frame_tcp_T=target_frame_tcp_T,
            q0_deg=q0_deg,
            rotation_weight_mm=rotation_weight_mm,
            max_nfev=max_nfev,
            raise_on_fail=raise_on_fail,
        )

    def expand_solution_turn_variants(self, q_deg: Sequence[float]) -> List[Tuple[np.ndarray, np.ndarray]]:
        """兼容保留：转交给独立数值求解器。"""
        from .numeric_solver import NumericIKSolver

        return NumericIKSolver(self).expand_solution_turn_variants(q_deg)

    def ik_all(
        self,
        target_frame_tcp_T: np.ndarray,
        target_space: str,
        seed_joints_deg: Optional[Sequence[float]],
        filter_lower_deg: Sequence[float],
        filter_upper_deg: Sequence[float],
        q2q3_seed_pairs_deg: Sequence[Sequence[float]],
        wrist_seed_triplets_deg: Sequence[Sequence[float]],
        rotation_weight_mm: float,
        max_nfev: int,
        arm_position_tolerance_mm: float,
        pose_position_tolerance_mm: float,
        pose_orientation_tolerance_deg: float,
        periodic_dedup_tolerance_deg: float,
    ) -> LocalIKSolutionSet:
        """兼容保留：转交给独立数值求解器。"""
        from .numeric_solver import NumericIKSolver

        return NumericIKSolver(self).solve_ik_all(
            target_frame_tcp_T=target_frame_tcp_T,
            target_space=target_space,
            seed_joints_deg=seed_joints_deg,
            filter_lower_deg=filter_lower_deg,
            filter_upper_deg=filter_upper_deg,
            q2q3_seed_pairs_deg=q2q3_seed_pairs_deg,
            wrist_seed_triplets_deg=wrist_seed_triplets_deg,
            rotation_weight_mm=rotation_weight_mm,
            max_nfev=max_nfev,
            arm_position_tolerance_mm=arm_position_tolerance_mm,
            pose_position_tolerance_mm=pose_position_tolerance_mm,
            pose_orientation_tolerance_deg=pose_orientation_tolerance_deg,
            periodic_dedup_tolerance_deg=periodic_dedup_tolerance_deg,
        )


def pick_preferred_local_solution(
    solution_set: LocalIKSolutionSet,
) -> Optional[LocalIKSolution]:
    """从过滤后的本地解集中挑一组首选解。"""
    if not solution_set.filtered_solutions:
        return None
    return solution_set.filtered_solutions[0]


def format_pose(T: np.ndarray) -> str:
    """把位姿矩阵格式化成终端易读的文本。"""
    values = pose_to_xyz_zyx(T)
    return (
        f"  X={values[0]: .3f} mm\n"
        f"  Y={values[1]: .3f} mm\n"
        f"  Z={values[2]: .3f} mm\n"
        f"  Rz={values[3]: .3f} deg\n"
        f"  Ry={values[4]: .3f} deg\n"
        f"  Rx={values[5]: .3f} deg"
    )


def print_pose(name: str, T: np.ndarray) -> None:
    """打印带标题的位姿块。"""
    print(f"{name}:")
    print(format_pose(T))
    print()


def print_joint_vector(name: str, q_deg: Sequence[float]) -> None:
    """统一打印关节角向量。"""
    q = as_joint_vector(q_deg, name)
    print(f"{name}:")
    print(np.round(q, 6))
    print()


def _print_joint_limit_block(title: str, lower_deg: np.ndarray, upper_deg: np.ndarray) -> None:
    """打印关节范围块。"""
    print(title)
    for index, (lower_value, upper_value) in enumerate(zip(lower_deg, upper_deg), start=1):
        print(f"  J{index}: [{lower_value: .3f}, {upper_value: .3f}] deg")
    print()


def print_fk_report(robot: RobotModel, q_deg: Sequence[float]) -> None:
    """打印一组关节角对应的 FK 结果。"""
    q = as_joint_vector(q_deg, "q_deg")
    print_joint_vector("Joint angles (deg)", q)
    print_pose("Flange pose relative to robot base", robot.fk_flange(q))
    print_pose("TCP pose relative to robot base", robot.fk_tcp_in_robot_base(q))
    print_pose("TCP pose relative to Frame 2", robot.fk_tcp_in_frame(q))


def print_ik_report(robot: RobotModel, target_pose: np.ndarray, ik_result: IKResult) -> None:
    """打印单次数值 IK 的结果和回代误差。"""
    q_solution = ik_result.q_deg
    solved_pose = robot.fk_tcp_in_frame(q_solution)
    error_xyz_zyx = pose_to_xyz_zyx(solved_pose) - pose_to_xyz_zyx(target_pose)
    position_error_norm = float(np.linalg.norm(target_pose[:3, 3] - solved_pose[:3, 3]))
    orientation_error = rotation_error_deg(solved_pose, target_pose)

    print("=== IK Solver Status ===")
    print(f"success: {ik_result.success}")
    print(f"message: {ik_result.message}")
    print(f"nfev: {ik_result.nfev}")
    print(f"cost: {ik_result.cost:.12f}")
    print(f"residual norm: {np.linalg.norm(ik_result.residual):.12f}")
    print()

    print_joint_vector("IK solution (deg)", q_solution)
    print_pose("Target TCP pose relative to Frame 2", target_pose)
    print_pose("TCP pose from IK solution, relative to Frame 2", solved_pose)

    print("=== Pose error summary ===")
    print(f"position error norm: {position_error_norm:.9f} mm")
    print(f"orientation error angle: {orientation_error:.9f} deg")
    print("display-space difference [dX, dY, dZ, dRz, dRy, dRx]:")
    print(np.round(error_xyz_zyx, 9))
    print()

    print("Within joint limits:")
    print(robot.within_joint_limits(q_solution))
    print()


def print_local_ik_solution_set(
    solution_set: LocalIKSolutionSet,
    preferred_solution: Optional[LocalIKSolution] = None,
) -> None:
    """打印本地多解 IK 结果。"""
    print("=== Local IK Summary ===")
    print(f"target space: {solution_set.target_space}")
    print(f"arm candidate count: {len(solution_set.arm_candidate_solutions_deg)}")
    print(f"unique branch count (mod 360): {len(solution_set.branch_solutions)}")
    print(f"expanded solution count: {len(solution_set.all_solutions)}")
    print(f"filtered solution count: {len(solution_set.filtered_solutions)}")
    print()

    print_pose("Target TCP pose", solution_set.target_pose)
    if solution_set.seed_joints_deg is not None:
        print_joint_vector("IK sorting seed (deg)", solution_set.seed_joints_deg)

    _print_joint_limit_block(
        "Robot limits used for solving",
        solution_set.robot_lower_limits_deg,
        solution_set.robot_upper_limits_deg,
    )
    _print_joint_limit_block(
        "Filter limits used for solution selection",
        solution_set.filter_lower_limits_deg,
        solution_set.filter_upper_limits_deg,
    )

    print("=== Arm Candidates (J1-J3) ===")
    if not solution_set.arm_candidate_solutions_deg:
        print("No valid arm candidates were found.")
        print()
    else:
        for index, arm_solution in enumerate(solution_set.arm_candidate_solutions_deg):
            print(f"[A{index:02d}] {np.round(arm_solution, 6)}")
        print()

    print("=== Unique Branch Solutions (mod 360) ===")
    if not solution_set.branch_solutions:
        print("No branch solution converged.")
        print()
    else:
        for solution in solution_set.branch_solutions:
            preferred_mark = (
                " PREFERRED"
                if preferred_solution and solution.branch_index == preferred_solution.branch_index
                else ""
            )
            distance_text = (
                "n/a"
                if solution.distance_to_seed_deg is None
                else f"{solution.distance_to_seed_deg:.6f}"
            )
            print(
                f"[B{solution.branch_index:02d}]{preferred_mark} "
                f"dist_to_seed={distance_text} "
                f"pos_err={solution.position_error_mm:.9f} mm "
                f"ori_err={solution.orientation_error_deg:.9f} deg "
                f"residual={solution.residual_norm:.9f}"
            )
            print(np.round(solution.joints_deg, 6))
            print()

    kept_indices = {solution.index for solution in solution_set.filtered_solutions}
    print("=== Expanded IK Solutions ===")
    if not solution_set.all_solutions:
        print("No solution remains after turn expansion.")
        print()
    else:
        for solution in solution_set.all_solutions:
            status = "KEEP" if solution.index in kept_indices else "SKIP"
            preferred_mark = (
                " PREFERRED" if preferred_solution and solution.index == preferred_solution.index else ""
            )
            distance_text = (
                "n/a"
                if solution.distance_to_seed_deg is None
                else f"{solution.distance_to_seed_deg:.6f}"
            )
            print(
                f"[{solution.index:02d}] {status}{preferred_mark} "
                f"branch=B{solution.branch_index:02d} "
                f"turns={solution.turn_offsets.tolist()} "
                f"filter_limits={solution.within_filter_limits} "
                f"dist_to_seed={distance_text}"
            )
            print(np.round(solution.joints_deg, 6))
            print()
