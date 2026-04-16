"""
RoboDK 联调模块。

这个模块的职责非常明确：
1. 连接 RoboDK。
2. 读取当前机器人真实状态。
3. 调用 RoboDK 自带的 FK / IK / SolveIK_All。
4. 把 RoboDK 返回值转换成项目内部统一的数据结构。
5. 按配置文件里的关节范围筛选解，并打印调试信息。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np

from . import config
from .kinematics import (
    JOINT_COUNT,
    LocalIKSolutionSet,
    RobotModel,
    as_joint_vector,
    as_transform,
    deduplicate_joint_vectors_periodic,
    joint_distance_deg,
    pose_to_xyz_zyx,
    print_joint_vector,
    print_pose,
    rotation_error_deg,
)

try:
    from robodk.robolink import ITEM_TYPE_ROBOT, Robolink
    from robodk.robomath import Mat
except ImportError:  # pragma: no cover
    ITEM_TYPE_ROBOT = None
    Robolink = None
    Mat = None


@dataclass
class JointConfiguration:
    """
    RoboDK 的关节构型标志。
    """

    rear: int
    lower: int
    flip: int
    turns: int

    def short_label(self) -> str:
        """生成简短配置标签，方便终端直接看。"""
        rear_label = "Rear" if self.rear else "Front"
        lower_label = "Lower" if self.lower else "Upper"
        flip_label = "Flip" if self.flip else "NonFlip"
        return f"{rear_label}/{lower_label}/{flip_label}/Turns={self.turns}"


@dataclass
class RoboDKJointSolution:
    """
    一组 RoboDK IK 解。
    """

    index: int
    joints_deg: np.ndarray
    extra_values: np.ndarray
    configuration: JointConfiguration
    within_robot_limits: bool
    within_filter_limits: bool
    distance_to_seed_deg: Optional[float]


@dataclass
class RoboDKIKSolutionSet:
    """
    RoboDK SolveIK_All 的结构化结果。
    """

    target_pose: np.ndarray
    target_space: str
    seed_joints_deg: Optional[np.ndarray]
    robot_lower_limits_deg: np.ndarray
    robot_upper_limits_deg: np.ndarray
    filter_lower_limits_deg: np.ndarray
    filter_upper_limits_deg: np.ndarray
    all_solutions: List[RoboDKJointSolution]
    filtered_solutions: List[RoboDKJointSolution]
    dof_count: int
    extra_value_count: int


@dataclass
class RoboDKLiveRobotState:
    """
    当前 RoboDK 机器人的实时状态快照。
    """

    robot_name: str
    current_joints_deg: np.ndarray
    tool_pose: np.ndarray
    frame_pose: np.ndarray
    lower_limits_deg: np.ndarray
    upper_limits_deg: np.ndarray


def _ensure_robodk_available() -> None:
    """检查 RoboDK Python API 是否可用。"""
    if Robolink is None or Mat is None:
        raise RuntimeError(
            "RoboDK Python API is not available. Install it with: pip install robodk"
        )


def _contains_joints(q_deg: Sequence[float], lower_deg: np.ndarray, upper_deg: np.ndarray) -> bool:
    """判断一组关节角是否落在指定范围内。"""
    q = as_joint_vector(q_deg)
    return bool(np.all(q >= lower_deg) and np.all(q <= upper_deg))


def _numpy_pose_to_robodk(T: np.ndarray) -> Mat:
    """把 numpy 4x4 变换矩阵转换成 RoboDK 的 `Mat`。"""
    T = as_transform(T)
    return Mat(T.tolist())


def _robodk_pose_to_numpy(T: Mat) -> np.ndarray:
    """把 RoboDK 的 `Mat` 转成 numpy 4x4 变换矩阵。"""
    matrix = np.array(
        [
            [T[row_index, col_index] for col_index in range(T.size(1))]
            for row_index in range(T.size(0))
        ],
        dtype=float,
    )
    return as_transform(matrix)


def _solution_matrix_to_list(solutions_mat: Mat, dof_count: int) -> List[np.ndarray]:
    """
    把 RoboDK 的二维解矩阵拆成逐列的解列表。
    """
    row_count = solutions_mat.size(0)
    col_count = solutions_mat.size(1)

    if row_count == 0 or col_count == 0:
        return []

    solutions: List[np.ndarray] = []
    for col_index in range(col_count):
        column_values = np.array(
            [solutions_mat[row_index, col_index] for row_index in range(row_count)],
            dtype=float,
        )
        if column_values.size < dof_count:
            continue
        solutions.append(column_values)

    return solutions


def _config_from_robodk_flags(flags: Sequence[float]) -> JointConfiguration:
    """把 RoboDK 的配置标志数组转成结构化对象。"""
    ints = [int(round(value)) for value in list(flags)[:4]]
    while len(ints) < 4:
        ints.append(0)
    return JointConfiguration(rear=ints[0], lower=ints[1], flip=ints[2], turns=ints[3])


def connect_robot_from_settings():
    """
    连接当前 RoboDK，并取回配置文件里指定名称的机器人。
    """
    _ensure_robodk_available()

    if config.ROBODK_PORT is None:
        rdk = Robolink(robodk_ip=config.ROBODK_HOST)
    else:
        rdk = Robolink(robodk_ip=config.ROBODK_HOST, port=config.ROBODK_PORT)

    if config.ROBODK_ROBOT_NAME:
        robot = rdk.Item(config.ROBODK_ROBOT_NAME, ITEM_TYPE_ROBOT)
    else:
        robots = rdk.ItemList(filter=ITEM_TYPE_ROBOT)
        if not robots:
            raise RuntimeError("No robot found in the current RoboDK station.")
        robot = robots[0]

    if not robot.Valid():
        raise RuntimeError(
            f"Robot '{config.ROBODK_ROBOT_NAME}' not found in the current RoboDK station."
        )

    return robot


def get_live_robot_state(robot) -> RoboDKLiveRobotState:
    """
    读取当前 RoboDK 机器人的实时状态。
    """
    current_joints = np.array(robot.Joints().list()[:JOINT_COUNT], dtype=float)
    tool_pose = _robodk_pose_to_numpy(robot.PoseTool())
    frame_pose = _robodk_pose_to_numpy(robot.PoseFrame())

    lower_limits_mat, upper_limits_mat, _ = robot.JointLimits()
    lower_limits = np.array(lower_limits_mat.list()[:JOINT_COUNT], dtype=float)
    upper_limits = np.array(upper_limits_mat.list()[:JOINT_COUNT], dtype=float)

    return RoboDKLiveRobotState(
        robot_name=robot.Name(),
        current_joints_deg=current_joints,
        tool_pose=tool_pose,
        frame_pose=frame_pose,
        lower_limits_deg=lower_limits,
        upper_limits_deg=upper_limits,
    )


def robodk_fk(
    robot,
    joints_deg: Sequence[float],
    tool_pose: Optional[np.ndarray],
    frame_pose: Optional[np.ndarray],
    target_space: str,
) -> np.ndarray:
    """
    调用 RoboDK 求 FK。
    """
    joints = as_joint_vector(joints_deg).tolist()

    tool = _numpy_pose_to_robodk(tool_pose) if tool_pose is not None else None
    reference = None
    if target_space == "frame" and frame_pose is not None:
        reference = _numpy_pose_to_robodk(frame_pose)

    pose = robot.SolveFK(joints, tool=tool, reference=reference)
    return _robodk_pose_to_numpy(pose)


def robodk_solve_all_ik(
    robot,
    target_pose: np.ndarray,
    target_space: str,
    tool_pose: np.ndarray,
    frame_pose: np.ndarray,
    filter_lower_deg: np.ndarray,
    filter_upper_deg: np.ndarray,
    seed_joints_deg: Optional[Sequence[float]],
    keep_extra_values: bool,
) -> RoboDKIKSolutionSet:
    """
    调用 RoboDK 的 `SolveIK_All`，并对所有解做结构化整理与范围过滤。
    """
    live_state = get_live_robot_state(robot)
    dof_count = live_state.current_joints_deg.size

    tool = _numpy_pose_to_robodk(tool_pose)
    reference = _numpy_pose_to_robodk(frame_pose) if target_space == "frame" else None
    pose = _numpy_pose_to_robodk(target_pose)

    seed = None if seed_joints_deg is None else as_joint_vector(seed_joints_deg, "seed_joints_deg")

    solutions_mat = robot.SolveIK_All(pose, tool=tool, reference=reference)
    raw_solutions = _solution_matrix_to_list(solutions_mat, dof_count=dof_count)

    all_solutions: List[RoboDKJointSolution] = []
    for solution_index, raw_solution in enumerate(raw_solutions):
        joints_deg = raw_solution[:dof_count]
        extra_values = raw_solution[dof_count:] if keep_extra_values else np.array([], dtype=float)
        config_flags = robot.JointsConfig(joints_deg.tolist()).list()
        configuration = _config_from_robodk_flags(config_flags)

        within_robot_limits = _contains_joints(
            joints_deg, live_state.lower_limits_deg, live_state.upper_limits_deg
        )
        within_filter_limits = _contains_joints(joints_deg, filter_lower_deg, filter_upper_deg)

        distance = None if seed is None else joint_distance_deg(joints_deg, seed)

        all_solutions.append(
            RoboDKJointSolution(
                index=solution_index,
                joints_deg=joints_deg,
                extra_values=extra_values,
                configuration=configuration,
                within_robot_limits=within_robot_limits,
                within_filter_limits=within_filter_limits,
                distance_to_seed_deg=distance,
            )
        )

    filtered_solutions = [
        solution
        for solution in all_solutions
        if solution.within_robot_limits and solution.within_filter_limits
    ]

    filtered_solutions = sorted(
        filtered_solutions,
        key=lambda item: (
            float("inf") if item.distance_to_seed_deg is None else item.distance_to_seed_deg,
            item.index,
        ),
    )
    all_solutions = sorted(
        all_solutions,
        key=lambda item: (
            float("inf") if item.distance_to_seed_deg is None else item.distance_to_seed_deg,
            item.index,
        ),
    )

    extra_count = 0 if not raw_solutions else max(0, raw_solutions[0].size - dof_count)

    return RoboDKIKSolutionSet(
        target_pose=as_transform(target_pose, "target_pose"),
        target_space=target_space,
        seed_joints_deg=seed,
        robot_lower_limits_deg=live_state.lower_limits_deg,
        robot_upper_limits_deg=live_state.upper_limits_deg,
        filter_lower_limits_deg=np.asarray(filter_lower_deg, dtype=float),
        filter_upper_limits_deg=np.asarray(filter_upper_deg, dtype=float),
        all_solutions=all_solutions,
        filtered_solutions=filtered_solutions,
        dof_count=dof_count,
        extra_value_count=extra_count,
    )


def pick_preferred_solution(solution_set: RoboDKIKSolutionSet) -> Optional[RoboDKJointSolution]:
    """
    从过滤后的解集中挑出一组“首选解”。
    """
    if not solution_set.filtered_solutions:
        return None
    return solution_set.filtered_solutions[0]


def apply_configured_setup_to_robodk(
    robot,
    tool_pose: np.ndarray,
    frame_pose: np.ndarray,
    lower_limits_deg: np.ndarray,
    upper_limits_deg: np.ndarray,
) -> None:
    """
    把配置文件中的 Tool / Frame / Joint Limits 应用到当前 RoboDK 机器人。
    """
    robot.setPoseTool(_numpy_pose_to_robodk(tool_pose))
    robot.setPoseFrame(_numpy_pose_to_robodk(frame_pose))
    robot.setJointLimits(lower_limits_deg.tolist(), upper_limits_deg.tolist())


def _print_joint_limit_block(title: str, lower_deg: np.ndarray, upper_deg: np.ndarray) -> None:
    """打印关节范围。"""
    print(title)
    for index, (lower_value, upper_value) in enumerate(zip(lower_deg, upper_deg), start=1):
        print(f"  J{index}: [{lower_value: .3f}, {upper_value: .3f}] deg")
    print()


def print_live_robot_state(live_state: RoboDKLiveRobotState) -> None:
    """
    打印 RoboDK 当前机器人的实时状态，同时附带配置文件里的对应值。
    """
    print(f"Live RoboDK robot: {live_state.robot_name}")
    print()
    print_joint_vector("Current joints in RoboDK (deg)", live_state.current_joints_deg)
    print_pose("Live Tool pose from RoboDK", live_state.tool_pose)
    print_pose("Live Frame pose from RoboDK", live_state.frame_pose)
    _print_joint_limit_block(
        "Live joint limits from RoboDK",
        live_state.lower_limits_deg,
        live_state.upper_limits_deg,
    )

    print_pose("Configured Tool pose", config.get_configured_tool_pose())
    print_pose("Configured Frame pose", config.get_configured_frame_pose())
    _print_joint_limit_block(
        "Configured joint limits",
        config.get_configured_lower_limits_deg(),
        config.get_configured_upper_limits_deg(),
    )
    _print_joint_limit_block(
        "Configured IK filter limits",
        config.get_filter_lower_limits_deg(),
        config.get_filter_upper_limits_deg(),
    )


def print_ik_solution_set(
    solution_set: RoboDKIKSolutionSet,
    preferred_solution: Optional[RoboDKJointSolution] = None,
) -> None:
    """
    打印 RoboDK 全解 IK 结果。
    """
    print("=== RoboDK IK Summary ===")
    print(f"target space: {solution_set.target_space}")
    print(f"raw solution count: {len(solution_set.all_solutions)}")
    print(f"filtered solution count: {len(solution_set.filtered_solutions)}")
    print(f"robot dof count: {solution_set.dof_count}")
    print(f"extra values per solution: {solution_set.extra_value_count}")
    print()

    print_pose("Target TCP pose", solution_set.target_pose)
    if solution_set.seed_joints_deg is not None:
        print_joint_vector("IK sorting seed (deg)", solution_set.seed_joints_deg)

    _print_joint_limit_block(
        "Robot limits used for validation",
        solution_set.robot_lower_limits_deg,
        solution_set.robot_upper_limits_deg,
    )
    _print_joint_limit_block(
        "Filter limits used for solution selection",
        solution_set.filter_lower_limits_deg,
        solution_set.filter_upper_limits_deg,
    )

    kept_indices = {solution.index for solution in solution_set.filtered_solutions}

    print("=== All RoboDK IK solutions ===")
    if not solution_set.all_solutions:
        print("No solution returned by RoboDK.")
        print()
    else:
        for solution in solution_set.all_solutions:
            status = "KEEP" if solution.index in kept_indices else "SKIP"
            distance_text = (
                "n/a"
                if solution.distance_to_seed_deg is None
                else f"{solution.distance_to_seed_deg:.6f}"
            )
            print(
                f"[{solution.index:02d}] {status} "
                f"cfg={solution.configuration.short_label()} "
                f"robot_limits={solution.within_robot_limits} "
                f"filter_limits={solution.within_filter_limits} "
                f"dist_to_seed={distance_text}"
            )
            print(np.round(solution.joints_deg, 6))
            if solution.extra_values.size > 0:
                print(f"extra values: {np.round(solution.extra_values, 6)}")
            print()

    print("=== Filtered Solutions ===")
    if not solution_set.filtered_solutions:
        print("No solution remains after applying robot limits and configured filter limits.")
        print()
    else:
        for rank, solution in enumerate(solution_set.filtered_solutions, start=1):
            print(
                f"rank={rank:02d} source_index={solution.index:02d} "
                f"cfg={solution.configuration.short_label()}"
            )
            print(np.round(solution.joints_deg, 6))
            if solution.extra_values.size > 0:
                print(f"extra values: {np.round(solution.extra_values, 6)}")
            print()

    print("=== Preferred Solution ===")
    if preferred_solution is None:
        print("No preferred solution available.")
        print()
    else:
        print(f"source_index={preferred_solution.index:02d}")
        print(f"configuration={preferred_solution.configuration.short_label()}")
        print(np.round(preferred_solution.joints_deg, 6))
        if preferred_solution.extra_values.size > 0:
            print(f"extra values: {np.round(preferred_solution.extra_values, 6)}")
        print()


def print_backend_comparison(
    local_robot: RobotModel,
    live_robot,
    live_state: RoboDKLiveRobotState,
    joints_deg: Sequence[float],
    target_space: str,
) -> None:
    """
    比较本地实验模型和 RoboDK 真值 FK 的差异。
    """
    q_deg = as_joint_vector(joints_deg, "joints_deg")
    if target_space == "frame":
        local_pose = local_robot.fk_tcp_in_frame(q_deg)
        robodk_pose = robodk_fk(
            live_robot,
            q_deg,
            tool_pose=live_state.tool_pose,
            frame_pose=live_state.frame_pose,
            target_space="frame",
        )
    else:
        local_pose = local_robot.fk_tcp_in_robot_base(q_deg)
        robodk_pose = robodk_fk(
            live_robot,
            q_deg,
            tool_pose=live_state.tool_pose,
            frame_pose=None,
            target_space="base",
        )

    pose_difference = pose_to_xyz_zyx(local_pose) - pose_to_xyz_zyx(robodk_pose)
    position_error_norm = float(np.linalg.norm(local_pose[:3, 3] - robodk_pose[:3, 3]))
    orientation_error = rotation_error_deg(local_pose, robodk_pose)

    print_joint_vector("Compared joints (deg)", q_deg)
    print_pose("Local model pose", local_pose)
    print_pose("RoboDK pose", robodk_pose)

    print("=== Local vs RoboDK FK Difference ===")
    print(f"target space: {target_space}")
    print(f"position error norm: {position_error_norm:.9f} mm")
    print(f"orientation error angle: {orientation_error:.9f} deg")
    print("display-space difference [dX, dY, dZ, dRz, dRy, dRx]:")
    print(np.round(pose_difference, 9))
    print()


def get_unique_filtered_robodk_joint_vectors(
    solution_set: RoboDKIKSolutionSet,
    periodic_dedup_tolerance_deg: float,
) -> List[np.ndarray]:
    """
    把 RoboDK 过滤后的解按 360 度周期去重。

    RoboDK 往往会把 `60 deg` 和 `-300 deg` 这种多圈等价解都列出来。
    本地解算器内部会先求“唯一分支”，再按硬限位展开圈数。
    因此做对照时，最好同时看：
    - 原始过滤后解数量
    - 周期去重后的唯一分支数量
    """
    return deduplicate_joint_vectors_periodic(
        [solution.joints_deg for solution in solution_set.filtered_solutions],
        tolerance_deg=periodic_dedup_tolerance_deg,
    )


def print_local_vs_robodk_ik_comparison(
    local_solution_set: LocalIKSolutionSet,
    robodk_solution_set: RoboDKIKSolutionSet,
    periodic_dedup_tolerance_deg: float,
) -> None:
    """
    对比本地 `ik-all` 和 RoboDK `SolveIK_All` 的结果。

    这个对比会同时显示：
    - 本地展开后的最终解数量
    - RoboDK 过滤后的原始解数量
    - 双方按 360 度周期去重后的唯一分支数量
    - 是否存在本地缺失解 / 额外解
    """
    local_unique = deduplicate_joint_vectors_periodic(
        [solution.joints_deg for solution in local_solution_set.filtered_solutions],
        tolerance_deg=periodic_dedup_tolerance_deg,
    )
    robodk_unique = get_unique_filtered_robodk_joint_vectors(
        robodk_solution_set,
        periodic_dedup_tolerance_deg=periodic_dedup_tolerance_deg,
    )

    print("=== Local vs RoboDK IK Comparison ===")
    print(f"local filtered solutions: {len(local_solution_set.filtered_solutions)}")
    print(f"robodk filtered solutions: {len(robodk_solution_set.filtered_solutions)}")
    print(f"local unique branches (mod 360): {len(local_unique)}")
    print(f"robodk unique branches (mod 360): {len(robodk_unique)}")
    print()

    missing_in_local = []
    for robodk_vector in robodk_unique:
        if not any(
            joint_distance_deg(robodk_vector, local_vector) <= periodic_dedup_tolerance_deg
            for local_vector in local_unique
        ):
            missing_in_local.append(robodk_vector)

    extra_in_local = []
    for local_vector in local_unique:
        if not any(
            joint_distance_deg(local_vector, robodk_vector) <= periodic_dedup_tolerance_deg
            for robodk_vector in robodk_unique
        ):
            extra_in_local.append(local_vector)

    print("Missing local branches compared with RoboDK:")
    if not missing_in_local:
        print("  None")
    else:
        for vector in missing_in_local:
            print(f"  {np.round(vector, 6)}")
    print()

    print("Extra local branches not found in RoboDK:")
    if not extra_in_local:
        print("  None")
    else:
        for vector in extra_in_local:
            print(f"  {np.round(vector, 6)}")
    print()
