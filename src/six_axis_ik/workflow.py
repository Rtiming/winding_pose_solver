"""
面向 `main.py` 的高层工作流模块。

目标：
1. 让 `main.py` 只保留用户最常改的运行参数。
2. 让 `python main.py` 直接执行一次完整任务。
3. 固定输出本地模型结果，并在可连接 RoboDK 时同时输出 RoboDK 对照结果。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Sequence

import numpy as np

from . import config
from .interface import SixAxisIKSolver
from .kinematics import (
    JOINT_COUNT,
    POSE_VECTOR_SIZE,
    LocalIKSolutionSet,
    as_joint_vector,
    make_pose_zyx,
    pick_preferred_local_solution,
    print_fk_report,
    print_joint_vector,
    print_local_ik_solution_set,
    print_pose,
)
from .robodk_bridge import (
    apply_configured_setup_to_robodk,
    connect_robot_from_settings,
    get_live_robot_state,
    pick_preferred_solution,
    print_backend_comparison,
    print_ik_solution_set,
    print_local_vs_robodk_ik_comparison,
    robodk_fk,
    robodk_solve_all_ik,
)

RunMode = Literal["fk", "ik"]
TargetSpace = Literal["frame", "base"]


@dataclass
class MainRunConfig:
    """
    `main.py` 使用的用户运行配置。

    这个对象只描述“当前这一轮你想跑什么”：
    - 跑 FK 还是 IK
    - 目标位姿 / 输入关节角
    - 参考坐标系
    - 本地 IK 初值
    - 是否同时连 RoboDK 对照
    """

    run_mode: RunMode
    target_space: TargetSpace
    fk_joints_deg: Sequence[float]
    ik_target_pose_xyz_rzyx_mm_deg: Sequence[float]
    ik_seed_joints_deg: Optional[Sequence[float]]
    compare_with_robodk: bool
    apply_config_to_robodk_before_run: bool
    local_rotation_weight_mm: float
    local_max_nfev: int
    ik_backend: str | None = "numeric"


def print_readme_hint() -> None:
    """在运行开始前提醒用户查看 README。"""
    print(config.README_HINT)
    print()


def _as_pose_vector(values: Sequence[float], name: str) -> np.ndarray:
    """把 6 维位姿向量整理成 numpy 向量。"""
    pose_vector = np.asarray(list(values), dtype=float).reshape(-1)
    if pose_vector.size != POSE_VECTOR_SIZE:
        raise ValueError(f"{name} must contain {POSE_VECTOR_SIZE} values, got {pose_vector.size}.")
    return pose_vector


def _pose_from_xyz_rzyx(values: Sequence[float], name: str) -> np.ndarray:
    """把 `[X, Y, Z, Rz, Ry, Rx]` 转成 4x4 齐次矩阵。"""
    pose_vector = _as_pose_vector(values, name)
    return make_pose_zyx(*pose_vector.tolist())


def _target_pose_for_local_backend(local_robot, target_pose: np.ndarray, space: str) -> np.ndarray:
    """把用户输入的目标位姿统一转换成 local backend 需要的 Frame 空间。"""
    if space == "frame":
        return target_pose
    return local_robot.frame_T_inv @ target_pose


def _resolve_seed_joints(seed_joints_deg: Optional[Sequence[float]]) -> np.ndarray:
    """把 main 里的可选初值整理成最终的 6 轴关节角。"""
    if seed_joints_deg is None:
        return config.get_default_seed_joints()
    return as_joint_vector(seed_joints_deg, "ik_seed_joints_deg")


def _connect_live_setup(apply_config_first: bool):
    """
    尝试连接 RoboDK 并读取实时状态。

    如果用户要求，可以先把 `config.py` 中的 Tool / Frame / 限位推送到 RoboDK。
    """
    live_robot = connect_robot_from_settings()
    if apply_config_first:
        apply_configured_setup_to_robodk(
            live_robot,
            tool_pose=config.get_configured_tool_pose(),
            frame_pose=config.get_configured_frame_pose(),
            lower_limits_deg=config.get_configured_lower_limits_deg(),
            upper_limits_deg=config.get_configured_upper_limits_deg(),
        )
    return live_robot, get_live_robot_state(live_robot)


def print_main_run_config(run_config: MainRunConfig) -> None:
    """把当前 `main.py` 的运行参数打印出来，便于确认。"""
    print("=== Main Run Config ===")
    print(f"run mode: {run_config.run_mode}")
    print(f"target space: {run_config.target_space}")
    print(f"compare with RoboDK: {run_config.compare_with_robodk}")
    print(f"apply config to RoboDK before run: {run_config.apply_config_to_robodk_before_run}")
    print(f"local rotation weight: {run_config.local_rotation_weight_mm}")
    print(f"local max_nfev: {run_config.local_max_nfev}")
    print(f"local IK backend: {run_config.ik_backend}")
    print()

    print_joint_vector("FK input joints (deg)", run_config.fk_joints_deg)
    print("IK target pose vector [X, Y, Z, Rz, Ry, Rx]:")
    print(np.round(_as_pose_vector(run_config.ik_target_pose_xyz_rzyx_mm_deg, "ik_target_pose"), 6))
    print()

    if run_config.ik_seed_joints_deg is None:
        print("IK seed joints (deg):")
        print("None -> use default zeros")
        print()
    else:
        print_joint_vector("IK seed joints (deg)", run_config.ik_seed_joints_deg)


def _run_fk_comparison(run_config: MainRunConfig) -> None:
    """执行一轮 FK，并在可能时附带 RoboDK 对照。"""
    local_robot = config.build_local_robot_model()
    fk_joints_deg = as_joint_vector(run_config.fk_joints_deg, "fk_joints_deg")

    print("=== Local FK ===")
    if run_config.target_space == "frame":
        print_fk_report(local_robot, fk_joints_deg)
    else:
        print_joint_vector("Joint angles (deg)", fk_joints_deg)
        print_pose("TCP pose relative to Base", local_robot.fk_tcp_in_robot_base(fk_joints_deg))

    if not run_config.compare_with_robodk:
        return

    try:
        live_robot, live_state = _connect_live_setup(
            apply_config_first=run_config.apply_config_to_robodk_before_run
        )
    except Exception as error:
        print("=== RoboDK FK Comparison ===")
        print("RoboDK comparison was skipped.")
        print(f"reason: {error}")
        print()
        return

    print("=== RoboDK FK ===")
    robodk_pose = robodk_fk(
        live_robot,
        fk_joints_deg,
        tool_pose=live_state.tool_pose,
        frame_pose=live_state.frame_pose if run_config.target_space == "frame" else None,
        target_space=run_config.target_space,
    )
    print_pose(f"TCP pose from RoboDK, relative to {run_config.target_space.title()}", robodk_pose)

    print_backend_comparison(
        local_robot=local_robot,
        live_robot=live_robot,
        live_state=live_state,
        joints_deg=fk_joints_deg,
        target_space=run_config.target_space,
    )


def _solve_local_ik(run_config: MainRunConfig) -> LocalIKSolutionSet:
    """执行本地多解 IK，并返回结构化结果。"""
    solver = SixAxisIKSolver.from_config(ik_backend=run_config.ik_backend)
    target_pose = _pose_from_xyz_rzyx(run_config.ik_target_pose_xyz_rzyx_mm_deg, "ik_target_pose")
    seed_joints_deg = _resolve_seed_joints(run_config.ik_seed_joints_deg)

    result = solver.solve_ik_all(
        target_pose=target_pose,
        target_space=run_config.target_space,
        seed_joints_deg=seed_joints_deg,
        filter_lower_deg=config.get_filter_lower_limits_deg(),
        filter_upper_deg=config.get_filter_upper_limits_deg(),
        rotation_weight_mm=run_config.local_rotation_weight_mm,
        max_nfev=run_config.local_max_nfev,
    )
    return result.to_local_solution_set()


def _run_ik_comparison(run_config: MainRunConfig) -> None:
    """执行一轮 IK，并在可能时附带 RoboDK 对照。"""
    local_solution_set = _solve_local_ik(run_config)
    preferred_local_solution = pick_preferred_local_solution(local_solution_set)

    print("=== Local IK ===")
    print_local_ik_solution_set(local_solution_set, preferred_solution=preferred_local_solution)

    if not run_config.compare_with_robodk:
        return

    try:
        live_robot, live_state = _connect_live_setup(
            apply_config_first=run_config.apply_config_to_robodk_before_run
        )
    except Exception as error:
        print("=== RoboDK IK Comparison ===")
        print("RoboDK comparison was skipped.")
        print(f"reason: {error}")
        print()
        return

    target_pose = _pose_from_xyz_rzyx(run_config.ik_target_pose_xyz_rzyx_mm_deg, "ik_target_pose")
    seed_joints_deg = _resolve_seed_joints(run_config.ik_seed_joints_deg)
    robodk_solution_set = robodk_solve_all_ik(
        live_robot,
        target_pose=target_pose,
        target_space=run_config.target_space,
        tool_pose=live_state.tool_pose,
        frame_pose=live_state.frame_pose,
        filter_lower_deg=config.get_filter_lower_limits_deg(),
        filter_upper_deg=config.get_filter_upper_limits_deg(),
        seed_joints_deg=seed_joints_deg,
        keep_extra_values=config.KEEP_EXTRA_ROBODK_COLUMNS,
    )
    preferred_robodk_solution = pick_preferred_solution(robodk_solution_set)

    print("=== RoboDK IK ===")
    print_ik_solution_set(robodk_solution_set, preferred_solution=preferred_robodk_solution)

    print_local_vs_robodk_ik_comparison(
        local_solution_set=local_solution_set,
        robodk_solution_set=robodk_solution_set,
        periodic_dedup_tolerance_deg=config.LOCAL_IK_PERIODIC_DEDUP_TOLERANCE_DEG,
    )


def run_main_case(run_config: MainRunConfig) -> None:
    """
    执行 `main.py` 指定的一轮任务。

    行为约定：
    - `run_mode == "fk"`: 输出本地 FK，并尽量输出 RoboDK FK 与差异。
    - `run_mode == "ik"`: 输出本地多解 IK，并尽量输出 RoboDK 解集与差异。
    """
    print_readme_hint()
    print_main_run_config(run_config)

    if run_config.run_mode == "fk":
        _run_fk_comparison(run_config)
        return

    if run_config.run_mode == "ik":
        _run_ik_comparison(run_config)
        return

    raise ValueError(f"Unsupported run_mode: {run_config.run_mode}")
