"""
SixAxisIK 核心包。

主要分三层：
1. `config`
   用户主要修改的参数集中地。
2. `kinematics`
   本地 POE 运动学、本地 IK、多解枚举与终端调试输出。
3. `robodk_bridge`
   RoboDK 通信、实时状态读取和校验工具。
"""

from . import config
from .backends import (
    AnalyticIKBackend,
    IKBackend,
    IKSolveAllRequest,
    IKSolveRequest,
    NumericIKBackend,
    build_ik_backend,
)
from .interface import IKDiagnostics, IKSolutionRecord, IKSolveResult, SixAxisIKSolver
from .kinematics import (
    IKResult,
    JOINT_COUNT,
    POSE_VECTOR_SIZE,
    LocalIKSolution,
    LocalIKSolutionSet,
    RobotModel,
    as_joint_vector,
    deduplicate_joint_vectors_periodic,
    format_pose,
    joint_distance_deg,
    make_pose_zyx,
    pick_preferred_local_solution,
    pose_to_xyz_zyx,
    print_fk_report,
    print_ik_report,
    print_joint_vector,
    print_local_ik_solution_set,
    print_pose,
)
from .robodk_bridge import (
    JointConfiguration,
    RoboDKIKSolutionSet,
    RoboDKJointSolution,
    RoboDKLiveRobotState,
)
from .workflow import MainRunConfig, run_main_case

__all__ = [
    "config",
    "AnalyticIKBackend",
    "IKBackend",
    "IKDiagnostics",
    "IKResult",
    "IKSolveAllRequest",
    "IKSolveRequest",
    "IKSolutionRecord",
    "IKSolveResult",
    "JOINT_COUNT",
    "POSE_VECTOR_SIZE",
    "LocalIKSolution",
    "LocalIKSolutionSet",
    "RobotModel",
    "JointConfiguration",
    "RoboDKIKSolutionSet",
    "RoboDKJointSolution",
    "RoboDKLiveRobotState",
    "MainRunConfig",
    "NumericIKBackend",
    "SixAxisIKSolver",
    "as_joint_vector",
    "build_ik_backend",
    "deduplicate_joint_vectors_periodic",
    "format_pose",
    "joint_distance_deg",
    "make_pose_zyx",
    "pick_preferred_local_solution",
    "pose_to_xyz_zyx",
    "print_fk_report",
    "print_ik_report",
    "print_joint_vector",
    "print_local_ik_solution_set",
    "print_pose",
    "run_main_case",
]
