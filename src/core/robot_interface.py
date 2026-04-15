from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence


class _ListWrapper:
    __slots__ = ("_values",)

    def __init__(self, values: Sequence[float]) -> None:
        self._values: list[float] = [float(value) for value in values]

    def list(self) -> list[float]:
        return list(self._values)

    def __iter__(self):
        return iter(self._values)

    def __len__(self) -> int:
        return len(self._values)

    def __getitem__(self, index):
        return self._values[index]


@dataclass(frozen=True)
class _JointGeometryCacheEntry:
    config_flags: tuple[int, int, int]
    arm_singularity_measure: float


_JOINT_GEOMETRY_CACHE_MAX_SIZE = 200000


class RoboDKRobotInterface:
    ik_seed_invariant = False

    def __init__(self, robot: Any) -> None:
        self._robot = robot

    def __getattr__(self, name: str) -> Any:
        return getattr(self._robot, name)

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


class SixAxisIKRobotInterface:
    ik_seed_invariant = True

    def __init__(self, robodk_robot: Any = None) -> None:
        from src.six_axis_ik import config as six_axis_config
        from src.six_axis_ik.interface import SixAxisIKSolver

        self._robodk_robot = robodk_robot
        self._solver = SixAxisIKSolver.from_config(ik_backend="pure_analytic")
        self._model = self._solver.robot_model
        self._config = six_axis_config
        self._cached_all_solutions: list[list[float]] = []
        self._cached_pose_key: tuple | None = None
        self._cached_branch_ids: dict[tuple[float, ...], tuple[int, ...]] = {}
        self._arm_link_2_home_angle_deg = _arm_link_2_home_angle_deg(self._model)
        self._joint_geometry_cache: dict[
            tuple[float, ...],
            _JointGeometryCacheEntry,
        ] = {}

    def JointLimits(self):
        lower = list(self._model.joint_min_deg)
        upper = list(self._model.joint_max_deg)
        return _ListWrapper(lower), _ListWrapper(upper), None

    def Joints(self):
        if self._robodk_robot is not None:
            try:
                return self._robodk_robot.Joints()
            except Exception:
                pass
        return _ListWrapper([0.0] * 6)

    def JointsHome(self):
        return _ListWrapper([0.0] * 6)

    def _pose_cache_key(self, pose: Any) -> tuple:
        try:
            rows = pose.rows
            return tuple(rows[row_index][column_index] for row_index in range(4) for column_index in range(4))
        except Exception:
            return (id(pose),)

    def SolveIK_All(self, pose: Any, tool_pose: Any, reference_pose: Any) -> list[list[float]]:
        self._cached_pose_key = self._pose_cache_key(pose)
        result = self._solver.solve_ik_all_joint_records(
            pose,
            target_space="frame",
            tool_pose=tool_pose,
            reference_pose=reference_pose,
        )
        self._cached_branch_ids = {
            tuple(float(value) for value in solution.joints_deg.tolist()): solution.branch_id
            for solution in result
        }
        self._cached_all_solutions = [solution.joints_deg.tolist() for solution in result]
        return list(self._cached_all_solutions)

    def SolveIK(self, pose: Any, seed: list[float], tool_pose: Any, reference_pose: Any) -> list[float]:
        key = self._pose_cache_key(pose)
        if key == self._cached_pose_key:
            if not self._cached_all_solutions:
                return []
            seed_vector = seed if isinstance(seed, list) else list(seed)
            return min(
                self._cached_all_solutions,
                key=lambda solution: sum((joint - seed_value) ** 2 for joint, seed_value in zip(solution, seed_vector)),
            )

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

    def JointsConfig(self, joints_list: list[float]) -> _ListWrapper:
        return _ListWrapper(list(self.JointGeometryMetrics(joints_list).config_flags))

    def JointBranchId(self, joints_list: Sequence[float]) -> tuple[int, ...] | None:
        cache_key = tuple(float(value) for value in joints_list)
        return self._cached_branch_ids.get(cache_key)

    def JointPoses(self, joints_list: list[float]) -> list:
        return _compute_link_poses(joints_list, self._model)

    def JointGeometryMetrics(
        self,
        joints_list: Sequence[float],
    ) -> _JointGeometryCacheEntry:
        cache_key = tuple(float(value) for value in joints_list)
        cached_entry = self._joint_geometry_cache.get(cache_key)
        if cached_entry is not None:
            return cached_entry

        metrics = _compute_joint_geometry_cache_entry(
            cache_key,
            self._model,
            arm_link_2_home_angle_deg=self._arm_link_2_home_angle_deg,
        )
        if len(self._joint_geometry_cache) >= _JOINT_GEOMETRY_CACHE_MAX_SIZE:
            self._joint_geometry_cache.pop(next(iter(self._joint_geometry_cache)))
        self._joint_geometry_cache[cache_key] = metrics
        return metrics

    def setJoints(self, joints_list) -> None:
        if self._robodk_robot is not None:
            self._robodk_robot.setJoints(joints_list)

    def __getattr__(self, name: str) -> Any:
        if self._robodk_robot is not None:
            return getattr(self._robodk_robot, name)
        raise AttributeError(
            f"SixAxisIKRobotInterface has no attribute '{name}' "
            "and no underlying RoboDK robot to delegate to."
        )


def build_robot_interface(
    ik_backend: str,
    robodk_robot: Any,
) -> RoboDKRobotInterface | SixAxisIKRobotInterface:
    if ik_backend == "six_axis_ik":
        print("[IK backend] Using SixAxisIK local solver.")
        return SixAxisIKRobotInterface(robodk_robot=robodk_robot)
    if ik_backend == "robodk":
        return RoboDKRobotInterface(robodk_robot)
    raise ValueError(
        f"Unknown IK backend '{ik_backend}'. Valid values: 'robodk', 'six_axis_ik'."
    )


def _compute_kuka_config_flags(
    joints_list: Sequence[float],
    model: Any,
) -> tuple[int, int, int]:
    return _compute_joint_geometry_cache_entry(
        joints_list,
        model,
        arm_link_2_home_angle_deg=_arm_link_2_home_angle_deg(model),
    ).config_flags


def _compute_joint_geometry_cache_entry(
    joints_list: Sequence[float],
    model: Any,
    *,
    arm_link_2_home_angle_deg: float,
) -> _JointGeometryCacheEntry:
    joints = [float(value) for value in joints_list]
    j3 = joints[2] if len(joints) > 2 else 0.0
    j5 = joints[4] if len(joints) > 4 else 0.0
    arm_singularity_measure = 1.0
    bit0 = 0

    try:
        wrist_position = model.fk_wrist_center_in_robot_base(joints)
        bit0 = 1 if float(wrist_position[0]) < 150.0 else 0
        arm_singularity_measure = abs(
            math.sin(
                math.radians(j3 + float(arm_link_2_home_angle_deg))
            )
        )
    except Exception:
        bit0 = 0

    return _JointGeometryCacheEntry(
        config_flags=(
            bit0,
            1 if j3 < 0.0 else 0,
            1 if j5 < 0.0 else 0,
        ),
        arm_singularity_measure=arm_singularity_measure,
    )


def _compute_link_poses(
    joints_list: Sequence[float],
    model: Any,
) -> list:
    import numpy as np

    joints = [float(value) for value in joints_list]
    j2_point_home = model.joint_axis_points_base_mm[1]
    j3_point_home = model.joint_axis_points_base_mm[2]

    transform_j1 = model.fk_partial(joints, n_joints=1)
    shoulder_position = (
        transform_j1
        @ np.array([j2_point_home[0], j2_point_home[1], j2_point_home[2], 1.0], dtype=float)
    )[:3]
    shoulder_pose = np.eye(4, dtype=float)
    shoulder_pose[:3, 3] = shoulder_position

    transform_j2 = model.fk_partial(joints, n_joints=2)
    elbow_position = (
        transform_j2
        @ np.array([j3_point_home[0], j3_point_home[1], j3_point_home[2], 1.0], dtype=float)
    )[:3]
    elbow_pose = np.eye(4, dtype=float)
    elbow_pose[:3, 3] = elbow_position

    wrist_position = model.fk_wrist_center_in_robot_base(joints)
    wrist_pose = np.eye(4, dtype=float)
    wrist_pose[:3, 3] = wrist_position

    return [None, shoulder_pose, elbow_pose, wrist_pose]


def _pose_from_position(
    position_mm: Sequence[float],
):
    import numpy as np

    pose = np.eye(4, dtype=float)
    pose[:3, 3] = [float(value) for value in position_mm[:3]]
    return pose


def _arm_link_2_home_angle_deg(model: Any) -> float:
    try:
        home_wrist_center = getattr(model, "home_wrist_center_base_mm", None)
        if home_wrist_center is None:
            home_wrist_center = model.joint_axis_points_base_mm[3]
        joint3_point = model.joint_axis_points_base_mm[2]
        vector = home_wrist_center - joint3_point
        return float(math.degrees(math.atan2(float(vector[0]), float(vector[2]))))
    except Exception:
        return 0.0
