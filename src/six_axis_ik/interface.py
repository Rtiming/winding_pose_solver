"""
SixAxisIK 对外可调用接口。

目标：
- 让上层以接近 RoboDK 的调用习惯直接请求单点 IK / 多解 IK
- 接受 RoboDK `Mat`、numpy 4x4、或嵌套列表位姿输入
- 返回结构化结果而不是裸数组
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Optional, Sequence

import numpy as np

from . import config
from .backends import IKBackend, IKSolveAllRequest, IKSolveRequest, build_ik_backend
from .kinematics import (
    IKResult,
    JOINT_COUNT,
    LocalIKSolution,
    LocalIKSolutionSet,
    RobotModel,
    as_joint_vector,
    as_transform,
    joint_distance_deg,
    rotation_error_deg,
)


def _coerce_pose_matrix(pose: Any, name: str) -> np.ndarray:
    """把 RoboDK `Mat` / numpy / 嵌套列表统一转成 4x4 numpy 矩阵。"""
    if hasattr(pose, "size") and callable(pose.size) and hasattr(pose, "__getitem__"):
        row_count = pose.size(0)
        col_count = pose.size(1)
        if row_count != 4 or col_count != 4:
            raise ValueError(f"{name} must be 4x4, got RoboDK Mat shape ({row_count}, {col_count}).")
        matrix = np.array(
            [[pose[row_index, col_index] for col_index in range(col_count)] for row_index in range(row_count)],
            dtype=float,
        )
        return as_transform(matrix, name)

    return as_transform(np.asarray(pose, dtype=float), name)


@dataclass(frozen=True)
class IKSolutionRecord:
    """统一后的单组 IK 解记录。"""

    joints_deg: np.ndarray
    distance_to_seed_deg: Optional[float]
    position_error_mm: float
    orientation_error_deg: float
    residual_norm: float
    turn_offsets: np.ndarray
    within_robot_limits: bool
    within_filter_limits: bool


@dataclass(frozen=True)
class IKDiagnostics:
    """单次接口调用的诊断信息。"""

    arm_candidate_count: int
    unique_branch_count: int
    expanded_solution_count: int
    filtered_solution_count: int
    settings_used: dict[str, object]
    backend_message: str | None = None


@dataclass(frozen=True)
class IKSolveResult:
    """接口统一返回结果。"""

    success: bool
    preferred_solution: Optional[IKSolutionRecord]
    filtered_solutions: list[IKSolutionRecord]
    all_solutions: list[IKSolutionRecord]
    failure_reason: str | None
    diagnostics: IKDiagnostics
    legacy_solution_set: LocalIKSolutionSet | None = field(default=None, repr=False)

    def to_local_solution_set(self) -> LocalIKSolutionSet:
        """为兼容旧 CLI / 打印逻辑返回内部的旧结构结果。"""
        if self.legacy_solution_set is None:
            raise RuntimeError("This IKSolveResult does not carry a LocalIKSolutionSet payload.")
        return self.legacy_solution_set


def _record_from_local_solution(solution: LocalIKSolution) -> IKSolutionRecord:
    """把旧的本地解结构映射到统一接口结构。"""
    return IKSolutionRecord(
        joints_deg=solution.joints_deg.copy(),
        distance_to_seed_deg=solution.distance_to_seed_deg,
        position_error_mm=solution.position_error_mm,
        orientation_error_deg=solution.orientation_error_deg,
        residual_norm=solution.residual_norm,
        turn_offsets=solution.turn_offsets.copy(),
        within_robot_limits=solution.within_robot_limits,
        within_filter_limits=solution.within_filter_limits,
    )


class SixAxisIKSolver:
    """SixAxisIK 的稳定对外入口。"""

    def __init__(
        self,
        robot_model: RobotModel,
        *,
        ik_backend: IKBackend | str | None = None,
        q2q3_seed_pairs_deg: Optional[Sequence[Sequence[float]]] = None,
        wrist_seed_triplets_deg: Optional[Sequence[Sequence[float]]] = None,
        rotation_weight_mm: float = config.NUMERIC_ROTATION_WEIGHT_MM,
        max_nfev: int = config.NUMERIC_MAX_NFEV,
        arm_position_tolerance_mm: float = config.LOCAL_ARM_POSITION_TOLERANCE_MM,
        pose_position_tolerance_mm: float = config.LOCAL_IK_POSITION_TOLERANCE_MM,
        pose_orientation_tolerance_deg: float = config.LOCAL_IK_ORIENTATION_TOLERANCE_DEG,
        periodic_dedup_tolerance_deg: float = config.LOCAL_IK_PERIODIC_DEDUP_TOLERANCE_DEG,
    ):
        self.robot_model = robot_model
        self.ik_backend = build_ik_backend(ik_backend)
        self.q2q3_seed_pairs_deg = (
            tuple(tuple(float(value) for value in pair) for pair in q2q3_seed_pairs_deg)
            if q2q3_seed_pairs_deg is not None
            else tuple(tuple(float(value) for value in pair) for pair in config.LOCAL_ARM_Q2Q3_SEEDS_DEG)
        )
        self.wrist_seed_triplets_deg = (
            tuple(tuple(float(value) for value in triplet) for triplet in wrist_seed_triplets_deg)
            if wrist_seed_triplets_deg is not None
            else tuple(tuple(float(value) for value in triplet) for triplet in config.LOCAL_WRIST_SEED_TRIPLETS_DEG)
        )
        self.rotation_weight_mm = float(rotation_weight_mm)
        self.max_nfev = int(max_nfev)
        self.arm_position_tolerance_mm = float(arm_position_tolerance_mm)
        self.pose_position_tolerance_mm = float(pose_position_tolerance_mm)
        self.pose_orientation_tolerance_deg = float(pose_orientation_tolerance_deg)
        self.periodic_dedup_tolerance_deg = float(periodic_dedup_tolerance_deg)

    @classmethod
    def from_config(
        cls,
        *,
        ik_backend: IKBackend | str | None = None,
    ) -> "SixAxisIKSolver":
        """用 `config.py` 默认模型构造一个接口实例。"""
        return cls(config.build_local_robot_model(), ik_backend=ik_backend)

    def _build_effective_robot_model(
        self,
        *,
        tool_pose: Any | None,
        reference_pose: Any | None,
        robot_lower_limits_deg: Sequence[float] | None,
        robot_upper_limits_deg: Sequence[float] | None,
    ) -> RobotModel:
        """按本次调用上下文生成实际使用的机器人模型。"""
        effective_model = self.robot_model

        if tool_pose is not None:
            effective_model = replace(
                effective_model,
                tool_T=_coerce_pose_matrix(tool_pose, "tool_pose"),
            )

        if reference_pose is not None:
            effective_model = replace(
                effective_model,
                frame_T=_coerce_pose_matrix(reference_pose, "reference_pose"),
            )

        if robot_lower_limits_deg is not None or robot_upper_limits_deg is not None:
            lower = (
                effective_model.joint_min_deg
                if robot_lower_limits_deg is None
                else as_joint_vector(robot_lower_limits_deg, "robot_lower_limits_deg")
            )
            upper = (
                effective_model.joint_max_deg
                if robot_upper_limits_deg is None
                else as_joint_vector(robot_upper_limits_deg, "robot_upper_limits_deg")
            )
            effective_model = replace(
                effective_model,
                joint_min_deg=lower,
                joint_max_deg=upper,
            )

        return effective_model

    def _target_pose_in_frame(
        self,
        effective_model: RobotModel,
        target_pose: Any,
        target_space: str,
    ) -> np.ndarray:
        """把目标位姿整理成 solver 需要的 Frame 空间。"""
        pose_matrix = _coerce_pose_matrix(target_pose, "target_pose")
        if target_space == "frame":
            return pose_matrix
        if target_space == "base":
            return effective_model.frame_T_inv @ pose_matrix
        raise ValueError(f"Unsupported target_space: {target_space}")

    def _settings_payload(
        self,
        *,
        target_space: str,
        filter_lower_deg: Sequence[float],
        filter_upper_deg: Sequence[float],
        seed_joints_deg: Sequence[float] | None,
        rotation_weight_mm: float,
        max_nfev: int,
    ) -> dict[str, object]:
        """压缩输出一次调用时实际使用的关键设置。"""
        return {
            "target_space": target_space,
            "filter_lower_deg": tuple(float(value) for value in filter_lower_deg),
            "filter_upper_deg": tuple(float(value) for value in filter_upper_deg),
            "seed_joints_deg": None
            if seed_joints_deg is None
            else tuple(float(value) for value in seed_joints_deg),
            "rotation_weight_mm": float(rotation_weight_mm),
            "max_nfev": int(max_nfev),
            "arm_position_tolerance_mm": self.arm_position_tolerance_mm,
            "pose_position_tolerance_mm": self.pose_position_tolerance_mm,
            "pose_orientation_tolerance_deg": self.pose_orientation_tolerance_deg,
            "periodic_dedup_tolerance_deg": self.periodic_dedup_tolerance_deg,
        }

    def solve_ik(
        self,
        target_pose: Any,
        *,
        target_space: str = "frame",
        seed_joints_deg: Optional[Sequence[float]] = None,
        tool_pose: Any | None = None,
        reference_pose: Any | None = None,
        robot_lower_limits_deg: Sequence[float] | None = None,
        robot_upper_limits_deg: Sequence[float] | None = None,
        filter_lower_deg: Sequence[float] | None = None,
        filter_upper_deg: Sequence[float] | None = None,
        rotation_weight_mm: float | None = None,
        max_nfev: int | None = None,
    ) -> IKSolveResult:
        """做一次单 seed IK，并以统一结构返回。"""
        effective_model = self._build_effective_robot_model(
            tool_pose=tool_pose,
            reference_pose=reference_pose,
            robot_lower_limits_deg=robot_lower_limits_deg,
            robot_upper_limits_deg=robot_upper_limits_deg,
        )
        target_frame_pose = self._target_pose_in_frame(effective_model, target_pose, target_space)

        seed = None if seed_joints_deg is None else as_joint_vector(seed_joints_deg, "seed_joints_deg")
        filter_lower = (
            effective_model.joint_min_deg.copy()
            if filter_lower_deg is None
            else as_joint_vector(filter_lower_deg, "filter_lower_deg")
        )
        filter_upper = (
            effective_model.joint_max_deg.copy()
            if filter_upper_deg is None
            else as_joint_vector(filter_upper_deg, "filter_upper_deg")
        )
        rotation_weight_value = self.rotation_weight_mm if rotation_weight_mm is None else float(rotation_weight_mm)
        max_nfev_value = self.max_nfev if max_nfev is None else int(max_nfev)

        ik_result: IKResult = self.ik_backend.solve(
            IKSolveRequest(
                robot_model=effective_model,
                target_frame_pose=target_frame_pose,
                seed_joints_deg=seed,
                rotation_weight_mm=rotation_weight_value,
                max_nfev=max_nfev_value,
            )
        )

        solved_pose = effective_model.fk_tcp_in_frame(ik_result.q_deg)
        position_error_mm = float(np.linalg.norm(target_frame_pose[:3, 3] - solved_pose[:3, 3]))
        orientation_error_deg = rotation_error_deg(solved_pose, target_frame_pose)
        residual_norm = float(np.linalg.norm(ik_result.residual))
        within_filter_limits = bool(np.all(ik_result.q_deg >= filter_lower) and np.all(ik_result.q_deg <= filter_upper))
        within_tolerance = (
            position_error_mm <= self.pose_position_tolerance_mm
            and orientation_error_deg <= self.pose_orientation_tolerance_deg
        )
        success = bool(ik_result.success and within_tolerance and within_filter_limits)

        solution_record = IKSolutionRecord(
            joints_deg=ik_result.q_deg.copy(),
            distance_to_seed_deg=None if seed is None else joint_distance_deg(ik_result.q_deg, seed),
            position_error_mm=position_error_mm,
            orientation_error_deg=orientation_error_deg,
            residual_norm=residual_norm,
            turn_offsets=np.zeros(JOINT_COUNT, dtype=int),
            within_robot_limits=effective_model.within_joint_limits(ik_result.q_deg),
            within_filter_limits=within_filter_limits,
        )
        failure_reason = None
        if not success:
            if not ik_result.success:
                failure_reason = ik_result.message
            elif not within_tolerance:
                failure_reason = (
                    "IK converged, but pose error exceeded tolerance "
                    f"(pos={position_error_mm:.6g} mm, ori={orientation_error_deg:.6g} deg)."
                )
            else:
                failure_reason = "IK converged, but the solution was outside the requested filter range."

        diagnostics = IKDiagnostics(
            arm_candidate_count=0,
            unique_branch_count=1 if success else 0,
            expanded_solution_count=1 if success else 0,
            filtered_solution_count=1 if success else 0,
            settings_used=self._settings_payload(
                target_space=target_space,
                filter_lower_deg=filter_lower,
                filter_upper_deg=filter_upper,
                seed_joints_deg=seed,
                rotation_weight_mm=rotation_weight_value,
                max_nfev=max_nfev_value,
            ),
            backend_message=ik_result.message,
        )

        return IKSolveResult(
            success=success,
            preferred_solution=solution_record if success else None,
            filtered_solutions=[solution_record] if success else [],
            all_solutions=[solution_record],
            failure_reason=failure_reason,
            diagnostics=diagnostics,
            legacy_solution_set=None,
        )

    def solve_ik_all(
        self,
        target_pose: Any,
        *,
        target_space: str = "frame",
        seed_joints_deg: Optional[Sequence[float]] = None,
        tool_pose: Any | None = None,
        reference_pose: Any | None = None,
        robot_lower_limits_deg: Sequence[float] | None = None,
        robot_upper_limits_deg: Sequence[float] | None = None,
        filter_lower_deg: Sequence[float] | None = None,
        filter_upper_deg: Sequence[float] | None = None,
        q2q3_seed_pairs_deg: Optional[Sequence[Sequence[float]]] = None,
        wrist_seed_triplets_deg: Optional[Sequence[Sequence[float]]] = None,
        rotation_weight_mm: float | None = None,
        max_nfev: int | None = None,
        arm_position_tolerance_mm: float | None = None,
        pose_position_tolerance_mm: float | None = None,
        pose_orientation_tolerance_deg: float | None = None,
        periodic_dedup_tolerance_deg: float | None = None,
    ) -> IKSolveResult:
        """做一次多解 IK，并以统一结构返回。"""
        (
            legacy_solution_set,
            seed,
            filter_lower,
            filter_upper,
            rotation_weight_value,
            max_nfev_value,
        ) = self._solve_ik_all_legacy(
            target_pose,
            target_space=target_space,
            seed_joints_deg=seed_joints_deg,
            tool_pose=tool_pose,
            reference_pose=reference_pose,
            robot_lower_limits_deg=robot_lower_limits_deg,
            robot_upper_limits_deg=robot_upper_limits_deg,
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

        filtered_records = [
            _record_from_local_solution(solution) for solution in legacy_solution_set.filtered_solutions
        ]
        all_records = [_record_from_local_solution(solution) for solution in legacy_solution_set.all_solutions]
        preferred_solution = filtered_records[0] if filtered_records else None

        failure_reason = None
        if not filtered_records:
            if not all_records:
                failure_reason = "No IK solution converged within pose tolerance."
            else:
                failure_reason = "IK solutions converged, but all solutions were filtered out."

        diagnostics = IKDiagnostics(
            arm_candidate_count=len(legacy_solution_set.arm_candidate_solutions_deg),
            unique_branch_count=len(legacy_solution_set.branch_solutions),
            expanded_solution_count=len(legacy_solution_set.all_solutions),
            filtered_solution_count=len(legacy_solution_set.filtered_solutions),
            settings_used=self._settings_payload(
                target_space=target_space,
                filter_lower_deg=filter_lower,
                filter_upper_deg=filter_upper,
                seed_joints_deg=seed,
                rotation_weight_mm=rotation_weight_value,
                max_nfev=max_nfev_value,
            ),
            backend_message=None,
        )

        return IKSolveResult(
            success=bool(filtered_records),
            preferred_solution=preferred_solution,
            filtered_solutions=filtered_records,
            all_solutions=all_records,
            failure_reason=failure_reason,
            diagnostics=diagnostics,
            legacy_solution_set=legacy_solution_set,
        )

    def solve_ik_all_local_solution_set(
        self,
        target_pose: Any,
        *,
        target_space: str = "frame",
        seed_joints_deg: Optional[Sequence[float]] = None,
        tool_pose: Any | None = None,
        reference_pose: Any | None = None,
        robot_lower_limits_deg: Sequence[float] | None = None,
        robot_upper_limits_deg: Sequence[float] | None = None,
        filter_lower_deg: Sequence[float] | None = None,
        filter_upper_deg: Sequence[float] | None = None,
        q2q3_seed_pairs_deg: Optional[Sequence[Sequence[float]]] = None,
        wrist_seed_triplets_deg: Optional[Sequence[Sequence[float]]] = None,
        rotation_weight_mm: float | None = None,
        max_nfev: int | None = None,
        arm_position_tolerance_mm: float | None = None,
        pose_position_tolerance_mm: float | None = None,
        pose_orientation_tolerance_deg: float | None = None,
        periodic_dedup_tolerance_deg: float | None = None,
    ) -> LocalIKSolutionSet:
        legacy_solution_set, *_ = self._solve_ik_all_legacy(
            target_pose,
            target_space=target_space,
            seed_joints_deg=seed_joints_deg,
            tool_pose=tool_pose,
            reference_pose=reference_pose,
            robot_lower_limits_deg=robot_lower_limits_deg,
            robot_upper_limits_deg=robot_upper_limits_deg,
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
        return legacy_solution_set

    def solve_ik_all_joint_vectors(
        self,
        target_pose: Any,
        *,
        target_space: str = "frame",
        seed_joints_deg: Optional[Sequence[float]] = None,
        tool_pose: Any | None = None,
        reference_pose: Any | None = None,
        robot_lower_limits_deg: Sequence[float] | None = None,
        robot_upper_limits_deg: Sequence[float] | None = None,
        filter_lower_deg: Sequence[float] | None = None,
        filter_upper_deg: Sequence[float] | None = None,
        q2q3_seed_pairs_deg: Optional[Sequence[Sequence[float]]] = None,
        wrist_seed_triplets_deg: Optional[Sequence[Sequence[float]]] = None,
        rotation_weight_mm: float | None = None,
        max_nfev: int | None = None,
        arm_position_tolerance_mm: float | None = None,
        pose_position_tolerance_mm: float | None = None,
        pose_orientation_tolerance_deg: float | None = None,
        periodic_dedup_tolerance_deg: float | None = None,
    ) -> list[np.ndarray]:
        effective_model = self._build_effective_robot_model(
            tool_pose=tool_pose,
            reference_pose=reference_pose,
            robot_lower_limits_deg=robot_lower_limits_deg,
            robot_upper_limits_deg=robot_upper_limits_deg,
        )
        target_frame_pose = self._target_pose_in_frame(effective_model, target_pose, target_space)

        seed = None if seed_joints_deg is None else as_joint_vector(seed_joints_deg, "seed_joints_deg")
        filter_lower = (
            effective_model.joint_min_deg.copy()
            if filter_lower_deg is None
            else as_joint_vector(filter_lower_deg, "filter_lower_deg")
        )
        filter_upper = (
            effective_model.joint_max_deg.copy()
            if filter_upper_deg is None
            else as_joint_vector(filter_upper_deg, "filter_upper_deg")
        )
        rotation_weight_value = self.rotation_weight_mm if rotation_weight_mm is None else float(rotation_weight_mm)
        max_nfev_value = self.max_nfev if max_nfev is None else int(max_nfev)
        arm_tolerance_value = (
            self.arm_position_tolerance_mm
            if arm_position_tolerance_mm is None
            else float(arm_position_tolerance_mm)
        )
        pose_pos_tolerance_value = (
            self.pose_position_tolerance_mm
            if pose_position_tolerance_mm is None
            else float(pose_position_tolerance_mm)
        )
        pose_ori_tolerance_value = (
            self.pose_orientation_tolerance_deg
            if pose_orientation_tolerance_deg is None
            else float(pose_orientation_tolerance_deg)
        )
        periodic_tolerance_value = (
            self.periodic_dedup_tolerance_deg
            if periodic_dedup_tolerance_deg is None
            else float(periodic_dedup_tolerance_deg)
        )

        request = IKSolveAllRequest(
            robot_model=effective_model,
            target_frame_pose=target_frame_pose,
            seed_joints_deg=seed,
            filter_lower_deg=filter_lower,
            filter_upper_deg=filter_upper,
            q2q3_seed_pairs_deg=(
                self.q2q3_seed_pairs_deg if q2q3_seed_pairs_deg is None else q2q3_seed_pairs_deg
            ),
            wrist_seed_triplets_deg=(
                self.wrist_seed_triplets_deg if wrist_seed_triplets_deg is None else wrist_seed_triplets_deg
            ),
            rotation_weight_mm=rotation_weight_value,
            max_nfev=max_nfev_value,
            arm_position_tolerance_mm=arm_tolerance_value,
            pose_position_tolerance_mm=pose_pos_tolerance_value,
            pose_orientation_tolerance_deg=pose_ori_tolerance_value,
            periodic_dedup_tolerance_deg=periodic_tolerance_value,
        )

        solve_all_joint_vectors = getattr(self.ik_backend, "solve_all_joint_vectors", None)
        if callable(solve_all_joint_vectors):
            return [solution.copy() for solution in solve_all_joint_vectors(request)]

        legacy_solution_set = self.ik_backend.solve_all(request)
        return [solution.joints_deg.copy() for solution in legacy_solution_set.filtered_solutions]

    def _solve_ik_all_legacy(
        self,
        target_pose: Any,
        *,
        target_space: str = "frame",
        seed_joints_deg: Optional[Sequence[float]] = None,
        tool_pose: Any | None = None,
        reference_pose: Any | None = None,
        robot_lower_limits_deg: Sequence[float] | None = None,
        robot_upper_limits_deg: Sequence[float] | None = None,
        filter_lower_deg: Sequence[float] | None = None,
        filter_upper_deg: Sequence[float] | None = None,
        q2q3_seed_pairs_deg: Optional[Sequence[Sequence[float]]] = None,
        wrist_seed_triplets_deg: Optional[Sequence[Sequence[float]]] = None,
        rotation_weight_mm: float | None = None,
        max_nfev: int | None = None,
        arm_position_tolerance_mm: float | None = None,
        pose_position_tolerance_mm: float | None = None,
        pose_orientation_tolerance_deg: float | None = None,
        periodic_dedup_tolerance_deg: float | None = None,
    ) -> tuple[
        LocalIKSolutionSet,
        np.ndarray | None,
        np.ndarray,
        np.ndarray,
        float,
        int,
    ]:
        effective_model = self._build_effective_robot_model(
            tool_pose=tool_pose,
            reference_pose=reference_pose,
            robot_lower_limits_deg=robot_lower_limits_deg,
            robot_upper_limits_deg=robot_upper_limits_deg,
        )
        target_frame_pose = self._target_pose_in_frame(effective_model, target_pose, target_space)

        seed = None if seed_joints_deg is None else as_joint_vector(seed_joints_deg, "seed_joints_deg")
        filter_lower = (
            effective_model.joint_min_deg.copy()
            if filter_lower_deg is None
            else as_joint_vector(filter_lower_deg, "filter_lower_deg")
        )
        filter_upper = (
            effective_model.joint_max_deg.copy()
            if filter_upper_deg is None
            else as_joint_vector(filter_upper_deg, "filter_upper_deg")
        )
        rotation_weight_value = self.rotation_weight_mm if rotation_weight_mm is None else float(rotation_weight_mm)
        max_nfev_value = self.max_nfev if max_nfev is None else int(max_nfev)
        arm_tolerance_value = (
            self.arm_position_tolerance_mm
            if arm_position_tolerance_mm is None
            else float(arm_position_tolerance_mm)
        )
        pose_pos_tolerance_value = (
            self.pose_position_tolerance_mm
            if pose_position_tolerance_mm is None
            else float(pose_position_tolerance_mm)
        )
        pose_ori_tolerance_value = (
            self.pose_orientation_tolerance_deg
            if pose_orientation_tolerance_deg is None
            else float(pose_orientation_tolerance_deg)
        )
        periodic_tolerance_value = (
            self.periodic_dedup_tolerance_deg
            if periodic_dedup_tolerance_deg is None
            else float(periodic_dedup_tolerance_deg)
        )

        legacy_solution_set = self.ik_backend.solve_all(
            IKSolveAllRequest(
                robot_model=effective_model,
                target_frame_pose=target_frame_pose,
                seed_joints_deg=seed,
                filter_lower_deg=filter_lower,
                filter_upper_deg=filter_upper,
                q2q3_seed_pairs_deg=(
                    self.q2q3_seed_pairs_deg if q2q3_seed_pairs_deg is None else q2q3_seed_pairs_deg
                ),
                wrist_seed_triplets_deg=(
                    self.wrist_seed_triplets_deg if wrist_seed_triplets_deg is None else wrist_seed_triplets_deg
                ),
                rotation_weight_mm=rotation_weight_value,
                max_nfev=max_nfev_value,
                arm_position_tolerance_mm=arm_tolerance_value,
                pose_position_tolerance_mm=pose_pos_tolerance_value,
                pose_orientation_tolerance_deg=pose_ori_tolerance_value,
                periodic_dedup_tolerance_deg=periodic_tolerance_value,
            )
        )
        return (
            legacy_solution_set,
            seed,
            filter_lower,
            filter_upper,
            rotation_weight_value,
            max_nfev_value,
        )
