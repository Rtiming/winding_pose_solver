"""
数值 IK 求解模块。

这里承接原先写在 `RobotModel` 内部的求解逻辑，
把“模型能力”和“求解策略”拆开：

- `RobotModel` 负责模型、FK、位姿变换、关节限位等基础能力
- `NumericIKSolver` 负责数值 IK、arm candidate 枚举、多 seed 扩展和 turn expansion
"""

from __future__ import annotations

from itertools import product
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import least_squares

from .kinematics import (
    ARM_JOINT_COUNT,
    IKResult,
    JOINT_COUNT,
    LocalIKSolution,
    LocalIKSolutionSet,
    RobotModel,
    as_joint_vector,
    as_transform,
    joint_distance_deg,
    joints_within_limits,
    normalize_angle_deg,
    rotation_error_deg,
)


class NumericIKSolver:
    """围绕单个 `RobotModel` 提供数值 IK 能力。"""

    def __init__(self, robot_model: RobotModel):
        self.robot_model = robot_model

    def _as_arm_candidate(self, q_arm_deg: Iterable[float], name: str = "q_arm_deg") -> np.ndarray:
        """把 1~3 轴候选整理成固定长度的向量。"""
        q_arm = np.asarray(list(q_arm_deg), dtype=float).reshape(-1)
        if q_arm.size != ARM_JOINT_COUNT:
            raise ValueError(f"{name} must contain {ARM_JOINT_COUNT} values, got {q_arm.size}.")
        return q_arm

    def _solve_position_only_ik_for_arm(
        self,
        q1_deg: float,
        target_wrist_center_mm: np.ndarray,
        q2q3_seed_deg: Sequence[float],
        max_nfev: int,
    ) -> Tuple[np.ndarray, float]:
        """只解 1~3 轴，让腕心位置对上目标腕心。"""
        q2q3_seed = np.asarray(list(q2q3_seed_deg), dtype=float).reshape(-1)
        if q2q3_seed.size != 2:
            raise ValueError(f"q2q3_seed_deg must contain 2 values, got {q2q3_seed.size}.")

        lower = self.robot_model.joint_min_deg[1:3]
        upper = self.robot_model.joint_max_deg[1:3]

        def residual(q2q3_deg: np.ndarray) -> np.ndarray:
            q_candidate = np.array([q1_deg, q2q3_deg[0], q2q3_deg[1], 0.0, 0.0, 0.0], dtype=float)
            wrist_center = self.robot_model.fk_wrist_center_in_robot_base(q_candidate)
            return wrist_center - target_wrist_center_mm

        result = least_squares(
            residual,
            np.clip(q2q3_seed, lower, upper),
            bounds=(lower, upper),
            xtol=1e-12,
            ftol=1e-12,
            gtol=1e-12,
            max_nfev=max_nfev,
        )
        q_arm = np.array([q1_deg, result.x[0], result.x[1]], dtype=float)
        residual_norm_mm = float(np.linalg.norm(result.fun))
        return q_arm, residual_norm_mm

    def solve_arm_position_candidates(
        self,
        target_flange_base_T: np.ndarray,
        q2q3_seed_pairs_deg: Sequence[Sequence[float]],
        position_tolerance_mm: float,
        dedup_tolerance_deg: float,
        max_nfev: int,
    ) -> List[np.ndarray]:
        """解出本地 `ik-all` 第一阶段的多组 1~3 轴候选。"""
        target_flange_base_T = as_transform(target_flange_base_T, "target_flange_base_T")
        wrist_center_mm = self.robot_model.wrist_center_from_flange_pose(target_flange_base_T)

        target_azimuth_deg = np.rad2deg(np.arctan2(wrist_center_mm[1], wrist_center_mm[0]))
        q1_candidates_deg = []
        for raw_q1_deg in (
            -target_azimuth_deg,
            -target_azimuth_deg + 180.0,
            -target_azimuth_deg - 180.0,
        ):
            q1_deg = normalize_angle_deg(raw_q1_deg)
            if not (self.robot_model.joint_min_deg[0] <= q1_deg <= self.robot_model.joint_max_deg[0]):
                continue
            if any(abs(q1_deg - saved_q1) <= 1e-9 for saved_q1 in q1_candidates_deg):
                continue
            q1_candidates_deg.append(q1_deg)

        arm_solutions: List[np.ndarray] = []
        for q1_deg in q1_candidates_deg:
            for q2q3_seed_deg in q2q3_seed_pairs_deg:
                q_arm_deg, residual_norm_mm = self._solve_position_only_ik_for_arm(
                    q1_deg=q1_deg,
                    target_wrist_center_mm=wrist_center_mm,
                    q2q3_seed_deg=q2q3_seed_deg,
                    max_nfev=max_nfev,
                )
                if residual_norm_mm > position_tolerance_mm:
                    continue
                if any(
                    joint_distance_deg(np.r_[q_arm_deg, [0.0, 0.0, 0.0]], np.r_[saved, [0.0, 0.0, 0.0]])
                    <= dedup_tolerance_deg
                    for saved in arm_solutions
                ):
                    continue
                arm_solutions.append(q_arm_deg)

        arm_solutions = sorted(arm_solutions, key=lambda item: tuple(np.round(item, 9).tolist()))
        return arm_solutions

    def build_numeric_seed_candidates(
        self,
        target_frame_tcp_T: np.ndarray,
        seed_joints_deg: Optional[Sequence[float]],
        q2q3_seed_pairs_deg: Sequence[Sequence[float]],
        wrist_seed_triplets_deg: Sequence[Sequence[float]],
        arm_position_tolerance_mm: float,
        periodic_dedup_tolerance_deg: float,
        max_nfev: int,
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """生成现有 numeric `ik-all` 使用的 arm candidates 和完整 6 轴 seeds。"""
        target_frame_tcp_T = as_transform(target_frame_tcp_T, "target_frame_tcp_T")
        seed = None if seed_joints_deg is None else as_joint_vector(seed_joints_deg, "seed_joints_deg")
        target_flange_base_T = self.robot_model.flange_target_from_frame_tcp_target(target_frame_tcp_T)

        arm_candidates = self.solve_arm_position_candidates(
            target_flange_base_T=target_flange_base_T,
            q2q3_seed_pairs_deg=q2q3_seed_pairs_deg,
            position_tolerance_mm=arm_position_tolerance_mm,
            dedup_tolerance_deg=periodic_dedup_tolerance_deg,
            max_nfev=max_nfev,
        )

        candidate_seeds: List[np.ndarray] = []
        if seed is not None:
            candidate_seeds.append(self.robot_model.clip_joints(seed))

        for arm_candidate in arm_candidates:
            if seed is not None:
                candidate_seeds.append(self.robot_model.clip_joints(np.r_[arm_candidate, seed[3:6]]))
            for wrist_seed in wrist_seed_triplets_deg:
                wrist_seed_vector = np.asarray(list(wrist_seed), dtype=float).reshape(-1)
                if wrist_seed_vector.size != JOINT_COUNT - ARM_JOINT_COUNT:
                    raise ValueError(
                        f"Each wrist seed must contain {JOINT_COUNT - ARM_JOINT_COUNT} values, "
                        f"got {wrist_seed_vector.size}."
                    )
                candidate_seeds.append(
                    self.robot_model.clip_joints(np.r_[arm_candidate, wrist_seed_vector])
                )

        if not candidate_seeds:
            candidate_seeds.append(np.zeros(JOINT_COUNT, dtype=float))

        return arm_candidates, candidate_seeds

    def deduplicate_seed_candidates(
        self,
        candidate_seeds: Sequence[Sequence[float]],
        tolerance_deg: float,
    ) -> List[np.ndarray]:
        """按 360 度周期对完整 6 轴 seed 去重，保留首次出现的顺序。"""
        unique_seeds: List[np.ndarray] = []
        for candidate in candidate_seeds:
            candidate_vector = self.robot_model.clip_joints(as_joint_vector(candidate, "candidate_seed"))
            if any(joint_distance_deg(candidate_vector, saved) <= tolerance_deg for saved in unique_seeds):
                continue
            unique_seeds.append(candidate_vector)
        return unique_seeds

    def deduplicate_seed_candidates_exact(
        self,
        candidate_seeds: Sequence[Sequence[float]],
    ) -> List[np.ndarray]:
        """只去掉逐元素完全相同的完整 6 轴 seeds，保留近似但可能有用的奇异邻域 seeds。"""
        unique_seeds: List[np.ndarray] = []
        for candidate in candidate_seeds:
            candidate_vector = self.robot_model.clip_joints(as_joint_vector(candidate, "candidate_seed"))
            if any(np.array_equal(candidate_vector, saved) for saved in unique_seeds):
                continue
            unique_seeds.append(candidate_vector)
        return unique_seeds

    def deduplicate_arm_candidates(
        self,
        arm_candidates_deg: Sequence[Sequence[float]],
        tolerance_deg: float,
    ) -> List[np.ndarray]:
        """对 1~3 轴候选按周期距离去重，保留首次出现的顺序。"""
        unique_arms: List[np.ndarray] = []
        for candidate in arm_candidates_deg:
            candidate_vector = np.array(
                [normalize_angle_deg(value) for value in self._as_arm_candidate(candidate, "arm_candidate_deg")],
                dtype=float,
            )
            if any(
                joint_distance_deg(
                    np.r_[candidate_vector, np.zeros(JOINT_COUNT - ARM_JOINT_COUNT, dtype=float)],
                    np.r_[saved, np.zeros(JOINT_COUNT - ARM_JOINT_COUNT, dtype=float)],
                )
                <= tolerance_deg
                for saved in unique_arms
            ):
                continue
            unique_arms.append(candidate_vector)
        return unique_arms

    def _solve_numeric_on_flange_target(
        self,
        target_flange_base_T: np.ndarray,
        q0_deg: Sequence[float],
        rotation_weight_mm: float,
        max_nfev: int,
    ) -> IKResult:
        """直接对目标法兰位姿做一次完整数值 IK。"""
        target_flange_base_T = as_transform(target_flange_base_T, "target_flange_base_T")
        q0 = self.robot_model.clip_joints(q0_deg)

        def residual(q: np.ndarray) -> np.ndarray:
            current_flange_T = self.robot_model.fk_flange(q)
            from .kinematics import pose_error_vector

            return pose_error_vector(current_flange_T, target_flange_base_T, rotation_weight_mm)

        result = least_squares(
            residual,
            q0,
            bounds=(self.robot_model.joint_min_deg, self.robot_model.joint_max_deg),
            xtol=1e-12,
            ftol=1e-12,
            gtol=1e-12,
            max_nfev=max_nfev,
        )

        return IKResult(
            q_deg=result.x,
            success=bool(result.success),
            message=str(result.message),
            nfev=int(result.nfev),
            cost=float(result.cost),
            residual=np.asarray(result.fun, dtype=float),
            target_flange_base_T=target_flange_base_T,
        )

    def solve_ik(
        self,
        target_frame_tcp_T: np.ndarray,
        q0_deg: Optional[Sequence[float]] = None,
        rotation_weight_mm: float = 200.0,
        max_nfev: int = 200,
        raise_on_fail: bool = True,
    ) -> IKResult:
        """单次数值 IK，主要用于调试单个初值。"""
        target_frame_tcp_T = as_transform(target_frame_tcp_T, "target_frame_tcp_T")
        target_flange_base_T = self.robot_model.flange_target_from_frame_tcp_target(target_frame_tcp_T)

        if q0_deg is None:
            q0 = np.zeros(JOINT_COUNT, dtype=float)
        else:
            q0 = self.robot_model.clip_joints(q0_deg)

        ik_result = self._solve_numeric_on_flange_target(
            target_flange_base_T=target_flange_base_T,
            q0_deg=q0,
            rotation_weight_mm=rotation_weight_mm,
            max_nfev=max_nfev,
        )

        if raise_on_fail and not ik_result.success:
            raise RuntimeError(f"IK failed: {ik_result.message}")

        return ik_result

    def expand_solution_turn_variants(self, q_deg: Sequence[float]) -> List[Tuple[np.ndarray, np.ndarray]]:
        """在机器人硬限位内展开 360 度等价解。"""
        q = as_joint_vector(q_deg, "q_deg")

        per_joint_values: List[List[Tuple[float, int]]] = []
        for joint_deg, lower_deg, upper_deg in zip(
            q, self.robot_model.joint_min_deg, self.robot_model.joint_max_deg
        ):
            min_turn = int(np.ceil((lower_deg - joint_deg) / 360.0 - 1e-12))
            max_turn = int(np.floor((upper_deg - joint_deg) / 360.0 + 1e-12))
            values_for_this_joint = [
                (joint_deg + 360.0 * turn_index, turn_index)
                for turn_index in range(min_turn, max_turn + 1)
            ]
            per_joint_values.append(values_for_this_joint)

        variants: List[Tuple[np.ndarray, np.ndarray]] = []
        for joint_combo in product(*per_joint_values):
            variant_joints = np.array([pair[0] for pair in joint_combo], dtype=float)
            turn_offsets = np.array([pair[1] for pair in joint_combo], dtype=int)
            variants.append((variant_joints, turn_offsets))
        return variants

    def refine_seed_candidates(
        self,
        target_frame_tcp_T: np.ndarray,
        target_space: str,
        seed_joints_deg: Optional[Sequence[float]],
        filter_lower_deg: Sequence[float],
        filter_upper_deg: Sequence[float],
        arm_candidate_solutions_deg: Sequence[Sequence[float]],
        candidate_seeds: Sequence[Sequence[float]],
        rotation_weight_mm: float,
        max_nfev: int,
        pose_position_tolerance_mm: float,
        pose_orientation_tolerance_deg: float,
        periodic_dedup_tolerance_deg: float,
    ) -> LocalIKSolutionSet:
        """把给定 seed 集合走完现有 numeric refinement / dedup / turn expansion 流程。"""
        target_frame_tcp_T = as_transform(target_frame_tcp_T, "target_frame_tcp_T")
        if target_space != "frame":
            raise ValueError(f"Local IK expects target_space='frame', got {target_space}.")

        filter_lower = as_joint_vector(filter_lower_deg, "filter_lower_deg")
        filter_upper = as_joint_vector(filter_upper_deg, "filter_upper_deg")
        if np.any(filter_lower > filter_upper):
            raise ValueError("Each filter lower limit must be <= the corresponding upper limit.")

        seed = None if seed_joints_deg is None else as_joint_vector(seed_joints_deg, "seed_joints_deg")
        target_flange_base_T = self.robot_model.flange_target_from_frame_tcp_target(target_frame_tcp_T)
        normalized_arm_candidates = [
            self._as_arm_candidate(candidate, "arm_candidate_solution_deg").copy()
            for candidate in arm_candidate_solutions_deg
        ]
        normalized_candidate_seeds = [
            self.robot_model.clip_joints(as_joint_vector(candidate, "candidate_seed"))
            for candidate in candidate_seeds
        ]
        if not normalized_candidate_seeds:
            normalized_candidate_seeds.append(np.zeros(JOINT_COUNT, dtype=float))

        branch_solutions: List[LocalIKSolution] = []
        for seed_vector in normalized_candidate_seeds:
            ik_result = self._solve_numeric_on_flange_target(
                target_flange_base_T=target_flange_base_T,
                q0_deg=seed_vector,
                rotation_weight_mm=rotation_weight_mm,
                max_nfev=max_nfev,
            )
            if not ik_result.success:
                continue

            solved_frame_tcp_T = self.robot_model.fk_tcp_in_frame(ik_result.q_deg)
            position_error_mm = float(
                np.linalg.norm(target_frame_tcp_T[:3, 3] - solved_frame_tcp_T[:3, 3])
            )
            orientation_error = rotation_error_deg(solved_frame_tcp_T, target_frame_tcp_T)
            residual_norm = float(np.linalg.norm(ik_result.residual))

            if position_error_mm > pose_position_tolerance_mm:
                continue
            if orientation_error > pose_orientation_tolerance_deg:
                continue

            if any(
                joint_distance_deg(ik_result.q_deg, solution.joints_deg) <= periodic_dedup_tolerance_deg
                for solution in branch_solutions
            ):
                continue

            branch_solutions.append(
                LocalIKSolution(
                    index=-1,
                    branch_index=-1,
                    joints_deg=ik_result.q_deg,
                    seed_joints_deg=seed_vector,
                    turn_offsets=np.zeros(JOINT_COUNT, dtype=int),
                    within_robot_limits=self.robot_model.within_joint_limits(ik_result.q_deg),
                    within_filter_limits=joints_within_limits(
                        ik_result.q_deg, filter_lower, filter_upper
                    ),
                    distance_to_seed_deg=None
                    if seed is None
                    else joint_distance_deg(ik_result.q_deg, seed),
                    residual_norm=residual_norm,
                    position_error_mm=position_error_mm,
                    orientation_error_deg=orientation_error,
                )
            )

        branch_solutions = sorted(
            branch_solutions,
            key=lambda solution: (
                float("inf") if solution.distance_to_seed_deg is None else solution.distance_to_seed_deg,
                tuple(np.round(solution.joints_deg, 9).tolist()),
            ),
        )
        branch_solutions = [
            LocalIKSolution(
                index=branch_index,
                branch_index=branch_index,
                joints_deg=solution.joints_deg,
                seed_joints_deg=solution.seed_joints_deg,
                turn_offsets=solution.turn_offsets,
                within_robot_limits=solution.within_robot_limits,
                within_filter_limits=solution.within_filter_limits,
                distance_to_seed_deg=solution.distance_to_seed_deg,
                residual_norm=solution.residual_norm,
                position_error_mm=solution.position_error_mm,
                orientation_error_deg=solution.orientation_error_deg,
            )
            for branch_index, solution in enumerate(branch_solutions)
        ]

        all_solutions: List[LocalIKSolution] = []
        for branch_solution in branch_solutions:
            for variant_joints_deg, turn_offsets in self.expand_solution_turn_variants(
                branch_solution.joints_deg
            ):
                within_filter_limits = joints_within_limits(
                    variant_joints_deg, filter_lower, filter_upper
                )
                distance_to_seed = (
                    None if seed is None else joint_distance_deg(variant_joints_deg, seed)
                )
                all_solutions.append(
                    LocalIKSolution(
                        index=-1,
                        branch_index=branch_solution.branch_index,
                        joints_deg=variant_joints_deg,
                        seed_joints_deg=branch_solution.seed_joints_deg,
                        turn_offsets=turn_offsets,
                        within_robot_limits=True,
                        within_filter_limits=within_filter_limits,
                        distance_to_seed_deg=distance_to_seed,
                        residual_norm=branch_solution.residual_norm,
                        position_error_mm=branch_solution.position_error_mm,
                        orientation_error_deg=branch_solution.orientation_error_deg,
                    )
                )

        all_solutions = sorted(
            all_solutions,
            key=lambda solution: (
                float("inf") if solution.distance_to_seed_deg is None else solution.distance_to_seed_deg,
                solution.branch_index,
                tuple(solution.turn_offsets.tolist()),
            ),
        )
        all_solutions = [
            LocalIKSolution(
                index=index,
                branch_index=solution.branch_index,
                joints_deg=solution.joints_deg,
                seed_joints_deg=solution.seed_joints_deg,
                turn_offsets=solution.turn_offsets,
                within_robot_limits=solution.within_robot_limits,
                within_filter_limits=solution.within_filter_limits,
                distance_to_seed_deg=solution.distance_to_seed_deg,
                residual_norm=solution.residual_norm,
                position_error_mm=solution.position_error_mm,
                orientation_error_deg=solution.orientation_error_deg,
            )
            for index, solution in enumerate(all_solutions)
        ]

        filtered_solutions = [
            solution for solution in all_solutions if solution.within_robot_limits and solution.within_filter_limits
        ]

        return LocalIKSolutionSet(
            target_pose=target_frame_tcp_T,
            target_space=target_space,
            seed_joints_deg=None if seed is None else seed.copy(),
            robot_lower_limits_deg=self.robot_model.joint_min_deg.copy(),
            robot_upper_limits_deg=self.robot_model.joint_max_deg.copy(),
            filter_lower_limits_deg=filter_lower.copy(),
            filter_upper_limits_deg=filter_upper.copy(),
            arm_candidate_solutions_deg=[candidate.copy() for candidate in normalized_arm_candidates],
            branch_solutions=branch_solutions,
            all_solutions=all_solutions,
            filtered_solutions=filtered_solutions,
        )

    def solve_ik_all(
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
        """
        本地多解 IK。

        流程分三步：
        1. 先解腕心位置，枚举 1~3 轴候选。
        2. 给每组 1~3 轴配多组腕部初值，跑完整 6 轴数值 IK。
        3. 在机器人硬限位内展开 360 度等价解，再按过滤范围筛选。
        """
        target_frame_tcp_T = as_transform(target_frame_tcp_T, "target_frame_tcp_T")
        if target_space != "frame":
            raise ValueError(f"Local IK expects target_space='frame', got {target_space}.")

        arm_candidates, candidate_seeds = self.build_numeric_seed_candidates(
            target_frame_tcp_T=target_frame_tcp_T,
            seed_joints_deg=seed_joints_deg,
            q2q3_seed_pairs_deg=q2q3_seed_pairs_deg,
            wrist_seed_triplets_deg=wrist_seed_triplets_deg,
            arm_position_tolerance_mm=arm_position_tolerance_mm,
            periodic_dedup_tolerance_deg=periodic_dedup_tolerance_deg,
            max_nfev=max_nfev,
        )

        return self.refine_seed_candidates(
            target_frame_tcp_T=target_frame_tcp_T,
            target_space=target_space,
            seed_joints_deg=seed_joints_deg,
            filter_lower_deg=filter_lower_deg,
            filter_upper_deg=filter_upper_deg,
            arm_candidate_solutions_deg=arm_candidates,
            candidate_seeds=candidate_seeds,
            rotation_weight_mm=rotation_weight_mm,
            max_nfev=max_nfev,
            pose_position_tolerance_mm=pose_position_tolerance_mm,
            pose_orientation_tolerance_deg=pose_orientation_tolerance_deg,
            periodic_dedup_tolerance_deg=periodic_dedup_tolerance_deg,
        )
