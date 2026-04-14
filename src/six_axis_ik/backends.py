from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence

import numpy as np

from . import config
from .analytic_solver import AnalyticSeedGenerator
from .kinematics import IKResult, JOINT_COUNT, LocalIKSolutionSet, RobotModel, joint_distance_deg
from .numeric_solver import NumericIKSolver


@dataclass(frozen=True)
class IKSolveRequest:
    """Single-solution IK request in normalized local Frame space."""

    robot_model: RobotModel
    target_frame_pose: np.ndarray
    seed_joints_deg: np.ndarray | None
    rotation_weight_mm: float
    max_nfev: int


@dataclass(frozen=True)
class IKSolveAllRequest:
    """Multi-solution IK request in normalized local Frame space."""

    robot_model: RobotModel
    target_frame_pose: np.ndarray
    seed_joints_deg: np.ndarray | None
    filter_lower_deg: np.ndarray
    filter_upper_deg: np.ndarray
    q2q3_seed_pairs_deg: Sequence[Sequence[float]]
    wrist_seed_triplets_deg: Sequence[Sequence[float]]
    rotation_weight_mm: float
    max_nfev: int
    arm_position_tolerance_mm: float
    pose_position_tolerance_mm: float
    pose_orientation_tolerance_deg: float
    periodic_dedup_tolerance_deg: float


class IKBackend(Protocol):
    """Stable low-level backend contract used by SixAxisIKSolver."""

    name: str

    def solve(self, request: IKSolveRequest) -> IKResult:
        """Solve one IK result near the provided seed."""

    def solve_all(self, request: IKSolveAllRequest) -> LocalIKSolutionSet:
        """Enumerate all reachable IK branches for one target pose."""


class NumericIKBackend:
    """Adapter that preserves the current numeric IK behavior."""

    name = "numeric"

    def solve(self, request: IKSolveRequest) -> IKResult:
        solver = NumericIKSolver(request.robot_model)
        return solver.solve_ik(
            request.target_frame_pose,
            q0_deg=request.seed_joints_deg,
            rotation_weight_mm=request.rotation_weight_mm,
            max_nfev=request.max_nfev,
            raise_on_fail=False,
        )

    def solve_all(self, request: IKSolveAllRequest) -> LocalIKSolutionSet:
        solver = NumericIKSolver(request.robot_model)
        return solver.solve_ik_all(
            target_frame_tcp_T=request.target_frame_pose,
            target_space="frame",
            seed_joints_deg=request.seed_joints_deg,
            filter_lower_deg=request.filter_lower_deg,
            filter_upper_deg=request.filter_upper_deg,
            q2q3_seed_pairs_deg=request.q2q3_seed_pairs_deg,
            wrist_seed_triplets_deg=request.wrist_seed_triplets_deg,
            rotation_weight_mm=request.rotation_weight_mm,
            max_nfev=request.max_nfev,
            arm_position_tolerance_mm=request.arm_position_tolerance_mm,
            pose_position_tolerance_mm=request.pose_position_tolerance_mm,
            pose_orientation_tolerance_deg=request.pose_orientation_tolerance_deg,
            periodic_dedup_tolerance_deg=request.periodic_dedup_tolerance_deg,
        )


class AnalyticIKBackend:
    """解析 IK v1：只生成 analytic seeds，最终解仍走现有 numeric refinement。"""

    name = "analytic"

    def _build_candidate_seed_sets(
        self,
        solver: NumericIKSolver,
        seed_generator: AnalyticSeedGenerator,
        *,
        target_frame_pose: np.ndarray,
        preferred_seed_joints_deg: np.ndarray | None,
        periodic_dedup_tolerance_deg: float,
        max_nfev: int,
        arm_position_tolerance_mm: float,
        q2q3_seed_pairs_deg: Sequence[Sequence[float]],
        wrist_seed_triplets_deg: Sequence[Sequence[float]],
    ) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """合并 analytic seeds 和现有 numeric seeds，保留 analytic-first 的顺序。"""
        analytic_arm_candidates, analytic_seed_candidates = seed_generator.generate_seed_candidates(
            target_frame_pose,
            preferred_seed_joints_deg=preferred_seed_joints_deg,
            periodic_dedup_tolerance_deg=periodic_dedup_tolerance_deg,
        )
        numeric_arm_candidates, numeric_seed_candidates = solver.build_numeric_seed_candidates(
            target_frame_tcp_T=target_frame_pose,
            seed_joints_deg=preferred_seed_joints_deg,
            q2q3_seed_pairs_deg=q2q3_seed_pairs_deg,
            wrist_seed_triplets_deg=wrist_seed_triplets_deg,
            arm_position_tolerance_mm=arm_position_tolerance_mm,
            periodic_dedup_tolerance_deg=periodic_dedup_tolerance_deg,
            max_nfev=max_nfev,
        )

        merged_arm_candidates = solver.deduplicate_arm_candidates(
            [*analytic_arm_candidates, *numeric_arm_candidates],
            tolerance_deg=periodic_dedup_tolerance_deg,
        )
        merged_seed_candidates = solver.deduplicate_seed_candidates_exact(
            [*analytic_seed_candidates, *numeric_seed_candidates]
        )
        return merged_arm_candidates, merged_seed_candidates

    def solve(self, request: IKSolveRequest) -> IKResult:
        solver = NumericIKSolver(request.robot_model)
        seed_generator = AnalyticSeedGenerator(request.robot_model)

        _, candidate_seeds = self._build_candidate_seed_sets(
            solver,
            seed_generator,
            target_frame_pose=request.target_frame_pose,
            preferred_seed_joints_deg=request.seed_joints_deg,
            periodic_dedup_tolerance_deg=config.LOCAL_IK_PERIODIC_DEDUP_TOLERANCE_DEG,
            max_nfev=request.max_nfev,
            arm_position_tolerance_mm=config.LOCAL_ARM_POSITION_TOLERANCE_MM,
            q2q3_seed_pairs_deg=config.LOCAL_ARM_Q2Q3_SEEDS_DEG,
            wrist_seed_triplets_deg=config.LOCAL_WRIST_SEED_TRIPLETS_DEG,
        )

        if request.seed_joints_deg is not None:
            candidate_seeds = solver.deduplicate_seed_candidates_exact(
                [request.seed_joints_deg.copy(), *candidate_seeds]
            )
        elif not candidate_seeds:
            candidate_seeds = [np.zeros(JOINT_COUNT, dtype=float)]

        if request.seed_joints_deg is not None:
            candidate_seeds = sorted(
                candidate_seeds,
                key=lambda candidate: (
                    joint_distance_deg(candidate, request.seed_joints_deg),
                    tuple(np.round(candidate, 9).tolist()),
                ),
            )

        best_failure_result: IKResult | None = None
        for seed_vector in candidate_seeds:
            ik_result = solver.solve_ik(
                request.target_frame_pose,
                q0_deg=seed_vector,
                rotation_weight_mm=request.rotation_weight_mm,
                max_nfev=request.max_nfev,
                raise_on_fail=False,
            )
            if ik_result.success:
                return ik_result
            if best_failure_result is None or ik_result.cost < best_failure_result.cost:
                best_failure_result = ik_result

        if best_failure_result is not None:
            return best_failure_result

        return solver.solve_ik(
            request.target_frame_pose,
            q0_deg=request.seed_joints_deg,
            rotation_weight_mm=request.rotation_weight_mm,
            max_nfev=request.max_nfev,
            raise_on_fail=False,
        )

    def solve_all(self, request: IKSolveAllRequest) -> LocalIKSolutionSet:
        solver = NumericIKSolver(request.robot_model)
        seed_generator = AnalyticSeedGenerator(request.robot_model)

        merged_arm_candidates, merged_seed_candidates = self._build_candidate_seed_sets(
            solver,
            seed_generator,
            target_frame_pose=request.target_frame_pose,
            preferred_seed_joints_deg=request.seed_joints_deg,
            periodic_dedup_tolerance_deg=request.periodic_dedup_tolerance_deg,
            max_nfev=request.max_nfev,
            arm_position_tolerance_mm=request.arm_position_tolerance_mm,
            q2q3_seed_pairs_deg=request.q2q3_seed_pairs_deg,
            wrist_seed_triplets_deg=request.wrist_seed_triplets_deg,
        )

        return solver.refine_seed_candidates(
            target_frame_tcp_T=request.target_frame_pose,
            target_space="frame",
            seed_joints_deg=request.seed_joints_deg,
            filter_lower_deg=request.filter_lower_deg,
            filter_upper_deg=request.filter_upper_deg,
            arm_candidate_solutions_deg=merged_arm_candidates,
            candidate_seeds=merged_seed_candidates,
            rotation_weight_mm=request.rotation_weight_mm,
            max_nfev=request.max_nfev,
            pose_position_tolerance_mm=request.pose_position_tolerance_mm,
            pose_orientation_tolerance_deg=request.pose_orientation_tolerance_deg,
            periodic_dedup_tolerance_deg=request.periodic_dedup_tolerance_deg,
        )


class PureAnalyticIKBackend:
    """Pure analytic IK backend — no scipy optimization, direct closed-form solution.

    For spherical-wrist robots the analytic solution is exact (FK residual ≈ 0).
    Skipping numeric refinement gives 50-100x speed improvement over NumericIKBackend.
    """

    name = "pure_analytic"

    def solve(self, request: IKSolveRequest) -> IKResult:
        from .numeric_solver import NumericIKSolver
        seed_generator = AnalyticSeedGenerator(request.robot_model)
        _, full_seeds = seed_generator.generate_seed_candidates(
            request.target_frame_pose,
            preferred_seed_joints_deg=request.seed_joints_deg,
            periodic_dedup_tolerance_deg=config.LOCAL_IK_PERIODIC_DEDUP_TOLERANCE_DEG,
        )
        if not full_seeds:
            solver = NumericIKSolver(request.robot_model)
            return solver.solve_ik(
                request.target_frame_pose,
                q0_deg=request.seed_joints_deg,
                rotation_weight_mm=request.rotation_weight_mm,
                max_nfev=request.max_nfev,
                raise_on_fail=False,
            )
        # Sort by distance to seed if provided
        if request.seed_joints_deg is not None:
            full_seeds = sorted(
                full_seeds,
                key=lambda s: joint_distance_deg(s, request.seed_joints_deg),
            )
        best = full_seeds[0]
        fk = request.robot_model.fk_tcp_in_frame(best)
        pos_err = float(np.linalg.norm(fk[:3, 3] - request.target_frame_pose[:3, 3]))
        residual = np.zeros(6, dtype=float)
        return IKResult(
            q_deg=best,
            success=pos_err < 0.1,
            message="analytic" if pos_err < 0.1 else "analytic_imprecise",
            nfev=0,
            cost=pos_err,
            residual=residual,
            target_flange_base_T=request.robot_model.flange_target_from_frame_tcp_target(
                request.target_frame_pose
            ),
        )

    def solve_all(self, request: IKSolveAllRequest) -> "LocalIKSolutionSet":
        from .numeric_solver import NumericIKSolver
        from .kinematics import (
            LocalIKSolution,
            LocalIKSolutionSet,
            joints_within_limits,
            joint_distance_deg as _jd,
            rotation_error_deg,
        )

        seed_generator = AnalyticSeedGenerator(request.robot_model)
        _, full_seeds = seed_generator.generate_seed_candidates(
            request.target_frame_pose,
            preferred_seed_joints_deg=request.seed_joints_deg,
            periodic_dedup_tolerance_deg=request.periodic_dedup_tolerance_deg,
        )

        filter_lower = np.asarray(request.filter_lower_deg, dtype=float)
        filter_upper = np.asarray(request.filter_upper_deg, dtype=float)
        seed = None if request.seed_joints_deg is None else np.asarray(request.seed_joints_deg, dtype=float)

        # Validate each analytic seed via FK (no optimization needed)
        numeric = NumericIKSolver(request.robot_model)
        branch_solutions: list[LocalIKSolution] = []
        seen_joints: list[np.ndarray] = []

        for branch_index, seed_vec in enumerate(full_seeds):
            fk = request.robot_model.fk_tcp_in_frame(seed_vec)
            pos_err = float(np.linalg.norm(fk[:3, 3] - request.target_frame_pose[:3, 3]))
            rot_err = rotation_error_deg(fk, request.target_frame_pose)

            if pos_err > request.pose_position_tolerance_mm:
                continue
            if rot_err > request.pose_orientation_tolerance_deg:
                continue

            # Deduplicate
            if any(_jd(seed_vec, seen) <= request.periodic_dedup_tolerance_deg for seen in seen_joints):
                continue
            seen_joints.append(seed_vec)

            branch_solutions.append(
                LocalIKSolution(
                    index=branch_index,
                    branch_index=branch_index,
                    joints_deg=seed_vec,
                    seed_joints_deg=seed_vec,
                    turn_offsets=np.zeros(len(seed_vec), dtype=int),
                    within_robot_limits=request.robot_model.within_joint_limits(seed_vec),
                    within_filter_limits=joints_within_limits(seed_vec, filter_lower, filter_upper),
                    distance_to_seed_deg=None if seed is None else _jd(seed_vec, seed),
                    residual_norm=pos_err,
                    position_error_mm=pos_err,
                    orientation_error_deg=rot_err,
                )
            )

        # Turn expansion
        all_solutions: list[LocalIKSolution] = []
        for branch_sol in branch_solutions:
            for variant_joints, turn_offsets in numeric.expand_solution_turn_variants(branch_sol.joints_deg):
                dist = None if seed is None else _jd(variant_joints, seed)
                all_solutions.append(
                    LocalIKSolution(
                        index=-1,
                        branch_index=branch_sol.branch_index,
                        joints_deg=variant_joints,
                        seed_joints_deg=branch_sol.seed_joints_deg,
                        turn_offsets=turn_offsets,
                        within_robot_limits=True,
                        within_filter_limits=joints_within_limits(variant_joints, filter_lower, filter_upper),
                        distance_to_seed_deg=dist,
                        residual_norm=branch_sol.residual_norm,
                        position_error_mm=branch_sol.position_error_mm,
                        orientation_error_deg=branch_sol.orientation_error_deg,
                    )
                )

        all_solutions.sort(
            key=lambda s: (
                float("inf") if s.distance_to_seed_deg is None else s.distance_to_seed_deg,
                s.branch_index,
                tuple(s.turn_offsets.tolist()),
            )
        )
        for i, sol in enumerate(all_solutions):
            all_solutions[i] = LocalIKSolution(
                index=i,
                branch_index=sol.branch_index,
                joints_deg=sol.joints_deg,
                seed_joints_deg=sol.seed_joints_deg,
                turn_offsets=sol.turn_offsets,
                within_robot_limits=sol.within_robot_limits,
                within_filter_limits=sol.within_filter_limits,
                distance_to_seed_deg=sol.distance_to_seed_deg,
                residual_norm=sol.residual_norm,
                position_error_mm=sol.position_error_mm,
                orientation_error_deg=sol.orientation_error_deg,
            )

        filtered_solutions = [
            s for s in all_solutions if s.within_robot_limits and s.within_filter_limits
        ]

        return LocalIKSolutionSet(
            target_pose=request.target_frame_pose,
            target_space="frame",
            seed_joints_deg=None if seed is None else seed.copy(),
            robot_lower_limits_deg=request.robot_model.joint_min_deg.copy(),
            robot_upper_limits_deg=request.robot_model.joint_max_deg.copy(),
            filter_lower_limits_deg=filter_lower.copy(),
            filter_upper_limits_deg=filter_upper.copy(),
            arm_candidate_solutions_deg=[],
            branch_solutions=branch_solutions,
            all_solutions=all_solutions,
            filtered_solutions=filtered_solutions,
        )

    def solve_all_joint_vectors(self, request: IKSolveAllRequest) -> list[np.ndarray]:
        from .numeric_solver import NumericIKSolver
        from .kinematics import (
            joint_distance_deg as _jd,
            joints_within_limits,
            rotation_error_deg,
        )

        seed_generator = AnalyticSeedGenerator(request.robot_model)
        _, full_seeds = seed_generator.generate_seed_candidates(
            request.target_frame_pose,
            preferred_seed_joints_deg=request.seed_joints_deg,
            periodic_dedup_tolerance_deg=request.periodic_dedup_tolerance_deg,
        )

        filter_lower = np.asarray(request.filter_lower_deg, dtype=float)
        filter_upper = np.asarray(request.filter_upper_deg, dtype=float)
        seed = None if request.seed_joints_deg is None else np.asarray(request.seed_joints_deg, dtype=float)

        numeric = NumericIKSolver(request.robot_model)
        branch_solutions: list[tuple[int, np.ndarray]] = []
        seen_joints: list[np.ndarray] = []

        for branch_index, seed_vec in enumerate(full_seeds):
            fk = request.robot_model.fk_tcp_in_frame(seed_vec)
            pos_err = float(np.linalg.norm(fk[:3, 3] - request.target_frame_pose[:3, 3]))
            rot_err = rotation_error_deg(fk, request.target_frame_pose)

            if pos_err > request.pose_position_tolerance_mm:
                continue
            if rot_err > request.pose_orientation_tolerance_deg:
                continue
            if any(_jd(seed_vec, seen) <= request.periodic_dedup_tolerance_deg for seen in seen_joints):
                continue

            seen_joints.append(seed_vec)
            branch_solutions.append((branch_index, seed_vec))

        variant_records: list[tuple[np.ndarray, int, np.ndarray, float | None]] = []
        for branch_index, branch_joints in branch_solutions:
            for variant_joints, turn_offsets in numeric.expand_solution_turn_variants(branch_joints):
                if not joints_within_limits(variant_joints, filter_lower, filter_upper):
                    continue
                variant_records.append(
                    (
                        variant_joints,
                        branch_index,
                        turn_offsets,
                        None if seed is None else _jd(variant_joints, seed),
                    )
                )

        variant_records.sort(
            key=lambda item: (
                float("inf") if item[3] is None else item[3],
                item[1],
                tuple(item[2].tolist()),
            )
        )
        return [variant_joints.copy() for variant_joints, _branch_index, _turn_offsets, _distance in variant_records]


def build_ik_backend(backend: IKBackend | str | None = None) -> IKBackend:
    """Resolve a backend instance from an injected object or a simple name."""

    if backend is None:
        return PureAnalyticIKBackend()

    if not isinstance(backend, str):
        return backend

    normalized = backend.strip().lower()
    if normalized in {"numeric", "numericikbackend"}:
        return NumericIKBackend()
    if normalized in {"analytic", "analyticikbackend"}:
        return AnalyticIKBackend()
    if normalized in {"pure_analytic", "pureanalyticikbackend"}:
        return PureAnalyticIKBackend()
    raise ValueError(f"Unsupported SixAxisIK backend: {backend}")
