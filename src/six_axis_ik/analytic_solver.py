"""
解析 IK seed 生成模块。

当前实现只负责为 numeric solver 生成更好的初值，不直接把解析结果作为最终 IK 输出。
"""

from __future__ import annotations

import warnings
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
from scipy.spatial.transform import Rotation as R

from .kinematics import (
    JOINT_COUNT,
    RobotModel,
    as_joint_vector,
    as_transform,
    joint_distance_deg,
    normalize_angle_deg,
    rotation_error_deg,
)

_GEOMETRY_ATOL = 1e-6
_REACHABILITY_EPS = 1e-9
_SINGULAR_ANGLE_EPS_DEG = 1e-4


def _rotation_z_deg(angle_deg: float) -> np.ndarray:
    """构造绕 Z 轴的旋转矩阵。"""
    angle_rad = np.deg2rad(angle_deg)
    cosine = np.cos(angle_rad)
    sine = np.sin(angle_rad)
    return np.array(
        [
            [cosine, -sine, 0.0],
            [sine, cosine, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )


class AnalyticSeedGenerator:
    """面向当前 SixAxisIK 几何模型生成解析 IK seeds。"""

    def __init__(self, robot_model: RobotModel):
        self.robot_model = robot_model
        self._validate_supported_geometry()

        self.joint2_point_mm = self.robot_model.joint_axis_points_base_mm[1].copy()
        self.joint3_point_mm = self.robot_model.joint_axis_points_base_mm[2].copy()
        self.home_wrist_center_mm = self.robot_model.home_wrist_center_base_mm.copy()
        self.arm_link_1_mm = float(np.linalg.norm(self.joint3_point_mm - self.joint2_point_mm))
        self.arm_link_2_vector_mm = self.home_wrist_center_mm - self.joint3_point_mm
        self.arm_link_2_mm = float(np.linalg.norm(self.arm_link_2_vector_mm))
        self.arm_link_2_home_angle_rad = float(
            np.arctan2(self.arm_link_2_vector_mm[0], self.arm_link_2_vector_mm[2])
        )
        self.home_flange_rotation = self.robot_model.home_flange_T[:3, :3].copy()
        self._ry_pi = R.from_euler("Y", 180.0, degrees=True).as_matrix()

    def generate_seed_candidates(
        self,
        target_frame_pose: np.ndarray,
        *,
        preferred_seed_joints_deg: Optional[Sequence[float]],
        periodic_dedup_tolerance_deg: float,
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """生成解析 arm candidates 和完整 6 轴 seed。"""
        target_frame_pose = as_transform(target_frame_pose, "target_frame_pose")
        preferred_seed = (
            None
            if preferred_seed_joints_deg is None
            else as_joint_vector(preferred_seed_joints_deg, "preferred_seed_joints_deg")
        )
        target_flange_base_T = self.robot_model.flange_target_from_frame_tcp_target(target_frame_pose)
        wrist_center_mm = self.robot_model.wrist_center_from_flange_pose(target_flange_base_T)

        arm_candidates = self._solve_arm_candidates(
            target_wrist_center_mm=wrist_center_mm,
            preferred_seed_joints_deg=preferred_seed,
            periodic_dedup_tolerance_deg=periodic_dedup_tolerance_deg,
        )
        full_joint_seeds: List[np.ndarray] = []
        for arm_candidate in arm_candidates:
            wrist_candidates = self._solve_wrist_candidates(
                target_flange_base_T=target_flange_base_T,
                arm_candidate_deg=arm_candidate,
                preferred_seed_joints_deg=preferred_seed,
                periodic_dedup_tolerance_deg=periodic_dedup_tolerance_deg,
            )
            for wrist_candidate in wrist_candidates:
                full_joint_seed = self._canonicalize_full_joint_seed(
                    np.r_[arm_candidate, wrist_candidate],
                    preferred_seed_joints_deg=preferred_seed,
                )
                if full_joint_seed is not None:
                    full_joint_seeds.append(full_joint_seed)

        arm_candidates = self._sort_arm_candidates(arm_candidates, preferred_seed)
        full_joint_seeds = self._sort_full_joint_seeds(
            self._deduplicate_periodic_joint_seeds(full_joint_seeds, periodic_dedup_tolerance_deg),
            preferred_seed,
            target_flange_base_T=target_flange_base_T,
        )
        return arm_candidates, full_joint_seeds

    def _validate_supported_geometry(self) -> None:
        """校验当前机器人是否满足 v1 解析 seed 的几何假设。"""
        expected_axes = np.array(
            [
                [0.0, 0.0, 1.0],
                [0.0, 1.0, 0.0],
                [0.0, 1.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [1.0, 0.0, 0.0],
            ],
            dtype=float,
        )
        if not np.allclose(self.robot_model.joint_axis_directions_base, expected_axes, atol=_GEOMETRY_ATOL):
            raise ValueError(
                "AnalyticIKBackend v1 only supports the current spherical-wrist axis layout."
            )

        wrist_points = self.robot_model.joint_axis_points_base_mm[3:6]
        if not np.allclose(wrist_points, wrist_points[0], atol=_GEOMETRY_ATOL):
            raise ValueError(
                "AnalyticIKBackend v1 expects joints 4/5/6 to share one wrist center point."
            )

        if not np.isclose(self.robot_model.joint_axis_points_base_mm[1, 0], self.robot_model.joint_axis_points_base_mm[2, 0], atol=_GEOMETRY_ATOL):
            raise ValueError("AnalyticIKBackend v1 expects joints 2 and 3 to lie on one arm plane.")
        if not np.allclose(self.robot_model.joint_axis_points_base_mm[1:4, 1], 0.0, atol=_GEOMETRY_ATOL):
            raise ValueError("AnalyticIKBackend v1 expects the calibrated arm geometry to lie in the XZ home plane.")

    def _solve_q1_candidates(
        self,
        target_wrist_center_mm: np.ndarray,
        preferred_seed_joints_deg: Optional[np.ndarray],
    ) -> List[float]:
        """枚举解析 shoulder 分支的 q1 候选。"""
        x_value = float(target_wrist_center_mm[0])
        y_value = float(target_wrist_center_mm[1])
        radial_distance = float(np.hypot(x_value, y_value))

        if radial_distance <= _REACHABILITY_EPS:
            preferred_q1 = 0.0 if preferred_seed_joints_deg is None else float(preferred_seed_joints_deg[0])
            raw_candidates = (preferred_q1, preferred_q1 + 180.0)
        else:
            azimuth_deg = float(np.rad2deg(np.arctan2(y_value, x_value)))
            raw_candidates = (-azimuth_deg, -azimuth_deg + 180.0, -azimuth_deg - 180.0)

        q1_candidates: List[float] = []
        for raw_q1_deg in raw_candidates:
            q1_deg = normalize_angle_deg(raw_q1_deg)
            if not (self.robot_model.joint_min_deg[0] <= q1_deg <= self.robot_model.joint_max_deg[0]):
                continue
            if any(abs(q1_deg - saved_q1) <= _GEOMETRY_ATOL for saved_q1 in q1_candidates):
                continue
            q1_candidates.append(q1_deg)

        if preferred_seed_joints_deg is None:
            return q1_candidates
        return sorted(
            q1_candidates,
            key=lambda candidate: abs(normalize_angle_deg(candidate - preferred_seed_joints_deg[0])),
        )

    def _equivalent_joint_values_within_limits(
        self,
        raw_angle_deg: float,
        joint_index: int,
    ) -> List[float]:
        """列出一个关节角在当前硬限位内的所有 360 度等价表示。"""
        lower_deg = float(self.robot_model.joint_min_deg[joint_index])
        upper_deg = float(self.robot_model.joint_max_deg[joint_index])
        min_turn = int(np.ceil((lower_deg - raw_angle_deg) / 360.0 - 1e-12))
        max_turn = int(np.floor((upper_deg - raw_angle_deg) / 360.0 + 1e-12))

        candidates: List[float] = []
        for turn_index in range(min_turn, max_turn + 1):
            candidate = float(raw_angle_deg + 360.0 * turn_index)
            if candidate < lower_deg - _GEOMETRY_ATOL or candidate > upper_deg + _GEOMETRY_ATOL:
                continue
            candidate = float(np.clip(candidate, lower_deg, upper_deg))
            if any(abs(candidate - saved) <= _GEOMETRY_ATOL for saved in candidates):
                continue
            candidates.append(candidate)
        return candidates

    def _select_joint_representation(
        self,
        raw_angle_deg: float,
        joint_index: int,
        preferred_angle_deg: Optional[float],
    ) -> Optional[float]:
        """在硬限位内挑出一个最适合作为 seed 的角度表示。"""
        candidates = self._equivalent_joint_values_within_limits(raw_angle_deg, joint_index)
        if not candidates:
            return None

        if preferred_angle_deg is not None:
            return min(
                candidates,
                key=lambda candidate: (
                    abs(candidate - preferred_angle_deg),
                    abs(normalize_angle_deg(candidate - preferred_angle_deg)),
                    abs(candidate),
                    candidate,
                ),
            )

        normalized_raw_angle_deg = normalize_angle_deg(raw_angle_deg)
        return min(
            candidates,
            key=lambda candidate: (
                abs(candidate),
                abs(candidate - normalized_raw_angle_deg),
                candidate,
            ),
        )

    def _fit_joint_values_to_limits(
        self,
        raw_joint_values_deg: Iterable[float],
        joint_indices: Sequence[int],
        preferred_seed_joints_deg: Optional[np.ndarray],
    ) -> Optional[np.ndarray]:
        """把一组解析角度映射成硬限位内的等价表示。"""
        raw_joint_values = np.asarray(list(raw_joint_values_deg), dtype=float).reshape(-1)
        if raw_joint_values.size != len(joint_indices):
            raise ValueError(
                f"raw_joint_values_deg must contain {len(joint_indices)} values, got {raw_joint_values.size}."
            )

        fitted_values: List[float] = []
        for value, joint_index in zip(raw_joint_values, joint_indices):
            preferred_value = (
                None if preferred_seed_joints_deg is None else float(preferred_seed_joints_deg[joint_index])
            )
            fitted_value = self._select_joint_representation(value, joint_index, preferred_value)
            if fitted_value is None:
                return None
            fitted_values.append(fitted_value)
        return np.asarray(fitted_values, dtype=float)

    def _solve_arm_candidates(
        self,
        target_wrist_center_mm: np.ndarray,
        preferred_seed_joints_deg: Optional[np.ndarray],
        periodic_dedup_tolerance_deg: float,
    ) -> List[np.ndarray]:
        """解析求解 1~3 轴 major branches。"""
        q1_candidates = self._solve_q1_candidates(target_wrist_center_mm, preferred_seed_joints_deg)
        arm_candidates: List[np.ndarray] = []

        for q1_deg in q1_candidates:
            wrist_in_arm_plane = _rotation_z_deg(q1_deg) @ target_wrist_center_mm
            target_x_mm = float(wrist_in_arm_plane[0] - self.joint2_point_mm[0])
            target_z_mm = float(wrist_in_arm_plane[2] - self.joint2_point_mm[2])
            radial_distance = float(np.hypot(target_x_mm, target_z_mm))

            cosine_beta = (
                radial_distance**2 - self.arm_link_1_mm**2 - self.arm_link_2_mm**2
            ) / (2.0 * self.arm_link_1_mm * self.arm_link_2_mm)
            if cosine_beta < -1.0 - _REACHABILITY_EPS or cosine_beta > 1.0 + _REACHABILITY_EPS:
                continue
            cosine_beta = float(np.clip(cosine_beta, -1.0, 1.0))

            target_angle_rad = float(np.arctan2(target_x_mm, target_z_mm))
            beta_candidates = (np.arccos(cosine_beta), -np.arccos(cosine_beta))
            for beta_rad in beta_candidates:
                q3_deg_raw = float(np.rad2deg(beta_rad - self.arm_link_2_home_angle_rad))
                q2_deg_raw = float(
                    np.rad2deg(
                        target_angle_rad
                        - np.arctan2(
                            self.arm_link_2_mm * np.sin(beta_rad),
                            self.arm_link_1_mm + self.arm_link_2_mm * np.cos(beta_rad),
                        )
                    )
                )
                candidate = self._fit_joint_values_to_limits(
                    [q1_deg, q2_deg_raw, q3_deg_raw],
                    joint_indices=(0, 1, 2),
                    preferred_seed_joints_deg=preferred_seed_joints_deg,
                )
                if candidate is None:
                    continue

                solved_wrist_center = self.robot_model.fk_wrist_center_in_robot_base(
                    np.r_[candidate, np.zeros(3, dtype=float)]
                )
                wrist_center_error_mm = solved_wrist_center - target_wrist_center_mm
                if float(wrist_center_error_mm @ wrist_center_error_mm) > 1e-10:
                    continue
                arm_candidates.append(candidate)

        return self._deduplicate_arm_candidates(arm_candidates, periodic_dedup_tolerance_deg)

    def _solve_wrist_candidates(
        self,
        target_flange_base_T: np.ndarray,
        arm_candidate_deg: np.ndarray,
        preferred_seed_joints_deg: Optional[np.ndarray],
        periodic_dedup_tolerance_deg: float,
    ) -> List[np.ndarray]:
        """根据 arm branch 解析生成 wrist seeds。"""
        zero_wrist_joints = np.r_[arm_candidate_deg, np.zeros(3, dtype=float)]
        zero_wrist_flange_T = self.robot_model.fk_flange(zero_wrist_joints)
        wrist_rotation = (
            self.home_flange_rotation
            @ zero_wrist_flange_T[:3, :3].T
            @ target_flange_base_T[:3, :3]
            @ self.home_flange_rotation.T
        )

        wrist_candidates: List[np.ndarray] = []
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            xyx_deg = np.array(R.from_matrix(wrist_rotation).as_euler("XYX", degrees=True), dtype=float)
        wrist_candidates.extend(self._wrist_candidates_from_xyx(xyx_deg))

        q5_magnitude = abs(normalize_angle_deg(float(xyx_deg[1])))
        if q5_magnitude <= _SINGULAR_ANGLE_EPS_DEG:
            wrist_candidates.extend(
                self._wrist_singular_fallback_candidates(
                    wrist_rotation=wrist_rotation,
                    q5_deg=0.0,
                    preferred_seed_joints_deg=preferred_seed_joints_deg,
                )
            )
        if abs(q5_magnitude - 180.0) <= _SINGULAR_ANGLE_EPS_DEG:
            wrist_candidates.extend(
                self._wrist_singular_fallback_candidates(
                    wrist_rotation=wrist_rotation,
                    q5_deg=180.0,
                    preferred_seed_joints_deg=preferred_seed_joints_deg,
                )
            )

        wrist_candidates = self._deduplicate_wrist_candidates(
            wrist_candidates,
            periodic_dedup_tolerance_deg=periodic_dedup_tolerance_deg,
        )
        return self._sort_wrist_candidates(wrist_candidates, preferred_seed_joints_deg)

    def _wrist_candidates_from_xyx(self, xyx_deg: np.ndarray) -> List[np.ndarray]:
        """把 XYX 欧拉角转换成 commanded wrist joints，并补齐另一组等价分支。"""
        alpha_deg, beta_deg, gamma_deg = (float(value) for value in xyx_deg)
        branches = [
            np.array([-alpha_deg, beta_deg, -gamma_deg], dtype=float),
            np.array([-(alpha_deg + 180.0), -beta_deg, -(gamma_deg + 180.0)], dtype=float),
        ]
        return [np.asarray(branch, dtype=float) for branch in branches]

    def _wrist_singular_fallback_candidates(
        self,
        wrist_rotation: np.ndarray,
        q5_deg: float,
        preferred_seed_joints_deg: Optional[np.ndarray],
    ) -> List[np.ndarray]:
        """在 wrist singularity 附近保留 arm branch，并生成一小组确定性的 fallback wrist seeds。"""
        seed_q4 = 0.0 if preferred_seed_joints_deg is None else normalize_angle_deg(preferred_seed_joints_deg[3])
        seed_q6 = 0.0 if preferred_seed_joints_deg is None else normalize_angle_deg(preferred_seed_joints_deg[5])

        if abs(q5_deg) <= _SINGULAR_ANGLE_EPS_DEG:
            combined_x_deg = float(
                np.rad2deg(np.arctan2(wrist_rotation[2, 1], wrist_rotation[1, 1]))
            )
            desired_sum_deg = normalize_angle_deg(-combined_x_deg)
            candidates = [
                np.array([0.0, 0.0, desired_sum_deg], dtype=float),
                np.array([desired_sum_deg, 0.0, 0.0], dtype=float),
                np.array([seed_q4, 0.0, desired_sum_deg - seed_q4], dtype=float),
                np.array([desired_sum_deg - seed_q6, 0.0, seed_q6], dtype=float),
            ]
        else:
            wrist_without_pi = wrist_rotation @ self._ry_pi
            difference_deg = float(
                np.rad2deg(np.arctan2(wrist_without_pi[2, 1], wrist_without_pi[1, 1]))
            )
            desired_difference_deg = normalize_angle_deg(difference_deg)
            candidates = [
                np.array([0.0, 180.0, desired_difference_deg], dtype=float),
                np.array([-desired_difference_deg, 180.0, 0.0], dtype=float),
                np.array([seed_q4, 180.0, seed_q4 + desired_difference_deg], dtype=float),
                np.array([seed_q6 - desired_difference_deg, 180.0, seed_q6], dtype=float),
            ]

        return [np.asarray(candidate, dtype=float) for candidate in candidates]

    def _canonicalize_wrist_seed(
        self,
        wrist_seed_deg: Sequence[float],
        preferred_seed_joints_deg: Optional[np.ndarray],
    ) -> Optional[np.ndarray]:
        """把 wrist seed 映射到当前机器人硬限位内的等价表示。"""
        wrist_seed = np.asarray(list(wrist_seed_deg), dtype=float).reshape(-1)
        if wrist_seed.size != 3:
            raise ValueError(f"wrist_seed_deg must contain 3 values, got {wrist_seed.size}.")
        return self._fit_joint_values_to_limits(
            wrist_seed,
            joint_indices=(3, 4, 5),
            preferred_seed_joints_deg=preferred_seed_joints_deg,
        )

    def _canonicalize_full_joint_seed(
        self,
        joint_seed_deg: Sequence[float],
        preferred_seed_joints_deg: Optional[np.ndarray],
    ) -> Optional[np.ndarray]:
        """把完整 6 轴 seed 映射到当前机器人硬限位内的等价表示。"""
        joint_seed = as_joint_vector(joint_seed_deg, "joint_seed_deg")
        return self._fit_joint_values_to_limits(
            joint_seed,
            joint_indices=tuple(range(JOINT_COUNT)),
            preferred_seed_joints_deg=preferred_seed_joints_deg,
        )

    def _deduplicate_arm_candidates(
        self,
        arm_candidates_deg: Sequence[Sequence[float]],
        periodic_dedup_tolerance_deg: float,
    ) -> List[np.ndarray]:
        """对解析 arm candidates 做周期去重。"""
        unique_candidates: List[np.ndarray] = []
        for candidate in arm_candidates_deg:
            candidate_vector = np.asarray(candidate, dtype=float).reshape(-1)
            if any(
                joint_distance_deg(
                    np.r_[candidate_vector, np.zeros(3, dtype=float)],
                    np.r_[saved, np.zeros(3, dtype=float)],
                )
                <= periodic_dedup_tolerance_deg
                for saved in unique_candidates
            ):
                continue
            unique_candidates.append(candidate_vector.copy())
        return unique_candidates

    def _deduplicate_wrist_candidates(
        self,
        wrist_candidates_deg: Sequence[Sequence[float]],
        periodic_dedup_tolerance_deg: float,
    ) -> List[np.ndarray]:
        """对 wrist seeds 做周期去重。"""
        unique_candidates: List[np.ndarray] = []
        for candidate in wrist_candidates_deg:
            candidate_vector = self._canonicalize_wrist_seed(
                candidate,
                preferred_seed_joints_deg=None,
            )
            if candidate_vector is None:
                continue
            if any(
                joint_distance_deg(
                    np.r_[np.zeros(3, dtype=float), candidate_vector],
                    np.r_[np.zeros(3, dtype=float), saved],
                )
                <= periodic_dedup_tolerance_deg
                for saved in unique_candidates
            ):
                continue
            unique_candidates.append(candidate_vector)
        return unique_candidates

    def _deduplicate_full_joint_seeds(
        self,
        joint_seeds_deg: Sequence[Sequence[float]],
        periodic_dedup_tolerance_deg: float,
    ) -> List[np.ndarray]:
        """对完整解析 seed 做周期去重。"""
        unique_seeds: List[np.ndarray] = []
        for candidate in joint_seeds_deg:
            candidate_vector = self._canonicalize_full_joint_seed(
                candidate,
                preferred_seed_joints_deg=None,
            )
            if candidate_vector is None:
                continue
            if any(
                joint_distance_deg(candidate_vector, saved) <= periodic_dedup_tolerance_deg
                for saved in unique_seeds
            ):
                continue
            unique_seeds.append(candidate_vector)
        return unique_seeds

    def _deduplicate_periodic_joint_seeds(
        self,
        joint_seeds_deg: Sequence[Sequence[float]],
        periodic_dedup_tolerance_deg: float,
    ) -> List[np.ndarray]:
        """对已经规范化过的完整 joint seeds 做周期去重。"""
        unique_seeds: List[np.ndarray] = []
        for candidate in joint_seeds_deg:
            candidate_vector = as_joint_vector(candidate, "candidate")
            if any(
                joint_distance_deg(candidate_vector, saved) <= periodic_dedup_tolerance_deg
                for saved in unique_seeds
            ):
                continue
            unique_seeds.append(candidate_vector.copy())
        return unique_seeds

    def _sort_arm_candidates(
        self,
        arm_candidates: Sequence[np.ndarray],
        preferred_seed_joints_deg: Optional[np.ndarray],
    ) -> List[np.ndarray]:
        """按 seed 距离或字典序排序 arm candidates。"""
        if preferred_seed_joints_deg is None:
            return sorted(arm_candidates, key=lambda candidate: tuple(np.round(candidate, 9).tolist()))
        return sorted(
            arm_candidates,
            key=lambda candidate: (
                joint_distance_deg(
                    np.r_[candidate, np.zeros(3, dtype=float)],
                    np.r_[preferred_seed_joints_deg[:3], np.zeros(3, dtype=float)],
                ),
                tuple(np.round(candidate, 9).tolist()),
            ),
        )

    def _sort_wrist_candidates(
        self,
        wrist_candidates: Sequence[np.ndarray],
        preferred_seed_joints_deg: Optional[np.ndarray],
    ) -> List[np.ndarray]:
        """按 seed 距离或字典序排序 wrist seeds。"""
        if preferred_seed_joints_deg is None:
            return sorted(wrist_candidates, key=lambda candidate: tuple(np.round(candidate, 9).tolist()))
        preferred_wrist = preferred_seed_joints_deg[3:6]
        return sorted(
            wrist_candidates,
            key=lambda candidate: (
                joint_distance_deg(
                    np.r_[np.zeros(3, dtype=float), candidate],
                    np.r_[np.zeros(3, dtype=float), preferred_wrist],
                ),
                tuple(np.round(candidate, 9).tolist()),
            ),
        )

    def _sort_full_joint_seeds(
        self,
        joint_seeds: Sequence[np.ndarray],
        preferred_seed_joints_deg: Optional[np.ndarray],
        target_flange_base_T: np.ndarray,
    ) -> List[np.ndarray]:
        """按 seed 距离或字典序排序完整 6 轴 seeds。"""
        if preferred_seed_joints_deg is None:
            return sorted(joint_seeds, key=lambda candidate: tuple(np.round(candidate, 9).tolist()))

        def seed_quality(candidate: np.ndarray) -> tuple[float, float]:
            solved_flange_base_T = self.robot_model.fk_flange(candidate)
            position_error_mm = float(
                np.linalg.norm(solved_flange_base_T[:3, 3] - target_flange_base_T[:3, 3])
            )
            orientation_error = float(rotation_error_deg(solved_flange_base_T, target_flange_base_T))
            return position_error_mm, orientation_error

        return sorted(
            joint_seeds,
            key=lambda candidate: (
                seed_quality(candidate),
                joint_distance_deg(candidate, preferred_seed_joints_deg),
                tuple(np.round(candidate, 9).tolist()),
            ),
        )
