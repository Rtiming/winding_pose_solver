from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class RoboDKMotionSettings:
    move_type: str = "MoveJ"
    linear_speed_mm_s: float = 200.0
    joint_speed_deg_s: float = 30.0
    linear_accel_mm_s2: float = 600.0
    joint_accel_deg_s2: float = 120.0
    rounding_mm: float = -1.0
    hide_targets_after_generation: bool = True

    enable_custom_smoothing_and_pose_selection: bool = True

    a1_min_deg: float = -150.0
    a1_max_deg: float = 30.0
    a2_max_deg: float = 115.0
    joint_constraint_tolerance_deg: float = 1e-6

    enable_joint_continuity_constraint: bool = True
    max_joint_step_deg: tuple[float, ...] = (5.0, 5.0, 5.0, 180.0, 100.0, 180.0)

    bridge_trigger_joint_delta_deg: float = 20.0
    bridge_step_deg: tuple[float, ...] = (2.0, 2.0, 2.0, 20.0, 10.0, 20.0)

    frame_a_origin_yz_envelope_schedule_mm: tuple[float, ...] = (6.0,)
    frame_a_origin_yz_step_schedule_mm: tuple[float, ...] = (4.0, 2.0, 1.0)
    frame_a_origin_yz_window_radius: int = 8
    frame_a_origin_yz_max_passes: int = 4
    frame_a_origin_yz_insertion_counts: tuple[int, ...] = (4, 8)

    wrist_phase_lock_threshold_deg: float = 12.0

    # Minimum meaningful joint delta for a config switch to be counted as a
    # continuity problem.  Config transitions with smaller joint steps (e.g. a
    # wrist-flip bit changing when J5 ≈ 0°) are benign and should not trigger
    # bridge repairs or invalidate the path.
    config_switch_min_joint_delta_deg: float = 5.0

    # Penalty added to the DP transition cost when the overhead/rear bit (KUKA
    # "Rear" bit, bit0) changes.  Set high together with lower/flip to force the
    # DP to commit to a single consistent config family throughout the path.
    rear_switch_penalty: float = 2000.0

    # Penalty added to the DP transition cost when the elbow-up/elbow-down bit
    # (KUKA "Lower" bit) changes.  Set very high to prevent the DP from ever
    # voluntarily switching elbow families; elbow-down is available in every row
    # of this winding path, so there is no need to switch.
    lower_switch_penalty: float = 2000.0

    # Penalty for the wrist-flip bit (KUKA J5 sign) changing.  Large wrist-flip
    # transitions involve ~167° J4/J6 jumps.  Set high to force the path to pick
    # ONE wrist-flip family and stay there; both flip and no-flip are available
    # in every row of this winding path.
    flip_switch_penalty: float = 2000.0

    # IK 后端选择："robodk" 使用 RoboDK 求解器，"six_axis_ik" 使用内置本地求解器。
    ik_backend: str = "robodk"


def validate_motion_settings(settings: RoboDKMotionSettings) -> RoboDKMotionSettings:
    if settings.move_type not in {"MoveL", "MoveJ"}:
        raise ValueError(f"Unsupported move type: {settings.move_type}")
    if settings.linear_speed_mm_s <= 0.0:
        raise ValueError("Linear speed must be positive.")
    if settings.joint_speed_deg_s <= 0.0:
        raise ValueError("Joint speed must be positive.")
    if settings.linear_accel_mm_s2 <= 0.0:
        raise ValueError("Linear acceleration must be positive.")
    if settings.joint_accel_deg_s2 <= 0.0:
        raise ValueError("Joint acceleration must be positive.")
    if settings.rounding_mm < -1.0:
        raise ValueError("Rounding must be -1 or greater.")
    if not settings.enable_custom_smoothing_and_pose_selection:
        return settings
    if settings.a2_max_deg <= 0.0:
        raise ValueError("A2 max constraint must be positive.")
    if settings.joint_constraint_tolerance_deg < 0.0:
        raise ValueError("Joint constraint tolerance must be non-negative.")
    if not settings.max_joint_step_deg:
        raise ValueError("Joint continuity step limits must not be empty.")
    if any(limit <= 0.0 for limit in settings.max_joint_step_deg):
        raise ValueError("Each joint continuity step limit must be positive.")
    if settings.bridge_trigger_joint_delta_deg <= 0.0:
        raise ValueError("Bridge trigger joint delta must be positive.")
    if not settings.bridge_step_deg:
        raise ValueError("Bridge step limits must not be empty.")
    if any(limit <= 0.0 for limit in settings.bridge_step_deg):
        raise ValueError("Each bridge-step reference limit must be positive.")
    if any(envelope < 0.0 for envelope in settings.frame_a_origin_yz_envelope_schedule_mm):
        raise ValueError("Each Frame-A origin Y/Z envelope must be non-negative.")
    if any(step <= 0.0 for step in settings.frame_a_origin_yz_step_schedule_mm):
        raise ValueError("Each Frame-A origin Y/Z search step must be positive.")
    if settings.frame_a_origin_yz_window_radius < 0:
        raise ValueError("Frame-A origin Y/Z window radius must be non-negative.")
    if settings.frame_a_origin_yz_max_passes < 0:
        raise ValueError("Frame-A origin Y/Z max passes must be non-negative.")
    if any(count <= 0 for count in settings.frame_a_origin_yz_insertion_counts):
        raise ValueError("Each insertion count must be positive.")
    if settings.wrist_phase_lock_threshold_deg <= 0.0:
        raise ValueError("Wrist phase-lock threshold must be positive.")
    if settings.config_switch_min_joint_delta_deg < 0.0:
        raise ValueError("Config-switch minimum joint delta must be non-negative.")
    if settings.rear_switch_penalty < 0.0:
        raise ValueError("Rear-switch penalty must be non-negative.")
    if settings.lower_switch_penalty < 0.0:
        raise ValueError("Lower-switch penalty must be non-negative.")
    if settings.flip_switch_penalty < 0.0:
        raise ValueError("Flip-switch penalty must be non-negative.")
    return settings


def build_motion_settings_from_dict(payload: dict[str, Any]) -> RoboDKMotionSettings:
    return validate_motion_settings(RoboDKMotionSettings(**payload))


def motion_settings_to_dict(settings: RoboDKMotionSettings) -> dict[str, Any]:
    return asdict(settings)
