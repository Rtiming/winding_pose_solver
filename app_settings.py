from __future__ import annotations

from pathlib import Path

from src.app_runner import AppRuntimeSettings
from src.frame_math import FrameBuildOptions
from src.robodk_program import RoboDKMotionSettings


# ---------------------------------------------------------------------------
# Advanced settings
# These are the lower-frequency tuning parameters for geometry, verification,
# visualization, and RoboDK execution.
# ---------------------------------------------------------------------------

# Local process-frame construction
ZERO_VECTOR_TOLERANCE = 1e-9
NEAR_PARALLEL_TOLERANCE = 1e-6
CONTINUITY_DOT_THRESHOLD = 0.0

# Visualization
VISUALIZATION_STEP = 8
VISUALIZATION_VECTOR_SCALE = 8.0
SHOW_TANGENT = True
SHOW_NORMAL = True
SHOW_SIDE = True
SHOW_BAD_ROWS = True

# Pose-solver verification
ENABLE_SOLVER_VERIFICATION = True
VERIFICATION_ROW_IDS: tuple[int, ...] | None = None
VERIFICATION_TOLERANCE = 1e-6

# RoboDK execution
ROBOT_MOVE_TYPE = "MoveJ"
ROBOT_LINEAR_SPEED_MM_S = 200.0
ROBOT_JOINT_SPEED_DEG_S = 30.0
ROBOT_LINEAR_ACCEL_MM_S2 = 600.0
ROBOT_JOINT_ACCEL_DEG_S2 = 120.0
ROBOT_ROUNDING_MM = -1.0
HIDE_TARGETS_AFTER_GENERATION = True

# Hard joint constraints
A1_MIN_DEG = -150.0
A1_MAX_DEG = 30.0
A2_MAX_DEG = 115.0
JOINT_CONSTRAINT_TOLERANCE_DEG = 1e-6

# Continuity constraints
ENABLE_JOINT_CONTINUITY_CONSTRAINT = True
MAX_JOINT_STEP_DEG = (5.0, 5.0, 5.0, 180.0, 100.0, 180.0)
BRIDGE_TRIGGER_JOINT_DELTA_DEG = 20.0
BRIDGE_STEP_DEG = (2.0, 2.0, 2.0, 20.0, 10.0, 20.0)

# Frame-A origin optimization in Frame 2
# The search starts with +/-6 mm and widens to +/-12 mm only if needed.
FRAME_A_ORIGIN_YZ_ENVELOPE_SCHEDULE_MM = (6.0, 12.0)
FRAME_A_ORIGIN_YZ_STEP_SCHEDULE_MM = (4.0, 2.0, 1.0)
FRAME_A_ORIGIN_YZ_WINDOW_RADIUS = 8
FRAME_A_ORIGIN_YZ_MAX_PASSES = 4
FRAME_A_ORIGIN_YZ_INSERTION_COUNTS = (4, 8)
WRIST_PHASE_LOCK_THRESHOLD_DEG = 12.0


def build_frame_options() -> FrameBuildOptions:
    return FrameBuildOptions(
        zero_tolerance=ZERO_VECTOR_TOLERANCE,
        parallel_tolerance=NEAR_PARALLEL_TOLERANCE,
        continuity_dot_threshold=CONTINUITY_DOT_THRESHOLD,
    )


def build_motion_settings(
    *,
    enable_custom_smoothing_and_pose_selection: bool,
) -> RoboDKMotionSettings:
    return RoboDKMotionSettings(
        move_type=ROBOT_MOVE_TYPE,
        linear_speed_mm_s=ROBOT_LINEAR_SPEED_MM_S,
        joint_speed_deg_s=ROBOT_JOINT_SPEED_DEG_S,
        linear_accel_mm_s2=ROBOT_LINEAR_ACCEL_MM_S2,
        joint_accel_deg_s2=ROBOT_JOINT_ACCEL_DEG_S2,
        rounding_mm=ROBOT_ROUNDING_MM,
        hide_targets_after_generation=HIDE_TARGETS_AFTER_GENERATION,
        enable_custom_smoothing_and_pose_selection=enable_custom_smoothing_and_pose_selection,
        a1_min_deg=A1_MIN_DEG,
        a1_max_deg=A1_MAX_DEG,
        a2_max_deg=A2_MAX_DEG,
        joint_constraint_tolerance_deg=JOINT_CONSTRAINT_TOLERANCE_DEG,
        enable_joint_continuity_constraint=ENABLE_JOINT_CONTINUITY_CONSTRAINT,
        max_joint_step_deg=MAX_JOINT_STEP_DEG,
        bridge_trigger_joint_delta_deg=BRIDGE_TRIGGER_JOINT_DELTA_DEG,
        bridge_step_deg=BRIDGE_STEP_DEG,
        frame_a_origin_yz_envelope_schedule_mm=FRAME_A_ORIGIN_YZ_ENVELOPE_SCHEDULE_MM,
        frame_a_origin_yz_step_schedule_mm=FRAME_A_ORIGIN_YZ_STEP_SCHEDULE_MM,
        frame_a_origin_yz_window_radius=FRAME_A_ORIGIN_YZ_WINDOW_RADIUS,
        frame_a_origin_yz_max_passes=FRAME_A_ORIGIN_YZ_MAX_PASSES,
        frame_a_origin_yz_insertion_counts=FRAME_A_ORIGIN_YZ_INSERTION_COUNTS,
        wrist_phase_lock_threshold_deg=WRIST_PHASE_LOCK_THRESHOLD_DEG,
    )


def build_app_runtime_settings(
    *,
    validation_centerline_csv: str | Path,
    tool_poses_frame2_csv: str | Path,
    target_frame_origin_mm: tuple[float, float, float],
    target_frame_rotation_xyz_deg: tuple[float, float, float],
    enable_custom_smoothing_and_pose_selection: bool,
    robot_name: str,
    frame_name: str,
    program_name: str,
) -> AppRuntimeSettings:
    return AppRuntimeSettings(
        validation_centerline_csv=Path(validation_centerline_csv),
        tool_poses_frame2_csv=Path(tool_poses_frame2_csv),
        target_frame_origin_mm=tuple(float(value) for value in target_frame_origin_mm),
        target_frame_rotation_xyz_deg=tuple(
            float(value) for value in target_frame_rotation_xyz_deg
        ),
        robot_name=robot_name,
        frame_name=frame_name,
        program_name=program_name,
        frame_build_options=build_frame_options(),
        motion_settings=build_motion_settings(
            enable_custom_smoothing_and_pose_selection=enable_custom_smoothing_and_pose_selection,
        ),
        enable_solver_verification=ENABLE_SOLVER_VERIFICATION,
        verification_row_ids=VERIFICATION_ROW_IDS,
        verification_tolerance=VERIFICATION_TOLERANCE,
        visualization_step=VISUALIZATION_STEP,
        visualization_vector_scale=VISUALIZATION_VECTOR_SCALE,
        show_tangent=SHOW_TANGENT,
        show_normal=SHOW_NORMAL,
        show_side=SHOW_SIDE,
        show_bad_rows=SHOW_BAD_ROWS,
    )
