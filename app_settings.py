from __future__ import annotations

from pathlib import Path

from src.core.frame_math import FrameBuildOptions
from src.core.motion_settings import RoboDKMotionSettings
from src.runtime.app import AppRuntimeSettings


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
# Hide generated program targets after each target has stored pose and joint
# payloads.  Runtime code hides target items individually instead of calling
# Program.ShowTargets(False), which can make RoboDK display "target undefined".
HIDE_TARGETS_AFTER_GENERATION = True

# Hard joint constraints
A1_MIN_DEG = -150.0
A1_MAX_DEG = 30.0
A2_MAX_DEG = 115.0
JOINT_CONSTRAINT_TOLERANCE_DEG = 1e-6

# Continuity constraints
ENABLE_JOINT_CONTINUITY_CONSTRAINT = True
MAX_JOINT_STEP_DEG = (5.0, 5.0, 5.0, 45.0, 30.0, 45.0)
IK_MAX_CANDIDATES_PER_CONFIG_FAMILY = 4
USE_GUIDED_CONFIG_PATH = True
BRIDGE_TRIGGER_JOINT_DELTA_DEG = 30.0
BRIDGE_STEP_DEG = (2.0, 2.0, 2.0, 20.0, 10.0, 20.0)

# Frame-A origin optimization in Frame 2
# The search starts with +/-6 mm and widens to +/-12 mm only if needed.
FRAME_A_ORIGIN_YZ_ENVELOPE_SCHEDULE_MM = (6.0, 12.0)
FRAME_A_ORIGIN_YZ_STEP_SCHEDULE_MM = (4.0, 2.0, 1.0)
FRAME_A_ORIGIN_YZ_WINDOW_RADIUS = 8
FRAME_A_ORIGIN_YZ_MAX_PASSES = 4
FRAME_A_ORIGIN_YZ_INSERTION_COUNTS = (4, 8)
LOCK_FRAME_A_ORIGIN_YZ_PROFILE_ENDPOINTS = True
WRIST_PHASE_LOCK_THRESHOLD_DEG = 12.0
ENABLE_JOINT_SPACE_BRIDGE_REPAIR = True
JOINT_SPACE_BRIDGE_MAX_INSERTIONS_PER_SEGMENT = 24
JOINT_SPACE_BRIDGE_MAX_TCP_DEVIATION_MM = 20.0
JOINT_SPACE_BRIDGE_MAX_TCP_PATH_RATIO = 4.0
LOCAL_PARALLEL_WORKERS = 0  # 0 = auto
LOCAL_PARALLEL_MIN_BATCH_SIZE = 8
BIG_CIRCLE_STEP_DEG_THRESHOLD = 170.0
BRANCH_FLIP_RATIO_THRESHOLD = 8.0
BRANCH_FLIP_RATIO_EPS_MM = 1e-3
OFFICIAL_WORST_JOINT_STEP_DEG_LIMIT = 60.0
ENABLE_SAME_FAMILY_SEGMENT_REPAIR = True
SAME_FAMILY_REPAIR_MAX_SEGMENTS = 8


def build_frame_options() -> FrameBuildOptions:
    return FrameBuildOptions(
        zero_tolerance=ZERO_VECTOR_TOLERANCE,
        parallel_tolerance=NEAR_PARALLEL_TOLERANCE,
        continuity_dot_threshold=CONTINUITY_DOT_THRESHOLD,
    )


def build_motion_settings(
    *,
    enable_custom_smoothing_and_pose_selection: bool,
    ik_backend: str = "robodk",
    local_parallel_workers: int = LOCAL_PARALLEL_WORKERS,
    local_parallel_min_batch_size: int = LOCAL_PARALLEL_MIN_BATCH_SIZE,
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
        ik_max_candidates_per_config_family=IK_MAX_CANDIDATES_PER_CONFIG_FAMILY,
        use_guided_config_path=USE_GUIDED_CONFIG_PATH,
        bridge_trigger_joint_delta_deg=BRIDGE_TRIGGER_JOINT_DELTA_DEG,
        bridge_step_deg=BRIDGE_STEP_DEG,
        frame_a_origin_yz_envelope_schedule_mm=FRAME_A_ORIGIN_YZ_ENVELOPE_SCHEDULE_MM,
        frame_a_origin_yz_step_schedule_mm=FRAME_A_ORIGIN_YZ_STEP_SCHEDULE_MM,
        frame_a_origin_yz_window_radius=FRAME_A_ORIGIN_YZ_WINDOW_RADIUS,
        frame_a_origin_yz_max_passes=FRAME_A_ORIGIN_YZ_MAX_PASSES,
        frame_a_origin_yz_insertion_counts=FRAME_A_ORIGIN_YZ_INSERTION_COUNTS,
        lock_frame_a_origin_yz_profile_endpoints=LOCK_FRAME_A_ORIGIN_YZ_PROFILE_ENDPOINTS,
        wrist_phase_lock_threshold_deg=WRIST_PHASE_LOCK_THRESHOLD_DEG,
        enable_joint_space_bridge_repair=ENABLE_JOINT_SPACE_BRIDGE_REPAIR,
        joint_space_bridge_max_insertions_per_segment=(
            JOINT_SPACE_BRIDGE_MAX_INSERTIONS_PER_SEGMENT
        ),
        joint_space_bridge_max_tcp_deviation_mm=JOINT_SPACE_BRIDGE_MAX_TCP_DEVIATION_MM,
        joint_space_bridge_max_tcp_path_ratio=JOINT_SPACE_BRIDGE_MAX_TCP_PATH_RATIO,
        local_parallel_workers=local_parallel_workers,
        local_parallel_min_batch_size=local_parallel_min_batch_size,
        big_circle_step_deg_threshold=BIG_CIRCLE_STEP_DEG_THRESHOLD,
        branch_flip_ratio_threshold=BRANCH_FLIP_RATIO_THRESHOLD,
        branch_flip_ratio_eps_mm=BRANCH_FLIP_RATIO_EPS_MM,
        official_worst_joint_step_deg_limit=OFFICIAL_WORST_JOINT_STEP_DEG_LIMIT,
        enable_same_family_segment_repair=ENABLE_SAME_FAMILY_SEGMENT_REPAIR,
        same_family_repair_max_segments=SAME_FAMILY_REPAIR_MAX_SEGMENTS,
        ik_backend=ik_backend,
    )


def build_app_runtime_settings(
    *,
    validation_centerline_csv: str | Path,
    tool_poses_frame2_csv: str | Path,
    append_start_as_terminal: bool = False,
    target_frame_origin_mm: tuple[float, float, float],
    target_frame_rotation_xyz_deg: tuple[float, float, float],
    enable_custom_smoothing_and_pose_selection: bool,
    robot_name: str,
    frame_name: str,
    program_name: str,
    ik_backend: str = "robodk",
    local_parallel_workers: int = LOCAL_PARALLEL_WORKERS,
    local_parallel_min_batch_size: int = LOCAL_PARALLEL_MIN_BATCH_SIZE,
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
            ik_backend=ik_backend,
            local_parallel_workers=local_parallel_workers,
            local_parallel_min_batch_size=local_parallel_min_batch_size,
        ),
        append_start_as_terminal=bool(append_start_as_terminal),
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
