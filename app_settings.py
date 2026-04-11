from __future__ import annotations

from pathlib import Path

from src.app_runner import AppRuntimeSettings
from src.frame_math import FrameBuildOptions
from src.robodk_program import RoboDKMotionSettings


# -----------------------------------------------------------------------------
# 高级参数区
# 这里放的是默认不需要频繁改动、但又希望集中管理的参数。
# `main.py` 只保留最常改的主参数，其他细节都在这里维护。
# -----------------------------------------------------------------------------

# 局部坐标系构建参数
ZERO_VECTOR_TOLERANCE = 1e-9  # 判断向量是否可视为零向量的阈值；过大可能掩盖脏数据，过小可能放大数值噪声。
NEAR_PARALLEL_TOLERANCE = 1e-6  # 判断两向量是否近似平行的阈值；用于构造局部坐标系时避免退化叉乘。
CONTINUITY_DOT_THRESHOLD = 0.0  # 相邻局部轴连续性的点积阈值；越大越强调方向连续，0 表示只要求不反向恶化。
FLIP_PROCESS_Y_AXIS = True  # 是否翻转工艺坐标系 Y 轴；用于把局部坐标系朝向对齐到工艺定义。
FLIP_PROCESS_Z_AXIS = True  # 是否翻转工艺坐标系 Z 轴；通常对应法向或工具主轴方向是否要反过来。

# 可视化参数
VISUALIZATION_STEP = 8  # 可视化时每隔多少个有效点抽一帧；值越大图越清爽，值越小越贴近原始密度。
VISUALIZATION_VECTOR_SCALE = 8.0  # 可视化箭头长度缩放；只影响显示效果，不影响求解结果。
SHOW_TANGENT = True  # 可视化时是否显示切向量。
SHOW_NORMAL = True  # 可视化时是否显示法向量。
SHOW_SIDE = True  # 可视化时是否显示侧向量。
SHOW_BAD_ROWS = True  # 可视化时是否高亮异常/无效行，便于排查输入数据问题。

# 位姿求解校验参数
ENABLE_SOLVER_VERIFICATION = True  # 求完位姿后是否做一次数值回代校验；建议默认开启。
VERIFICATION_ROW_IDS: tuple[int, ...] | None = None  # 指定只校验哪些行；None 表示自动选取开头/中间/末尾有效行。
VERIFICATION_TOLERANCE = 1e-6  # 位姿求解数学回代容差；这是几何闭环校验阈值，不是 RoboDK 运动精度参数。

# RoboDK 运动学与执行参数
ROBOT_MOVE_TYPE = "MoveJ"  # RoboDK 指令类型；MoveJ 走关节空间，MoveL 更强调 TCP 直线轨迹。
ROBOT_LINEAR_SPEED_MM_S = 200.0  # 笛卡尔速度设定，单位 mm/s；主要对 MoveL 更敏感。
ROBOT_JOINT_SPEED_DEG_S = 30.0  # 关节速度设定，单位 deg/s；主要对 MoveJ 更敏感。
ROBOT_LINEAR_ACCEL_MM_S2 = 600.0  # 笛卡尔加速度设定，单位 mm/s^2。
ROBOT_JOINT_ACCEL_DEG_S2 = 120.0  # 关节加速度设定，单位 deg/s^2。
ROBOT_ROUNDING_MM = -1.0  # RoboDK 圆角/过渡半径；-1 常用于精确停点，正值会更平滑但会降低贴点精度。
HIDE_TARGETS_AFTER_GENERATION = True  # 程序生成后是否自动隐藏 target，避免站点树和 3D 视图过于杂乱。

# 自定义优化的硬约束与搜索参数
A1_MIN_DEG = -150.0  # 自定义优化阶段允许的 A1 最小值，单位度。
A1_MAX_DEG = 30.0  # 自定义优化阶段允许的 A1 最大值，单位度。
A2_MAX_DEG = 115.0  # 自定义优化阶段允许的 A2 上限，单位度。
JOINT_CONSTRAINT_TOLERANCE_DEG = 1e-6  # 用户关节硬约束的数值容差，避免浮点误差导致边界误判。

ENABLE_JOINT_CONTINUITY_CONSTRAINT = True  # 是否把相邻点最大关节步长当成硬约束。
MAX_JOINT_STEP_DEG = (5.0, 5.0, 5.0, 180.0, 100.0, 180.0)  # 各轴允许的单步最大跳变；前 3 轴更严格，后 3 轴更宽松。

ENABLE_POSE_BRIDGE = True  # 是否允许在坏段之间自动插入桥接点补救姿态切换。
BRIDGE_TRIGGER_JOINT_DELTA_DEG = 20.0  # 相邻点单关节跳变超过多少度后，认定该段需要桥接或额外处理。
BRIDGE_STEP_DEG = (2.0, 2.0, 2.0, 20.0, 10.0, 20.0)  # 桥接点之间的最大关节步长；用于控制桥接段离散密度。
BRIDGE_TRANSLATION_SEARCH_MM = (0.0, 1.0, 2.0)  # 桥接搜索时允许尝试的平移扰动集合，单位 mm。
BRIDGE_ROTATION_SEARCH_DEG = (0.0, 0.5, 1.0)  # 桥接搜索时允许尝试的姿态扰动集合，单位度。

# 整路径统一位姿偏置搜索
# 目的不是改单点，而是把整条路径一起绕工艺锚点做小角度刚体旋转，
# 让 exact-pose 全局 IK 更容易保持在同一条连续位形分支上。
ENABLE_GLOBAL_POSE_SEARCH = True  # 是否启用整条路径统一旋转搜索；常用于把整条路径整体挪进更稳定的 IK 走廊。
GLOBAL_POSE_SEARCH_MAX_OFFSET_DEG = 18.0  # 全局搜索允许的最大姿态偏置角度；过大可能偏离原始工艺定义。
GLOBAL_POSE_SEARCH_STEP_SCHEDULE_DEG = (8.0, 4.0, 2.0, 1.0)  # 全局搜索的粗到细步长序列；越细越慢，但更容易找到更优姿态族。

# 腕奇异细化
# 当 A5 接近 0 或腕部跳变过大时，会做二次 IK 重采样，
# 优先尝试保住 A6 相位连续，减少奇异附近的大幅甩动。
ENABLE_WRIST_SINGULARITY_REFINEMENT = True  # 是否在腕奇异附近做二次细化搜索。
WRIST_REFINEMENT_A5_THRESHOLD_DEG = 12.0  # 把 A5 视为”接近奇异”的阈值；小于该值时会额外关注 A6 相位连续性。
WRIST_REFINEMENT_LARGE_WRIST_STEP_DEG = 20.0  # 认定腕部跳变过大的阈值；超出后更积极触发细化。
# 窗口半径扩大到 8：确保奇异区前后各 8 个点都被覆盖，给 DP 足够的”平滑过渡余地”。
# 若 A4 需要在奇异附近从 38° 爬升到 198°（160° 总量），在 16 步内完成则每步仅 10°，完全平滑。
WRIST_REFINEMENT_WINDOW_RADIUS = 8
# 相位偏置集合扩展到 ±180°：原来最大只有 ±90°，无法覆盖 160° 级别的反向半球。
# 现在增加 ±120°、±150°、±180°，确保能找到”另一侧半球”的候选解。
WRIST_REFINEMENT_PHASE_OFFSETS_DEG = (30.0, 60.0, 90.0, 120.0, 150.0, 180.0)

# 解丰富窗口姿态重分配
# 这里只改姿态，不改 TCP 位置；目标是把 unavoidable 的位形切换
# 尽量摊到解空间更丰富的窗口里，减少最终桥接补救压力。
ENABLE_SOLUTION_RICH_ORIENTATION_REDISTRIBUTION = True  # 是否启用“只改姿态、不改位置”的局部平滑重分配。
SOLUTION_RICH_ORIENTATION_WINDOW_RADIUS = 4  # 围绕坏段向前后扩多少个点来寻找“解丰富窗口”。
SOLUTION_RICH_ORIENTATION_OFFSET_DEG = 4.0  # 在候选窗口里尝试施加的姿态偏置幅度，单位度。


def build_frame_options() -> FrameBuildOptions:
    """根据高级配置构造局部坐标系参数对象。"""

    return FrameBuildOptions(
        zero_tolerance=ZERO_VECTOR_TOLERANCE,
        parallel_tolerance=NEAR_PARALLEL_TOLERANCE,
        continuity_dot_threshold=CONTINUITY_DOT_THRESHOLD,
        flip_process_y_axis=FLIP_PROCESS_Y_AXIS,
        flip_surface_normal=FLIP_PROCESS_Z_AXIS,
    )


def build_motion_settings(
    *,
    target_frame_origin_mm: tuple[float, float, float],
    enable_custom_smoothing_and_pose_selection: bool,
) -> RoboDKMotionSettings:
    """根据高级配置构造 RoboDK 程序生成参数对象。"""

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
        enable_pose_bridge=ENABLE_POSE_BRIDGE,
        bridge_trigger_joint_delta_deg=BRIDGE_TRIGGER_JOINT_DELTA_DEG,
        bridge_step_deg=BRIDGE_STEP_DEG,
        bridge_translation_search_mm=BRIDGE_TRANSLATION_SEARCH_MM,
        bridge_rotation_search_deg=BRIDGE_ROTATION_SEARCH_DEG,
        enable_global_pose_search=ENABLE_GLOBAL_POSE_SEARCH,
        global_pose_search_origin_mm=target_frame_origin_mm,
        global_pose_search_max_offset_deg=GLOBAL_POSE_SEARCH_MAX_OFFSET_DEG,
        global_pose_search_step_schedule_deg=GLOBAL_POSE_SEARCH_STEP_SCHEDULE_DEG,
        enable_wrist_singularity_refinement=ENABLE_WRIST_SINGULARITY_REFINEMENT,
        wrist_refinement_a5_threshold_deg=WRIST_REFINEMENT_A5_THRESHOLD_DEG,
        wrist_refinement_large_wrist_step_deg=WRIST_REFINEMENT_LARGE_WRIST_STEP_DEG,
        wrist_refinement_window_radius=WRIST_REFINEMENT_WINDOW_RADIUS,
        wrist_refinement_phase_offsets_deg=WRIST_REFINEMENT_PHASE_OFFSETS_DEG,
        enable_solution_rich_orientation_redistribution=ENABLE_SOLUTION_RICH_ORIENTATION_REDISTRIBUTION,
        solution_rich_orientation_window_radius=SOLUTION_RICH_ORIENTATION_WINDOW_RADIUS,
        solution_rich_orientation_offset_deg=SOLUTION_RICH_ORIENTATION_OFFSET_DEG,
    )


def build_app_runtime_settings(
    *,
    validation_centerline_csv: str | Path,
    tool_poses_frame2_csv: str | Path,
    target_frame_origin_mm: tuple[float, float, float],
    enable_custom_smoothing_and_pose_selection: bool,
    robot_name: str,
    frame_name: str,
    program_name: str,
) -> AppRuntimeSettings:
    """把主参数与高级参数合并成运行时设置对象。"""

    return AppRuntimeSettings(
        validation_centerline_csv=Path(validation_centerline_csv),
        tool_poses_frame2_csv=Path(tool_poses_frame2_csv),
        target_frame_origin_mm=tuple(float(value) for value in target_frame_origin_mm),
        robot_name=robot_name,
        frame_name=frame_name,
        program_name=program_name,
        frame_build_options=build_frame_options(),
        motion_settings=build_motion_settings(
            target_frame_origin_mm=target_frame_origin_mm,
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
