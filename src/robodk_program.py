from __future__ import annotations

import csv
import math
from dataclasses import dataclass, replace
from functools import lru_cache
from pathlib import Path
from typing import Sequence


# 生成 RoboDK 程序时，输入 CSV 至少要包含这些位姿列。
REQUIRED_COLUMNS = (
    "x_mm",
    "y_mm",
    "z_mm",
    "r11",
    "r12",
    "r13",
    "r21",
    "r22",
    "r23",
    "r31",
    "r32",
    "r33",
)

# RoboDK 的 JointsConfig() 前 3 个标志位分别表示：
# 1. rear/front
# 2. lower/upper
# 3. flip/non-flip
_CONFIG_FLAG_COUNT = 3

# 对 IK 解去重时使用的角度保留小数位数。
_IK_DEDUP_DECIMALS = 6

# 姿态桥接层的候选数量上限。
# 这里刻意限制每层候选数，避免少数坏段触发桥接后导致局部 DP 爆炸。
# 同时会始终保留“关节线性插值 + FK”生成的兜底候选，保证桥接段始终可构造。
_BRIDGE_LAYER_CANDIDATE_LIMIT = 18

# “固定法兰位置”的姿态桥接段参数。
# 这里的桥接不再优先追求关节空间插值，而是优先让法兰原点固定不动，
# 通过一串原地 MoveL 姿态点完成姿态切换，再继续走原始路径。
_POSITION_LOCK_BRIDGE_MIN_SEGMENTS = 6
_POSITION_LOCK_BRIDGE_ORIENTATION_STEP_DEG = 1.0

# 按 joint tuple 缓存 RoboDK 的派生量，避免在全局搜索与局部重分配时反复查询同一批结果。
_ROBOT_CONFIG_FLAGS_CACHE: dict[tuple[int, tuple[float, ...]], tuple[int, ...]] = {}
_ROBOT_SINGULARITY_PENALTY_CACHE: dict[
    tuple[int, tuple[float, ...], "_PathOptimizerSettings"],
    float,
] = {}


@dataclass(frozen=True)
class RoboDKMotionSettings:
    """RoboDK 程序生成参数。

    这里同时放两类配置：
    1. 运动学参数：速度、加速度、圆角等；
    2. 关节硬约束：A1、A2 必须满足的范围。
    """

    # 运动指令类型，当前支持 MoveJ / MoveL。
    move_type: str = "MoveJ"

    # 常规运动参数。
    linear_speed_mm_s: float = 200.0
    joint_speed_deg_s: float = 30.0
    linear_accel_mm_s2: float = 600.0
    joint_accel_deg_s2: float = 120.0
    rounding_mm: float = -1.0
    hide_targets_after_generation: bool = True

    # 本项目自定义的“多解 IK + 全局路径优化 + 平滑/桥接”总开关。
    # 关闭后不再做任何自定义位姿选择与平滑，直接把 CSV 位姿写成 RoboDK 笛卡尔目标，
    # 由 RoboDK 自己决定 IK 分支、关节连续性和程序执行路径。
    enable_custom_smoothing_and_pose_selection: bool = True

    # 用户指定的硬约束。
    # A1 强制在 [-150, 30] 内。
    a1_min_deg: float = -150.0
    a1_max_deg: float = 30.0

    # A2 强制小于该阈值。实现时会带一个极小容差，避免浮点误差误伤。
    a2_max_deg: float = 115.0
    joint_constraint_tolerance_deg: float = 1e-6

    # 点与点之间的连续性硬约束。
    # 如果启用，则相邻路径点之间任一关节变化超过阈值时，这条转移会被直接判定为不可用。
    enable_joint_continuity_constraint: bool = True
    max_joint_step_deg: tuple[float, ...] = (5.0, 5.0, 5.0, 180.0, 100.0, 180.0)

    # 姿态重构桥接参数。
    # 当相邻两点之间关节跳变过大，或配置标志发生切换时，
    # 会自动插入一串桥接点，尽量让法兰位姿在调整过程中保持接近原始目标位姿。
    enable_pose_bridge: bool = True
    bridge_trigger_joint_delta_deg: float = 20.0
    bridge_step_deg: tuple[float, ...] = (2.0, 2.0, 2.0, 20.0, 10.0, 20.0)
    bridge_translation_search_mm: tuple[float, ...] = (0.0, 1.0, 2.0)
    bridge_rotation_search_deg: tuple[float, ...] = (0.0, 0.5, 1.0)

    # 全局位姿微调搜索。
    # 这里不是对单点随意改目标，而是把整条路径的所有法兰位姿一起绕同一个锚点做刚体旋转，
    # 相当于统一调整“目标工艺坐标系 A 在 Frame 2 中的朝向”，从源头上寻找更稳定的位形分支。
    # 这种做法仍然保持整条路径使用同一套几何关系，不会在点与点之间引入额外精度破坏。
    enable_global_pose_search: bool = True
    global_pose_search_origin_mm: tuple[float, float, float] | None = None
    global_pose_search_max_offset_deg: float = 18.0
    global_pose_search_step_schedule_deg: tuple[float, ...] = (8.0, 4.0, 2.0, 1.0)

    # 参考 FINA11.src 的处理思路：在腕奇异附近尽量不要让 A6 大幅甩动，
    # 而是优先通过同分支内的腕部相位重分配，把变化摊到更容易连续的关节上。
    enable_wrist_singularity_refinement: bool = True
    wrist_refinement_a5_threshold_deg: float = 12.0
    wrist_refinement_large_wrist_step_deg: float = 20.0
    wrist_refinement_window_radius: int = 2
    wrist_refinement_phase_offsets_deg: tuple[float, ...] = (15.0, 30.0, 45.0, 60.0, 90.0)

    # 继续参考 FINA11.src 的“提前在解丰富区处理位形”的思路。
    # 这里不是插桥接点，也不是改 TCP 位置，而是在残留坏段前后的多解窗口里，
    # 对后续整段路径做一个“位置锁定、只改姿态”的平滑重分配搜索。
    # 目标是把必须发生的位形切换尽量摊到更宽的窗口里，并优先放在解空间更丰富的位置发生。
    enable_solution_rich_orientation_redistribution: bool = True
    solution_rich_orientation_window_radius: int = 4
    solution_rich_orientation_offset_deg: float = 4.0


@dataclass(frozen=True)
class _PathOptimizerSettings:
    """动态规划路径优化时使用的代价权重。"""

    # 各关节的基础权重。后面的大轴通常更容易导致大幅姿态变化，因此权重略高。
    joint_delta_weights: tuple[float, ...]

    # 关节变化的 L1 / L2 混合代价。
    abs_delta_weight: float = 1.0
    squared_delta_weight: float = 0.015

    # 配置切换惩罚。
    rear_switch_penalty: float = 40.0
    lower_switch_penalty: float = 55.0
    flip_switch_penalty: float = 140.0

    # 额外抑制腕部翻转、J6 大幅旋转和大跳变。
    wrist_flip_sign_penalty: float = 100.0
    joint6_spin_threshold_deg: float = 120.0
    joint6_spin_penalty_per_deg: float = 1.5
    large_jump_threshold_deg: float = 35.0
    large_jump_penalty_weight: float = 0.3

    # 逼近关节限位时的惩罚。
    joint_limit_margin_ratio: float = 0.10
    joint_limit_penalty_weight: float = 180.0

    # 奇异附近惩罚。
    wrist_singularity_threshold_deg: float = 12.0
    wrist_singularity_penalty_weight: float = 110.0
    arm_singularity_threshold: float = 0.12
    arm_singularity_penalty_weight: float = 90.0

    # MoveL 额外代价。
    move_l_branch_mismatch_weight: float = 0.6
    move_l_unreachable_penalty: float = math.inf

    # 点与点之间的连续性硬约束。
    enable_joint_continuity_constraint: bool = True
    max_joint_step_deg: tuple[float, ...] = (5.0, 5.0, 5.0, 60.0, 45.0, 90.0)

    # 第一个点与机器人当前姿态的关系只作为轻量参考，避免当前站点姿态把整条路径“带歪”。
    start_transition_weight: float = 0.20

    # 候选点自身的“近限位 / 近奇异”代价仍保留，但只占一部分权重，
    # 以免局部节点惩罚过大，反而逼着 DP 提前离开整条更平顺的位形走廊。
    node_penalty_scale: float = 0.22

    # “优选连续性”不是硬约束，而是用来识别整条路径中那些能长距离维持同一位形族的候选走廊。
    preferred_joint_step_deg: tuple[float, ...] = (5.0, 5.0, 5.0, 25.0, 25.0, 25.0)
    preferred_transition_bonus: float = 12.0
    same_config_stay_bonus: float = 8.0
    corridor_bonus_per_step: float = 3.0
    corridor_bonus_cap: float = 20.0

    # 先在 config_flags 层面做一遍“位形族路径规划”时使用的切换惩罚。
    # 这个值刻意比普通转移代价大很多，目的是：
    # 1. 先问“能不能整段不切族”；
    # 2. 如果不能，再问“在哪个层切族最划算”。
    family_switch_penalty: float = 4000.0

    # 参考 FINA11.src：当 A5 接近 0 时，优先保持 A6 相位连续，
    # 尽量不要在腕奇异附近把姿态突变甩给 A6。
    wrist_phase_lock_threshold_deg: float = 12.0
    wrist_phase_lock_penalty_per_deg: float = 6.0


@dataclass(frozen=True)
class _IKCandidate:
    """某一个路径点的一组候选 IK 解。"""

    joints: tuple[float, ...]
    config_flags: tuple[int, ...]
    joint_limit_penalty: float
    singularity_penalty: float


@dataclass(frozen=True)
class _IKLayer:
    """路径中的一个离散点。

    pose:
        该点的笛卡尔位姿。
    candidates:
        该点所有可行的关节候选解。
    """

    pose: object
    candidates: tuple[_IKCandidate, ...]


@dataclass(frozen=True)
class _ProgramWaypoint:
    """最终写入 RoboDK 程序的离散路点。

    这里既包括原始路径点，也包括自动插入的桥接点。
    """

    name: str
    pose: object
    joints: tuple[float, ...]
    move_type: str
    is_bridge: bool


@dataclass(frozen=True)
class _BridgeCandidate:
    """桥接层中的一个候选状态。"""

    pose: object
    joints: tuple[float, ...]
    config_flags: tuple[int, ...]
    node_cost: float


@dataclass(frozen=True)
class _PathSearchResult:
    """一次完整 exact-pose 全局搜索的结果。

    这个结果同时记录：
    1. 当前这套全局位姿偏置下的 pose rows；
    2. 该 pose rows 经过多解 IK + 全局 DP 后得到的最优关节路径；
    3. 便于比较不同偏置方案优劣的统计指标。
    """

    pose_rows: tuple[dict[str, float], ...]
    ik_layers: tuple[_IKLayer, ...]
    selected_path: tuple[_IKCandidate, ...]
    total_cost: float
    rotation_offset_deg: tuple[float, float, float]
    config_switches: int
    bridge_like_segments: int
    worst_joint_step_deg: float
    mean_joint_step_deg: float


def create_program_from_csv(
    csv_path: str | Path,
    *,
    robot_name: str,
    frame_name: str,
    program_name: str,
    motion_settings: RoboDKMotionSettings,
) -> object:
    """根据位姿 CSV 生成 RoboDK 程序。

    支持两种模式：
    1. 默认模式：使用本项目自定义的多解 IK + 全局路径优化 + 平滑/桥接；
    2. RoboDK 原生模式：直接把 CSV 位姿写成笛卡尔目标，完全交给 RoboDK 自己选 IK。
    """

    settings = _validate_motion_settings(motion_settings)
    pose_rows = tuple(load_pose_rows(csv_path))
    api = _import_robodk_api()

    rdk = api["Robolink"]()
    robot = _require_item(rdk, robot_name, api["ITEM_TYPE_ROBOT"], "Robot")
    frame = _require_item(rdk, frame_name, api["ITEM_TYPE_FRAME"], "Reference frame")
    _delete_stale_bridge_targets(rdk, api["ITEM_TYPE_TARGET"])

    # 记录机器人当前关节，后面测试 IK / MoveL 时可能临时改变机器人状态，
    # 最终统一恢复，避免打乱用户当前站点画面。
    current_joints_list = robot.Joints().list()
    joint_count = len(current_joints_list)
    original_joints = _trim_joint_vector(current_joints_list, joint_count)
    lower_limits_raw, upper_limits_raw, _ = robot.JointLimits()
    lower_limits = _trim_joint_vector(lower_limits_raw.list(), joint_count)
    upper_limits = _trim_joint_vector(upper_limits_raw.list(), joint_count)

    # 这里必须使用 robot.PoseFrame() 作为 IK 的 reference。
    # 实测当前 RoboDK 站点下，直接把 frame.Pose() 传给 SolveIK_All 会拿不到完整解集。
    robot.setPoseFrame(frame)
    tool_pose = robot.PoseTool()
    reference_pose = robot.PoseFrame()

    # A1 范围允许用户在配置里写成任意顺序，这里统一归一化为 [lower, upper]。
    a1_lower_deg, a1_upper_deg = _normalize_angle_range(settings.a1_min_deg, settings.a1_max_deg)
    optimizer_settings = _build_optimizer_settings(joint_count, settings)

    rdk.Render(False)
    try:
        if not settings.enable_custom_smoothing_and_pose_selection:
            return _create_program_from_pose_rows_with_robodk_defaults(
                pose_rows,
                rdk=rdk,
                robot=robot,
                frame=frame,
                mat_type=api["Mat"],
                item_type_program=api["ITEM_TYPE_PROGRAM"],
                item_type_target=api["ITEM_TYPE_TARGET"],
                program_name=program_name,
                frame_name=frame_name,
                motion_settings=settings,
            )

        search_result = _search_best_exact_pose_path(
            pose_rows,
            robot=robot,
            mat_type=api["Mat"],
            move_type=settings.move_type,
            start_joints=original_joints,
            tool_pose=tool_pose,
            reference_pose=reference_pose,
            joint_count=joint_count,
            motion_settings=settings,
            optimizer_settings=optimizer_settings,
            a1_lower_deg=a1_lower_deg,
            a1_upper_deg=a1_upper_deg,
            a2_max_deg=settings.a2_max_deg,
            joint_constraint_tolerance_deg=settings.joint_constraint_tolerance_deg,
        )
        ik_layers = search_result.ik_layers
        selected_path = search_result.selected_path
        total_cost = search_result.total_cost

        # 如果整条 exact path 已经稳定地落在同一位形走廊内，
        # 就不要再做针对奇异段和坏段的二次搜索了。
        # 这样既避免把已经很干净的结果重新扰动坏，也能显著缩短运行时间。
        path_already_clean = _path_is_clean_enough_for_program_generation(
            selected_path,
            bridge_trigger_joint_delta_deg=settings.bridge_trigger_joint_delta_deg,
            preferred_joint_step_deg=optimizer_settings.preferred_joint_step_deg,
        )
        if not path_already_clean:
            ik_layers, selected_path, total_cost = _refine_path_near_wrist_singularity(
                ik_layers,
                selected_path,
                total_cost=total_cost,
                start_joints=original_joints,
                robot=robot,
                tool_pose=tool_pose,
                reference_pose=reference_pose,
                joint_count=joint_count,
                motion_settings=settings,
                optimizer_settings=optimizer_settings,
                lower_limits=lower_limits,
                upper_limits=upper_limits,
                a1_lower_deg=a1_lower_deg,
                a1_upper_deg=a1_upper_deg,
                a2_max_deg=settings.a2_max_deg,
                joint_constraint_tolerance_deg=settings.joint_constraint_tolerance_deg,
            )

            path_already_clean = _path_is_clean_enough_for_program_generation(
                selected_path,
                bridge_trigger_joint_delta_deg=settings.bridge_trigger_joint_delta_deg,
                preferred_joint_step_deg=optimizer_settings.preferred_joint_step_deg,
            )
            if not path_already_clean:
                ik_layers, selected_path, total_cost = (
                    _redistribute_orientation_in_solution_rich_window(
                        search_result.pose_rows,
                        ik_layers,
                        selected_path,
                        total_cost=total_cost,
                        start_joints=original_joints,
                        robot=robot,
                        mat_type=api["Mat"],
                        tool_pose=tool_pose,
                        reference_pose=reference_pose,
                        joint_count=joint_count,
                        motion_settings=settings,
                        optimizer_settings=optimizer_settings,
                        lower_limits=lower_limits,
                        upper_limits=upper_limits,
                        a1_lower_deg=a1_lower_deg,
                        a1_upper_deg=a1_upper_deg,
                        a2_max_deg=settings.a2_max_deg,
                        joint_constraint_tolerance_deg=settings.joint_constraint_tolerance_deg,
                        base_rotation_offset_deg=search_result.rotation_offset_deg,
                    )
                )
        else:
            print(
                "Exact path is already continuous in a single configuration corridor; "
                "skip targeted refinement passes."
            )

        (
            final_config_switches,
            final_bridge_like_segments,
            final_worst_joint_step_deg,
            _final_mean_joint_step_deg,
        ) = _summarize_selected_path(
            selected_path,
            bridge_trigger_joint_delta_deg=settings.bridge_trigger_joint_delta_deg,
        )

        program_waypoints = _build_program_waypoints(
            ik_layers,
            selected_path,
            robot=robot,
            tool_pose=tool_pose,
            reference_pose=reference_pose,
            motion_settings=settings,
            optimizer_settings=optimizer_settings,
            lower_limits=lower_limits,
            upper_limits=upper_limits,
            a1_lower_deg=a1_lower_deg,
            a1_upper_deg=a1_upper_deg,
        )

        # 结束搜索后先把机器人状态恢复，再开始正式创建程序对象。
        robot.setJoints(list(original_joints))

        existing_program = rdk.Item(program_name, api["ITEM_TYPE_PROGRAM"])
        if existing_program.Valid():
            existing_program.Delete()

        program = rdk.AddProgram(program_name, robot)
        program.setRobot(robot)
        if hasattr(program, "setPoseFrame"):
            program.setPoseFrame(frame)
        if hasattr(program, "setPoseTool"):
            program.setPoseTool(robot.PoseTool())
        _apply_motion_settings(program, settings)

        for waypoint in program_waypoints:
            existing_target = rdk.Item(waypoint.name, api["ITEM_TYPE_TARGET"])
            if existing_target.Valid():
                existing_target.Delete()

            target = rdk.AddTarget(waypoint.name, frame, robot)
            target.setRobot(robot)
            _apply_selected_target(
                target,
                pose=waypoint.pose,
                joints=waypoint.joints,
                move_type=waypoint.move_type,
            )
            _append_move_instruction(program, target, waypoint.move_type)
    finally:
        rdk.Render(True)
        if original_joints:
            robot.setJoints(list(original_joints))

    if settings.hide_targets_after_generation and hasattr(program, "ShowTargets"):
        program.ShowTargets(False)

    total_candidates = sum(len(layer.candidates) for layer in ik_layers)
    bridge_count = sum(1 for waypoint in program_waypoints if waypoint.is_bridge)
    print(
        "Hard joint constraints: "
        f"A1 in [{a1_lower_deg:.1f}, {a1_upper_deg:.1f}] deg, "
        f"A2 < {settings.a2_max_deg:.1f} deg."
    )
    if settings.enable_joint_continuity_constraint:
        continuity_limits = _normalize_step_limits(settings.max_joint_step_deg, joint_count)
        print(
            "Hard continuity constraints: "
            f"max joint step = {[round(value, 3) for value in continuity_limits]} deg."
        )
    if settings.enable_global_pose_search and settings.global_pose_search_origin_mm is not None:
        print(
            "Global pose search: "
            f"selected offset xyz(deg) = {[round(value, 3) for value in search_result.rotation_offset_deg]}, "
            f"bridge_like_segments={final_bridge_like_segments}, "
            f"worst_joint_step={final_worst_joint_step_deg:.3f} deg."
        )
    print(
        f"Optimized {len(ik_layers)} target(s) using {total_candidates} IK candidate(s); "
        f"selected path cost={total_cost:.3f}, config_switches={final_config_switches}."
    )
    if bridge_count > 0:
        print(f"Inserted {bridge_count} bridge waypoint(s) for posture adjustment.")
    print(
        f"Created RoboDK program '{program.Name()}' with {len(program_waypoints)} target(s) "
        f"({len(selected_path)} original + {bridge_count} bridge) using {settings.move_type} "
        f"in frame '{frame_name}'."
    )
    return program


def _create_program_from_pose_rows_with_robodk_defaults(
    pose_rows: Sequence[dict[str, float]],
    *,
    rdk,
    robot,
    frame,
    mat_type,
    item_type_program: int,
    item_type_target: int,
    program_name: str,
    frame_name: str,
    motion_settings: RoboDKMotionSettings,
) -> object:
    """直接把 CSV 位姿写成 RoboDK 笛卡尔目标，由 RoboDK 自己处理 IK。"""

    existing_program = rdk.Item(program_name, item_type_program)
    if existing_program.Valid():
        existing_program.Delete()

    program = rdk.AddProgram(program_name, robot)
    program.setRobot(robot)
    if hasattr(program, "setPoseFrame"):
        program.setPoseFrame(frame)
    if hasattr(program, "setPoseTool"):
        program.setPoseTool(robot.PoseTool())
    _apply_motion_settings(program, motion_settings)

    target_index_width = max(3, len(str(max(0, len(pose_rows) - 1))))
    for index, row in enumerate(pose_rows):
        target_name = f"P_{index:0{target_index_width}d}"
        existing_target = rdk.Item(target_name, item_type_target)
        if existing_target.Valid():
            existing_target.Delete()

        target = rdk.AddTarget(target_name, frame, robot)
        target.setRobot(robot)
        _apply_robodk_native_target(target, pose=_build_pose(row, mat_type))
        _append_move_instruction(program, target, motion_settings.move_type)

    if motion_settings.hide_targets_after_generation and hasattr(program, "ShowTargets"):
        program.ShowTargets(False)

    print(
        "Custom smoothing and pose selection disabled; using RoboDK native Cartesian targets "
        "and native IK/config selection."
    )
    print(
        f"Created RoboDK program '{program.Name()}' with {len(pose_rows)} target(s) using "
        f"{motion_settings.move_type} in frame '{frame_name}'."
    )
    return program


def load_pose_rows(csv_path: str | Path) -> list[dict[str, float]]:
    """读取位姿 CSV，并把必需列转换成浮点数。"""

    csv_path = Path(csv_path)
    if not csv_path.is_file():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    with csv_path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"CSV file has no header row: {csv_path}")

        missing_columns = [column for column in REQUIRED_COLUMNS if column not in reader.fieldnames]
        if missing_columns:
            raise ValueError(
                "CSV file is missing required column(s): " + ", ".join(missing_columns)
            )

        pose_rows: list[dict[str, float]] = []
        for line_number, row in enumerate(reader, start=2):
            if _row_is_empty(row) or _row_is_marked_invalid(row):
                continue

            values: dict[str, float] = {}
            for column in REQUIRED_COLUMNS:
                raw_value = row.get(column, "")
                try:
                    numeric_value = float(raw_value)
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        f"Invalid numeric value for '{column}' at CSV line {line_number}: {raw_value!r}"
                    ) from exc

                if not math.isfinite(numeric_value):
                    raise ValueError(
                        f"Non-finite value for '{column}' at CSV line {line_number}: {raw_value!r}"
                    )

                values[column] = numeric_value

            pose_rows.append(values)

    if not pose_rows:
        raise ValueError(f"No valid pose rows found in CSV file: {csv_path}")

    return pose_rows


def _search_best_exact_pose_path(
    pose_rows: Sequence[dict[str, float]],
    *,
    robot,
    mat_type,
    move_type: str,
    start_joints: tuple[float, ...],
    tool_pose,
    reference_pose,
    joint_count: int,
    motion_settings: RoboDKMotionSettings,
    optimizer_settings: _PathOptimizerSettings,
    a1_lower_deg: float,
    a1_upper_deg: float,
    a2_max_deg: float,
    joint_constraint_tolerance_deg: float,
) -> _PathSearchResult:
    """搜索“整条路径统一刚体旋转”后的最优 exact-pose 解。

    这里不做逐点自由修改，而是只允许把整条路径一起绕固定锚点旋转。
    这样做的好处是：
    1. 保持整条路径使用同一套工艺几何关系；
    2. 不会把问题继续推给桥接层；
    3. 常常能在源头上把整条路径放进同一条连续的 IK 分支。
    """

    cache: dict[tuple[float, float, float], _PathSearchResult | None] = {}
    preview_cache: dict[tuple[float, float, float], _PathSearchResult | None] = {}
    preview_pose_rows = _build_global_pose_search_preview_rows(pose_rows)
    lower_limits_raw, upper_limits_raw, _ = robot.JointLimits()
    lower_limits = _trim_joint_vector(lower_limits_raw.list(), joint_count)
    upper_limits = _trim_joint_vector(upper_limits_raw.list(), joint_count)
    search_seed_joints = _build_seed_joint_strategies(
        robot=robot,
        lower_limits=lower_limits,
        upper_limits=upper_limits,
        joint_count=joint_count,
    )
    preview_seed_joints = _build_preview_seed_joint_strategies(
        robot=robot,
        lower_limits=lower_limits,
        upper_limits=upper_limits,
        joint_count=joint_count,
    )
    preview_optimizer_settings = replace(
        optimizer_settings,
        enable_joint_continuity_constraint=False,
    )

    def evaluate(rotation_offset_deg: tuple[float, float, float]) -> _PathSearchResult | None:
        cache_key = tuple(round(value, 6) for value in rotation_offset_deg)
        if cache_key in cache:
            return cache[cache_key]

        candidate_pose_rows = tuple(
            _apply_global_pose_rotation_to_pose_row(row, rotation_offset_deg, motion_settings)
            for row in pose_rows
        )
        try:
            ik_layers = tuple(
                _build_ik_layers(
                    candidate_pose_rows,
                    robot=robot,
                    mat_type=mat_type,
                    tool_pose=tool_pose,
                    reference_pose=reference_pose,
                    joint_count=joint_count,
                    optimizer_settings=optimizer_settings,
                    a1_lower_deg=a1_lower_deg,
                    a1_upper_deg=a1_upper_deg,
                    a2_max_deg=a2_max_deg,
                    joint_constraint_tolerance_deg=joint_constraint_tolerance_deg,
                    seed_joints_override=search_seed_joints,
                    lower_limits_override=lower_limits,
                    upper_limits_override=upper_limits,
                    log_summary=False,
                )
            )
            selected_path_list, total_cost = _optimize_joint_path(
                ik_layers,
                robot=robot,
                move_type=move_type,
                start_joints=start_joints,
                optimizer_settings=optimizer_settings,
            )
        except RuntimeError:
            cache[cache_key] = None
            return None

        selected_path = tuple(selected_path_list)
        config_switches, bridge_like_segments, worst_joint_step_deg, mean_joint_step_deg = (
            _summarize_selected_path(
                selected_path,
                bridge_trigger_joint_delta_deg=motion_settings.bridge_trigger_joint_delta_deg,
            )
        )
        result = _PathSearchResult(
            pose_rows=candidate_pose_rows,
            ik_layers=ik_layers,
            selected_path=selected_path,
            total_cost=total_cost,
            rotation_offset_deg=rotation_offset_deg,
            config_switches=config_switches,
            bridge_like_segments=bridge_like_segments,
            worst_joint_step_deg=worst_joint_step_deg,
            mean_joint_step_deg=mean_joint_step_deg,
        )
        cache[cache_key] = result
        return result

    def evaluate_preview(rotation_offset_deg: tuple[float, float, float]) -> _PathSearchResult | None:
        cache_key = tuple(round(value, 6) for value in rotation_offset_deg)
        if cache_key in preview_cache:
            return preview_cache[cache_key]

        candidate_preview_rows = tuple(
            _apply_global_pose_rotation_to_pose_row(row, rotation_offset_deg, motion_settings)
            for row in preview_pose_rows
        )
        try:
            preview_layers = tuple(
                _build_ik_layers(
                    candidate_preview_rows,
                    robot=robot,
                    mat_type=mat_type,
                    tool_pose=tool_pose,
                    reference_pose=reference_pose,
                    joint_count=joint_count,
                    optimizer_settings=preview_optimizer_settings,
                    a1_lower_deg=a1_lower_deg,
                    a1_upper_deg=a1_upper_deg,
                    a2_max_deg=a2_max_deg,
                    joint_constraint_tolerance_deg=joint_constraint_tolerance_deg,
                    seed_joints_override=preview_seed_joints,
                    lower_limits_override=lower_limits,
                    upper_limits_override=upper_limits,
                    log_summary=False,
                )
            )
            preview_path_list, preview_cost = _optimize_joint_path(
                preview_layers,
                robot=robot,
                move_type=move_type,
                start_joints=start_joints,
                optimizer_settings=preview_optimizer_settings,
            )
        except RuntimeError:
            preview_cache[cache_key] = None
            return None

        preview_path = tuple(preview_path_list)
        config_switches, bridge_like_segments, worst_joint_step_deg, mean_joint_step_deg = (
            _summarize_selected_path(
                preview_path,
                bridge_trigger_joint_delta_deg=motion_settings.bridge_trigger_joint_delta_deg,
            )
        )
        preview_result = _PathSearchResult(
            pose_rows=candidate_preview_rows,
            ik_layers=preview_layers,
            selected_path=preview_path,
            total_cost=preview_cost,
            rotation_offset_deg=rotation_offset_deg,
            config_switches=config_switches,
            bridge_like_segments=bridge_like_segments,
            worst_joint_step_deg=worst_joint_step_deg,
            mean_joint_step_deg=mean_joint_step_deg,
        )
        preview_cache[cache_key] = preview_result
        return preview_result

    search_enabled = (
        motion_settings.enable_global_pose_search
        and motion_settings.global_pose_search_origin_mm is not None
        and bool(motion_settings.global_pose_search_step_schedule_deg)
    )
    baseline_result = (
        evaluate((0.0, 0.0, 0.0))
        if not search_enabled or evaluate_preview((0.0, 0.0, 0.0)) is not None
        else None
    )

    if not search_enabled:
        if baseline_result is None:
            raise RuntimeError("The original pose rows do not yield any globally feasible path.")
        return baseline_result

    if best_result := baseline_result:
        if _path_search_goal_is_satisfied(best_result, optimizer_settings=optimizer_settings):
            return best_result

    best_result = baseline_result
    search_center = (0.0, 0.0, 0.0)
    family_seed_tried = False
    max_offset_deg = motion_settings.global_pose_search_max_offset_deg
    best_offset = list(best_result.rotation_offset_deg) if best_result is not None else [0.0, 0.0, 0.0]

    for step_deg in motion_settings.global_pose_search_step_schedule_deg:
        if step_deg <= 0.0:
            continue

        if best_result is None:
            seed_candidates: list[_PathSearchResult] = []
            family_seed_used = False
            for offset_shell in _iter_global_pose_search_shell_offsets(
                step_deg,
                max_offset_deg=max_offset_deg,
            ):
                preview_candidates: list[_PathSearchResult] = []
                for candidate_offset in offset_shell:
                    preview_result = evaluate_preview(candidate_offset)
                    if preview_result is None:
                        continue
                    preview_candidates.append(preview_result)

                if not preview_candidates:
                    continue

                preview_candidates.sort(key=_path_search_sort_key)
                for preview_result in preview_candidates[: min(6, len(preview_candidates))]:
                    candidate_result = evaluate(preview_result.rotation_offset_deg)
                    if candidate_result is None:
                        continue
                    seed_candidates.append(candidate_result)

                if seed_candidates:
                    break

            if not seed_candidates and not family_seed_tried:
                family_seed_tried = True
                family_preview_candidates: list[_PathSearchResult] = []
                for family_offset in _iter_global_pose_search_family_seed_offsets():
                    preview_result = evaluate_preview(family_offset)
                    if preview_result is None:
                        continue
                    family_preview_candidates.append(preview_result)

                family_preview_candidates.sort(key=_path_search_sort_key)
                for preview_result in family_preview_candidates[: min(6, len(family_preview_candidates))]:
                    candidate_result = evaluate(preview_result.rotation_offset_deg)
                    if candidate_result is None:
                        continue
                    seed_candidates.append(candidate_result)

                if seed_candidates:
                    family_seed_used = True

            if not seed_candidates:
                continue

            best_result = min(seed_candidates, key=_path_search_sort_key)
            best_offset = list(best_result.rotation_offset_deg)
            search_center = (
                tuple(best_result.rotation_offset_deg)
                if family_seed_used
                else (0.0, 0.0, 0.0)
            )
            print(
                "Global pose search seeded offset "
                f"{[round(value, 3) for value in best_result.rotation_offset_deg]} deg: "
                f"bridge_like_segments={best_result.bridge_like_segments}, "
                f"config_switches={best_result.config_switches}, "
                f"worst_joint_step={best_result.worst_joint_step_deg:.3f} deg."
            )
            if _path_search_goal_is_satisfied(best_result, optimizer_settings=optimizer_settings):
                return best_result

        improved = True
        while improved:
            improved = False
            for axis_index in range(3):
                for direction in (-1.0, 1.0):
                    candidate_offset = list(best_offset)
                    candidate_offset[axis_index] += direction * step_deg
                    if (
                        abs(candidate_offset[axis_index] - search_center[axis_index])
                        > max_offset_deg + 1e-9
                    ):
                        continue

                    candidate_result = evaluate(tuple(candidate_offset))
                    if candidate_result is None:
                        continue
                    if _is_path_search_result_better(candidate_result, best_result):
                        best_result = candidate_result
                        best_offset = list(candidate_result.rotation_offset_deg)
                        improved = True
                        print(
                            "Global pose search accepted offset "
                            f"{[round(value, 3) for value in candidate_result.rotation_offset_deg]} deg: "
                            f"bridge_like_segments={candidate_result.bridge_like_segments}, "
                            f"config_switches={candidate_result.config_switches}, "
                            f"worst_joint_step={candidate_result.worst_joint_step_deg:.3f} deg."
                        )

        if best_result is not None and _path_search_goal_is_satisfied(
            best_result,
            optimizer_settings=optimizer_settings,
        ):
            return best_result

    if best_result is None:
        raise RuntimeError(
            "No globally feasible path was found, even after searching rigid orientation "
            "offsets for the whole path."
        )
    return best_result


def _is_path_search_result_better(
    candidate_result: _PathSearchResult,
    reference_result: _PathSearchResult,
) -> bool:
    """比较两组整路径姿态偏置搜索结果的优先级。"""

    return _path_search_sort_key(candidate_result) < _path_search_sort_key(reference_result)


def _path_search_sort_key(result: _PathSearchResult) -> tuple[float, ...]:
    """把搜索结果压成稳定的排序键。

    排序优先级从高到低是：
    1. 少触发桥接；
    2. 少切换位形分支；
    3. 单步最坏关节跳变更小；
    4. 平均关节变化更小；
    5. DP 总成本更小；
    6. 在完全持平时，优先较小的全局姿态偏置。
    """

    total_offset_deg = sum(abs(value) for value in result.rotation_offset_deg)
    return (
        float(result.bridge_like_segments),
        float(result.config_switches),
        float(result.worst_joint_step_deg),
        float(result.mean_joint_step_deg),
        float(result.total_cost),
        float(total_offset_deg),
    )


def _iter_global_pose_search_shell_offsets(
    step_deg: float,
    *,
    max_offset_deg: float,
) -> tuple[tuple[tuple[float, float, float], ...], ...]:
    """在当前 coarse 步长下枚举一圈全路径统一姿态偏置候选。

    这里不是只测单轴 ±step，而是把当前步长下的三轴组合都覆盖掉。
    这样即使零偏置无解，只要附近存在一个协同偏置解，也能把搜索启动起来。
    """

    if step_deg <= 0.0 or max_offset_deg <= 0.0:
        return ()

    max_shell = int(math.floor(max_offset_deg / step_deg + 1e-9))
    offset_shells: list[tuple[tuple[float, float, float], ...]] = []
    for shell_index in range(1, max_shell + 1):
        shell_offsets: list[tuple[float, float, float]] = []
        for x_index in range(-shell_index, shell_index + 1):
            for y_index in range(-shell_index, shell_index + 1):
                for z_index in range(-shell_index, shell_index + 1):
                    if max(abs(x_index), abs(y_index), abs(z_index)) != shell_index:
                        continue
                    offset = (
                        x_index * step_deg,
                        y_index * step_deg,
                        z_index * step_deg,
                    )
                    if any(abs(value) > max_offset_deg + 1e-9 for value in offset):
                        continue
                    shell_offsets.append(offset)

        # 先测多轴协同旋转，尽量贴近 FINA11 那种“整条位形一起挪进同一走廊”的效果。
        shell_offsets.sort(
            key=lambda offset: (
                -sum(1 for value in offset if abs(value) > 1e-9),
                sum(abs(value) for value in offset),
                offset,
            )
        )
        if shell_offsets:
            offset_shells.append(tuple(shell_offsets))
    return tuple(offset_shells)


def _build_global_pose_search_preview_rows(
    pose_rows: Sequence[dict[str, float]],
    *,
    target_count: int = 24,
) -> tuple[dict[str, float], ...]:
    """为全局位姿起始搜索构造一个稀疏预览路径。

    这一步只用于“先挑偏置方向”，不是最终落地结果。
    因此可以安全地抽样，先用更低成本排掉明显不行的偏置，
    再把全量 IK 留给少数最有希望的候选。
    """

    if len(pose_rows) <= target_count:
        return tuple(dict(row) for row in pose_rows)

    step = max(1, math.ceil((len(pose_rows) - 1) / max(1, target_count - 1)))
    indices = list(range(0, len(pose_rows), step))
    if indices[-1] != len(pose_rows) - 1:
        indices.append(len(pose_rows) - 1)
    return tuple(dict(pose_rows[index]) for index in indices)


def _iter_global_pose_search_family_seed_offsets() -> tuple[tuple[float, float, float], ...]:
    """枚举一组 180 度翻转的姿态族中心。

    这一步对应的是“先选一个正确的全局姿态族”，
    用来处理当前 identity 目标朝向离真实可行走廊太远的情况。
    """

    family_offsets: list[tuple[float, float, float]] = []
    for x_offset in (0.0, -180.0):
        for y_offset in (0.0, -180.0):
            for z_offset in (0.0, -180.0):
                offset = (x_offset, y_offset, z_offset)
                if not any(abs(value) > 1e-9 for value in offset):
                    continue
                family_offsets.append(offset)

    family_offsets.sort(
        key=lambda offset: (
            -sum(1 for value in offset if abs(value) > 1e-9),
            sum(abs(value) for value in offset),
            offset,
        )
    )
    return tuple(family_offsets)


def _build_preview_seed_joint_strategies(
    *,
    robot,
    lower_limits: tuple[float, ...],
    upper_limits: tuple[float, ...],
    joint_count: int,
) -> tuple[tuple[float, ...], ...]:
    """为全局位姿 preview 准备一组更轻量的 seed。

    preview 的目标只是找到“哪一侧的整路径刚体偏置更像有戏”，
    不需要像正式求解那样把所有局部分支都挖出来。
    这里保留少量代表性 seed，同时额外放入 A6=0 的腕部 seed，
    让搜索更贴近 FINA11 那种尽量稳定腕部相位的思路。
    """

    seeds: list[tuple[float, ...]] = []

    current_joints = _trim_joint_vector(robot.Joints().list(), joint_count)
    if current_joints:
        seeds.append(current_joints)

    home_joints = _trim_joint_vector(robot.JointsHome().list(), joint_count)
    if home_joints:
        seeds.append(home_joints)

    midpoint = tuple((lower + upper) * 0.5 for lower, upper in zip(lower_limits, upper_limits))
    seeds.append(midpoint)
    seeds.append(_clip_seed_to_limits((0.0,) * joint_count, lower_limits, upper_limits))

    if joint_count >= 6:
        for joint5 in (-60.0, 60.0):
            wrist_seed = list(midpoint)
            wrist_seed[4] = joint5
            wrist_seed[5] = 0.0
            seeds.append(_clip_seed_to_limits(wrist_seed, lower_limits, upper_limits))

    deduped_seeds: list[tuple[float, ...]] = []
    seen: set[tuple[float, ...]] = set()
    for seed in seeds:
        dedup_key = tuple(round(value, _IK_DEDUP_DECIMALS) for value in seed)
        if dedup_key in seen:
            continue
        seen.add(dedup_key)
        deduped_seeds.append(seed)

    return tuple(deduped_seeds)


def _path_search_goal_is_satisfied(
    result: _PathSearchResult,
    *,
    optimizer_settings: _PathOptimizerSettings,
) -> bool:
    """判断当前 exact path 是否已经好到可以停止继续搜整条刚体偏置。"""

    if result.bridge_like_segments != 0 or result.config_switches != 0:
        return False
    preferred_limit = max(optimizer_settings.preferred_joint_step_deg, default=0.0)
    return result.worst_joint_step_deg <= preferred_limit + 1e-9


def _summarize_selected_path(
    selected_path: Sequence[_IKCandidate],
    *,
    bridge_trigger_joint_delta_deg: float,
) -> tuple[int, int, float, float]:
    """统计一条已选 joint path 的关键连续性指标。"""

    if len(selected_path) <= 1:
        return 0, 0, 0.0, 0.0

    config_switches = 0
    bridge_like_segments = 0
    worst_joint_step_deg = 0.0
    mean_joint_step_sum = 0.0

    for previous_candidate, current_candidate in zip(selected_path, selected_path[1:]):
        joint_deltas = [
            abs(current - previous)
            for previous, current in zip(previous_candidate.joints, current_candidate.joints)
        ]
        max_joint_delta = max(joint_deltas, default=0.0)
        worst_joint_step_deg = max(worst_joint_step_deg, max_joint_delta)
        mean_joint_step_sum += _mean_abs_joint_delta(
            previous_candidate.joints,
            current_candidate.joints,
        )

        config_changed = previous_candidate.config_flags != current_candidate.config_flags
        if config_changed:
            config_switches += 1
        if config_changed or max_joint_delta > bridge_trigger_joint_delta_deg:
            bridge_like_segments += 1

    mean_joint_step_deg = mean_joint_step_sum / max(1, len(selected_path) - 1)
    return config_switches, bridge_like_segments, worst_joint_step_deg, mean_joint_step_deg


def _apply_global_pose_rotation_to_pose_row(
    pose_row: dict[str, float],
    rotation_offset_deg: Sequence[float],
    motion_settings: RoboDKMotionSettings,
) -> dict[str, float]:
    """把单个 pose row 绕统一锚点做刚体旋转。

    注意：
    - 这里只是统一旋转整条路径的所有法兰位姿；
    - 不会逐点独立篡改目标；
    - 当未启用全局搜索或偏置为 0 时，直接返回当前 row 的副本。
    """

    if (
        motion_settings.global_pose_search_origin_mm is None
        or not any(abs(value) > 1e-9 for value in rotation_offset_deg)
    ):
        return dict(pose_row)

    base_rotation, base_translation = _pose_row_to_rotation_translation(pose_row)
    search_rotation = _rotation_matrix_from_xyz_offset_deg(rotation_offset_deg)
    origin = motion_settings.global_pose_search_origin_mm

    rotated_rotation = _multiply_rotation_matrices(search_rotation, base_rotation)
    translated_from_origin = _subtract_vectors(base_translation, origin)
    rotated_translation = _add_vectors(
        _multiply_rotation_vector(search_rotation, translated_from_origin),
        origin,
    )
    return _pose_row_from_rotation_translation(rotated_rotation, rotated_translation)


def _build_pose(row: dict[str, float], mat_type) -> object:
    """把一行 CSV 位姿组装成 RoboDK 使用的 4x4 齐次矩阵。"""

    return mat_type(
        [
            [row["r11"], row["r12"], row["r13"], row["x_mm"]],
            [row["r21"], row["r22"], row["r23"], row["y_mm"]],
            [row["r31"], row["r32"], row["r33"], row["z_mm"]],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )


def _build_optimizer_settings(
    joint_count: int,
    motion_settings: RoboDKMotionSettings,
) -> _PathOptimizerSettings:
    """根据机器人轴数构造 DP 代价权重。"""

    base_weights = (1.0, 1.0, 1.1, 1.4, 2.0, 2.7)
    preferred_step_defaults = (5.0, 5.0, 5.0, 25.0, 25.0, 25.0)
    if joint_count <= len(base_weights):
        weights = base_weights[:joint_count]
    else:
        weights = base_weights + (base_weights[-1],) * (joint_count - len(base_weights))

    hard_step_limits = _normalize_step_limits(motion_settings.max_joint_step_deg, joint_count)
    if joint_count <= len(preferred_step_defaults):
        preferred_step_limits = preferred_step_defaults[:joint_count]
    else:
        preferred_step_limits = preferred_step_defaults + (
            preferred_step_defaults[-1],
        ) * (joint_count - len(preferred_step_defaults))

    # “优选连续性阈值”不能宽于真正的硬连续性阈值，否则走廊评分会鼓励一条实际上不可走的边。
    preferred_step_limits = tuple(
        min(hard_limit, preferred_limit)
        for hard_limit, preferred_limit in zip(hard_step_limits, preferred_step_limits)
    )

    return _PathOptimizerSettings(
        joint_delta_weights=weights,
        enable_joint_continuity_constraint=motion_settings.enable_joint_continuity_constraint,
        max_joint_step_deg=hard_step_limits,
        preferred_joint_step_deg=preferred_step_limits,
        wrist_phase_lock_threshold_deg=motion_settings.wrist_refinement_a5_threshold_deg,
    )


def _build_ik_layers(
    pose_rows: Sequence[dict[str, float]],
    *,
    robot,
    mat_type,
    tool_pose,
    reference_pose,
    joint_count: int,
    optimizer_settings: _PathOptimizerSettings,
    a1_lower_deg: float,
    a1_upper_deg: float,
    a2_max_deg: float,
    joint_constraint_tolerance_deg: float,
    seed_joints_override: Sequence[tuple[float, ...]] | None = None,
    lower_limits_override: Sequence[float] | None = None,
    upper_limits_override: Sequence[float] | None = None,
    log_summary: bool = True,
) -> list[_IKLayer]:
    """为整条路径逐点收集候选 IK 解。"""

    if lower_limits_override is None or upper_limits_override is None:
        lower_limits_raw, upper_limits_raw, _ = robot.JointLimits()
        lower_limits = _trim_joint_vector(lower_limits_raw.list(), joint_count)
        upper_limits = _trim_joint_vector(upper_limits_raw.list(), joint_count)
    else:
        lower_limits = tuple(float(value) for value in lower_limits_override[:joint_count])
        upper_limits = tuple(float(value) for value in upper_limits_override[:joint_count])

    # 通过不同 seed 诱导 RoboDK 返回不同分支附近的解。
    if seed_joints_override is None:
        seed_joints = _build_seed_joint_strategies(
            robot=robot,
            lower_limits=lower_limits,
            upper_limits=upper_limits,
            joint_count=joint_count,
        )
    else:
        seed_joints = tuple(seed_joints_override)

    ik_layers: list[_IKLayer] = []
    total_candidates = 0
    for row_index, row in enumerate(pose_rows):
        pose = _build_pose(row, mat_type)
        candidates = _collect_ik_candidates(
            robot,
            pose,
            tool_pose=tool_pose,
            reference_pose=reference_pose,
            lower_limits=lower_limits,
            upper_limits=upper_limits,
            seed_joints=seed_joints,
            joint_count=joint_count,
            optimizer_settings=optimizer_settings,
            a1_lower_deg=a1_lower_deg,
            a1_upper_deg=a1_upper_deg,
            a2_max_deg=a2_max_deg,
            joint_constraint_tolerance_deg=joint_constraint_tolerance_deg,
        )
        if not candidates:
            raise RuntimeError(
                f"No IK candidates remain for CSV pose row {row_index} after applying hard "
                f"constraints: A1 in [{a1_lower_deg:.1f}, {a1_upper_deg:.1f}] deg, "
                f"A2 < {a2_max_deg:.1f} deg."
            )

        ik_layers.append(_IKLayer(pose=pose, candidates=tuple(candidates)))
        total_candidates += len(candidates)

    if log_summary:
        print(
            f"Collected {total_candidates} IK candidate(s) across {len(ik_layers)} target pose(s)."
        )
    return ik_layers


def _refine_path_near_wrist_singularity(
    ik_layers: Sequence[_IKLayer],
    selected_path: Sequence[_IKCandidate],
    *,
    total_cost: float,
    start_joints: tuple[float, ...],
    robot,
    tool_pose,
    reference_pose,
    joint_count: int,
    motion_settings: RoboDKMotionSettings,
    optimizer_settings: _PathOptimizerSettings,
    lower_limits: Sequence[float],
    upper_limits: Sequence[float],
    a1_lower_deg: float,
    a1_upper_deg: float,
    a2_max_deg: float,
    joint_constraint_tolerance_deg: float,
) -> tuple[tuple[_IKLayer, ...], tuple[_IKCandidate, ...], float]:
    """在腕奇异窗口附近做一次 targeted IK refinement。"""

    if (
        not motion_settings.enable_wrist_singularity_refinement
        or joint_count < 6
        or not ik_layers
        or len(ik_layers) != len(selected_path)
    ):
        return tuple(ik_layers), tuple(selected_path), total_cost

    refine_indices = _collect_wrist_refinement_indices(selected_path, motion_settings)
    if not refine_indices:
        return tuple(ik_layers), tuple(selected_path), total_cost

    refine_index_set = set(refine_indices)
    augmented_layers: list[_IKLayer] = []
    new_candidate_count = 0
    lower_limits_tuple = tuple(lower_limits)
    upper_limits_tuple = tuple(upper_limits)

    for layer_index, layer in enumerate(ik_layers):
        if layer_index not in refine_index_set:
            augmented_layers.append(layer)
            continue

        candidates = list(layer.candidates)
        seen = {
            tuple(round(value, _IK_DEDUP_DECIMALS) for value in candidate.joints)
            for candidate in candidates
        }
        seed_strategies = _build_wrist_refinement_seed_strategies(
            selected_path,
            layer_index,
            motion_settings=motion_settings,
            lower_limits=lower_limits_tuple,
            upper_limits=upper_limits_tuple,
        )
        before_count = len(candidates)
        for seed in seed_strategies:
            raw_solution = robot.SolveIK(layer.pose, list(seed), tool_pose, reference_pose)
            _append_candidate_if_unique(
                candidates,
                seen,
                robot=robot,
                raw_joints=raw_solution,
                lower_limits=lower_limits_tuple,
                upper_limits=upper_limits_tuple,
                joint_count=joint_count,
                optimizer_settings=optimizer_settings,
                a1_lower_deg=a1_lower_deg,
                a1_upper_deg=a1_upper_deg,
                a2_max_deg=a2_max_deg,
                joint_constraint_tolerance_deg=joint_constraint_tolerance_deg,
            )

        candidates.sort(
            key=lambda candidate: (
                candidate.config_flags,
                candidate.joint_limit_penalty + candidate.singularity_penalty,
                candidate.joints,
            )
        )
        augmented_layers.append(_IKLayer(pose=layer.pose, candidates=tuple(candidates)))
        new_candidate_count += len(candidates) - before_count

    if new_candidate_count <= 0:
        return tuple(ik_layers), tuple(selected_path), total_cost

    refined_path_list, refined_cost = _optimize_joint_path(
        augmented_layers,
        robot=robot,
        move_type=motion_settings.move_type,
        start_joints=start_joints,
        optimizer_settings=optimizer_settings,
    )
    refined_path = tuple(refined_path_list)

    previous_quality = _selected_path_quality_key(
        selected_path,
        total_cost=total_cost,
        bridge_trigger_joint_delta_deg=motion_settings.bridge_trigger_joint_delta_deg,
    )
    refined_quality = _selected_path_quality_key(
        refined_path,
        total_cost=refined_cost,
        bridge_trigger_joint_delta_deg=motion_settings.bridge_trigger_joint_delta_deg,
    )
    if refined_quality >= previous_quality:
        return tuple(ik_layers), tuple(selected_path), total_cost

    print(
        "Accepted wrist singularity refinement: "
        f"window_count={len(refine_indices)}, added_candidates={new_candidate_count}, "
        f"path_cost={refined_cost:.3f}."
    )
    return tuple(augmented_layers), refined_path, refined_cost


def _collect_wrist_refinement_indices(
    selected_path: Sequence[_IKCandidate],
    motion_settings: RoboDKMotionSettings,
) -> tuple[int, ...]:
    """找出需要补做腕部 refinement 的路径窗口。"""

    if not selected_path:
        return ()

    refine_indices: set[int] = set()
    radius = max(0, motion_settings.wrist_refinement_window_radius)
    for index, candidate in enumerate(selected_path):
        if len(candidate.joints) < 6:
            continue

        should_refine = abs(candidate.joints[4]) < motion_settings.wrist_refinement_a5_threshold_deg
        if not should_refine:
            for neighbor_index in (index - 1, index + 1):
                if neighbor_index < 0 or neighbor_index >= len(selected_path):
                    continue
                neighbor_candidate = selected_path[neighbor_index]
                if len(neighbor_candidate.joints) < 6:
                    continue
                large_wrist_step = max(
                    abs(candidate.joints[3] - neighbor_candidate.joints[3]),
                    abs(candidate.joints[5] - neighbor_candidate.joints[5]),
                )
                if (
                    large_wrist_step > motion_settings.wrist_refinement_large_wrist_step_deg
                    or candidate.config_flags != neighbor_candidate.config_flags
                ):
                    should_refine = True
                    break

        if not should_refine:
            continue

        for expand_index in range(index - radius, index + radius + 1):
            if 0 <= expand_index < len(selected_path):
                refine_indices.add(expand_index)

    return tuple(sorted(refine_indices))


def _build_wrist_refinement_seed_strategies(
    selected_path: Sequence[_IKCandidate],
    target_index: int,
    *,
    motion_settings: RoboDKMotionSettings,
    lower_limits: Sequence[float],
    upper_limits: Sequence[float],
) -> tuple[tuple[float, ...], ...]:
    """构造用于奇异窗口二次求解的 seed 集。"""

    seen: set[tuple[float, ...]] = set()
    seeds: list[tuple[float, ...]] = []
    radius = max(1, motion_settings.wrist_refinement_window_radius + 1)
    reference_a6 = _estimate_reference_a6_for_wrist_refinement(
        selected_path,
        target_index,
        a5_threshold_deg=motion_settings.wrist_refinement_a5_threshold_deg,
    )
    a6_targets = [reference_a6, 0.0]
    for phase_offset_deg in motion_settings.wrist_refinement_phase_offsets_deg:
        a6_targets.append(reference_a6 - phase_offset_deg)
        a6_targets.append(reference_a6 + phase_offset_deg)

    for neighbor_index in range(
        max(0, target_index - radius),
        min(len(selected_path), target_index + radius + 1),
    ):
        base_seed = selected_path[neighbor_index].joints
        _append_seed_if_unique(seeds, seen, _clip_seed_to_limits(base_seed, lower_limits, upper_limits))
        if len(base_seed) < 6:
            continue

        for target_a6 in a6_targets:
            compensation_deg = base_seed[5] - target_a6
            variant = list(base_seed)
            variant[3] = variant[3] + compensation_deg
            variant[5] = target_a6
            _append_seed_if_unique(
                seeds,
                seen,
                _clip_seed_to_limits(variant, lower_limits, upper_limits),
            )

        for phase_offset_deg in motion_settings.wrist_refinement_phase_offsets_deg:
            for direction in (-1.0, 1.0):
                variant = list(base_seed)
                phase_shift = direction * phase_offset_deg
                variant[3] = variant[3] + phase_shift
                variant[5] = variant[5] - phase_shift
                _append_seed_if_unique(
                    seeds,
                    seen,
                    _clip_seed_to_limits(variant, lower_limits, upper_limits),
                )

    return tuple(seeds)


def _estimate_reference_a6_for_wrist_refinement(
    selected_path: Sequence[_IKCandidate],
    target_index: int,
    *,
    a5_threshold_deg: float,
) -> float:
    """估计当前奇异窗口希望维持的 A6 相位参考值。"""

    candidate_values: list[float] = []
    for search_radius in range(0, len(selected_path)):
        left_index = target_index - search_radius
        right_index = target_index + search_radius
        for index in (left_index, right_index):
            if index < 0 or index >= len(selected_path):
                continue
            candidate = selected_path[index]
            if len(candidate.joints) < 6:
                continue
            if abs(candidate.joints[4]) >= a5_threshold_deg:
                candidate_values.append(candidate.joints[5])
        if candidate_values:
            break

    if candidate_values:
        return sum(candidate_values) / len(candidate_values)
    return selected_path[target_index].joints[5]


def _append_seed_if_unique(
    seeds: list[tuple[float, ...]],
    seen: set[tuple[float, ...]],
    seed: Sequence[float],
) -> None:
    """把 seed 去重后加入列表。"""

    normalized_seed = tuple(float(value) for value in seed)
    dedup_key = tuple(round(value, _IK_DEDUP_DECIMALS) for value in normalized_seed)
    if dedup_key in seen:
        return
    seen.add(dedup_key)
    seeds.append(normalized_seed)


def _selected_path_quality_key(
    selected_path: Sequence[_IKCandidate],
    *,
    total_cost: float,
    bridge_trigger_joint_delta_deg: float,
) -> tuple[float, ...]:
    """把最终路径压成一个可比较的质量排序键。"""

    (
        config_switches,
        bridge_like_segments,
        worst_joint_step_deg,
        mean_joint_step_deg,
    ) = _summarize_selected_path(
        selected_path,
        bridge_trigger_joint_delta_deg=bridge_trigger_joint_delta_deg,
    )
    return (
        float(bridge_like_segments),
        float(config_switches),
        float(worst_joint_step_deg),
        float(mean_joint_step_deg),
        float(total_cost),
    )


def _path_is_clean_enough_for_program_generation(
    selected_path: Sequence[_IKCandidate],
    *,
    bridge_trigger_joint_delta_deg: float,
    preferred_joint_step_deg: Sequence[float],
) -> bool:
    """判断 exact path 是否已经足够稳定，可以直接进入程序落地阶段。"""

    if not selected_path:
        return False

    problem_segments = _collect_problem_segments(
        selected_path,
        bridge_trigger_joint_delta_deg=bridge_trigger_joint_delta_deg,
    )
    if problem_segments:
        return False

    _config_switches, _bridge_like_segments, worst_joint_step_deg, _mean_joint_step_deg = (
        _summarize_selected_path(
            selected_path,
            bridge_trigger_joint_delta_deg=bridge_trigger_joint_delta_deg,
        )
    )
    preferred_limit = max(preferred_joint_step_deg, default=0.0)
    return worst_joint_step_deg <= preferred_limit + 1e-9


def _redistribute_orientation_in_solution_rich_window(
    base_pose_rows: Sequence[dict[str, float]],
    ik_layers: Sequence[_IKLayer],
    selected_path: Sequence[_IKCandidate],
    *,
    total_cost: float,
    start_joints: tuple[float, ...],
    robot,
    mat_type,
    tool_pose,
    reference_pose,
    joint_count: int,
    motion_settings: RoboDKMotionSettings,
    optimizer_settings: _PathOptimizerSettings,
    lower_limits: Sequence[float],
    upper_limits: Sequence[float],
    a1_lower_deg: float,
    a1_upper_deg: float,
    a2_max_deg: float,
    joint_constraint_tolerance_deg: float,
    base_rotation_offset_deg: Sequence[float],
) -> tuple[tuple[_IKLayer, ...], tuple[_IKCandidate, ...], float]:
    """在“解比较多”的窗口里搜索纯姿态重分配方案。

    这一步的设计目标有两个：
    1. 不改 TCP 位置，避免牺牲路径精度；
    2. 如果 exact-pose 全局 DP 仍然留下残余坏段，就尝试把“后续整段姿态”
       在一个多解窗口里平滑重分配，从而把必须发生的位形切换提前摊开。

    这和之前的桥接不同：
    - 桥接是已经出现坏段后，再插额外点补救；
    - 这里是直接改原始路径点的姿态分布，再重新跑整条 exact-pose 求解。
    """

    if (
        not motion_settings.enable_solution_rich_orientation_redistribution
        or len(selected_path) <= 1
    ):
        return tuple(ik_layers), tuple(selected_path), total_cost

    problem_segments = _collect_problem_segments(
        selected_path,
        bridge_trigger_joint_delta_deg=motion_settings.bridge_trigger_joint_delta_deg,
    )
    if not problem_segments:
        return tuple(ik_layers), tuple(selected_path), total_cost

    candidate_windows = _build_solution_rich_window_candidates(
        ik_layers,
        problem_segment_index=problem_segments[0][0],
        window_radius=motion_settings.solution_rich_orientation_window_radius,
    )
    candidate_offsets = _build_solution_rich_orientation_offsets(
        base_rotation_offset_deg,
        offset_deg=motion_settings.solution_rich_orientation_offset_deg,
    )
    if not candidate_windows or not candidate_offsets:
        return tuple(ik_layers), tuple(selected_path), total_cost

    lower_limits_tuple = tuple(float(value) for value in lower_limits[:joint_count])
    upper_limits_tuple = tuple(float(value) for value in upper_limits[:joint_count])
    search_seed_joints = _build_seed_joint_strategies(
        robot=robot,
        lower_limits=lower_limits_tuple,
        upper_limits=upper_limits_tuple,
        joint_count=joint_count,
    )
    quick_baseline_quality = _orientation_redistribution_candidate_sort_key(
        ik_layers,
        selected_path,
        total_cost=total_cost,
        window_start=-1,
        window_end=-1,
        robot=robot,
        tool_pose=tool_pose,
        reference_pose=reference_pose,
        motion_settings=motion_settings,
        optimizer_settings=optimizer_settings,
        lower_limits=lower_limits,
        upper_limits=upper_limits,
        a1_lower_deg=a1_lower_deg,
        a1_upper_deg=a1_upper_deg,
        evaluate_fixed_position_bridge=False,
    )
    candidate_results: list[
        tuple[
            tuple[float, ...],
            tuple[_IKLayer, ...],
            tuple[_IKCandidate, ...],
            float,
            tuple[int, int],
            tuple[float, ...],
        ]
    ] = []

    for window_start, window_end in candidate_windows:
        for rotation_offset_deg in candidate_offsets:
            candidate_pose_rows = _apply_orientation_redistribution_window(
                base_pose_rows,
                window_start=window_start,
                window_end=window_end,
                rotation_offset_deg=rotation_offset_deg,
            )
            try:
                reused_prefix_layers = tuple(ik_layers[:window_start])
                rebuilt_suffix_layers = tuple(
                    _build_ik_layers(
                        candidate_pose_rows[window_start:],
                        robot=robot,
                        mat_type=mat_type,
                        tool_pose=tool_pose,
                        reference_pose=reference_pose,
                        joint_count=joint_count,
                        optimizer_settings=optimizer_settings,
                        a1_lower_deg=a1_lower_deg,
                        a1_upper_deg=a1_upper_deg,
                        a2_max_deg=a2_max_deg,
                        joint_constraint_tolerance_deg=joint_constraint_tolerance_deg,
                        seed_joints_override=search_seed_joints,
                        lower_limits_override=lower_limits_tuple,
                        upper_limits_override=upper_limits_tuple,
                        log_summary=False,
                    )
                )
                candidate_layers = reused_prefix_layers + rebuilt_suffix_layers
                candidate_path_list, candidate_cost = _optimize_joint_path(
                    candidate_layers,
                    robot=robot,
                    move_type=motion_settings.move_type,
                    start_joints=start_joints,
                    optimizer_settings=optimizer_settings,
                )
            except RuntimeError:
                continue

            candidate_path = tuple(candidate_path_list)
            candidate_quality = _orientation_redistribution_candidate_sort_key(
                candidate_layers,
                candidate_path,
                total_cost=candidate_cost,
                window_start=window_start,
                window_end=window_end,
                robot=robot,
                tool_pose=tool_pose,
                reference_pose=reference_pose,
                motion_settings=motion_settings,
                optimizer_settings=optimizer_settings,
                lower_limits=lower_limits,
                upper_limits=upper_limits,
                a1_lower_deg=a1_lower_deg,
                a1_upper_deg=a1_upper_deg,
                evaluate_fixed_position_bridge=False,
            )
            if candidate_quality >= quick_baseline_quality:
                continue

            candidate_results.append(
                (
                    candidate_quality,
                    candidate_layers,
                    candidate_path,
                    candidate_cost,
                    (window_start, window_end),
                    rotation_offset_deg,
                )
            )

    if not candidate_results:
        return tuple(ik_layers), tuple(selected_path), total_cost

    candidate_results.sort(key=lambda item: item[0])
    (
        quick_best_quality,
        quick_best_layers,
        quick_best_path,
        quick_best_cost,
        quick_best_window,
        quick_best_offset,
    ) = candidate_results[0]
    if quick_best_quality >= quick_baseline_quality:
        return tuple(ik_layers), tuple(selected_path), total_cost
    best_result = (
        quick_best_layers,
        quick_best_path,
        quick_best_cost,
        quick_best_window,
        quick_best_offset,
    )

    refined_layers, refined_path, refined_cost, refined_window, refined_offset = best_result
    print(
        "Accepted solution-rich orientation redistribution: "
        f"segment={problem_segments[0][0]}->{problem_segments[0][0] + 1}, "
        f"window={refined_window[0]}->{refined_window[1]}, "
        f"offset xyz(deg)={[round(value, 3) for value in refined_offset]}, "
        f"path_cost={refined_cost:.3f}."
    )
    return refined_layers, refined_path, refined_cost


def _orientation_redistribution_candidate_sort_key(
    ik_layers: Sequence[_IKLayer],
    selected_path: Sequence[_IKCandidate],
    *,
    total_cost: float,
    window_start: int,
    window_end: int,
    robot,
    tool_pose,
    reference_pose,
    motion_settings: RoboDKMotionSettings,
    optimizer_settings: _PathOptimizerSettings,
    lower_limits: Sequence[float],
    upper_limits: Sequence[float],
    a1_lower_deg: float,
    a1_upper_deg: float,
    evaluate_fixed_position_bridge: bool,
) -> tuple[float, ...]:
    """给“解丰富窗口姿态重分配”候选打分。

    这里只看 exact path 质量还不够。
    用户真正关心的是：
    1. 位形切换最好落在选定的驻留窗口里；
    2. 该切换段最好能直接做“固定位置桥接”；
    3. 在此基础上，再比较桥接触发次数、最坏跳变和总成本。
    """

    problem_segments = _collect_problem_segments(
        selected_path,
        bridge_trigger_joint_delta_deg=motion_settings.bridge_trigger_joint_delta_deg,
    )
    switch_segments = [
        segment_index
        for segment_index, config_changed, _max_joint_delta, _mean_joint_delta in problem_segments
        if config_changed
    ]
    switch_in_window = any(
        window_start <= segment_index + 1 <= window_end for segment_index in switch_segments
    )

    fixed_position_feasible = False
    if evaluate_fixed_position_bridge:
        candidate_segments_to_check = switch_segments or [
            segment_index
            for segment_index, _config_changed, _max_joint_delta, _mean_joint_delta in problem_segments
        ]
        for segment_index in candidate_segments_to_check:
            if segment_index < 0 or segment_index + 1 >= len(ik_layers):
                continue
            if _position_locked_bridge_is_feasible_for_segment(
                ik_layers,
                selected_path,
                segment_index=segment_index,
                robot=robot,
                tool_pose=tool_pose,
                reference_pose=reference_pose,
                motion_settings=motion_settings,
                optimizer_settings=optimizer_settings,
                lower_limits=lower_limits,
                upper_limits=upper_limits,
                a1_lower_deg=a1_lower_deg,
                a1_upper_deg=a1_upper_deg,
            ):
                fixed_position_feasible = True
                break

    (
        config_switches,
        bridge_like_segments,
        worst_joint_step_deg,
        mean_joint_step_deg,
    ) = _summarize_selected_path(
        selected_path,
        bridge_trigger_joint_delta_deg=motion_settings.bridge_trigger_joint_delta_deg,
    )
    return (
        0.0 if fixed_position_feasible else 1.0,
        0.0 if switch_in_window else 1.0,
        float(bridge_like_segments),
        float(config_switches),
        float(worst_joint_step_deg),
        float(mean_joint_step_deg),
        float(total_cost),
    )


def _position_locked_bridge_is_feasible_for_segment(
    ik_layers: Sequence[_IKLayer],
    selected_path: Sequence[_IKCandidate],
    *,
    segment_index: int,
    robot,
    tool_pose,
    reference_pose,
    motion_settings: RoboDKMotionSettings,
    optimizer_settings: _PathOptimizerSettings,
    lower_limits: Sequence[float],
    upper_limits: Sequence[float],
    a1_lower_deg: float,
    a1_upper_deg: float,
) -> bool:
    """快速判断某个相邻段是否存在固定位置桥接解。"""

    try:
        _build_position_locked_bridge_segment(
            segment_index=segment_index,
            target_index_width=3,
            current_target_name="TMP",
            previous_pose=ik_layers[segment_index].pose,
            current_pose=ik_layers[segment_index + 1].pose,
            previous_candidate=selected_path[segment_index],
            current_candidate=selected_path[segment_index + 1],
            robot=robot,
            tool_pose=tool_pose,
            reference_pose=reference_pose,
            motion_settings=motion_settings,
            optimizer_settings=optimizer_settings,
            lower_limits=lower_limits,
            upper_limits=upper_limits,
            a1_lower_deg=a1_lower_deg,
            a1_upper_deg=a1_upper_deg,
        )
    except RuntimeError:
        return False
    return True


def _collect_problem_segments(
    selected_path: Sequence[_IKCandidate],
    *,
    bridge_trigger_joint_delta_deg: float,
) -> tuple[tuple[int, bool, float, float], ...]:
    """找出当前 exact path 里最值得进一步处理的坏段。

    返回值中的每个元素依次表示：
    - 坏段起点索引 `i`，实际段为 `i -> i+1`
    - 该段是否发生了 `config_flags` 切换
    - 该段单步最大关节跳变
    - 该段平均关节跳变
    """

    segments: list[tuple[int, bool, float, float]] = []
    for segment_index, (previous_candidate, current_candidate) in enumerate(
        zip(selected_path, selected_path[1:])
    ):
        joint_deltas = [
            abs(current - previous)
            for previous, current in zip(previous_candidate.joints, current_candidate.joints)
        ]
        max_joint_delta = max(joint_deltas, default=0.0)
        mean_joint_delta = _mean_abs_joint_delta(
            previous_candidate.joints,
            current_candidate.joints,
        )
        config_changed = previous_candidate.config_flags != current_candidate.config_flags
        if config_changed or max_joint_delta > bridge_trigger_joint_delta_deg:
            segments.append(
                (
                    segment_index,
                    config_changed,
                    max_joint_delta,
                    mean_joint_delta,
                )
            )

    # 处理优先级：
    # 1. 先处理真的发生位形切换的段；
    # 2. 再看单步最坏跳变；
    # 3. 最后看平均跳变。
    segments.sort(
        key=lambda item: (
            -int(item[1]),
            -float(item[2]),
            -float(item[3]),
        )
    )
    return tuple(segments)


def _build_solution_rich_window_candidates(
    ik_layers: Sequence[_IKLayer],
    *,
    problem_segment_index: int,
    window_radius: int,
) -> tuple[tuple[int, int], ...]:
    """围绕坏段构造几个“解丰富窗口”候选。

    窗口并不是越窄越好。通常需要同时准备：
    - 一个尽量宽的窗口，把姿态变化摊开；
    - 一个中等宽度窗口，避免过度扰动整段后续路径；
    - 一个很窄的窗口，作为最保守的兜底方案。
    """

    if not ik_layers:
        return ()

    start_lower_bound = max(0, problem_segment_index - max(0, window_radius))
    end_upper_bound = min(len(ik_layers) - 1, problem_segment_index + 1 + max(0, window_radius))
    left_indices = [
        index
        for index in range(start_lower_bound, problem_segment_index + 1)
        if _layer_is_solution_rich(ik_layers[index])
    ]
    right_indices = [
        index
        for index in range(problem_segment_index + 1, end_upper_bound + 1)
        if _layer_is_solution_rich(ik_layers[index])
    ]
    if not left_indices:
        left_indices = [start_lower_bound, problem_segment_index]
    if not right_indices:
        right_indices = [problem_segment_index + 1, end_upper_bound]

    candidate_windows = [
        (left_indices[0], right_indices[-1]),
        (left_indices[len(left_indices) // 2], right_indices[len(right_indices) // 2]),
        (left_indices[-1], right_indices[0]),
    ]

    unique_windows: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    for window_start, window_end in candidate_windows:
        normalized_window = (min(window_start, window_end), max(window_start, window_end))
        if normalized_window[0] == normalized_window[1]:
            continue
        if normalized_window in seen:
            continue
        seen.add(normalized_window)
        unique_windows.append(normalized_window)
    return tuple(unique_windows)


def _layer_is_solution_rich(layer: _IKLayer) -> bool:
    """判断某一层是否适合作为“提前重分配姿态”的窗口点。"""

    config_groups = {candidate.config_flags for candidate in layer.candidates}
    return len(layer.candidates) >= 4 or len(config_groups) >= 2


def _build_solution_rich_orientation_offsets(
    base_rotation_offset_deg: Sequence[float],
    *,
    offset_deg: float,
) -> tuple[tuple[float, float, float], ...]:
    """根据当前整路径姿态偏置方向，构造局部姿态重分配的候选偏置。"""

    if offset_deg <= 0.0:
        return ()

    axis_signs = []
    for value in base_rotation_offset_deg[:3]:
        if value > 1e-9:
            axis_signs.append(1.0)
        elif value < -1e-9:
            axis_signs.append(-1.0)
        else:
            axis_signs.append(1.0)

    sx, sy, sz = axis_signs
    raw_offsets = (
        (sx * offset_deg, 0.0, 0.0),
        (0.0, sy * offset_deg, 0.0),
        (0.0, 0.0, sz * offset_deg),
        (sx * offset_deg, sy * offset_deg, 0.0),
        (sx * offset_deg, 0.0, sz * offset_deg),
        (0.0, sy * offset_deg, sz * offset_deg),
        (sx * offset_deg, sy * offset_deg, sz * offset_deg),
    )

    offsets: list[tuple[float, float, float]] = []
    seen: set[tuple[float, float, float]] = set()
    for rotation_offset_deg in raw_offsets:
        normalized_offset = tuple(float(value) for value in rotation_offset_deg)
        if not any(abs(value) > 1e-9 for value in normalized_offset):
            continue
        if normalized_offset in seen:
            continue
        seen.add(normalized_offset)
        offsets.append(normalized_offset)
    return tuple(offsets)


def _apply_orientation_redistribution_window(
    base_pose_rows: Sequence[dict[str, float]],
    *,
    window_start: int,
    window_end: int,
    rotation_offset_deg: Sequence[float],
) -> tuple[dict[str, float], ...]:
    """对一段窗口后的所有路径点做“位置锁定、姿态渐变”的重分配。

    具体做法是：
    - 在 `window_start` 之前，保持原始姿态不变；
    - 在 `[window_start, window_end]` 内，把旋转偏置从 0 平滑拉到目标偏置；
    - 在 `window_end` 之后，整段维持新的姿态偏置；
    - 整个过程中 TCP 平移始终不变，只更新旋转矩阵。
    """

    if window_end < window_start:
        return tuple(dict(row) for row in base_pose_rows)

    redistributed_rows: list[dict[str, float]] = []
    transition_span = max(1, window_end - window_start + 1)
    for row_index, pose_row in enumerate(base_pose_rows):
        if row_index < window_start:
            redistributed_rows.append(dict(pose_row))
            continue

        if row_index > window_end:
            interpolation_ratio = 1.0
        else:
            interpolation_ratio = (row_index - window_start + 1) / transition_span

        local_rotation_offset_deg = tuple(
            float(value) * interpolation_ratio for value in rotation_offset_deg
        )
        local_rotation = _rotation_matrix_from_xyz_offset_deg(local_rotation_offset_deg)
        base_rotation, base_translation = _pose_row_to_rotation_translation(pose_row)
        redistributed_rows.append(
            _pose_row_from_rotation_translation(
                _multiply_rotation_matrices(local_rotation, base_rotation),
                base_translation,
            )
        )
    return tuple(redistributed_rows)


def _build_seed_joint_strategies(
    *,
    robot,
    lower_limits: tuple[float, ...],
    upper_limits: tuple[float, ...],
    joint_count: int,
) -> tuple[tuple[float, ...], ...]:
    """构造一组 seed，用来诱导 SolveIK 返回不同支路附近的解。"""

    seeds: list[tuple[float, ...]] = []

    current_joints = _trim_joint_vector(robot.Joints().list(), joint_count)
    if current_joints:
        seeds.append(current_joints)

    home_joints = _trim_joint_vector(robot.JointsHome().list(), joint_count)
    if home_joints:
        seeds.append(home_joints)

    midpoint = tuple((lower + upper) * 0.5 for lower, upper in zip(lower_limits, upper_limits))
    seeds.append(midpoint)
    seeds.append(_clip_seed_to_limits((0.0,) * joint_count, lower_limits, upper_limits))

    # 在关节空间中再撒几组均匀点，增加命中不同支路的概率。
    for ratio in (0.15, 0.50, 0.85):
        seeds.append(
            tuple(
                lower + (upper - lower) * ratio
                for lower, upper in zip(lower_limits, upper_limits)
            )
        )

    # 对 6 轴机器人，额外人为扰动 J5 / J6，尽量诱导出 wrist flip 等不同分支。
    if joint_count >= 6:
        for joint5 in (-90.0, 90.0):
            for joint6 in (-180.0, 0.0, 180.0):
                wrist_seed = list(midpoint)
                wrist_seed[4] = joint5
                wrist_seed[5] = joint6
                seeds.append(_clip_seed_to_limits(wrist_seed, lower_limits, upper_limits))

    # 去重，避免重复种子造成无意义的 IK 调用。
    unique_seeds: list[tuple[float, ...]] = []
    seen: set[tuple[float, ...]] = set()
    for seed in seeds:
        rounded = tuple(round(value, _IK_DEDUP_DECIMALS) for value in seed)
        if rounded in seen:
            continue
        seen.add(rounded)
        unique_seeds.append(seed)
    return tuple(unique_seeds)


def _collect_ik_candidates(
    robot,
    pose,
    *,
    tool_pose,
    reference_pose,
    lower_limits: tuple[float, ...],
    upper_limits: tuple[float, ...],
    seed_joints: Sequence[tuple[float, ...]],
    joint_count: int,
    optimizer_settings: _PathOptimizerSettings,
    a1_lower_deg: float,
    a1_upper_deg: float,
    a2_max_deg: float,
    joint_constraint_tolerance_deg: float,
) -> list[_IKCandidate]:
    """收集一个路径点的所有候选 IK 解。

    这里同时使用两类入口：
    1. SolveIK_All：让 RoboDK 直接枚举当前能给出的所有支路；
    2. SolveIK + 多 seed：进一步诱导 RoboDK 靠近不同局部支路求解。
    """

    candidates: list[_IKCandidate] = []
    seen: set[tuple[float, ...]] = set()

    all_solutions = robot.SolveIK_All(pose, tool_pose, reference_pose)
    for raw_solution in all_solutions:
        _append_candidate_if_unique(
            candidates,
            seen,
            robot=robot,
            raw_joints=raw_solution,
            lower_limits=lower_limits,
            upper_limits=upper_limits,
            joint_count=joint_count,
            optimizer_settings=optimizer_settings,
            a1_lower_deg=a1_lower_deg,
            a1_upper_deg=a1_upper_deg,
            a2_max_deg=a2_max_deg,
            joint_constraint_tolerance_deg=joint_constraint_tolerance_deg,
        )

    for seed in seed_joints:
        raw_solution = robot.SolveIK(pose, list(seed), tool_pose, reference_pose)
        _append_candidate_if_unique(
            candidates,
            seen,
            robot=robot,
            raw_joints=raw_solution,
            lower_limits=lower_limits,
            upper_limits=upper_limits,
            joint_count=joint_count,
            optimizer_settings=optimizer_settings,
            a1_lower_deg=a1_lower_deg,
            a1_upper_deg=a1_upper_deg,
            a2_max_deg=a2_max_deg,
            joint_constraint_tolerance_deg=joint_constraint_tolerance_deg,
        )

    # 先按配置标志、再按节点代价排序，方便调试时观察候选。
    candidates.sort(
        key=lambda candidate: (
            candidate.config_flags,
            candidate.joint_limit_penalty + candidate.singularity_penalty,
            candidate.joints,
        )
    )
    return candidates


def _append_candidate_if_unique(
    candidates: list[_IKCandidate],
    seen: set[tuple[float, ...]],
    *,
    robot,
    raw_joints,
    lower_limits: tuple[float, ...],
    upper_limits: tuple[float, ...],
    joint_count: int,
    optimizer_settings: _PathOptimizerSettings,
    a1_lower_deg: float,
    a1_upper_deg: float,
    a2_max_deg: float,
    joint_constraint_tolerance_deg: float,
) -> None:
    """把一个 IK 解加入候选集合。

    会按以下顺序过滤：
    1. 空解 / 非法解；
    2. 超出机器人自身关节限位；
    3. 不满足用户指定的 A1 / A2 硬约束；
    4. 与已有候选重复。
    """

    joints = _extract_joint_tuple(raw_joints, joint_count)
    if not joints:
        return

    if not _is_within_joint_limits(joints, lower_limits, upper_limits):
        return

    if not _passes_user_joint_constraints(
        joints,
        a1_lower_deg=a1_lower_deg,
        a1_upper_deg=a1_upper_deg,
        a2_max_deg=a2_max_deg,
        tolerance_deg=joint_constraint_tolerance_deg,
    ):
        return

    dedup_key = tuple(round(value, _IK_DEDUP_DECIMALS) for value in joints)
    if dedup_key in seen:
        return
    seen.add(dedup_key)

    config_flags = _candidate_config_flags(robot, joints)
    candidates.append(
        _IKCandidate(
            joints=joints,
            config_flags=config_flags,
            joint_limit_penalty=_joint_limit_penalty(
                joints,
                lower_limits,
                upper_limits,
                optimizer_settings,
            ),
            singularity_penalty=_singularity_penalty(robot, joints, optimizer_settings),
        )
    )


def _candidate_config_flags(robot, joints: tuple[float, ...]) -> tuple[int, ...]:
    """缓存 RoboDK `JointsConfig()` 结果，减少重复查询。"""

    cache_key = (id(robot), joints)
    cached_flags = _ROBOT_CONFIG_FLAGS_CACHE.get(cache_key)
    if cached_flags is not None:
        return cached_flags

    config_values = robot.JointsConfig(list(joints)).list()
    config_flags = tuple(int(round(value)) for value in config_values[:_CONFIG_FLAG_COUNT])
    _ROBOT_CONFIG_FLAGS_CACHE[cache_key] = config_flags
    return config_flags


def _build_program_waypoints(
    ik_layers: Sequence[_IKLayer],
    selected_path: Sequence[_IKCandidate],
    *,
    robot,
    tool_pose,
    reference_pose,
    motion_settings: RoboDKMotionSettings,
    optimizer_settings: _PathOptimizerSettings,
    lower_limits: Sequence[float],
    upper_limits: Sequence[float],
    a1_lower_deg: float,
    a1_upper_deg: float,
) -> list[_ProgramWaypoint]:
    """把最终关节路径展开成真正要写入程序的路点序列。

    除了原始路径点外，还会在大跳变段之间插入桥接点。
    """

    if not ik_layers or not selected_path:
        return []

    target_index_width = max(3, len(str(len(ik_layers) - 1)))
    waypoints = [
        _ProgramWaypoint(
            name=f"P_{0:0{target_index_width}d}",
            pose=ik_layers[0].pose,
            joints=selected_path[0].joints,
            move_type=motion_settings.move_type,
            is_bridge=False,
        )
    ]

    for index in range(1, len(ik_layers)):
        previous_layer = ik_layers[index - 1]
        current_layer = ik_layers[index]
        previous_candidate = selected_path[index - 1]
        current_candidate = selected_path[index]

        if _needs_pose_bridge(previous_candidate, current_candidate, motion_settings):
            try:
                bridge_waypoints, current_waypoint = _build_position_locked_bridge_segment(
                    segment_index=index - 1,
                    target_index_width=target_index_width,
                    current_target_name=f"P_{index:0{target_index_width}d}",
                    previous_pose=previous_layer.pose,
                    current_pose=current_layer.pose,
                    previous_candidate=previous_candidate,
                    current_candidate=current_candidate,
                    robot=robot,
                    tool_pose=tool_pose,
                    reference_pose=reference_pose,
                    motion_settings=motion_settings,
                    optimizer_settings=optimizer_settings,
                    lower_limits=lower_limits,
                    upper_limits=upper_limits,
                    a1_lower_deg=a1_lower_deg,
                    a1_upper_deg=a1_upper_deg,
                )
                waypoints.extend(bridge_waypoints)
                waypoints.append(current_waypoint)
                continue
            except RuntimeError as exc:
                print(
                    "Warning: exact fixed-position posture bridge is infeasible for "
                    f"segment {index - 1}->{index}; falling back to best-effort bridge. "
                    f"Reason: {exc}"
                )
                waypoints.extend(
                    _build_bridge_waypoints(
                        segment_index=index - 1,
                        target_index_width=target_index_width,
                        previous_pose=previous_layer.pose,
                        current_pose=current_layer.pose,
                        previous_candidate=previous_candidate,
                        current_candidate=current_candidate,
                        robot=robot,
                        tool_pose=tool_pose,
                        reference_pose=reference_pose,
                        motion_settings=motion_settings,
                        optimizer_settings=optimizer_settings,
                        lower_limits=lower_limits,
                        upper_limits=upper_limits,
                        a1_lower_deg=a1_lower_deg,
                        a1_upper_deg=a1_upper_deg,
                    )
                )

        waypoints.append(
            _ProgramWaypoint(
                name=f"P_{index:0{target_index_width}d}",
                pose=current_layer.pose,
                joints=current_candidate.joints,
                move_type=motion_settings.move_type,
                is_bridge=False,
            )
        )

    return waypoints


def _needs_pose_bridge(
    previous_candidate: _IKCandidate,
    current_candidate: _IKCandidate,
    motion_settings: RoboDKMotionSettings,
) -> bool:
    """判断相邻两点之间是否需要插入姿态重构桥接点。"""

    if not motion_settings.enable_pose_bridge:
        return False

    joint_deltas = [
        abs(current - previous)
        for previous, current in zip(previous_candidate.joints, current_candidate.joints)
    ]
    if joint_deltas and max(joint_deltas) > motion_settings.bridge_trigger_joint_delta_deg:
        return True

    return previous_candidate.config_flags != current_candidate.config_flags


def _build_position_locked_bridge_segment(
    *,
    segment_index: int,
    target_index_width: int,
    current_target_name: str,
    previous_pose,
    current_pose,
    previous_candidate: _IKCandidate,
    current_candidate: _IKCandidate,
    robot,
    tool_pose,
    reference_pose,
    motion_settings: RoboDKMotionSettings,
    optimizer_settings: _PathOptimizerSettings,
    lower_limits: Sequence[float],
    upper_limits: Sequence[float],
    a1_lower_deg: float,
    a1_upper_deg: float,
) -> tuple[list[_ProgramWaypoint], _ProgramWaypoint]:
    """构造“法兰位置锁定”的姿态桥接段。

    这里会依次尝试两种策略：
    1. 先在前一个点原地变姿态，再线性走到下一个点；
    2. 先线性走到下一个点的位置，再在下一个点原地变姿态。

    两种策略都满足“姿态变化时法兰位置固定”，只是固定的位置不同。
    """

    first_error: RuntimeError | None = None
    for lock_to_current_position in (False, True):
        try:
            return _build_position_locked_bridge_segment_for_anchor(
                segment_index=segment_index,
                target_index_width=target_index_width,
                current_target_name=current_target_name,
                previous_pose=previous_pose,
                current_pose=current_pose,
                previous_candidate=previous_candidate,
                current_candidate=current_candidate,
                robot=robot,
                tool_pose=tool_pose,
                reference_pose=reference_pose,
                motion_settings=motion_settings,
                optimizer_settings=optimizer_settings,
                lower_limits=lower_limits,
                upper_limits=upper_limits,
                a1_lower_deg=a1_lower_deg,
                a1_upper_deg=a1_upper_deg,
                lock_to_current_position=lock_to_current_position,
            )
        except RuntimeError as exc:
            if first_error is None:
                first_error = exc

    raise RuntimeError(
        "No feasible fixed-position posture bridge could be found for either anchor strategy."
    ) from first_error


def _build_position_locked_bridge_segment_for_anchor(
    *,
    segment_index: int,
    target_index_width: int,
    current_target_name: str,
    previous_pose,
    current_pose,
    previous_candidate: _IKCandidate,
    current_candidate: _IKCandidate,
    robot,
    tool_pose,
    reference_pose,
    motion_settings: RoboDKMotionSettings,
    optimizer_settings: _PathOptimizerSettings,
    lower_limits: Sequence[float],
    upper_limits: Sequence[float],
    a1_lower_deg: float,
    a1_upper_deg: float,
    lock_to_current_position: bool,
) -> tuple[list[_ProgramWaypoint], _ProgramWaypoint]:
    """按指定锚点位置构造固定位置桥接段。"""

    bridge_step_limits = _normalize_step_limits(
        motion_settings.bridge_step_deg,
        len(previous_candidate.joints),
    )
    joint_segment_count = max(
        2,
        math.ceil(
            max(
                abs(current - previous) / limit
                for previous, current, limit in zip(
                    previous_candidate.joints,
                    current_candidate.joints,
                    bridge_step_limits,
                )
            )
        ),
    )
    orientation_delta_deg = _pose_rotation_distance_deg(previous_pose, current_pose)
    orientation_segment_count = max(
        1,
        math.ceil(orientation_delta_deg / _POSITION_LOCK_BRIDGE_ORIENTATION_STEP_DEG),
    )
    segment_count = max(
        joint_segment_count,
        orientation_segment_count,
        _POSITION_LOCK_BRIDGE_MIN_SEGMENTS,
    )
    if previous_candidate.config_flags != current_candidate.config_flags:
        segment_count = max(segment_count, 12)

    locked_translation = _translation_from_pose(
        current_pose if lock_to_current_position else previous_pose
    )
    ratio_indices = range(0, segment_count) if lock_to_current_position else range(1, segment_count + 1)
    rotation_search_deg = _build_position_locked_rotation_search_levels(
        motion_settings.bridge_rotation_search_deg,
        previous_candidate,
        current_candidate,
    )

    bridge_layers: list[tuple[_BridgeCandidate, ...]] = []
    for ratio_index in ratio_indices:
        interpolation_ratio = ratio_index / segment_count
        desired_joints = _interpolate_joints(
            previous_candidate.joints,
            current_candidate.joints,
            interpolation_ratio,
        )
        desired_pose = robot.SolveFK(list(desired_joints), tool_pose, reference_pose)
        base_pose = _build_pose_from_rotation_translation(
            desired_pose,
            _rotation_from_pose(desired_pose),
            locked_translation,
        )
        bridge_layers.append(
            _collect_position_locked_bridge_candidates(
                base_pose=base_pose,
                desired_joints=desired_joints,
                previous_candidate=previous_candidate,
                current_candidate=current_candidate,
                robot=robot,
                tool_pose=tool_pose,
                reference_pose=reference_pose,
                motion_settings=motion_settings,
                rotation_search_deg=rotation_search_deg,
                optimizer_settings=optimizer_settings,
                lower_limits=lower_limits,
                upper_limits=upper_limits,
                a1_lower_deg=a1_lower_deg,
                a1_upper_deg=a1_upper_deg,
            )
        )

    selected_bridge_states, final_reached_joints = _optimize_position_locked_bridge_layers(
        bridge_layers,
        final_pose=current_pose,
        final_preferred_joints=current_candidate.joints,
        start_pose=previous_pose,
        start_joints=previous_candidate.joints,
        robot=robot,
        optimizer_settings=optimizer_settings,
    )

    bridge_waypoints = [
        _ProgramWaypoint(
            name=f"P_{segment_index:0{target_index_width}d}_BR_{bridge_index:02d}",
            pose=bridge_state.pose,
            joints=bridge_state.joints,
            move_type="MoveL",
            is_bridge=True,
        )
        for bridge_index, bridge_state in enumerate(selected_bridge_states, start=1)
    ]
    current_waypoint = _ProgramWaypoint(
        name=current_target_name,
        pose=current_pose,
        joints=final_reached_joints,
        move_type="MoveL",
        is_bridge=False,
    )
    return bridge_waypoints, current_waypoint


def _collect_position_locked_bridge_candidates(
    *,
    base_pose,
    desired_joints: tuple[float, ...],
    previous_candidate: _IKCandidate,
    current_candidate: _IKCandidate,
    robot,
    tool_pose,
    reference_pose,
    motion_settings: RoboDKMotionSettings,
    rotation_search_deg: Sequence[float],
    optimizer_settings: _PathOptimizerSettings,
    lower_limits: Sequence[float],
    upper_limits: Sequence[float],
    a1_lower_deg: float,
    a1_upper_deg: float,
) -> tuple[_BridgeCandidate, ...]:
    """为单个“固定位置桥接层”收集姿态候选。"""

    candidates: list[_BridgeCandidate] = []
    seen_joint_keys: set[tuple[float, ...]] = set()
    seed_joints_options = _build_position_locked_seed_strategies(
        desired_joints=desired_joints,
        previous_candidate=previous_candidate,
        current_candidate=current_candidate,
        motion_settings=motion_settings,
        lower_limits=lower_limits,
        upper_limits=upper_limits,
    )

    for pose_candidate in _iter_position_locked_pose_candidates(
        base_pose,
        rotation_search_deg=rotation_search_deg,
        previous_candidate=previous_candidate,
        current_candidate=current_candidate,
        motion_settings=motion_settings,
    ):
        for joint_candidate in _iter_joint_solutions_for_pose_with_seeds(
            robot,
            pose_candidate,
            seed_joints_options,
            tool_pose,
            reference_pose,
        ):
            if not _is_within_joint_limits(joint_candidate, lower_limits, upper_limits):
                continue
            if not _passes_user_joint_constraints(
                joint_candidate,
                a1_lower_deg=a1_lower_deg,
                a1_upper_deg=a1_upper_deg,
                a2_max_deg=motion_settings.a2_max_deg,
                tolerance_deg=motion_settings.joint_constraint_tolerance_deg,
            ):
                continue

            dedup_key = tuple(round(value, _IK_DEDUP_DECIMALS) for value in joint_candidate)
            if dedup_key in seen_joint_keys:
                continue
            seen_joint_keys.add(dedup_key)
            candidates.append(
                _build_position_locked_bridge_candidate(
                    robot=robot,
                    candidate_pose=pose_candidate,
                    candidate_joints=joint_candidate,
                    base_pose=base_pose,
                    desired_joints=desired_joints,
                    optimizer_settings=optimizer_settings,
                    lower_limits=lower_limits,
                    upper_limits=upper_limits,
                )
            )

    if not candidates:
        raise RuntimeError("No fixed-position bridge candidates remain after applying constraints.")

    candidates.sort(
        key=lambda candidate: (
            candidate.node_cost,
            candidate.config_flags,
            candidate.joints,
        )
    )
    return tuple(candidates[:_BRIDGE_LAYER_CANDIDATE_LIMIT])


def _build_position_locked_seed_strategies(
    *,
    desired_joints: Sequence[float],
    previous_candidate: _IKCandidate,
    current_candidate: _IKCandidate,
    motion_settings: RoboDKMotionSettings,
    lower_limits: Sequence[float],
    upper_limits: Sequence[float],
) -> tuple[tuple[float, ...], ...]:
    """为固定位置桥接构造更偏向腕奇异重构的 seed 集。

    这里直接借鉴 FINA11 的经验：
    - 尽量稳住 A6 相位；
    - 通过 A4/A6 的补偿式 seed 去诱导 RoboDK 给出更多同位姿腕部分解；
    - 在需要时允许较大的 wrist phase shift，把姿态重构摊给 A4。
    """

    seeds: list[tuple[float, ...]] = []
    seen: set[tuple[float, ...]] = set()
    base_seeds = (
        tuple(float(value) for value in desired_joints),
        previous_candidate.joints,
        current_candidate.joints,
    )
    phase_offsets = tuple(
        sorted(
            {
                15.0,
                30.0,
                45.0,
                60.0,
                90.0,
                *(
                    abs(float(value))
                    for value in motion_settings.wrist_refinement_phase_offsets_deg
                ),
            }
        )
    )
    reference_a6_values = [
        base_seed[5]
        for base_seed in base_seeds
        if len(base_seed) >= 6
    ]
    reference_a6_values.append(0.0)
    preferred_a6 = sum(reference_a6_values) / max(1, len(reference_a6_values))

    for base_seed in base_seeds:
        clipped_seed = _clip_seed_to_limits(base_seed, lower_limits, upper_limits)
        _append_seed_if_unique(seeds, seen, clipped_seed)
        if len(base_seed) < 6:
            continue

        for target_a6 in (
            preferred_a6,
            previous_candidate.joints[5],
            current_candidate.joints[5],
            0.0,
        ):
            compensation_deg = base_seed[5] - target_a6
            variant = list(base_seed)
            variant[3] = variant[3] + compensation_deg
            variant[5] = target_a6
            _append_seed_if_unique(
                seeds,
                seen,
                _clip_seed_to_limits(variant, lower_limits, upper_limits),
            )

        for phase_offset_deg in phase_offsets:
            for direction in (-1.0, 1.0):
                phase_shift = direction * phase_offset_deg
                variant = list(base_seed)
                variant[3] = variant[3] + phase_shift
                variant[5] = variant[5] - phase_shift
                _append_seed_if_unique(
                    seeds,
                    seen,
                    _clip_seed_to_limits(variant, lower_limits, upper_limits),
                )

    return tuple(seeds)


def _build_position_locked_rotation_search_levels(
    base_levels_deg: Sequence[float],
    previous_candidate: _IKCandidate,
    current_candidate: _IKCandidate,
) -> tuple[float, ...]:
    """为固定位置桥接构造更激进的姿态搜索角度集。"""

    search_levels = {abs(float(value)) for value in base_levels_deg}
    joint_deltas = [
        abs(current - previous)
        for previous, current in zip(previous_candidate.joints, current_candidate.joints)
    ]
    max_joint_delta = max(joint_deltas, default=0.0)
    near_wrist_singularity = (
        len(previous_candidate.joints) >= 5
        and len(current_candidate.joints) >= 5
        and min(abs(previous_candidate.joints[4]), abs(current_candidate.joints[4])) <= 15.0
    )
    if previous_candidate.config_flags != current_candidate.config_flags or max_joint_delta >= 150.0:
        search_levels.update({1.0, 2.0, 5.0, 10.0, 15.0, 20.0, 40.0, 60.0, 90.0})
    elif max_joint_delta >= 90.0:
        search_levels.update({1.0, 2.0, 5.0, 10.0, 15.0, 20.0, 30.0, 45.0})
    else:
        search_levels.update({1.0, 2.0, 5.0, 10.0})
    if near_wrist_singularity:
        search_levels.update({3.0, 7.5, 12.0, 25.0, 35.0})

    return tuple(sorted(search_levels))


def _iter_position_locked_pose_candidates(
    base_pose,
    *,
    rotation_search_deg: Sequence[float],
    previous_candidate: _IKCandidate,
    current_candidate: _IKCandidate,
    motion_settings: RoboDKMotionSettings,
):
    """为固定位置桥接生成更丰富的纯姿态候选。

    原有 `_iter_bridge_pose_candidates` 只覆盖单轴扰动。
    对腕奇异重构来说，这通常不够，因为真正有用的候选往往是
    “两轴叠加的小姿态偏转 + 同一位置下的不同腕部分解”。
    """

    from robodk.robomath import rotx, roty, rotz

    yielded: set[tuple[float, ...]] = set()

    def yield_if_new(pose_candidate) -> None:
        signature = tuple(
            round(float(pose_candidate[row, column]), 6)
            for row in range(4)
            for column in range(4)
        )
        if signature in yielded:
            return
        yielded.add(signature)
        nonlocal_generated.append(pose_candidate)

    nonlocal_generated: list[object] = []
    for pose_candidate in _iter_bridge_pose_candidates(
        base_pose,
        (0.0,),
        rotation_search_deg,
    ):
        yield_if_new(pose_candidate)

    compound_levels_deg = tuple(
        level_deg
        for level_deg in sorted({abs(float(value)) for value in rotation_search_deg if value != 0.0})
        if level_deg <= 30.0
    )
    if (
        len(previous_candidate.joints) >= 5
        and len(current_candidate.joints) >= 5
        and min(abs(previous_candidate.joints[4]), abs(current_candidate.joints[4]))
        <= motion_settings.wrist_refinement_a5_threshold_deg
    ):
        for angle_deg in compound_levels_deg:
            angle_rad = math.radians(angle_deg)
            for first_sign in (-1.0, 1.0):
                for second_sign in (-1.0, 1.0):
                    yield_if_new(base_pose * rotx(first_sign * angle_rad) * rotz(second_sign * angle_rad))
                    yield_if_new(base_pose * roty(first_sign * angle_rad) * rotz(second_sign * angle_rad))
                    yield_if_new(base_pose * rotx(first_sign * angle_rad) * roty(second_sign * angle_rad))

    for pose_candidate in nonlocal_generated:
        yield pose_candidate


def _build_position_locked_bridge_candidate(
    *,
    robot,
    candidate_pose,
    candidate_joints: tuple[float, ...],
    base_pose,
    desired_joints: Sequence[float],
    optimizer_settings: _PathOptimizerSettings,
    lower_limits: Sequence[float],
    upper_limits: Sequence[float],
) -> _BridgeCandidate:
    """构造固定位置桥接层的一个候选状态。"""

    config_values = robot.JointsConfig(list(candidate_joints)).list()
    config_flags = tuple(int(round(value)) for value in config_values[:_CONFIG_FLAG_COUNT])
    joint_limit_penalty = _joint_limit_penalty(
        candidate_joints,
        lower_limits,
        upper_limits,
        optimizer_settings,
    )
    singularity_penalty = _singularity_penalty(robot, candidate_joints, optimizer_settings)
    return _BridgeCandidate(
        pose=candidate_pose,
        joints=candidate_joints,
        config_flags=config_flags,
        node_cost=(
            80.0 * _pose_rotation_distance_deg(base_pose, candidate_pose)
            + 0.05 * joint_limit_penalty
            + 0.05 * singularity_penalty
            + 0.03 * _joint_transition_penalty(
                desired_joints,
                candidate_joints,
                optimizer_settings,
            )
        ),
    )


def _optimize_position_locked_bridge_layers(
    bridge_layers: Sequence[Sequence[_BridgeCandidate]],
    *,
    final_pose,
    final_preferred_joints: tuple[float, ...],
    start_pose,
    start_joints: tuple[float, ...],
    robot,
    optimizer_settings: _PathOptimizerSettings,
) -> tuple[list[_BridgeCandidate], tuple[float, ...]]:
    """对固定位置桥接层做 DP，转移可行性由 MoveL_Test 决定。"""

    if not bridge_layers:
        return [], start_joints

    start_state = _build_runtime_bridge_state(robot, start_pose, start_joints)
    previous_states: list[_BridgeCandidate | None] = [start_state]
    previous_costs = [0.0]
    backpointers: list[list[int]] = []
    state_layers: list[list[_BridgeCandidate | None]] = []

    for layer in bridge_layers:
        current_costs = [math.inf] * len(layer)
        current_backpointers = [-1] * len(layer)
        current_states: list[_BridgeCandidate | None] = [None] * len(layer)

        for current_index, current_candidate in enumerate(layer):
            best_cost = math.inf
            best_previous_index = -1
            best_state: _BridgeCandidate | None = None

            for previous_index, previous_state in enumerate(previous_states):
                if previous_state is None or not math.isfinite(previous_costs[previous_index]):
                    continue

                linear_penalty, reached_joints = _evaluate_move_l_transition(
                    robot,
                    start_joints=previous_state.joints,
                    target_pose=current_candidate.pose,
                    joint_count=len(previous_state.joints),
                    optimizer_settings=optimizer_settings,
                )
                if not math.isfinite(linear_penalty) or reached_joints is None:
                    continue

                reached_state = _build_runtime_bridge_state(
                    robot,
                    current_candidate.pose,
                    reached_joints,
                )
                total_cost = (
                    previous_costs[previous_index]
                    + current_candidate.node_cost
                    + linear_penalty
                    + _position_locked_bridge_transition_cost(
                        previous_state,
                        reached_state,
                        preferred_target_joints=current_candidate.joints,
                        optimizer_settings=optimizer_settings,
                    )
                )
                if total_cost < best_cost:
                    best_cost = total_cost
                    best_previous_index = previous_index
                    best_state = reached_state

            current_costs[current_index] = best_cost
            current_backpointers[current_index] = best_previous_index
            current_states[current_index] = best_state

        if not any(math.isfinite(cost) for cost in current_costs):
            raise RuntimeError("No feasible fixed-position posture bridge could be found.")

        previous_states = current_states
        previous_costs = current_costs
        backpointers.append(current_backpointers)
        state_layers.append(current_states)

    best_total_cost = math.inf
    best_last_index = -1
    best_final_joints: tuple[float, ...] | None = None
    for previous_index, previous_state in enumerate(previous_states):
        if previous_state is None or not math.isfinite(previous_costs[previous_index]):
            continue

        linear_penalty, reached_joints = _evaluate_move_l_transition(
            robot,
            start_joints=previous_state.joints,
            target_pose=final_pose,
            joint_count=len(previous_state.joints),
            optimizer_settings=optimizer_settings,
        )
        if not math.isfinite(linear_penalty) or reached_joints is None:
            continue

        final_state = _build_runtime_bridge_state(robot, final_pose, reached_joints)
        total_cost = (
            previous_costs[previous_index]
            + linear_penalty
            + _position_locked_bridge_transition_cost(
                previous_state,
                final_state,
                preferred_target_joints=final_preferred_joints,
                optimizer_settings=optimizer_settings,
            )
        )
        if total_cost < best_total_cost:
            best_total_cost = total_cost
            best_last_index = previous_index
            best_final_joints = reached_joints

    if best_last_index < 0 or best_final_joints is None:
        raise RuntimeError("The fixed-position bridge cannot connect linearly back to the next path point.")

    selected_states = [state_layers[-1][best_last_index]]
    for layer_index in range(len(state_layers) - 1, 0, -1):
        best_last_index = backpointers[layer_index][best_last_index]
        selected_states.append(state_layers[layer_index - 1][best_last_index])

    selected_states.reverse()
    return [state for state in selected_states if state is not None], best_final_joints


def _build_runtime_bridge_state(robot, pose, joints: tuple[float, ...]) -> _BridgeCandidate:
    """把 MoveL_Test 的实际到达关节封装成运行态桥接状态。"""

    config_values = robot.JointsConfig(list(joints)).list()
    config_flags = tuple(int(round(value)) for value in config_values[:_CONFIG_FLAG_COUNT])
    return _BridgeCandidate(
        pose=pose,
        joints=joints,
        config_flags=config_flags,
        node_cost=0.0,
    )


def _position_locked_bridge_transition_cost(
    previous_state: _BridgeCandidate,
    current_state: _BridgeCandidate,
    *,
    preferred_target_joints: Sequence[float],
    optimizer_settings: _PathOptimizerSettings,
) -> float:
    """固定位置桥接段的转移代价。"""

    return (
        0.15
        * _candidate_transition_penalty(
            previous_state,
            current_state,
            optimizer_settings,
        )
        + 12.0 * _pose_rotation_distance_deg(previous_state.pose, current_state.pose)
        + 0.08
        * _joint_transition_penalty(
            preferred_target_joints,
            current_state.joints,
            optimizer_settings,
        )
    )


def _build_bridge_waypoints(
    *,
    segment_index: int,
    target_index_width: int,
    previous_pose,
    current_pose,
    previous_candidate: _IKCandidate,
    current_candidate: _IKCandidate,
    robot,
    tool_pose,
    reference_pose,
    motion_settings: RoboDKMotionSettings,
    optimizer_settings: _PathOptimizerSettings,
    lower_limits: Sequence[float],
    upper_limits: Sequence[float],
    a1_lower_deg: float,
    a1_upper_deg: float,
) -> list[_ProgramWaypoint]:
    """在一个坏段中插入桥接点，尽量把法兰位姿改动压小。"""

    bridge_step_limits = _normalize_step_limits(
        motion_settings.bridge_step_deg,
        len(previous_candidate.joints),
    )
    segment_count = max(
        2,
        math.ceil(
            max(
                abs(current - previous) / limit
                for previous, current, limit in zip(
                    previous_candidate.joints,
                    current_candidate.joints,
                    bridge_step_limits,
                )
            )
        ),
    )

    bridge_layers: list[tuple[_BridgeCandidate, ...]] = []
    for bridge_index in range(1, segment_count):
        interpolation_ratio = bridge_index / segment_count
        desired_joints = _interpolate_joints(
            previous_candidate.joints,
            current_candidate.joints,
            interpolation_ratio,
        )
        bridge_layers.append(
            _collect_bridge_candidates_for_layer(
                previous_pose=previous_pose,
                current_pose=current_pose,
                previous_candidate=previous_candidate,
                current_candidate=current_candidate,
                desired_joints=desired_joints,
                interpolation_ratio=interpolation_ratio,
                robot=robot,
                tool_pose=tool_pose,
                reference_pose=reference_pose,
                motion_settings=motion_settings,
                optimizer_settings=optimizer_settings,
                lower_limits=lower_limits,
                upper_limits=upper_limits,
                a1_lower_deg=a1_lower_deg,
                a1_upper_deg=a1_upper_deg,
            )
        )

    selected_bridge_path = _optimize_bridge_candidate_layers(
        bridge_layers,
        previous_pose=previous_pose,
        current_pose=current_pose,
        previous_candidate=previous_candidate,
        current_candidate=current_candidate,
        optimizer_settings=optimizer_settings,
        bridge_step_limits=bridge_step_limits,
    )

    return [
        _ProgramWaypoint(
            name=f"P_{segment_index:0{target_index_width}d}_BR_{bridge_index:02d}",
            pose=bridge_candidate.pose,
            joints=bridge_candidate.joints,
            move_type="MoveJ",
            is_bridge=True,
        )
        for bridge_index, bridge_candidate in enumerate(selected_bridge_path, start=1)
    ]


def _solve_bridge_waypoint(
    *,
    anchor_pose,
    desired_joints: tuple[float, ...],
    previous_bridge_joints: tuple[float, ...],
    robot,
    tool_pose,
    reference_pose,
    motion_settings: RoboDKMotionSettings,
    optimizer_settings: _PathOptimizerSettings,
    lower_limits: Sequence[float],
    upper_limits: Sequence[float],
    a1_lower_deg: float,
    a1_upper_deg: float,
    bridge_step_limits: Sequence[float],
) -> tuple[object, tuple[float, ...]]:
    """为单个桥接采样点求一组更稳的过渡位姿/关节。"""

    fallback_pose = robot.SolveFK(list(desired_joints), tool_pose, reference_pose)
    best_pose = fallback_pose
    best_joints = desired_joints
    best_cost = _bridge_candidate_cost(
        anchor_pose=anchor_pose,
        candidate_pose=fallback_pose,
        desired_joints=desired_joints,
        candidate_joints=desired_joints,
        previous_bridge_joints=previous_bridge_joints,
        optimizer_settings=optimizer_settings,
    )

    seen_joint_keys: set[tuple[float, ...]] = set()
    for pose_candidate in _iter_bridge_pose_candidates(
        anchor_pose,
        motion_settings.bridge_translation_search_mm,
        motion_settings.bridge_rotation_search_deg,
    ):
        for joint_candidate in _iter_joint_solutions_for_pose(
            robot,
            pose_candidate,
            desired_joints,
            tool_pose,
            reference_pose,
        ):
            if not _is_within_joint_limits(joint_candidate, lower_limits, upper_limits):
                continue
            if not _passes_user_joint_constraints(
                joint_candidate,
                a1_lower_deg=a1_lower_deg,
                a1_upper_deg=a1_upper_deg,
                a2_max_deg=motion_settings.a2_max_deg,
                tolerance_deg=motion_settings.joint_constraint_tolerance_deg,
            ):
                continue
            if not _passes_step_limit(previous_bridge_joints, joint_candidate, bridge_step_limits):
                continue

            dedup_key = tuple(round(value, _IK_DEDUP_DECIMALS) for value in joint_candidate)
            if dedup_key in seen_joint_keys:
                continue
            seen_joint_keys.add(dedup_key)

            actual_pose = robot.SolveFK(list(joint_candidate), tool_pose, reference_pose)
            candidate_cost = _bridge_candidate_cost(
                anchor_pose=anchor_pose,
                candidate_pose=actual_pose,
                desired_joints=desired_joints,
                candidate_joints=joint_candidate,
                previous_bridge_joints=previous_bridge_joints,
                optimizer_settings=optimizer_settings,
            )
            if candidate_cost < best_cost:
                best_cost = candidate_cost
                best_pose = actual_pose
                best_joints = joint_candidate

    return best_pose, best_joints


def _iter_bridge_pose_candidates(
    anchor_pose,
    translation_search_mm: Sequence[float],
    rotation_search_deg: Sequence[float],
):
    """围绕锚点位姿生成一组小扰动候选。"""

    from robodk.robomath import rotx, roty, rotz, transl

    yield anchor_pose

    translation_levels = sorted({abs(float(value)) for value in translation_search_mm if value != 0.0})
    rotation_levels_deg = sorted({abs(float(value)) for value in rotation_search_deg if value != 0.0})

    for distance_mm in translation_levels:
        for sign in (-1.0, 1.0):
            yield anchor_pose * transl(sign * distance_mm, 0.0, 0.0)
            yield anchor_pose * transl(0.0, sign * distance_mm, 0.0)
            yield anchor_pose * transl(0.0, 0.0, sign * distance_mm)

    for angle_deg in rotation_levels_deg:
        angle_rad = math.radians(angle_deg)
        for sign in (-1.0, 1.0):
            yield anchor_pose * rotx(sign * angle_rad)
            yield anchor_pose * roty(sign * angle_rad)
            yield anchor_pose * rotz(sign * angle_rad)


def _iter_joint_solutions_for_pose(
    robot,
    pose_candidate,
    desired_joints: Sequence[float],
    tool_pose,
    reference_pose,
):
    """为候选位姿枚举一组关节解。"""

    yielded: set[tuple[float, ...]] = set()

    seed_solution = _extract_joint_tuple(
        robot.SolveIK(pose_candidate, list(desired_joints), tool_pose, reference_pose),
        len(desired_joints),
    )
    if seed_solution:
        dedup_key = tuple(round(value, _IK_DEDUP_DECIMALS) for value in seed_solution)
        yielded.add(dedup_key)
        yield seed_solution

    for raw_solution in robot.SolveIK_All(pose_candidate, tool_pose, reference_pose):
        joint_solution = _extract_joint_tuple(raw_solution, len(desired_joints))
        if not joint_solution:
            continue
        dedup_key = tuple(round(value, _IK_DEDUP_DECIMALS) for value in joint_solution)
        if dedup_key in yielded:
            continue
        yielded.add(dedup_key)
        yield joint_solution


def _bridge_candidate_cost(
    *,
    anchor_pose,
    candidate_pose,
    desired_joints: Sequence[float],
    candidate_joints: Sequence[float],
    previous_bridge_joints: Sequence[float],
    optimizer_settings: _PathOptimizerSettings,
) -> float:
    """桥接候选评分：先保位姿，再保关节平滑。"""

    translation_error_mm = _pose_translation_distance_mm(anchor_pose, candidate_pose)
    rotation_error_deg = _pose_rotation_distance_deg(anchor_pose, candidate_pose)
    desired_joint_cost = _mean_abs_joint_delta(desired_joints, candidate_joints)
    step_joint_cost = _mean_abs_joint_delta(previous_bridge_joints, candidate_joints)

    return (
        30.0 * translation_error_mm
        + 12.0 * rotation_error_deg
        + 0.25 * desired_joint_cost
        + 0.10 * step_joint_cost
        + 0.005 * _joint_transition_penalty(
            desired_joints,
            candidate_joints,
            optimizer_settings,
        )
    )


def _collect_bridge_candidates_for_layer(
    *,
    previous_pose,
    current_pose,
    previous_candidate: _IKCandidate,
    current_candidate: _IKCandidate,
    desired_joints: tuple[float, ...],
    interpolation_ratio: float,
    robot,
    tool_pose,
    reference_pose,
    motion_settings: RoboDKMotionSettings,
    optimizer_settings: _PathOptimizerSettings,
    lower_limits: Sequence[float],
    upper_limits: Sequence[float],
    a1_lower_deg: float,
    a1_upper_deg: float,
) -> tuple[_BridgeCandidate, ...]:
    """为单个桥接层先收集候选，后续再统一交给桥接 DP。"""

    desired_pose = robot.SolveFK(list(desired_joints), tool_pose, reference_pose)
    fallback_candidate = _build_bridge_candidate_state(
        robot=robot,
        candidate_pose=desired_pose,
        candidate_joints=desired_joints,
        desired_joints=desired_joints,
        previous_pose=previous_pose,
        current_pose=current_pose,
        interpolation_ratio=interpolation_ratio,
        optimizer_settings=optimizer_settings,
        lower_limits=lower_limits,
        upper_limits=upper_limits,
    )

    other_candidates: list[_BridgeCandidate] = []
    seen_joint_keys = {
        tuple(round(value, _IK_DEDUP_DECIMALS) for value in fallback_candidate.joints)
    }
    seed_joints_options = (
        desired_joints,
        previous_candidate.joints,
        current_candidate.joints,
    )

    for anchor_pose in _iter_bridge_anchor_poses(
        previous_pose,
        current_pose,
        desired_pose,
    ):
        for pose_candidate in _iter_bridge_pose_candidates(
            anchor_pose,
            motion_settings.bridge_translation_search_mm,
            motion_settings.bridge_rotation_search_deg,
        ):
            for joint_candidate in _iter_joint_solutions_for_pose_with_seeds(
                robot,
                pose_candidate,
                seed_joints_options,
                tool_pose,
                reference_pose,
            ):
                if not _is_within_joint_limits(joint_candidate, lower_limits, upper_limits):
                    continue
                if not _passes_user_joint_constraints(
                    joint_candidate,
                    a1_lower_deg=a1_lower_deg,
                    a1_upper_deg=a1_upper_deg,
                    a2_max_deg=motion_settings.a2_max_deg,
                    tolerance_deg=motion_settings.joint_constraint_tolerance_deg,
                ):
                    continue

                dedup_key = tuple(round(value, _IK_DEDUP_DECIMALS) for value in joint_candidate)
                if dedup_key in seen_joint_keys:
                    continue
                seen_joint_keys.add(dedup_key)

                actual_pose = robot.SolveFK(list(joint_candidate), tool_pose, reference_pose)
                other_candidates.append(
                    _build_bridge_candidate_state(
                        robot=robot,
                        candidate_pose=actual_pose,
                        candidate_joints=joint_candidate,
                        desired_joints=desired_joints,
                        previous_pose=previous_pose,
                        current_pose=current_pose,
                        interpolation_ratio=interpolation_ratio,
                        optimizer_settings=optimizer_settings,
                        lower_limits=lower_limits,
                        upper_limits=upper_limits,
                    )
                )

    other_candidates.sort(
        key=lambda candidate: (
            candidate.node_cost,
            candidate.config_flags,
            candidate.joints,
        )
    )
    limited_candidates = [fallback_candidate]
    limited_candidates.extend(
        other_candidates[: max(0, _BRIDGE_LAYER_CANDIDATE_LIMIT - 1)]
    )
    return tuple(limited_candidates)


def _build_bridge_candidate_state(
    *,
    robot,
    candidate_pose,
    candidate_joints: tuple[float, ...],
    desired_joints: Sequence[float],
    previous_pose,
    current_pose,
    interpolation_ratio: float,
    optimizer_settings: _PathOptimizerSettings,
    lower_limits: Sequence[float],
    upper_limits: Sequence[float],
) -> _BridgeCandidate:
    """把桥接候选封装成统一结构，便于桥接 DP 使用。"""

    config_values = robot.JointsConfig(list(candidate_joints)).list()
    config_flags = tuple(int(round(value)) for value in config_values[:_CONFIG_FLAG_COUNT])
    joint_limit_penalty = _joint_limit_penalty(
        candidate_joints,
        lower_limits,
        upper_limits,
        optimizer_settings,
    )
    singularity_penalty = _singularity_penalty(robot, candidate_joints, optimizer_settings)
    return _BridgeCandidate(
        pose=candidate_pose,
        joints=candidate_joints,
        config_flags=config_flags,
        node_cost=_bridge_layer_candidate_cost(
            previous_pose=previous_pose,
            current_pose=current_pose,
            candidate_pose=candidate_pose,
            desired_joints=desired_joints,
            candidate_joints=candidate_joints,
            interpolation_ratio=interpolation_ratio,
            optimizer_settings=optimizer_settings,
            joint_limit_penalty=joint_limit_penalty,
            singularity_penalty=singularity_penalty,
        ),
    )


def _iter_bridge_anchor_poses(previous_pose, current_pose, desired_pose):
    """同时围绕前点、后点和关节插值 FK 位姿做桥接搜索。"""

    yielded: set[tuple[float, ...]] = set()
    for pose in (previous_pose, current_pose, desired_pose):
        signature = tuple(round(float(pose[row, column]), 6) for row in range(4) for column in range(4))
        if signature in yielded:
            continue
        yielded.add(signature)
        yield pose


def _iter_joint_solutions_for_pose_with_seeds(
    robot,
    pose_candidate,
    seed_joints_options: Sequence[Sequence[float]],
    tool_pose,
    reference_pose,
):
    """针对桥接层的同一个位姿候选，用多组 seed 尽量诱导多个 IK 分支。"""

    yielded: set[tuple[float, ...]] = set()
    joint_count = len(seed_joints_options[0]) if seed_joints_options else 0

    for seed_joints in seed_joints_options:
        joint_solution = _extract_joint_tuple(
            robot.SolveIK(pose_candidate, list(seed_joints), tool_pose, reference_pose),
            joint_count,
        )
        if not joint_solution:
            continue
        dedup_key = tuple(round(value, _IK_DEDUP_DECIMALS) for value in joint_solution)
        if dedup_key in yielded:
            continue
        yielded.add(dedup_key)
        yield joint_solution

    for raw_solution in robot.SolveIK_All(pose_candidate, tool_pose, reference_pose):
        joint_solution = _extract_joint_tuple(raw_solution, joint_count)
        if not joint_solution:
            continue
        dedup_key = tuple(round(value, _IK_DEDUP_DECIMALS) for value in joint_solution)
        if dedup_key in yielded:
            continue
        yielded.add(dedup_key)
        yield joint_solution


def _bridge_layer_candidate_cost(
    *,
    previous_pose,
    current_pose,
    candidate_pose,
    desired_joints: Sequence[float],
    candidate_joints: Sequence[float],
    interpolation_ratio: float,
    optimizer_settings: _PathOptimizerSettings,
    joint_limit_penalty: float,
    singularity_penalty: float,
) -> float:
    """桥接层节点代价：优先尽量保持法兰位姿不变，再兼顾关节平滑。"""

    previous_translation_error_mm = _pose_translation_distance_mm(previous_pose, candidate_pose)
    previous_rotation_error_deg = _pose_rotation_distance_deg(previous_pose, candidate_pose)
    current_translation_error_mm = _pose_translation_distance_mm(current_pose, candidate_pose)
    current_rotation_error_deg = _pose_rotation_distance_deg(current_pose, candidate_pose)
    desired_joint_cost = _mean_abs_joint_delta(desired_joints, candidate_joints)

    return (
        40.0 * min(previous_translation_error_mm, current_translation_error_mm)
        + 15.0 * min(previous_rotation_error_deg, current_rotation_error_deg)
        + 12.0
        * (
            (1.0 - interpolation_ratio) * previous_translation_error_mm
            + interpolation_ratio * current_translation_error_mm
        )
        + 4.0
        * (
            (1.0 - interpolation_ratio) * previous_rotation_error_deg
            + interpolation_ratio * current_rotation_error_deg
        )
        + 0.20 * desired_joint_cost
        + 0.006 * _joint_transition_penalty(
            desired_joints,
            candidate_joints,
            optimizer_settings,
        )
        + 0.04 * joint_limit_penalty
        + 0.04 * singularity_penalty
    )


def _optimize_bridge_candidate_layers(
    bridge_layers: Sequence[Sequence[_BridgeCandidate]],
    *,
    previous_pose,
    current_pose,
    previous_candidate: _IKCandidate,
    current_candidate: _IKCandidate,
    optimizer_settings: _PathOptimizerSettings,
    bridge_step_limits: Sequence[float],
) -> list[_BridgeCandidate]:
    """对整段桥接层做一次局部 DP，而不是逐点贪心。"""

    if not bridge_layers:
        return []

    start_state = _BridgeCandidate(
        pose=previous_pose,
        joints=previous_candidate.joints,
        config_flags=previous_candidate.config_flags,
        node_cost=0.0,
    )
    end_state = _BridgeCandidate(
        pose=current_pose,
        joints=current_candidate.joints,
        config_flags=current_candidate.config_flags,
        node_cost=0.0,
    )

    previous_layer: Sequence[_BridgeCandidate] = (start_state,)
    previous_costs = [0.0]
    backpointers: list[list[int]] = []

    for layer in bridge_layers:
        current_costs = [math.inf] * len(layer)
        current_backpointers = [-1] * len(layer)

        for current_index, current_state in enumerate(layer):
            best_cost = math.inf
            best_previous_index = -1

            for previous_index, previous_state in enumerate(previous_layer):
                if not _passes_step_limit(
                    previous_state.joints,
                    current_state.joints,
                    bridge_step_limits,
                ):
                    continue

                total_cost = (
                    previous_costs[previous_index]
                    + _bridge_layer_transition_cost(
                        previous_state,
                        current_state,
                        optimizer_settings,
                    )
                    + current_state.node_cost
                )
                if total_cost < best_cost:
                    best_cost = total_cost
                    best_previous_index = previous_index

            current_costs[current_index] = best_cost
            current_backpointers[current_index] = best_previous_index

        if not any(math.isfinite(cost) for cost in current_costs):
            raise RuntimeError("No feasible bridge candidate sequence could be found.")

        previous_layer = layer
        previous_costs = current_costs
        backpointers.append(current_backpointers)

    best_total_cost = math.inf
    best_last_index = -1
    for previous_index, previous_state in enumerate(previous_layer):
        if not _passes_step_limit(
            previous_state.joints,
            end_state.joints,
            bridge_step_limits,
        ):
            continue

        total_cost = previous_costs[previous_index] + _bridge_layer_transition_cost(
            previous_state,
            end_state,
            optimizer_settings,
        )
        if total_cost < best_total_cost:
            best_total_cost = total_cost
            best_last_index = previous_index

    if best_last_index < 0:
        raise RuntimeError("Bridge candidates could not connect back to the final path point.")

    selected_path = [bridge_layers[-1][best_last_index]]
    for layer_index in range(len(bridge_layers) - 1, 0, -1):
        best_last_index = backpointers[layer_index][best_last_index]
        selected_path.append(bridge_layers[layer_index - 1][best_last_index])

    selected_path.reverse()
    return selected_path


def _bridge_layer_transition_cost(
    previous_state: _BridgeCandidate,
    current_state: _BridgeCandidate,
    optimizer_settings: _PathOptimizerSettings,
) -> float:
    """桥接层转移代价：同时压关节变化和法兰位姿变化。"""

    translation_delta_mm = _pose_translation_distance_mm(previous_state.pose, current_state.pose)
    rotation_delta_deg = _pose_rotation_distance_deg(previous_state.pose, current_state.pose)
    return (
        0.18
        * _candidate_transition_penalty(
            previous_state,
            current_state,
            optimizer_settings,
        )
        + 28.0 * translation_delta_mm
        + 10.0 * rotation_delta_deg
    )


def _optimize_joint_path(
    ik_layers: Sequence[_IKLayer],
    *,
    robot,
    move_type: str,
    start_joints: tuple[float, ...],
    optimizer_settings: _PathOptimizerSettings,
) -> tuple[list[_IKCandidate], float]:
    """使用动态规划在整条路径上选出一条全局代价最小的关节序列。"""

    if not ik_layers:
        return [], 0.0

    corridor_scores = _compute_candidate_corridor_scores(ik_layers, optimizer_settings)
    guided_config_path = _build_guided_config_path(
        ik_layers,
        start_joints=start_joints,
        optimizer_settings=optimizer_settings,
    )
    if (
        guided_config_path is not None
        and not _guided_config_path_is_feasible(
            ik_layers,
            guided_config_path=guided_config_path,
            optimizer_settings=optimizer_settings,
        )
    ):
        guided_config_path = None

    # 第 0 层的代价 = 从当前机器人关节走到该候选的转移代价 + 该候选自己的节点代价。
    previous_costs = []
    for candidate_index, candidate in enumerate(ik_layers[0].candidates):
        if guided_config_path is not None and candidate.config_flags != guided_config_path[0]:
            previous_costs.append(math.inf)
            continue
        previous_costs.append(
            _candidate_node_cost(
                candidate,
                corridor_score=corridor_scores[0][candidate_index],
                optimizer_settings=optimizer_settings,
            )
            + optimizer_settings.start_transition_weight
            * _joint_transition_penalty(start_joints, candidate.joints, optimizer_settings)
        )
    backpointers: list[list[int]] = []

    previous_layer = ik_layers[0]
    for layer_index in range(1, len(ik_layers)):
        current_layer = ik_layers[layer_index]

        # MoveL 的可行性跟“前一个候选关节”强相关，因此这里先缓存：
        # 从上一层每个候选出发，走线性移动到当前笛卡尔位姿是否可达。
        move_l_cache: list[tuple[float, tuple[float, ...] | None]] | None = None
        if move_type == "MoveL":
            move_l_cache = [
                _evaluate_move_l_transition(
                    robot,
                    start_joints=candidate.joints,
                    target_pose=current_layer.pose,
                    joint_count=len(candidate.joints),
                    optimizer_settings=optimizer_settings,
                )
                for candidate in previous_layer.candidates
            ]

        current_costs = [math.inf] * len(current_layer.candidates)
        current_backpointers = [-1] * len(current_layer.candidates)

        for current_index, current_candidate in enumerate(current_layer.candidates):
            if (
                guided_config_path is not None
                and current_candidate.config_flags != guided_config_path[layer_index]
            ):
                continue
            node_cost = _candidate_node_cost(
                current_candidate,
                corridor_score=corridor_scores[layer_index][current_index],
                optimizer_settings=optimizer_settings,
            )
            best_cost = math.inf
            best_previous_index = -1

            for previous_index, previous_candidate in enumerate(previous_layer.candidates):
                if not math.isfinite(previous_costs[previous_index]):
                    continue
                if (
                    guided_config_path is not None
                    and previous_candidate.config_flags != guided_config_path[layer_index - 1]
                ):
                    continue
                if not _passes_joint_continuity_constraint(
                    previous_candidate.joints,
                    current_candidate.joints,
                    optimizer_settings,
                ):
                    continue

                transition_cost = _candidate_transition_penalty(
                    previous_candidate,
                    current_candidate,
                    optimizer_settings,
                )

                if move_l_cache is not None:
                    linear_penalty, reached_joints = move_l_cache[previous_index]
                    if not math.isfinite(linear_penalty):
                        continue
                    transition_cost += linear_penalty

                    # 即便 RoboDK 线性插补能到达目标点，线性到达时的实际末端关节也可能
                    # 跟当前候选关节有偏差，因此这里再给一个分支不一致惩罚。
                    if reached_joints is not None:
                        transition_cost += optimizer_settings.move_l_branch_mismatch_weight * (
                            _joint_transition_penalty(
                                reached_joints,
                                current_candidate.joints,
                                optimizer_settings,
                            )
                        )

                total_cost = previous_costs[previous_index] + transition_cost + node_cost
                if total_cost < best_cost:
                    best_cost = total_cost
                    best_previous_index = previous_index

            current_costs[current_index] = best_cost
            current_backpointers[current_index] = best_previous_index

        if not any(math.isfinite(cost) for cost in current_costs):
            message = (
                f"No globally feasible {move_type} sequence could be found for target index "
                f"{layer_index}."
            )
            if guided_config_path is not None:
                message += " The guided config-family path is too restrictive."
            if optimizer_settings.enable_joint_continuity_constraint:
                message += (
                    " The joint continuity constraint may be too strict for the current path."
                )
            raise RuntimeError(message)

        backpointers.append(current_backpointers)
        previous_layer = current_layer
        previous_costs = current_costs

    end_index = min(range(len(previous_costs)), key=previous_costs.__getitem__)
    total_cost = previous_costs[end_index]
    selected_path = [ik_layers[-1].candidates[end_index]]

    # 通过回溯指针得到整条最优路径。
    for layer_index in range(len(ik_layers) - 2, -1, -1):
        end_index = backpointers[layer_index][end_index]
        selected_path.append(ik_layers[layer_index].candidates[end_index])

    selected_path.reverse()
    return selected_path, total_cost


def _build_guided_config_path(
    ik_layers: Sequence[_IKLayer],
    *,
    start_joints: Sequence[float],
    optimizer_settings: _PathOptimizerSettings,
) -> tuple[tuple[int, ...], ...] | None:
    """先在 `config_flags` 层面求一条“位形族路径”。

    候选层 DP 的问题是：同一个 config family 内部通常还有很多具体关节候选，
    如果直接在细粒度层面优化，局部节点代价可能会把整条路径带离本来可连续的族。

    所以这里先做一层更粗的规划：
    - 只决定每一层优先属于哪个 family；
    - 以“少切族”为最高优先级；
    - 如果必须切，就把切换放在跨族连接代价最小的层。
    """

    if not ik_layers:
        return ()

    candidate_groups_by_layer = [
        _group_candidates_by_config(layer.candidates)
        for layer in ik_layers
    ]

    previous_costs: dict[tuple[int, ...], float] = {}
    for config_flags, candidates in candidate_groups_by_layer[0].items():
        previous_costs[config_flags] = min(
            optimizer_settings.start_transition_weight
            * _joint_transition_penalty(start_joints, candidate.joints, optimizer_settings)
            for candidate in candidates
        )

    backpointers: list[dict[tuple[int, ...], tuple[int, ...]]] = []
    for layer_index in range(1, len(candidate_groups_by_layer)):
        previous_groups = candidate_groups_by_layer[layer_index - 1]
        current_groups = candidate_groups_by_layer[layer_index]
        current_costs: dict[tuple[int, ...], float] = {}
        current_backpointers: dict[tuple[int, ...], tuple[int, ...]] = {}

        for current_flags, current_candidates in current_groups.items():
            best_cost = math.inf
            best_previous_flags: tuple[int, ...] | None = None
            for previous_flags, previous_candidates in previous_groups.items():
                if previous_flags not in previous_costs:
                    continue

                transition_cost = _best_config_group_transition_cost(
                    previous_candidates,
                    current_candidates,
                    optimizer_settings,
                )
                if not math.isfinite(transition_cost):
                    continue

                total_cost = previous_costs[previous_flags] + transition_cost
                if previous_flags != current_flags:
                    total_cost += optimizer_settings.family_switch_penalty
                else:
                    total_cost -= optimizer_settings.same_config_stay_bonus

                if total_cost < best_cost:
                    best_cost = total_cost
                    best_previous_flags = previous_flags

            if best_previous_flags is not None:
                current_costs[current_flags] = best_cost
                current_backpointers[current_flags] = best_previous_flags

        if not current_costs:
            return None

        previous_costs = current_costs
        backpointers.append(current_backpointers)

    end_flags = min(previous_costs, key=previous_costs.__getitem__)
    guided_flags = [end_flags]
    for layer_index in range(len(backpointers) - 1, -1, -1):
        end_flags = backpointers[layer_index][end_flags]
        guided_flags.append(end_flags)
    guided_flags.reverse()
    return tuple(guided_flags)


def _group_candidates_by_config(
    candidates: Sequence[_IKCandidate],
) -> dict[tuple[int, ...], tuple[_IKCandidate, ...]]:
    """按 `config_flags` 把同一层候选分组。"""

    grouped: dict[tuple[int, ...], list[_IKCandidate]] = {}
    for candidate in candidates:
        grouped.setdefault(candidate.config_flags, []).append(candidate)
    return {config_flags: tuple(group) for config_flags, group in grouped.items()}


def _best_config_group_transition_cost(
    previous_candidates: Sequence[_IKCandidate],
    current_candidates: Sequence[_IKCandidate],
    optimizer_settings: _PathOptimizerSettings,
) -> float:
    """估计两个 config-family 组之间的最优连接代价。"""

    best_cost = math.inf
    for previous_candidate in previous_candidates:
        for current_candidate in current_candidates:
            if not _passes_joint_continuity_constraint(
                previous_candidate.joints,
                current_candidate.joints,
                optimizer_settings,
            ):
                continue
            best_cost = min(
                best_cost,
                _joint_transition_penalty(
                    previous_candidate.joints,
                    current_candidate.joints,
                    optimizer_settings,
                ),
            )
    return best_cost


def _guided_config_path_is_feasible(
    ik_layers: Sequence[_IKLayer],
    *,
    guided_config_path: Sequence[tuple[int, ...]],
    optimizer_settings: _PathOptimizerSettings,
) -> bool:
    """检查 family guidance 是否真能落到一条具体候选链上。"""

    if len(guided_config_path) != len(ik_layers):
        return False

    reachable_indices = [
        index
        for index, candidate in enumerate(ik_layers[0].candidates)
        if candidate.config_flags == guided_config_path[0]
    ]
    if not reachable_indices:
        return False

    for layer_index in range(1, len(ik_layers)):
        current_reachable: list[int] = []
        for current_index, current_candidate in enumerate(ik_layers[layer_index].candidates):
            if current_candidate.config_flags != guided_config_path[layer_index]:
                continue
            for previous_index in reachable_indices:
                previous_candidate = ik_layers[layer_index - 1].candidates[previous_index]
                if _passes_joint_continuity_constraint(
                    previous_candidate.joints,
                    current_candidate.joints,
                    optimizer_settings,
                ):
                    current_reachable.append(current_index)
                    break
        if not current_reachable:
            return False
        reachable_indices = current_reachable

    return True


def _evaluate_move_l_transition(
    robot,
    *,
    start_joints: tuple[float, ...],
    target_pose,
    joint_count: int,
    optimizer_settings: _PathOptimizerSettings,
) -> tuple[float, tuple[float, ...] | None]:
    """评估从某个起始关节出发执行 MoveL 是否可行。"""

    status = robot.MoveL_Test(list(start_joints), target_pose)
    if status != 0:
        return optimizer_settings.move_l_unreachable_penalty, None

    reached_joints = _trim_joint_vector(robot.Joints().list(), joint_count)
    return 0.0, reached_joints


def _candidate_transition_penalty(
    previous_candidate: _IKCandidate,
    current_candidate: _IKCandidate,
    optimizer_settings: _PathOptimizerSettings,
) -> float:
    """计算两个候选之间的转移代价。"""

    cost = _joint_transition_penalty(
        previous_candidate.joints,
        current_candidate.joints,
        optimizer_settings,
    )

    previous_flags = previous_candidate.config_flags
    current_flags = current_candidate.config_flags
    if len(previous_flags) >= 1 and len(current_flags) >= 1 and previous_flags[0] != current_flags[0]:
        cost += optimizer_settings.rear_switch_penalty
    if len(previous_flags) >= 2 and len(current_flags) >= 2 and previous_flags[1] != current_flags[1]:
        cost += optimizer_settings.lower_switch_penalty
    if len(previous_flags) >= 3 and len(current_flags) >= 3 and previous_flags[2] != current_flags[2]:
        cost += optimizer_settings.flip_switch_penalty

    # J5 正负号改变通常对应 wrist flip，额外惩罚。
    if len(previous_candidate.joints) >= 5:
        if previous_candidate.joints[4] * current_candidate.joints[4] < 0.0:
            cost += optimizer_settings.wrist_flip_sign_penalty

    # 大幅转 J6 往往是“能到，但没必要”的多圈旋转，单独抑制。
    if len(previous_candidate.joints) >= 6:
        joint6_delta = abs(current_candidate.joints[5] - previous_candidate.joints[5])
        if joint6_delta > optimizer_settings.joint6_spin_threshold_deg:
            cost += (
                joint6_delta - optimizer_settings.joint6_spin_threshold_deg
            ) * optimizer_settings.joint6_spin_penalty_per_deg

        # 参考 FINA11.src：在 A5 接近 0 的腕奇异区，优先锁住 A6 的相位连续性。
        # 这样即便必须穿过奇异附近，也尽量让 A4 去连续变化，而不是让 A6 突然翻转。
        min_abs_a5_deg = math.inf
        if len(previous_candidate.joints) >= 5:
            min_abs_a5_deg = min(
                abs(previous_candidate.joints[4]),
                abs(current_candidate.joints[4]),
            )
        if min_abs_a5_deg < optimizer_settings.wrist_phase_lock_threshold_deg:
            normalized = (
                optimizer_settings.wrist_phase_lock_threshold_deg - min_abs_a5_deg
            ) / optimizer_settings.wrist_phase_lock_threshold_deg
            cost += (
                optimizer_settings.wrist_phase_lock_penalty_per_deg
                * normalized
                * joint6_delta
            )

    # 如果相邻两点本来就能用同一配置标志并且小步平滑连接，则给一个奖励，
    # 让 DP 更愿意待在这条“自然连续”的 exact-pose 走廊里。
    if previous_flags == current_flags:
        cost -= optimizer_settings.same_config_stay_bonus
        if _passes_preferred_continuity(
            previous_candidate.joints,
            current_candidate.joints,
            optimizer_settings,
        ):
            cost -= optimizer_settings.preferred_transition_bonus

    return cost


def _passes_joint_continuity_constraint(
    previous_joints: Sequence[float],
    current_joints: Sequence[float],
    optimizer_settings: _PathOptimizerSettings,
) -> bool:
    """检查相邻两点之间是否满足连续性硬约束。"""

    if not optimizer_settings.enable_joint_continuity_constraint:
        return True

    for delta_deg, limit_deg in zip(
        (abs(current - previous) for previous, current in zip(previous_joints, current_joints)),
        optimizer_settings.max_joint_step_deg,
    ):
        if delta_deg > limit_deg:
            return False

    return True


def _passes_preferred_continuity(
    previous_joints: Sequence[float],
    current_joints: Sequence[float],
    optimizer_settings: _PathOptimizerSettings,
) -> bool:
    """检查相邻两点是否满足“优选连续性”。

    这不是硬约束，失败也不代表不可用。
    它只用于识别那些能长距离保持同一位形族的 exact-pose 候选走廊。
    """

    for delta_deg, limit_deg in zip(
        (abs(current - previous) for previous, current in zip(previous_joints, current_joints)),
        optimizer_settings.preferred_joint_step_deg,
    ):
        if delta_deg > limit_deg:
            return False
    return True


def _candidate_node_cost(
    candidate: _IKCandidate,
    *,
    corridor_score: float,
    optimizer_settings: _PathOptimizerSettings,
) -> float:
    """计算单个候选节点的总代价。

    节点代价由两部分组成：
    1. 近限位 / 近奇异惩罚，但做缩放，避免局部代价压倒整条路径连续性；
    2. 连续位形走廊奖励，鼓励 DP 留在能跨越更多层的平滑分支里。
    """

    raw_penalty = candidate.joint_limit_penalty + candidate.singularity_penalty
    corridor_bonus = min(optimizer_settings.corridor_bonus_cap, corridor_score) * (
        optimizer_settings.corridor_bonus_per_step
    )
    return optimizer_settings.node_penalty_scale * raw_penalty - corridor_bonus


def _compute_candidate_corridor_scores(
    ik_layers: Sequence[_IKLayer],
    optimizer_settings: _PathOptimizerSettings,
) -> list[list[float]]:
    """估计每个候选点处在“连续位形走廊”中的强度。

    做法：
    1. 只考虑“同一 config_flags 且满足优选连续性”的边；
    2. 分别向前、向后做最长链长度 DP；
    3. 把前后长度合并成节点走廊分数。

    这个分数越高，说明该候选越像一条长距离稳定分支上的中间点。
    """

    if not ik_layers:
        return []

    forward_lengths: list[list[int]] = [
        [1] * len(layer.candidates) for layer in ik_layers
    ]
    backward_lengths: list[list[int]] = [
        [1] * len(layer.candidates) for layer in ik_layers
    ]

    for layer_index in range(len(ik_layers) - 2, -1, -1):
        current_layer = ik_layers[layer_index]
        next_layer = ik_layers[layer_index + 1]
        for current_index, current_candidate in enumerate(current_layer.candidates):
            best_reach = 0
            for next_index, next_candidate in enumerate(next_layer.candidates):
                if current_candidate.config_flags != next_candidate.config_flags:
                    continue
                if not _passes_preferred_continuity(
                    current_candidate.joints,
                    next_candidate.joints,
                    optimizer_settings,
                ):
                    continue
                best_reach = max(best_reach, forward_lengths[layer_index + 1][next_index])
            forward_lengths[layer_index][current_index] = 1 + best_reach

    for layer_index in range(1, len(ik_layers)):
        previous_layer = ik_layers[layer_index - 1]
        current_layer = ik_layers[layer_index]
        for current_index, current_candidate in enumerate(current_layer.candidates):
            best_reach = 0
            for previous_index, previous_candidate in enumerate(previous_layer.candidates):
                if previous_candidate.config_flags != current_candidate.config_flags:
                    continue
                if not _passes_preferred_continuity(
                    previous_candidate.joints,
                    current_candidate.joints,
                    optimizer_settings,
                ):
                    continue
                best_reach = max(best_reach, backward_lengths[layer_index - 1][previous_index])
            backward_lengths[layer_index][current_index] = 1 + best_reach

    corridor_scores: list[list[float]] = []
    for layer_index, layer in enumerate(ik_layers):
        layer_scores = []
        for candidate_index, _candidate in enumerate(layer.candidates):
            forward_length = forward_lengths[layer_index][candidate_index]
            backward_length = backward_lengths[layer_index][candidate_index]
            layer_scores.append(float(forward_length + backward_length - 2))
        corridor_scores.append(layer_scores)

    return corridor_scores


def _joint_transition_penalty(
    previous_joints: Sequence[float],
    current_joints: Sequence[float],
    optimizer_settings: _PathOptimizerSettings,
) -> float:
    """计算关节变化代价。

    这里同时考虑：
    1. 绝对变化量之和；
    2. 大变化的平方惩罚；
    3. 单步最大跳变额外惩罚。
    """

    deltas = [current - previous for previous, current in zip(previous_joints, current_joints)]
    abs_cost = sum(
        weight * abs(delta)
        for weight, delta in zip(optimizer_settings.joint_delta_weights, deltas)
    )
    squared_cost = sum(
        weight * delta * delta
        for weight, delta in zip(optimizer_settings.joint_delta_weights, deltas)
    )
    cost = (
        optimizer_settings.abs_delta_weight * abs_cost
        + optimizer_settings.squared_delta_weight * squared_cost
    )

    if deltas:
        max_delta = max(abs(delta) for delta in deltas)
        if max_delta > optimizer_settings.large_jump_threshold_deg:
            excess = max_delta - optimizer_settings.large_jump_threshold_deg
            cost += optimizer_settings.large_jump_penalty_weight * excess * excess

    return cost


def _joint_limit_penalty(
    joints: Sequence[float],
    lower_limits: Sequence[float],
    upper_limits: Sequence[float],
    optimizer_settings: _PathOptimizerSettings,
) -> float:
    """计算关节接近机器人自身限位时的惩罚。"""

    penalty = 0.0
    for joint, lower, upper in zip(joints, lower_limits, upper_limits):
        span = upper - lower
        if span <= 0.0:
            continue

        margin = min(joint - lower, upper - joint)
        margin_ratio = margin / span
        if margin_ratio < optimizer_settings.joint_limit_margin_ratio:
            normalized = (
                optimizer_settings.joint_limit_margin_ratio - margin_ratio
            ) / optimizer_settings.joint_limit_margin_ratio
            penalty += optimizer_settings.joint_limit_penalty_weight * normalized * normalized
    return penalty


def _singularity_penalty(
    robot,
    joints: Sequence[float],
    optimizer_settings: _PathOptimizerSettings,
) -> float:
    """估算奇异附近惩罚。

    这里不依赖完整雅可比矩阵，而是用两个简单但稳定的近似指标：
    1. J5 接近 0 度时，腕部奇异风险升高；
    2. 肩-肘-腕三点几乎共线时，手臂奇异风险升高。
    """

    joints_key = tuple(float(value) for value in joints)
    cache_key = (id(robot), joints_key, optimizer_settings)
    cached_penalty = _ROBOT_SINGULARITY_PENALTY_CACHE.get(cache_key)
    if cached_penalty is not None:
        return cached_penalty

    penalty = 0.0

    if len(joints) >= 5:
        wrist_measure = abs(math.sin(math.radians(joints[4])))
        threshold = math.sin(math.radians(optimizer_settings.wrist_singularity_threshold_deg))
        if wrist_measure < threshold:
            normalized = (threshold - wrist_measure) / threshold
            penalty += (
                optimizer_settings.wrist_singularity_penalty_weight * normalized * normalized
            )

    joint_poses = robot.JointPoses(list(joints))
    if len(joint_poses) >= 4:
        shoulder = _translation_from_pose(joint_poses[1])
        elbow = _translation_from_pose(joint_poses[2])
        wrist = _translation_from_pose(joint_poses[3])
        arm_measure = _normalized_cross_measure(
            _subtract_vectors(elbow, shoulder),
            _subtract_vectors(wrist, elbow),
        )
        if arm_measure < optimizer_settings.arm_singularity_threshold:
            normalized = (
                optimizer_settings.arm_singularity_threshold - arm_measure
            ) / optimizer_settings.arm_singularity_threshold
            penalty += optimizer_settings.arm_singularity_penalty_weight * normalized * normalized

    _ROBOT_SINGULARITY_PENALTY_CACHE[cache_key] = penalty
    return penalty


def _interpolate_joints(
    start_joints: Sequence[float],
    end_joints: Sequence[float],
    ratio: float,
) -> tuple[float, ...]:
    """在两组关节之间做线性插值。"""

    return tuple(
        start + (end - start) * ratio
        for start, end in zip(start_joints, end_joints)
    )


def _passes_step_limit(
    previous_joints: Sequence[float],
    current_joints: Sequence[float],
    step_limits: Sequence[float],
) -> bool:
    """检查一步关节变化是否落在给定阈值内。"""

    return all(
        abs(current - previous) <= limit
        for previous, current, limit in zip(previous_joints, current_joints, step_limits)
    )


def _pose_translation_distance_mm(first_pose, second_pose) -> float:
    """计算两个位姿之间的平移偏差。"""

    delta_x = float(first_pose[0, 3] - second_pose[0, 3])
    delta_y = float(first_pose[1, 3] - second_pose[1, 3])
    delta_z = float(first_pose[2, 3] - second_pose[2, 3])
    return math.sqrt(delta_x * delta_x + delta_y * delta_y + delta_z * delta_z)


def _pose_rotation_distance_deg(first_pose, second_pose) -> float:
    """计算两个位姿之间的旋转偏差角。"""

    rotation_error = [[0.0] * 3 for _ in range(3)]
    for row in range(3):
        for column in range(3):
            rotation_error[row][column] = sum(
                float(first_pose[k, row]) * float(second_pose[k, column])
                for k in range(3)
            )

    trace_value = rotation_error[0][0] + rotation_error[1][1] + rotation_error[2][2]
    cosine_value = max(-1.0, min(1.0, (trace_value - 1.0) * 0.5))
    return math.degrees(math.acos(cosine_value))


def _interpolate_pose_with_locked_translation(
    *,
    previous_pose,
    current_pose,
    interpolation_ratio: float,
    locked_translation: Sequence[float],
):
    """在固定平移的前提下，对两个姿态做球面插值。"""

    previous_quaternion = _rotation_matrix_to_quaternion(_rotation_from_pose(previous_pose))
    current_quaternion = _rotation_matrix_to_quaternion(_rotation_from_pose(current_pose))
    interpolated_quaternion = _slerp_quaternion(
        previous_quaternion,
        current_quaternion,
        interpolation_ratio,
    )
    interpolated_rotation = _quaternion_to_rotation_matrix(interpolated_quaternion)
    return _build_pose_from_rotation_translation(
        previous_pose,
        interpolated_rotation,
        locked_translation,
    )


def _rotation_from_pose(pose_matrix) -> tuple[tuple[float, float, float], ...]:
    """从 4x4 位姿中提取 3x3 旋转矩阵。"""

    return tuple(
        tuple(float(pose_matrix[row, column]) for column in range(3))
        for row in range(3)
    )


def _build_pose_from_rotation_translation(reference_pose, rotation, translation):
    """根据旋转和平移重建一个 RoboDK 位姿。"""

    pose_type = type(reference_pose)
    return pose_type(
        [
            [rotation[0][0], rotation[0][1], rotation[0][2], float(translation[0])],
            [rotation[1][0], rotation[1][1], rotation[1][2], float(translation[1])],
            [rotation[2][0], rotation[2][1], rotation[2][2], float(translation[2])],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )


def _pose_row_to_rotation_translation(
    pose_row: dict[str, float],
) -> tuple[tuple[tuple[float, float, float], ...], tuple[float, float, float]]:
    """把 CSV row 解析成 3x3 旋转矩阵和 3x1 平移向量。"""

    rotation = (
        (float(pose_row["r11"]), float(pose_row["r12"]), float(pose_row["r13"])),
        (float(pose_row["r21"]), float(pose_row["r22"]), float(pose_row["r23"])),
        (float(pose_row["r31"]), float(pose_row["r32"]), float(pose_row["r33"])),
    )
    translation = (
        float(pose_row["x_mm"]),
        float(pose_row["y_mm"]),
        float(pose_row["z_mm"]),
    )
    return rotation, translation


def _pose_row_from_rotation_translation(
    rotation: Sequence[Sequence[float]],
    translation: Sequence[float],
) -> dict[str, float]:
    """把旋转矩阵和平移向量重新打包成 pose row。"""

    return {
        "x_mm": float(translation[0]),
        "y_mm": float(translation[1]),
        "z_mm": float(translation[2]),
        "r11": float(rotation[0][0]),
        "r12": float(rotation[0][1]),
        "r13": float(rotation[0][2]),
        "r21": float(rotation[1][0]),
        "r22": float(rotation[1][1]),
        "r23": float(rotation[1][2]),
        "r31": float(rotation[2][0]),
        "r32": float(rotation[2][1]),
        "r33": float(rotation[2][2]),
    }


def _rotation_matrix_from_xyz_offset_deg(
    rotation_offset_deg: Sequence[float],
) -> tuple[tuple[float, float, float], ...]:
    """构造基于 Frame 2 固定轴的 XYZ 旋转矩阵。"""

    normalized_offset = (
        float(rotation_offset_deg[0]),
        float(rotation_offset_deg[1]),
        float(rotation_offset_deg[2]),
    )
    return _rotation_matrix_from_xyz_offset_deg_cached(normalized_offset)


@lru_cache(maxsize=512)
def _rotation_matrix_from_xyz_offset_deg_cached(
    rotation_offset_deg: tuple[float, float, float],
) -> tuple[tuple[float, float, float], ...]:
    """缓存常用姿态偏置旋转矩阵，避免在搜索中重复构造。"""

    rx_deg, ry_deg, rz_deg = rotation_offset_deg
    rx = math.radians(rx_deg)
    ry = math.radians(ry_deg)
    rz = math.radians(rz_deg)

    rotation_x = (
        (1.0, 0.0, 0.0),
        (0.0, math.cos(rx), -math.sin(rx)),
        (0.0, math.sin(rx), math.cos(rx)),
    )
    rotation_y = (
        (math.cos(ry), 0.0, math.sin(ry)),
        (0.0, 1.0, 0.0),
        (-math.sin(ry), 0.0, math.cos(ry)),
    )
    rotation_z = (
        (math.cos(rz), -math.sin(rz), 0.0),
        (math.sin(rz), math.cos(rz), 0.0),
        (0.0, 0.0, 1.0),
    )
    return _multiply_rotation_matrices(
        rotation_z,
        _multiply_rotation_matrices(rotation_y, rotation_x),
    )


def _multiply_rotation_matrices(
    first_rotation: Sequence[Sequence[float]],
    second_rotation: Sequence[Sequence[float]],
) -> tuple[tuple[float, float, float], ...]:
    """3x3 旋转矩阵相乘。"""

    return tuple(
        tuple(
            float(
                sum(
                    first_rotation[row_index][inner_index]
                    * second_rotation[inner_index][column_index]
                    for inner_index in range(3)
                )
            )
            for column_index in range(3)
        )
        for row_index in range(3)
    )


def _multiply_rotation_vector(
    rotation: Sequence[Sequence[float]],
    vector: Sequence[float],
) -> tuple[float, float, float]:
    """3x3 旋转矩阵乘三维向量。"""

    return tuple(
        float(
            sum(rotation[row_index][column_index] * vector[column_index] for column_index in range(3))
        )
        for row_index in range(3)
    )


def _rotation_matrix_to_quaternion(
    rotation: Sequence[Sequence[float]],
) -> tuple[float, float, float, float]:
    """把 3x3 旋转矩阵转成单位四元数。"""

    m00, m01, m02 = rotation[0]
    m10, m11, m12 = rotation[1]
    m20, m21, m22 = rotation[2]
    trace_value = m00 + m11 + m22

    if trace_value > 0.0:
        scale = math.sqrt(trace_value + 1.0) * 2.0
        qw = 0.25 * scale
        qx = (m21 - m12) / scale
        qy = (m02 - m20) / scale
        qz = (m10 - m01) / scale
    elif m00 > m11 and m00 > m22:
        scale = math.sqrt(1.0 + m00 - m11 - m22) * 2.0
        qw = (m21 - m12) / scale
        qx = 0.25 * scale
        qy = (m01 + m10) / scale
        qz = (m02 + m20) / scale
    elif m11 > m22:
        scale = math.sqrt(1.0 + m11 - m00 - m22) * 2.0
        qw = (m02 - m20) / scale
        qx = (m01 + m10) / scale
        qy = 0.25 * scale
        qz = (m12 + m21) / scale
    else:
        scale = math.sqrt(1.0 + m22 - m00 - m11) * 2.0
        qw = (m10 - m01) / scale
        qx = (m02 + m20) / scale
        qy = (m12 + m21) / scale
        qz = 0.25 * scale

    return _normalize_quaternion((qw, qx, qy, qz))


def _quaternion_to_rotation_matrix(
    quaternion: Sequence[float],
) -> tuple[tuple[float, float, float], ...]:
    """把单位四元数转成 3x3 旋转矩阵。"""

    qw, qx, qy, qz = _normalize_quaternion(quaternion)
    return (
        (
            1.0 - 2.0 * (qy * qy + qz * qz),
            2.0 * (qx * qy - qz * qw),
            2.0 * (qx * qz + qy * qw),
        ),
        (
            2.0 * (qx * qy + qz * qw),
            1.0 - 2.0 * (qx * qx + qz * qz),
            2.0 * (qy * qz - qx * qw),
        ),
        (
            2.0 * (qx * qz - qy * qw),
            2.0 * (qy * qz + qx * qw),
            1.0 - 2.0 * (qx * qx + qy * qy),
        ),
    )


def _slerp_quaternion(
    first_quaternion: Sequence[float],
    second_quaternion: Sequence[float],
    interpolation_ratio: float,
) -> tuple[float, float, float, float]:
    """对两个单位四元数做球面插值。"""

    first = _normalize_quaternion(first_quaternion)
    second = _normalize_quaternion(second_quaternion)
    dot_product = sum(a * b for a, b in zip(first, second))

    if dot_product < 0.0:
        second = tuple(-value for value in second)
        dot_product = -dot_product

    if dot_product > 0.9995:
        blended = tuple(
            (1.0 - interpolation_ratio) * first_value + interpolation_ratio * second_value
            for first_value, second_value in zip(first, second)
        )
        return _normalize_quaternion(blended)

    theta_0 = math.acos(max(-1.0, min(1.0, dot_product)))
    sin_theta_0 = math.sin(theta_0)
    theta = theta_0 * interpolation_ratio
    sin_theta = math.sin(theta)
    scale_first = math.sin(theta_0 - theta) / sin_theta_0
    scale_second = sin_theta / sin_theta_0
    return _normalize_quaternion(
        tuple(
            scale_first * first_value + scale_second * second_value
            for first_value, second_value in zip(first, second)
        )
    )


def _normalize_quaternion(
    quaternion: Sequence[float],
) -> tuple[float, float, float, float]:
    """把四元数归一化到单位长度。"""

    norm = math.sqrt(sum(value * value for value in quaternion))
    if norm <= 0.0:
        return (1.0, 0.0, 0.0, 0.0)
    return tuple(value / norm for value in quaternion)


def _mean_abs_joint_delta(
    first_joints: Sequence[float],
    second_joints: Sequence[float],
) -> float:
    """计算两组关节之间的平均绝对差值。"""

    deltas = [abs(second - first) for first, second in zip(first_joints, second_joints)]
    if not deltas:
        return 0.0
    return sum(deltas) / len(deltas)


def _normalize_angle_range(first_deg: float, second_deg: float) -> tuple[float, float]:
    """把角度范围归一化为 [lower, upper]。"""

    return (min(first_deg, second_deg), max(first_deg, second_deg))


def _normalize_step_limits(step_limits: Sequence[float], joint_count: int) -> tuple[float, ...]:
    """把连续性阈值补齐/裁剪到机器人实际轴数长度。"""

    if not step_limits:
        raise ValueError("Joint continuity step limits must not be empty.")

    normalized = [float(value) for value in step_limits]
    if len(normalized) >= joint_count:
        return tuple(normalized[:joint_count])

    # 如果提供的阈值数量少于轴数，则用最后一个阈值补齐剩余关节。
    normalized.extend([normalized[-1]] * (joint_count - len(normalized)))
    return tuple(normalized)


def _passes_user_joint_constraints(
    joints: Sequence[float],
    *,
    a1_lower_deg: float,
    a1_upper_deg: float,
    a2_max_deg: float,
    tolerance_deg: float,
) -> bool:
    """检查是否满足用户指定的 A1 / A2 硬约束。"""

    if len(joints) < 2:
        return False

    a1_value = joints[0]
    a2_value = joints[1]

    if a1_value < a1_lower_deg - tolerance_deg or a1_value > a1_upper_deg + tolerance_deg:
        return False

    # 用户要求 A2 小于 115。这里允许极小容差，避免 114.999999 / 115.0000001 这类数值噪声。
    if a2_value >= a2_max_deg + tolerance_deg:
        return False

    return True


def _translation_from_pose(pose_matrix) -> tuple[float, float, float]:
    """从 4x4 位姿矩阵中提取平移部分。"""

    return (float(pose_matrix[0, 3]), float(pose_matrix[1, 3]), float(pose_matrix[2, 3]))


def _add_vectors(
    first_vector: Sequence[float],
    second_vector: Sequence[float],
) -> tuple[float, float, float]:
    """三维向量相加。"""

    return (
        first_vector[0] + second_vector[0],
        first_vector[1] + second_vector[1],
        first_vector[2] + second_vector[2],
    )


def _subtract_vectors(
    first_vector: Sequence[float],
    second_vector: Sequence[float],
) -> tuple[float, float, float]:
    """三维向量相减。"""

    return (
        first_vector[0] - second_vector[0],
        first_vector[1] - second_vector[1],
        first_vector[2] - second_vector[2],
    )


def _normalized_cross_measure(
    first_vector: Sequence[float],
    second_vector: Sequence[float],
) -> float:
    """返回两个向量叉积模长的归一化值。

    值越接近 0，说明越接近共线。
    """

    first_norm = math.sqrt(sum(component * component for component in first_vector))
    second_norm = math.sqrt(sum(component * component for component in second_vector))
    if first_norm <= 0.0 or second_norm <= 0.0:
        return 1.0

    cross_x = first_vector[1] * second_vector[2] - first_vector[2] * second_vector[1]
    cross_y = first_vector[2] * second_vector[0] - first_vector[0] * second_vector[2]
    cross_z = first_vector[0] * second_vector[1] - first_vector[1] * second_vector[0]
    cross_norm = math.sqrt(cross_x * cross_x + cross_y * cross_y + cross_z * cross_z)
    return cross_norm / (first_norm * second_norm)


def _extract_joint_tuple(raw_joints, joint_count: int) -> tuple[float, ...]:
    """把 RoboDK 返回的 Mat / list 统一提取成固定长度关节元组。"""

    if hasattr(raw_joints, "list"):
        values = list(raw_joints.list())
    else:
        values = list(raw_joints)

    if not values:
        return ()

    trimmed = _trim_joint_vector(values, joint_count)
    if len(trimmed) != joint_count or not all(math.isfinite(value) for value in trimmed):
        return ()
    return trimmed


def _trim_joint_vector(values: Sequence[float], joint_count: int) -> tuple[float, ...]:
    """把关节向量裁成机器人实际轴数长度。"""

    return tuple(float(values[index]) for index in range(min(joint_count, len(values))))


def _clip_seed_to_limits(
    seed: Sequence[float],
    lower_limits: Sequence[float],
    upper_limits: Sequence[float],
) -> tuple[float, ...]:
    """把 seed 限制在机器人关节硬限位内。"""

    return tuple(
        min(max(float(value), lower), upper)
        for value, lower, upper in zip(seed, lower_limits, upper_limits)
    )


def _is_within_joint_limits(
    joints: Sequence[float],
    lower_limits: Sequence[float],
    upper_limits: Sequence[float],
    *,
    tolerance: float = 1e-6,
) -> bool:
    """检查是否落在机器人自身的关节限位内。"""

    return all(
        lower - tolerance <= joint <= upper + tolerance
        for joint, lower, upper in zip(joints, lower_limits, upper_limits)
    )


def _apply_selected_target(target, *, pose, joints: Sequence[float], move_type: str) -> None:
    """把已经全局选好的结果写入 RoboDK target。

    MoveJ:
        直接写成 joint target，避免 RoboDK 再次自由选 IK。
    MoveL:
        仍需保留笛卡尔目标，但会同时写入关节作为首选配置。
    """

    if move_type == "MoveJ":
        target.setAsJointTarget()
        target.setJoints(list(joints))
        return

    target.setAsCartesianTarget()
    target.setPose(pose)
    target.setJoints(list(joints))


def _apply_robodk_native_target(target, *, pose) -> None:
    """把 target 写成纯笛卡尔目标，不给 RoboDK 预设关节解。"""

    target.setAsCartesianTarget()
    target.setPose(pose)


def _import_robodk_api() -> dict[str, object]:
    """延迟导入 RoboDK API。

    这样在没有 RoboDK Python 包的环境里，只有真正执行程序生成时才会报错。
    """

    try:
        from robodk.robolink import (
            ITEM_TYPE_FRAME,
            ITEM_TYPE_PROGRAM,
            ITEM_TYPE_ROBOT,
            ITEM_TYPE_TARGET,
            Robolink,
        )
        from robodk.robomath import Mat
    except ImportError as exc:
        raise RuntimeError(
            "RoboDK Python API is not available. Run this script inside RoboDK or use a "
            "Python interpreter that has the 'robodk' package installed."
        ) from exc

    return {
        "ITEM_TYPE_FRAME": ITEM_TYPE_FRAME,
        "ITEM_TYPE_PROGRAM": ITEM_TYPE_PROGRAM,
        "ITEM_TYPE_ROBOT": ITEM_TYPE_ROBOT,
        "ITEM_TYPE_TARGET": ITEM_TYPE_TARGET,
        "Mat": Mat,
        "Robolink": Robolink,
    }


def _delete_stale_bridge_targets(rdk, item_type_target: int) -> None:
    """删除上次生成但这次可能不会复用的桥接点目标。"""

    try:
        target_names = rdk.ItemList(item_type_target, True)
    except Exception:
        return

    for target_name in target_names:
        if not isinstance(target_name, str):
            continue
        if "_BR_" not in target_name:
            continue
        target = rdk.Item(target_name, item_type_target)
        if target.Valid():
            target.Delete()


def _validate_motion_settings(settings: RoboDKMotionSettings) -> RoboDKMotionSettings:
    """校验用户配置，尽早发现明显错误。"""

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
        raise ValueError("Each bridge step limit must be positive.")
    if settings.global_pose_search_max_offset_deg < 0.0:
        raise ValueError("Global pose search max offset must be non-negative.")
    if any(step <= 0.0 for step in settings.global_pose_search_step_schedule_deg):
        raise ValueError("Each global pose search step must be positive.")
    if settings.wrist_refinement_a5_threshold_deg <= 0.0:
        raise ValueError("Wrist refinement A5 threshold must be positive.")
    if settings.wrist_refinement_large_wrist_step_deg <= 0.0:
        raise ValueError("Wrist refinement large wrist step must be positive.")
    if settings.wrist_refinement_window_radius < 0:
        raise ValueError("Wrist refinement window radius must be non-negative.")
    if any(step <= 0.0 for step in settings.wrist_refinement_phase_offsets_deg):
        raise ValueError("Each wrist refinement phase offset must be positive.")
    if settings.solution_rich_orientation_window_radius < 0:
        raise ValueError("Solution-rich orientation window radius must be non-negative.")
    if settings.solution_rich_orientation_offset_deg <= 0.0:
        raise ValueError("Solution-rich orientation offset must be positive.")
    return settings


def _apply_motion_settings(program, settings: RoboDKMotionSettings) -> None:
    """把速度、加速度、圆角等参数写入 RoboDK 程序。"""

    if (
        hasattr(program, "setSpeedJoints")
        and hasattr(program, "setAcceleration")
        and hasattr(program, "setAccelerationJoints")
    ):
        program.setSpeed(settings.linear_speed_mm_s)
        program.setSpeedJoints(settings.joint_speed_deg_s)
        program.setAcceleration(settings.linear_accel_mm_s2)
        program.setAccelerationJoints(settings.joint_accel_deg_s2)
    else:
        program.setSpeed(
            settings.linear_speed_mm_s,
            settings.joint_speed_deg_s,
            settings.linear_accel_mm_s2,
            settings.joint_accel_deg_s2,
        )

    if hasattr(program, "setRounding"):
        program.setRounding(settings.rounding_mm)
    elif hasattr(program, "setZoneData"):
        program.setZoneData(settings.rounding_mm)


def _append_move_instruction(program, target, move_type: str) -> None:
    """向程序追加一条运动指令。"""

    if move_type == "MoveJ":
        program.MoveJ(target)
    else:
        program.MoveL(target)


def _require_item(rdk, name: str, item_type: int, label: str):
    """从 RoboDK 站点中获取对象；如果不存在就直接抛错。"""

    item = rdk.Item(name, item_type)
    if not item.Valid():
        raise RuntimeError(f"{label} '{name}' was not found in the RoboDK station.")
    return item


def _row_is_empty(row: dict[str, str | None]) -> bool:
    """判断 CSV 行是否为空行。"""

    return not any((value or "").strip() for value in row.values())


def _row_is_marked_invalid(row: dict[str, str | None]) -> bool:
    """判断 CSV 行是否被 valid 列显式标记为无效。"""

    raw_valid = row.get("valid")
    if raw_valid is None:
        return False
    return raw_valid.strip().lower() in {"0", "false", "no"}
