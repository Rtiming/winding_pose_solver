from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from src.types import (
    _IKCandidate,
    _IKLayer,
    _PathOptimizerSettings,
    _ProgramWaypoint,
    _BridgeCandidate,
    _PathSearchResult,
)
from src.geometry import (
    _normalize_angle_range,
    _normalize_step_limits,
    _trim_joint_vector,
    _build_pose,
)
from src.path_optimizer import (
    _build_optimizer_settings,
    _summarize_selected_path,
    _path_is_clean_enough_for_program_generation,
    _optimize_joint_path,
)
from src.ik_collection import (
    _build_ik_layers,
)
from src.bridge_builder import (
    _needs_pose_bridge,
    _build_position_locked_bridge_segment,
)
from src.local_repair import (
    _refine_path_near_wrist_singularity,
    _redistribute_orientation_in_solution_rich_window,
)
from src.global_search import (
    _search_best_exact_pose_path,
)


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

    # 本项目自定义的"多解 IK + 全局路径优化 + 平滑/桥接"总开关。
    enable_custom_smoothing_and_pose_selection: bool = True

    # 用户指定的硬约束。
    a1_min_deg: float = -150.0
    a1_max_deg: float = 30.0

    # A2 强制小于该阈值。
    a2_max_deg: float = 115.0
    joint_constraint_tolerance_deg: float = 1e-6

    # 点与点之间的连续性硬约束。
    enable_joint_continuity_constraint: bool = True
    max_joint_step_deg: tuple[float, ...] = (5.0, 5.0, 5.0, 180.0, 100.0, 180.0)

    # 姿态重构桥接参数。
    enable_pose_bridge: bool = True
    bridge_trigger_joint_delta_deg: float = 20.0
    bridge_step_deg: tuple[float, ...] = (2.0, 2.0, 2.0, 20.0, 10.0, 20.0)
    bridge_translation_search_mm: tuple[float, ...] = (0.0, 1.0, 2.0)
    bridge_rotation_search_deg: tuple[float, ...] = (0.0, 0.5, 1.0)

    # 全局位姿微调搜索。
    enable_global_pose_search: bool = True
    global_pose_search_origin_mm: tuple[float, float, float] | None = None
    global_pose_search_max_offset_deg: float = 18.0
    global_pose_search_step_schedule_deg: tuple[float, ...] = (8.0, 4.0, 2.0, 1.0)

    # 腕奇异 refinement。
    enable_wrist_singularity_refinement: bool = True
    wrist_refinement_a5_threshold_deg: float = 12.0
    wrist_refinement_large_wrist_step_deg: float = 20.0
    wrist_refinement_window_radius: int = 2
    wrist_refinement_phase_offsets_deg: tuple[float, ...] = (15.0, 30.0, 45.0, 60.0, 90.0)

    # 解丰富区姿态重分配。
    enable_solution_rich_orientation_redistribution: bool = True
    solution_rich_orientation_window_radius: int = 4
    solution_rich_orientation_offset_deg: float = 4.0


def create_program_from_csv(
    csv_path: str | Path,
    *,
    robot_name: str,
    frame_name: str,
    program_name: str,
    motion_settings: RoboDKMotionSettings,
) -> object:
    """根据位姿 CSV 生成 RoboDK 程序。"""

    settings = _validate_motion_settings(motion_settings)
    pose_rows = tuple(load_pose_rows(csv_path))
    api = _import_robodk_api()

    rdk = api["Robolink"]()
    robot = _require_item(rdk, robot_name, api["ITEM_TYPE_ROBOT"], "Robot")
    frame = _require_item(rdk, frame_name, api["ITEM_TYPE_FRAME"], "Reference frame")
    _delete_stale_bridge_targets(rdk, api["ITEM_TYPE_TARGET"])

    current_joints_list = robot.Joints().list()
    joint_count = len(current_joints_list)
    original_joints = _trim_joint_vector(current_joints_list, joint_count)
    lower_limits_raw, upper_limits_raw, _ = robot.JointLimits()
    lower_limits = _trim_joint_vector(lower_limits_raw.list(), joint_count)
    upper_limits = _trim_joint_vector(upper_limits_raw.list(), joint_count)

    robot.setPoseFrame(frame)
    tool_pose = robot.PoseTool()
    reference_pose = robot.PoseFrame()

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

        # ── 紧约束二次 DP ──────────────────────────────────────────────────
        # 在所有 refinement 完成后，尝试以"紧腕部约束"（A4/A5/A6 ≤30°/步）
        # 重新在已扩充的 IK 候选层上跑一遍 DP。
        # 目的：
        #   1. 如果 refinement 在奇异窗口附近成功填入了替代候选，
        #      紧约束 DP 能找到一条每步腕部变化 ≤30° 的平滑路径，彻底消除大跳变。
        #   2. 如果紧约束 DP 失败，说明当前工艺几何与目标坐标系组合下，
        #      不存在满足约束的连续路径，程序应当明确报错而非输出错误结果。
        # 30°/步的上限足够允许奇异附近的腕部渐变（FINA11.src 的 A4 变化约 0.6°/步平均），
        # 同时严格拒绝 160° 级别的配置切换大跳变。
        _TIGHT_WRIST_STEP_DEG = 30.0
        tight_step = tuple(
            _TIGHT_WRIST_STEP_DEG if i >= 3 else lim
            for i, lim in enumerate(optimizer_settings.max_joint_step_deg)
        )
        from dataclasses import replace as _dataclasses_replace
        tight_optimizer = _dataclasses_replace(optimizer_settings, max_joint_step_deg=tight_step)
        # 先计算当前路径的最差关节跳变，无论紧约束 DP 是否成功都需要这个值作报告。
        _, _, _loose_worst, _ = _summarize_selected_path(
            selected_path,
            bridge_trigger_joint_delta_deg=settings.bridge_trigger_joint_delta_deg,
        )
        try:
            tight_path_list, tight_cost = _optimize_joint_path(
                ik_layers,
                robot=robot,
                move_type=settings.move_type,
                start_joints=original_joints,
                optimizer_settings=tight_optimizer,
            )
            tight_path = tuple(tight_path_list)
            _, _, tight_worst, _ = _summarize_selected_path(
                tight_path,
                bridge_trigger_joint_delta_deg=settings.bridge_trigger_joint_delta_deg,
            )
            # 只有当紧约束路径确实更平滑时才替换。
            if tight_worst < _loose_worst - 0.1:
                print(
                    f"Tight-wrist re-optimization accepted: "
                    f"worst_joint_step {_loose_worst:.2f}° → {tight_worst:.2f}°."
                )
                selected_path = tight_path
                total_cost = tight_cost
            else:
                print(
                    f"Tight-wrist re-optimization: no improvement "
                    f"({_loose_worst:.2f}° → {tight_worst:.2f}°), keeping original path."
                )
        except RuntimeError as exc:
            # 紧约束 DP 未找到无跳变路径 — 这说明路径中仍有无法消除的 IK 分支切换。
            # 这不是致命错误：后续的位置锁定桥接（position-locked bridge）会处理大跳变。
            # 如果桥接也失败，那才是真正的报错。
            # 这里只做警告，继续用宽松 DP 路径进入桥接阶段。
            print(
                f"[INFO] 紧腕部约束 DP (A4/A5/A6 <={_TIGHT_WRIST_STEP_DEG:.0f} deg/step) "
                f"未找到无跳变路径 (worst={_loose_worst:.2f} deg, reason={exc})。"
                "将继续使用宽松路径并交由位置锁定桥接处理大跳变。"
            )
        # ── 紧约束二次 DP 结束 ─────────────────────────────────────────────

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
    """把最终关节路径展开成真正要写入程序的路点序列。"""

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
            prev_joints_str = ", ".join(f"{j:.1f}" for j in previous_candidate.joints)
            curr_joints_str = ", ".join(f"{j:.1f}" for j in current_candidate.joints)
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
            except RuntimeError as exc:
                # 严格固定位置桥接失败。
                # 此前的 best-effort 回退（关节空间线性插值）会在笛卡尔空间产生大幅绕行弧，
                # 违反工艺要求，因此不再执行该回退。
                # 正确做法是在 DP 阶段就避免产生此类需要桥接的坏段；
                # 如果无法避免，程序必须明确报错而不是静默输出错误结果。
                joint_deltas = [
                    abs(c - p)
                    for p, c in zip(previous_candidate.joints, current_candidate.joints)
                ]
                max_delta = max(joint_deltas, default=0.0)
                raise RuntimeError(
                    f"路径段 {index - 1}→{index} 的位置锁定桥接失败，"
                    f"且无安全回退方案可用。\n"
                    f"  前一关节: [{prev_joints_str}]\n"
                    f"  当前关节: [{curr_joints_str}]\n"
                    f"  最大单轴跳变: {max_delta:.1f}°，"
                    f"配置标志: {previous_candidate.config_flags}→{current_candidate.config_flags}\n"
                    f"  根本原因: {exc}\n"
                    f"建议：检查全局搜索参数是否覆盖了这段路径所在的 IK 分支，"
                    f"或适当调整目标工艺坐标系原点以避免分支切换。"
                ) from exc
            waypoints.extend(bridge_waypoints)
            waypoints.append(current_waypoint)
            continue

        waypoints.append(
            _ProgramWaypoint(
                name=f"P_{index:0{target_index_width}d}",
                pose=current_layer.pose,
                joints=current_candidate.joints,
                move_type=motion_settings.move_type,
                is_bridge=False,
            )
        )

    _validate_waypoint_joint_continuity(waypoints, motion_settings)
    return waypoints


def _validate_waypoint_joint_continuity(
    waypoints: list[_ProgramWaypoint],
    motion_settings: RoboDKMotionSettings,
) -> None:
    """对最终路点序列做一次关节连续性验证，确保不存在超限跳变。

    这是最后一道防线：如果路点序列中仍然存在超过桥接触发阈值的关节跳变，
    说明前面的桥接逻辑有遗漏，需要立即报错，而不是静默输出错误程序。
    """

    trigger = motion_settings.bridge_trigger_joint_delta_deg
    joint_names = ("A1", "A2", "A3", "A4", "A5", "A6")

    for i in range(1, len(waypoints)):
        prev_joints = waypoints[i - 1].joints
        curr_joints = waypoints[i].joints
        if not prev_joints or not curr_joints or len(prev_joints) != len(curr_joints):
            continue

        for axis_idx, (prev_j, curr_j) in enumerate(zip(prev_joints, curr_joints)):
            delta = abs(curr_j - prev_j)
            if delta > trigger + 1e-6:
                axis_name = joint_names[axis_idx] if axis_idx < len(joint_names) else f"A{axis_idx + 1}"
                raise RuntimeError(
                    f"最终路点校验失败：路点 {waypoints[i - 1].name}→{waypoints[i].name} "
                    f"的 {axis_name} 跳变 {delta:.2f}° 超过桥接触发阈值 {trigger:.1f}°。\n"
                    f"  前一关节: [{', '.join(f'{j:.2f}' for j in prev_joints)}]\n"
                    f"  当前关节: [{', '.join(f'{j:.2f}' for j in curr_joints)}]\n"
                    f"该路径无法安全执行（会在笛卡尔空间产生大幅绕行），"
                    f"程序终止以避免输出错误结果。"
                )


def _apply_selected_target(target, *, pose, joints: Sequence[float], move_type: str) -> None:
    """把已经全局选好的结果写入 RoboDK target。"""

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
    """延迟导入 RoboDK API。"""

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
