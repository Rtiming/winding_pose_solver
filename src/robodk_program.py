from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from src.geometry import (
    _build_pose,
    _normalize_angle_range,
    _normalize_step_limits,
    _trim_joint_vector,
)
from src.global_search import _search_best_exact_pose_path
from src.local_repair import (
    _attempt_inserted_transition_repair,
    _collect_problem_segments,
    _format_failure_diagnostics,
    _format_focus_segment_report,
    _refine_path_with_frame_a_origin_profile,
)
from src.path_optimizer import _build_optimizer_settings
from src.types import _ProgramWaypoint


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


def create_program_from_csv(
    csv_path: str | Path,
    *,
    robot_name: str,
    frame_name: str,
    program_name: str,
    motion_settings: RoboDKMotionSettings,
) -> object:
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

        search_result = _refine_path_with_frame_a_origin_profile(
            search_result,
            robot=robot,
            mat_type=api["Mat"],
            move_type=settings.move_type,
            start_joints=original_joints,
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

        if _collect_problem_segments(
            search_result.selected_path,
            bridge_trigger_joint_delta_deg=settings.bridge_trigger_joint_delta_deg,
        ):
            inserted_result = _attempt_inserted_transition_repair(
                search_result,
                robot=robot,
                mat_type=api["Mat"],
                move_type=settings.move_type,
                start_joints=original_joints,
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
            if inserted_result is not None:
                search_result = inserted_result

        print(_format_focus_segment_report(search_result))
        _ensure_final_path_is_valid_or_raise(
            search_result,
            settings=settings,
        )

        _write_optimized_pose_csv(csv_path, search_result)
        program_waypoints = _build_program_waypoints(search_result, motion_settings=settings)

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

    total_candidates = sum(len(layer.candidates) for layer in search_result.ik_layers)
    inserted_count = sum(1 for flag in search_result.inserted_flags if flag)
    profile_y_values = [offset[0] for offset in search_result.frame_a_origin_yz_profile_mm]
    profile_z_values = [offset[1] for offset in search_result.frame_a_origin_yz_profile_mm]

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
    print(
        "Optimized Frame-A origin profile in Frame 2: "
        f"dy_range=[{min(profile_y_values):.3f}, {max(profile_y_values):.3f}] mm, "
        f"dz_range=[{min(profile_z_values):.3f}, {max(profile_z_values):.3f}] mm, "
        f"max_abs_offset={search_result.max_abs_offset_mm:.3f} mm."
    )
    print(
        f"Optimized {len(search_result.ik_layers)} target(s) using {total_candidates} IK candidate(s); "
        f"path_cost={search_result.total_cost:.3f}, "
        f"config_switches={search_result.config_switches}, "
        f"worst_joint_step={search_result.worst_joint_step_deg:.3f} deg."
    )
    print(
        f"Created RoboDK program '{program.Name()}' with {len(program_waypoints)} target(s) "
        f"({inserted_count} inserted transition sample(s)) using {settings.move_type} "
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

            for optional_column in ("source_row", "index"):
                raw_value = row.get(optional_column, "")
                if raw_value == "":
                    continue
                try:
                    values[optional_column] = float(raw_value)
                except (TypeError, ValueError):
                    continue

            pose_rows.append(values)

    if not pose_rows:
        raise ValueError(f"No valid pose rows found in CSV file: {csv_path}")

    return pose_rows


def _build_program_waypoints(
    search_result,
    *,
    motion_settings: RoboDKMotionSettings,
) -> list[_ProgramWaypoint]:
    if not search_result.ik_layers or not search_result.selected_path:
        return []

    target_index_width = max(3, len(str(max(0, len(search_result.ik_layers) - 1))))
    waypoints: list[_ProgramWaypoint] = []
    for index, (layer, candidate) in enumerate(
        zip(search_result.ik_layers, search_result.selected_path)
    ):
        row_label = search_result.row_labels[index]
        target_name = _target_name_from_row_label(
            row_label,
            fallback_index=index,
            width=target_index_width,
        )
        waypoints.append(
            _ProgramWaypoint(
                name=target_name,
                pose=layer.pose,
                joints=candidate.joints,
                move_type=motion_settings.move_type,
                is_bridge=bool(search_result.inserted_flags[index]),
            )
        )

    _validate_waypoint_joint_continuity(waypoints, motion_settings)
    return waypoints


def _ensure_final_path_is_valid_or_raise(
    search_result,
    *,
    settings: RoboDKMotionSettings,
) -> None:
    if search_result.ik_empty_row_count > 0 or not search_result.selected_path:
        raise RuntimeError(
            _format_failure_diagnostics(
                search_result,
                bridge_trigger_joint_delta_deg=settings.bridge_trigger_joint_delta_deg,
            )
        )

    problem_segments = _collect_problem_segments(
        search_result.selected_path,
        bridge_trigger_joint_delta_deg=settings.bridge_trigger_joint_delta_deg,
    )
    if problem_segments:
        raise RuntimeError(
            _format_failure_diagnostics(
                search_result,
                bridge_trigger_joint_delta_deg=settings.bridge_trigger_joint_delta_deg,
            )
        )


def _write_optimized_pose_csv(csv_path: str | Path, search_result) -> Path:
    csv_path = Path(csv_path)
    optimized_csv_path = csv_path.with_name(f"{csv_path.stem}_optimized.csv")

    fieldnames = [
        "row_label",
        "inserted_transition_point",
        "frame_a_origin_dy_mm",
        "frame_a_origin_dz_mm",
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
    ]

    with optimized_csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row_label, inserted_flag, (dy_mm, dz_mm), pose_row in zip(
            search_result.row_labels,
            search_result.inserted_flags,
            search_result.frame_a_origin_yz_profile_mm,
            search_result.pose_rows,
        ):
            writer.writerow(
                {
                    "row_label": row_label,
                    "inserted_transition_point": int(bool(inserted_flag)),
                    "frame_a_origin_dy_mm": float(dy_mm),
                    "frame_a_origin_dz_mm": float(dz_mm),
                    **{column: float(pose_row[column]) for column in REQUIRED_COLUMNS},
                }
            )

    print(f"Wrote optimized pose CSV: {optimized_csv_path}")
    return optimized_csv_path


def _target_name_from_row_label(row_label: str, *, fallback_index: int, width: int) -> str:
    sanitized_label = str(row_label).replace("-", "_")
    if sanitized_label.isdigit():
        return f"P_{int(sanitized_label):0{width}d}"
    return f"P_{fallback_index:0{width}d}_{sanitized_label}"


def _validate_waypoint_joint_continuity(
    waypoints: list[_ProgramWaypoint],
    motion_settings: RoboDKMotionSettings,
) -> None:
    trigger = motion_settings.bridge_trigger_joint_delta_deg
    joint_names = ("A1", "A2", "A3", "A4", "A5", "A6")

    for index in range(1, len(waypoints)):
        previous_joints = waypoints[index - 1].joints
        current_joints = waypoints[index].joints
        if not previous_joints or not current_joints or len(previous_joints) != len(current_joints):
            continue

        for axis_index, (previous_joint, current_joint) in enumerate(
            zip(previous_joints, current_joints)
        ):
            delta = abs(current_joint - previous_joint)
            if delta > trigger + 1e-6:
                axis_name = (
                    joint_names[axis_index]
                    if axis_index < len(joint_names)
                    else f"A{axis_index + 1}"
                )
                raise RuntimeError(
                    f"Final waypoint continuity validation failed: "
                    f"{waypoints[index - 1].name}->{waypoints[index].name} "
                    f"{axis_name} delta {delta:.2f} deg exceeds {trigger:.1f} deg."
                )


def _apply_selected_target(target, *, pose, joints: Sequence[float], move_type: str) -> None:
    if move_type == "MoveJ":
        target.setAsJointTarget()
        target.setJoints(list(joints))
        return

    target.setAsCartesianTarget()
    target.setPose(pose)
    target.setJoints(list(joints))


def _apply_robodk_native_target(target, *, pose) -> None:
    target.setAsCartesianTarget()
    target.setPose(pose)


def _import_robodk_api() -> dict[str, object]:
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
    return settings


def _apply_motion_settings(program, settings: RoboDKMotionSettings) -> None:
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
    if move_type == "MoveJ":
        program.MoveJ(target)
    else:
        program.MoveL(target)


def _require_item(rdk, name: str, item_type: int, label: str):
    item = rdk.Item(name, item_type)
    if not item.Valid():
        raise RuntimeError(f"{label} '{name}' was not found in the RoboDK station.")
    return item


def _row_is_empty(row: dict[str, str | None]) -> bool:
    return not any((value or "").strip() for value in row.values())


def _row_is_marked_invalid(row: dict[str, str | None]) -> bool:
    raw_valid = row.get("valid")
    if raw_valid is None:
        return False
    return raw_valid.strip().lower() in {"0", "false", "no"}
