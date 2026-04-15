from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Sequence

from src.core.geometry import (
    _build_pose,
    _normalize_angle_range,
    _normalize_step_limits,
)
from src.search.local_repair import _collect_problem_segments, _format_failure_diagnostics
from src.core.motion_settings import RoboDKMotionSettings, motion_settings_to_dict, validate_motion_settings
from src.core.pose_csv import REQUIRED_COLUMNS, load_pose_rows
from src.core.types import _ProgramWaypoint


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
    from src.core.collab_models import ProfileEvaluationRequest
    from src.search.global_search import _extract_row_labels
    from src.robodk_runtime.eval_worker import (
        evaluate_request,
        materialize_program,
        open_live_station_context,
    )

    context = open_live_station_context(
        robot_name=robot_name,
        frame_name=frame_name,
    )
    _delete_stale_bridge_targets(context.rdk, context.api["ITEM_TYPE_TARGET"])

    search_result = None
    program = None
    program_waypoints: list[_ProgramWaypoint] = []
    a1_lower_deg, a1_upper_deg = _normalize_angle_range(settings.a1_min_deg, settings.a1_max_deg)
    context.rdk.Render(False)
    try:
        if not settings.enable_custom_smoothing_and_pose_selection:
            program = _create_program_from_pose_rows_with_robodk_defaults(
                pose_rows,
                rdk=context.rdk,
                robot=context.robot,
                frame=context.frame,
                mat_type=context.mat_type,
                item_type_program=context.api["ITEM_TYPE_PROGRAM"],
                item_type_target=context.api["ITEM_TYPE_TARGET"],
                program_name=program_name,
                frame_name=frame_name,
                motion_settings=settings,
            )
        else:
            request = ProfileEvaluationRequest(
                request_id=f"{program_name}_full_search",
                robot_name=robot_name,
                frame_name=frame_name,
                motion_settings=motion_settings_to_dict(settings),
                reference_pose_rows=pose_rows,
                frame_a_origin_yz_profile_mm=tuple((0.0, 0.0) for _ in pose_rows),
                row_labels=_extract_row_labels(pose_rows),
                inserted_flags=tuple(False for _ in pose_rows),
                strategy="full_search",
                start_joints=context.original_joints,
                run_window_repair=True,
                run_inserted_repair=True,
                include_pose_rows_in_result=False,
                create_program=False,
                program_name=program_name,
                optimized_csv_path=None,
                metadata={"entrypoint": "single_machine"},
            )
            result, search_result = evaluate_request(request, context)
            print(result.focus_report)
            _ensure_final_path_is_valid_or_raise(search_result, settings=settings)
            _write_optimized_pose_csv(csv_path, search_result)
            program_name_out = materialize_program(context, request, search_result, settings)
            program = context.rdk.Item(program_name_out, context.api["ITEM_TYPE_PROGRAM"])
            program_waypoints = _build_program_waypoints(search_result, motion_settings=settings)
    finally:
        context.rdk.Render(True)
        if context.original_joints:
            context.robot.setJoints(list(context.original_joints))

    if settings.hide_targets_after_generation and hasattr(program, "ShowTargets"):
        program.ShowTargets(False)

    if search_result is None:
        return program

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
        continuity_limits = _normalize_step_limits(
            settings.max_joint_step_deg,
            context.joint_count,
        )
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
    return validate_motion_settings(settings)


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


