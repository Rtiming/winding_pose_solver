from __future__ import annotations

import csv
import math
import re
from pathlib import Path
from typing import Sequence

from src.core.geometry import (
    _build_pose,
    _normalize_angle_range,
    _normalize_step_limits,
)
from src.search.continuity_metrics import summarize_branch_jump_metrics
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
    _delete_stale_bridge_targets(
        context.rdk,
        context.api["ITEM_TYPE_TARGET"],
        prefix=_target_prefix_from_program_name(program_name),
    )

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
            program_waypoints = _build_program_waypoints(
                search_result,
                motion_settings=settings,
                target_prefix=program_name_out,
            )
    finally:
        context.rdk.Render(True)
        if context.original_joints:
            context.robot.setJoints(list(context.original_joints))

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
    target_prefix = _target_prefix_from_program_name(program_name)
    seed_joints = _robodk_joints_to_tuple(robot.Joints())
    for index, row in enumerate(pose_rows):
        target_name = f"{target_prefix}_P_{index:0{target_index_width}d}"
        existing_target = rdk.Item(target_name, item_type_target)
        if existing_target.Valid():
            existing_target.Delete()

        pose = _build_pose(row, mat_type)
        solved_joints = _solve_target_joints(
            robot,
            pose=pose,
            seed_joints=seed_joints,
        )
        if solved_joints:
            seed_joints = solved_joints
        target = rdk.AddTarget(target_name, frame, robot)
        target.setRobot(robot)
        _apply_robodk_native_target(
            target,
            pose=pose,
            joints=solved_joints,
            move_type=motion_settings.move_type,
        )
        _append_move_instruction(program, target, motion_settings.move_type)
        if motion_settings.hide_targets_after_generation:
            _set_item_visible(target, False)

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
    validate_joint_continuity: bool = True,
    target_prefix: str | None = None,
) -> list[_ProgramWaypoint]:
    if not search_result.ik_layers or not search_result.selected_path:
        return []

    target_index_width = max(3, len(str(max(0, len(search_result.ik_layers) - 1))))
    waypoints: list[_ProgramWaypoint] = []
    previous_was_bridge = False
    for index, (layer, candidate) in enumerate(
        zip(search_result.ik_layers, search_result.selected_path)
    ):
        row_label = search_result.row_labels[index]
        is_bridge = bool(search_result.inserted_flags[index])
        is_joint_space_bridge = _is_joint_space_bridge_label(row_label)
        move_type = (
            "MoveJ"
            if is_joint_space_bridge
            else "MoveL"
            if is_bridge or previous_was_bridge
            else motion_settings.move_type
        )
        target_name = _target_name_from_row_label(
            row_label,
            fallback_index=index,
            width=target_index_width,
            prefix=target_prefix,
        )
        waypoints.append(
            _ProgramWaypoint(
                name=target_name,
                pose=layer.pose,
                joints=candidate.joints,
                move_type=move_type,
                is_bridge=is_bridge,
            )
        )
        previous_was_bridge = is_bridge and not is_joint_space_bridge

    if validate_joint_continuity:
        _validate_waypoint_joint_continuity(waypoints, motion_settings)
    return waypoints


def _ensure_final_path_is_valid_or_raise(
    search_result,
    *,
    settings: RoboDKMotionSettings,
    enforce_continuity_gate: bool = True,
) -> None:
    selected_path = getattr(search_result, "selected_path", None)
    row_labels = getattr(search_result, "row_labels", ())
    if (
        int(getattr(search_result, "invalid_row_count", 0)) > 0
        or int(getattr(search_result, "ik_empty_row_count", 0)) > 0
        or not selected_path
        or len(selected_path) != len(row_labels)
    ):
        raise RuntimeError(
            _format_failure_diagnostics(
                search_result,
                bridge_trigger_joint_delta_deg=settings.bridge_trigger_joint_delta_deg,
            )
        )

    closed_terminal_report = closed_winding_terminal_report(search_result, settings=settings)
    if bool(closed_terminal_report.get("closed_path")) and not bool(
        closed_terminal_report.get("passed")
    ):
        raise RuntimeError(
            "Closed winding terminal constraint failed: "
            + str(closed_terminal_report.get("message", "unknown terminal mismatch"))
        )

    branch_jump_metrics = summarize_branch_jump_metrics(
        search_result.selected_path,
        row_labels=search_result.row_labels,
        pose_rows=search_result.pose_rows,
        big_circle_step_deg_threshold=float(
            getattr(settings, "big_circle_step_deg_threshold", 170.0)
        ),
        branch_flip_ratio_threshold=float(
            getattr(settings, "branch_flip_ratio_threshold", 8.0)
        ),
        ratio_eps_mm=float(getattr(settings, "branch_flip_ratio_eps_mm", 1e-3)),
    )
    big_circle_step_count = int(branch_jump_metrics.get("big_circle_step_count", 0))
    worst_step_limit = float(
        getattr(settings, "official_worst_joint_step_deg_limit", 60.0)
    )
    block_reasons: list[str] = []
    if int(getattr(search_result, "bridge_like_segments", 0)) > 0:
        block_reasons.append(
            f"bridge_like_segments={int(getattr(search_result, 'bridge_like_segments', 0))}"
        )
    if big_circle_step_count > 0:
        block_reasons.append(f"big_circle_step_count={big_circle_step_count}")
    if float(getattr(search_result, "worst_joint_step_deg", 0.0)) > worst_step_limit + 1e-9:
        block_reasons.append(
            "worst_joint_step_deg="
            f"{float(getattr(search_result, 'worst_joint_step_deg', 0.0)):.3f}"
            f">{worst_step_limit:.3f}"
        )
    if block_reasons:
        if not enforce_continuity_gate:
            print(
                "[program] Continuity diagnostics found (not blocking in evaluation-only mode): "
                + "; ".join(block_reasons)
            )
            return
        problem_segments = _collect_problem_segments(
            search_result.selected_path,
            bridge_trigger_joint_delta_deg=settings.bridge_trigger_joint_delta_deg,
            max_segments=6,
        )
        segment_note = ", ".join(
            f"{search_result.row_labels[segment_index]}->{search_result.row_labels[segment_index + 1]}"
            for segment_index, _cfg, _mx, _mn in problem_segments
        )
        raise RuntimeError(
            "Continuity delivery gate failed: "
            + "; ".join(block_reasons)
            + ("" if not segment_note else f"; segments={segment_note}")
        )


def closed_winding_terminal_report(
    search_result,
    *,
    settings: RoboDKMotionSettings,
) -> dict[str, object]:
    """Validate the user-promoted closed winding terminal hard rule.

    The rule only applies when the reference path explicitly appends the start
    row as the terminal row.  In that case terminal I1-I5 must match the start
    I1-I5 and I6 must differ by the configured number of full turns.
    """

    reference_pose_rows = getattr(search_result, "reference_pose_rows", ())
    if not _reference_path_has_terminal_start_copy(reference_pose_rows):
        return {
            "closed_path": False,
            "passed": True,
            "message": "not a closed winding path",
        }

    selected_path = tuple(getattr(search_result, "selected_path", ()) or ())
    if len(selected_path) < 2:
        return {
            "closed_path": True,
            "passed": False,
            "message": "selected path is empty or incomplete",
        }

    first_joints = tuple(float(value) for value in selected_path[0].joints)
    terminal_joints = tuple(float(value) for value in selected_path[-1].joints)
    if len(first_joints) < 6 or len(terminal_joints) < 6:
        return {
            "closed_path": True,
            "passed": False,
            "message": "selected path does not contain six robot axes",
        }

    tolerance_deg = float(settings.closed_path_joint6_turn_tolerance_deg)
    axis_deltas = tuple(
        terminal_joint - first_joint
        for first_joint, terminal_joint in zip(first_joints, terminal_joints)
    )
    mismatched_axes = tuple(
        axis_index + 1
        for axis_index, delta_deg in enumerate(axis_deltas[:5])
        if abs(float(delta_deg)) > tolerance_deg
    )
    if mismatched_axes:
        return {
            "closed_path": True,
            "passed": False,
            "message": (
                "terminal I1-I5 must match start I1-I5; mismatched axes="
                f"{mismatched_axes}"
            ),
            "axis_deltas_deg": [float(value) for value in axis_deltas],
            "tolerance_deg": tolerance_deg,
        }

    joint6_delta_deg = float(axis_deltas[5])
    nearest_turns = int(round(joint6_delta_deg / 360.0))
    turn_error_deg = abs(joint6_delta_deg - 360.0 * nearest_turns)
    required_turns = int(settings.closed_path_joint6_turns)
    if turn_error_deg > tolerance_deg or abs(nearest_turns) != required_turns:
        return {
            "closed_path": True,
            "passed": False,
            "message": (
                "terminal I6 must differ from start I6 by exactly "
                f"{required_turns} full turn(s)"
            ),
            "axis_deltas_deg": [float(value) for value in axis_deltas],
            "joint6_delta_deg": joint6_delta_deg,
            "signed_turns": nearest_turns,
            "turn_error_deg": turn_error_deg,
            "tolerance_deg": tolerance_deg,
        }

    inferred_direction = _infer_selected_path_joint6_direction(
        selected_path,
        sample_count=int(settings.closed_path_joint6_direction_sample_count),
        min_delta_deg=float(settings.closed_path_joint6_direction_min_delta_deg),
    )
    if required_turns and inferred_direction and nearest_turns != inferred_direction * required_turns:
        return {
            "closed_path": True,
            "passed": False,
            "message": (
                "terminal I6 full-turn sign does not match the early selected-path "
                "I6 trend"
            ),
            "axis_deltas_deg": [float(value) for value in axis_deltas],
            "joint6_delta_deg": joint6_delta_deg,
            "signed_turns": nearest_turns,
            "inferred_direction": inferred_direction,
            "tolerance_deg": tolerance_deg,
        }

    return {
        "closed_path": True,
        "passed": True,
        "message": "terminal I1-I5 match and I6 closes by the required full turn",
        "axis_deltas_deg": [float(value) for value in axis_deltas],
        "joint6_delta_deg": joint6_delta_deg,
        "signed_turns": nearest_turns,
        "inferred_direction": inferred_direction,
        "tolerance_deg": tolerance_deg,
    }


def _infer_selected_path_joint6_direction(
    selected_path: Sequence[object],
    *,
    sample_count: int,
    min_delta_deg: float,
) -> int:
    sample_count = min(len(selected_path), max(2, int(sample_count)))
    if sample_count < 2:
        return 0
    first_joints = tuple(float(value) for value in selected_path[0].joints)
    sampled_joints = tuple(float(value) for value in selected_path[sample_count - 1].joints)
    if len(first_joints) < 6 or len(sampled_joints) < 6:
        return 0
    delta_deg = float(sampled_joints[5] - first_joints[5])
    if abs(delta_deg) < float(min_delta_deg):
        return 0
    return 1 if delta_deg > 0.0 else -1


def _reference_path_has_terminal_start_copy(
    reference_pose_rows: Sequence[dict[str, float]],
) -> bool:
    if len(reference_pose_rows) < 2:
        return False
    first_row = reference_pose_rows[0]
    terminal_row = reference_pose_rows[-1]
    if "source_row" not in first_row or "source_row" not in terminal_row:
        return False
    if int(float(first_row["source_row"])) != int(float(terminal_row["source_row"])):
        return False
    return _pose_rows_match(first_row, terminal_row)


def _pose_rows_match(
    first_row: dict[str, float],
    second_row: dict[str, float],
    *,
    tolerance: float = 1e-9,
) -> bool:
    pose_columns = (
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
    return all(
        abs(float(first_row.get(column, math.nan)) - float(second_row.get(column, math.nan)))
        <= tolerance
        for column in pose_columns
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


def _target_name_from_row_label(
    row_label: str,
    *,
    fallback_index: int,
    width: int,
    prefix: str | None = None,
) -> str:
    sanitized_label = _sanitize_target_token(str(row_label))
    target_prefix = _target_prefix_from_program_name(prefix) if prefix else ""
    if sanitized_label.isdigit():
        target_name = f"P_{int(sanitized_label):0{width}d}"
    else:
        target_name = f"P_{fallback_index:0{width}d}_{sanitized_label}"
    return f"{target_prefix}_{target_name}" if target_prefix else target_name


def _target_prefix_from_program_name(program_name: str | None) -> str:
    return _sanitize_target_token(program_name or "Path")


def _sanitize_target_token(value: object) -> str:
    sanitized = re.sub(r"[^0-9A-Za-z_]+", "_", str(value).strip())
    sanitized = re.sub(r"_+", "_", sanitized).strip("_")
    return sanitized or "item"


def _is_joint_space_bridge_label(row_label: object) -> bool:
    return "_JBR_" in str(row_label)


def _validate_waypoint_joint_continuity(
    waypoints: list[_ProgramWaypoint],
    motion_settings: RoboDKMotionSettings,
) -> None:
    trigger = motion_settings.bridge_trigger_joint_delta_deg
    joint_names = ("A1", "A2", "A3", "A4", "A5", "A6")
    warnings: list[str] = []

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
                warnings.append(
                    f"{waypoints[index - 1].name}->{waypoints[index].name} "
                    f"{axis_name} delta {delta:.2f} deg exceeds {trigger:.1f} deg"
                )

    if warnings:
        preview = "; ".join(warnings[:3])
        suffix = "" if len(warnings) <= 3 else f"; ... +{len(warnings) - 3} more"
        print(
            "[program] Waypoint continuity warnings kept as diagnostics: "
            f"{preview}{suffix}."
        )


def _apply_selected_target(target, *, pose, joints: Sequence[float], move_type: str) -> None:
    # Always store both pose and joints. RoboDK program instructions can become
    # "Target undefined" when a MoveJ target only carries a name without a
    # materialized pose/joint payload in the station.
    if move_type == "MoveJ":
        target.setAsJointTarget()
        target.setPose(pose)
        target.setJoints(list(joints))
        return

    target.setAsCartesianTarget()
    target.setPose(pose)
    target.setJoints(list(joints))


def _set_item_visible(item, visible: bool) -> None:
    """Hide/show a RoboDK item without changing program target references."""
    for method_name in ("Visible", "setVisible"):
        method = getattr(item, method_name, None)
        if not callable(method):
            continue
        try:
            method(bool(visible))
            return
        except Exception:
            continue


def _apply_robodk_native_target(
    target,
    *,
    pose,
    joints: Sequence[float] | None,
    move_type: str,
) -> None:
    if move_type == "MoveJ" and joints:
        target.setAsJointTarget()
        target.setPose(pose)
        target.setJoints(list(joints))
        return

    target.setAsCartesianTarget()
    target.setPose(pose)
    if joints:
        target.setJoints(list(joints))


def _solve_target_joints(
    robot,
    *,
    pose,
    seed_joints: Sequence[float] | None,
) -> tuple[float, ...] | None:
    seed = list(seed_joints or ())
    solve_attempts = (
        lambda: robot.SolveIK(pose, seed, robot.PoseTool(), robot.PoseFrame()),
        lambda: robot.SolveIK(pose, seed),
        lambda: robot.SolveIK(pose),
    )
    for solve_attempt in solve_attempts:
        try:
            joints = _robodk_joints_to_tuple(solve_attempt())
        except Exception:
            continue
        if joints:
            return joints
    return None


def _robodk_joints_to_tuple(raw_joints) -> tuple[float, ...]:
    if raw_joints is None:
        return ()
    if hasattr(raw_joints, "list"):
        raw_joints = raw_joints.list()
    try:
        joints = tuple(float(value) for value in raw_joints)
    except TypeError:
        return ()
    if not joints or any(not math.isfinite(value) for value in joints):
        return ()
    return joints


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


def _delete_stale_bridge_targets(
    rdk,
    item_type_target: int,
    *,
    prefix: str | None = None,
) -> None:
    try:
        target_names = rdk.ItemList(item_type_target, True)
    except Exception:
        return

    target_prefix = _target_prefix_from_program_name(prefix) if prefix else None
    for target_name in target_names:
        if not isinstance(target_name, str):
            continue
        if "_BR_" not in target_name:
            continue
        if target_prefix is not None and not target_name.startswith(f"{target_prefix}_"):
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
