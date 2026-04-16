from __future__ import annotations

import re
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from src.core.collab_models import (
    EvaluationBatchResult,
    ProfileEvaluationResult,
    load_json_file,
)
from src.core.geometry import _build_pose
from src.core.motion_settings import build_motion_settings_from_dict
from src.robodk_runtime.eval_worker import open_live_station_context
from src.robodk_runtime.program import (
    _append_move_instruction,
    _apply_motion_settings,
    _apply_selected_target,
)


@dataclass(frozen=True)
class ProfileResultImportSummary:
    program_name: str | None
    marker_count: int
    program_target_count: int
    robot_name: str
    frame_name: str
    prefix: str


def load_profile_result(path: str | Path, *, result_index: int = 0) -> ProfileEvaluationResult:
    payload = load_json_file(path)
    if "results" in payload:
        batch = EvaluationBatchResult.from_dict(payload)
        if result_index < 0 or result_index >= len(batch.results):
            raise IndexError(
                f"Result index {result_index} is outside batch size {len(batch.results)}."
            )
        return batch.results[result_index]
    return ProfileEvaluationResult.from_dict(payload)


def import_profile_result_to_robodk(
    result: ProfileEvaluationResult,
    *,
    robot_name: str | None = None,
    frame_name: str | None = None,
    program_name: str | None = None,
    prefix: str = "YZNS",
    clear_prefix: bool = True,
    create_program: bool = True,
    create_cartesian_markers: bool = True,
    focus_start_label: str | None = None,
    focus_end_label: str | None = None,
    move_type: str | None = None,
) -> ProfileResultImportSummary:
    if result.pose_rows is None:
        raise ValueError(
            "The result does not include pose_rows. Re-run the validation with "
            "include_pose_rows_in_result=True."
        )
    if len(result.pose_rows) != len(result.selected_path):
        raise ValueError(
            "Result pose_rows and selected_path lengths differ: "
            f"{len(result.pose_rows)} vs {len(result.selected_path)}."
        )

    resolved_robot_name = robot_name or _metadata_string(
        result.motion_settings,
        "robot_name",
        default="KUKA",
    )
    resolved_frame_name = frame_name or "Frame 2"
    safe_prefix = _sanitize_name(prefix)
    resolved_program_name = (
        _sanitize_name(program_name)
        if program_name
        else f"{safe_prefix}_{_sanitize_name(result.request_id)}_Program"
    )

    settings = build_motion_settings_from_dict(result.motion_settings)
    if move_type is not None:
        settings = replace(settings, move_type=str(move_type))

    context = open_live_station_context(
        robot_name=resolved_robot_name,
        frame_name=resolved_frame_name,
    )

    marker_count = 0
    program_target_count = 0
    program = None
    context.rdk.Render(False)
    try:
        if clear_prefix:
            _delete_items_with_prefix(
                context.rdk,
                item_type=context.api["ITEM_TYPE_TARGET"],
                prefix=safe_prefix,
            )
            _delete_items_with_prefix(
                context.rdk,
                item_type=context.api["ITEM_TYPE_PROGRAM"],
                prefix=safe_prefix,
            )

        if create_program:
            existing_program = context.rdk.Item(
                resolved_program_name,
                context.api["ITEM_TYPE_PROGRAM"],
            )
            if existing_program.Valid():
                existing_program.Delete()
            program = context.rdk.AddProgram(resolved_program_name, context.robot)
            program.setRobot(context.robot)
            if hasattr(program, "setPoseFrame"):
                program.setPoseFrame(context.frame)
            if hasattr(program, "setPoseTool"):
                program.setPoseTool(context.robot.PoseTool())
            _apply_motion_settings(program, settings)

        previous_inserted = False
        target_index_width = max(3, len(str(max(0, len(result.row_labels) - 1))))
        for index, (row_label, inserted_flag, pose_row, selected_entry) in enumerate(
            zip(
                result.row_labels,
                result.inserted_flags,
                result.pose_rows,
                result.selected_path,
            )
        ):
            pose = _build_pose(pose_row, context.mat_type)
            label_token = _sanitize_name(str(row_label))
            focus = _label_is_in_focus(
                str(row_label),
                focus_start_label=focus_start_label,
                focus_end_label=focus_end_label,
            )

            if create_cartesian_markers:
                marker_name = f"{safe_prefix}_C_{index:0{target_index_width}d}_{label_token}"
                _delete_item_if_exists(
                    context.rdk,
                    marker_name,
                    context.api["ITEM_TYPE_TARGET"],
                )
                marker = context.rdk.AddTarget(marker_name, context.frame, context.robot)
                marker.setRobot(context.robot)
                marker.setAsCartesianTarget()
                marker.setPose(pose)
                _try_set_color(
                    marker,
                    [1.0, 0.2, 0.2, 1.0] if focus else [0.2, 0.7, 1.0, 1.0],
                )
                marker_count += 1

            if program is not None:
                target_name = f"{safe_prefix}_J_{index:0{target_index_width}d}_{label_token}"
                _delete_item_if_exists(
                    context.rdk,
                    target_name,
                    context.api["ITEM_TYPE_TARGET"],
                )
                target = context.rdk.AddTarget(target_name, context.frame, context.robot)
                target.setRobot(context.robot)
                target_move_type = (
                    "MoveL" if bool(inserted_flag) or previous_inserted else settings.move_type
                )
                _apply_selected_target(
                    target,
                    pose=pose,
                    joints=tuple(float(value) for value in selected_entry.joints),
                    move_type=target_move_type,
                )
                _append_move_instruction(program, target, target_move_type)
                program_target_count += 1

            previous_inserted = bool(inserted_flag)

        if (
            program is not None
            and settings.hide_targets_after_generation
            and hasattr(program, "ShowTargets")
        ):
            program.ShowTargets(False)
    finally:
        context.rdk.Render(True)
        if context.original_joints:
            context.robot.setJoints(list(context.original_joints))

    return ProfileResultImportSummary(
        program_name=str(program.Name()) if program is not None else None,
        marker_count=marker_count,
        program_target_count=program_target_count,
        robot_name=resolved_robot_name,
        frame_name=resolved_frame_name,
        prefix=safe_prefix,
    )


def _metadata_string(payload: dict[str, Any], key: str, *, default: str) -> str:
    value = payload.get(key)
    if value is None:
        return default
    return str(value)


def _delete_items_with_prefix(rdk, *, item_type: int, prefix: str) -> None:
    try:
        item_names = rdk.ItemList(item_type, True)
    except Exception:
        return

    for item_name in item_names:
        if not isinstance(item_name, str):
            continue
        if not item_name.startswith(prefix):
            continue
        _delete_item_if_exists(rdk, item_name, item_type)


def _delete_item_if_exists(rdk, item_name: str, item_type: int) -> None:
    item = rdk.Item(item_name, item_type)
    if item.Valid():
        item.Delete()


def _try_set_color(target, rgba: list[float]) -> None:
    if not hasattr(target, "setColor"):
        return
    try:
        target.setColor(rgba)
    except Exception:
        pass


def _sanitize_name(value: str) -> str:
    sanitized = re.sub(r"[^0-9A-Za-z_]+", "_", str(value).strip())
    sanitized = re.sub(r"_+", "_", sanitized).strip("_")
    return sanitized or "item"


def _label_is_in_focus(
    label: str,
    *,
    focus_start_label: str | None,
    focus_end_label: str | None,
) -> bool:
    if focus_start_label is None or focus_end_label is None:
        return False
    label_number = _try_parse_float(label)
    start_number = _try_parse_float(focus_start_label)
    end_number = _try_parse_float(focus_end_label)
    if label_number is not None and start_number is not None and end_number is not None:
        lower = min(start_number, end_number)
        upper = max(start_number, end_number)
        return lower <= label_number <= upper
    return str(label) in {str(focus_start_label), str(focus_end_label)}


def _try_parse_float(value: str) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
