from __future__ import annotations

import csv
import re
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from src.core.collab_models import (
    EvaluationBatchResult,
    ProfileEvaluationRequest,
    ProfileEvaluationResult,
    SelectedPathEntry,
    load_json_file,
)
from src.core.geometry import _build_pose
from src.core.motion_settings import build_motion_settings_from_dict
from src.core.pose_csv import REQUIRED_COLUMNS
from src.robodk_runtime.eval_worker import open_live_station_context
from src.robodk_runtime.program import (
    _append_move_instruction,
    _apply_motion_settings,
    _apply_selected_target,
    _is_joint_space_bridge_label,
    _set_item_visible,
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


def profile_result_from_handoff_payload(
    handoff_payload: dict[str, Any],
) -> ProfileEvaluationResult:
    """Rebuild the server-selected result carried by an online handoff package.

    The receiver should not run IK/search again: the server already selected a
    concrete pose row sequence and joint path. This helper turns that handoff
    data back into the normal ProfileEvaluationResult shape used by RoboDK
    import utilities.
    """

    receiver_request = ProfileEvaluationRequest.from_dict(
        dict(handoff_payload["receiver_request"])
    )
    selection = _coerce_handoff_object(handoff_payload.get("selection", {}), field_name="selection")
    selected_profile = _coerce_handoff_object(
        handoff_payload.get("selected_profile", {}),
        field_name="selected_profile",
    )
    selected_joint_path_payload = _coerce_handoff_rows(
        handoff_payload.get("selected_joint_path", ()),
        field_name="selected_joint_path",
    )
    pose_rows_payload = handoff_payload.get("optimized_pose_rows")
    if pose_rows_payload is None:
        pose_rows_payload = handoff_payload.get("reference_pose_rows")

    pose_rows = _coerce_pose_rows(pose_rows_payload or ())
    if not pose_rows:
        raise ValueError("Handoff package does not include optimized or reference pose rows.")

    row_label_payload = _selected_profile_value(
        selected_profile,
        "row_labels",
        [item.get("row_label", index) for index, item in enumerate(selected_joint_path_payload)],
    )
    row_labels = tuple(
        str(label)
        for label in row_label_payload
    )
    inserted_flag_payload = _selected_profile_value(
        selected_profile,
        "inserted_flags",
        [
            item.get("inserted_transition_point", False)
            for item in selected_joint_path_payload
        ],
    )
    inserted_flags = tuple(
        bool(flag)
        for flag in inserted_flag_payload
    )
    origin_profile_payload = _selected_profile_value(
        selected_profile,
        "frame_a_origin_yz_profile_mm",
        receiver_request.frame_a_origin_yz_profile_mm,
    )
    frame_a_origin_yz_profile_mm = tuple(
        (float(item[0]), float(item[1]))
        for item in origin_profile_payload
    )
    selected_path = tuple(
        SelectedPathEntry(
            joints=tuple(
                float(value)
                for value in item.get("joints_deg", item.get("joints", ()))
            ),
            config_flags=tuple(int(value) for value in item.get("config_flags", ())),
        )
        for item in selected_joint_path_payload
    )

    expected_count = len(row_labels)
    for name, count in (
        ("pose_rows", len(pose_rows)),
        ("inserted_flags", len(inserted_flags)),
        ("frame_a_origin_yz_profile_mm", len(frame_a_origin_yz_profile_mm)),
        ("selected_path", len(selected_path)),
    ):
        if count != expected_count:
            raise ValueError(
                "Handoff package has inconsistent selected path lengths: "
                f"row_labels={expected_count}, {name}={count}."
            )

    metadata = dict(receiver_request.metadata)
    metadata.update(
        {
            "handoff_run_id": handoff_payload.get("run_id"),
            "handoff_package_kind": handoff_payload.get("package_kind"),
            "receiver_materialization_mode": "direct_handoff_import",
            "source_request_id": selection.get("request_id", metadata.get("source_request_id")),
            "server_selection": selection,
        }
    )

    result = ProfileEvaluationResult(
        request_id=str(selection.get("request_id", receiver_request.request_id)),
        status=str(selection.get("status", "valid")),
        timing_seconds=0.0,
        motion_settings=dict(receiver_request.motion_settings),
        total_candidates=int(selection.get("total_candidates", 0)),
        invalid_row_count=int(selection.get("invalid_row_count", 0)),
        ik_empty_row_count=int(selection.get("ik_empty_row_count", 0)),
        config_switches=int(selection.get("config_switches", 0)),
        bridge_like_segments=int(selection.get("bridge_like_segments", 0)),
        big_circle_step_count=int(selection.get("big_circle_step_count", 0)),
        branch_flip_ratio=float(selection.get("branch_flip_ratio", 0.0)),
        violent_branch_segments=tuple(
            (
                dict(item)
                if isinstance(item, dict)
                else {"value": item}
            )
            for item in selection.get("violent_branch_segments", ())
        ),
        worst_joint_step_deg=float(selection.get("worst_joint_step_deg", 0.0)),
        mean_joint_step_deg=float(selection.get("mean_joint_step_deg", 0.0)),
        total_cost=float(selection.get("total_cost", 0.0)),
        row_labels=row_labels,
        inserted_flags=inserted_flags,
        frame_a_origin_yz_profile_mm=frame_a_origin_yz_profile_mm,
        selected_path=selected_path,
        failing_segments=(),
        ik_empty_rows=(),
        focus_report="",
        diagnostics=None,
        error_message=None,
        profiling={},
        gate_tier=str(selection.get("gate_tier", "diagnostic")),
        block_reasons=tuple(
            (
                dict(item)
                if isinstance(item, dict)
                else {"code": "legacy_reason", "message": str(item)}
            )
            for item in selection.get("block_reasons", ())
        ),
        metadata=metadata,
        pose_rows=pose_rows,
    )
    _validate_closed_winding_result(result)
    return result


def _validate_closed_winding_result(result: ProfileEvaluationResult) -> None:
    if result.pose_rows is None or not _pose_rows_form_closed_path(result.pose_rows):
        return
    if not result.selected_path:
        raise ValueError("Closed winding handoff has no selected joint path.")
    if len(result.selected_path) != len(result.pose_rows):
        raise ValueError(
            "Closed winding handoff selected_path length does not match pose rows: "
            f"{len(result.selected_path)} vs {len(result.pose_rows)}."
        )

    settings = build_motion_settings_from_dict(result.motion_settings)
    first = result.selected_path[0]
    terminal = result.selected_path[-1]
    tolerance = float(settings.closed_path_joint6_turn_tolerance_deg)
    _validate_closed_winding_terminal_joints(
        first.joints,
        terminal.joints,
        required_turns=int(settings.closed_path_joint6_turns),
        tolerance_deg=tolerance,
    )
    start_flags = first.config_flags
    locked_indices = (
        tuple(range(len(start_flags)))
        if bool(settings.closed_path_single_config)
        else tuple(int(value) for value in settings.closed_path_locked_config_indices)
    )
    for index, entry in enumerate(result.selected_path):
        for config_index in locked_indices:
            if config_index >= len(start_flags) or config_index >= len(entry.config_flags):
                label = result.row_labels[index] if index < len(result.row_labels) else index
                raise ValueError(
                    "Closed winding handoff is missing a locked config flag at "
                    f"row {label}, config index {config_index}."
                )
            if entry.config_flags[config_index] != start_flags[config_index]:
                label = result.row_labels[index] if index < len(result.row_labels) else index
                raise ValueError(
                    "Closed winding handoff changes a locked config family flag at "
                    f"row {label}, config index {config_index}: "
                    f"{entry.config_flags} != {start_flags}."
                )


def _pose_rows_form_closed_path(pose_rows: tuple[dict[str, float], ...]) -> bool:
    if len(pose_rows) < 2:
        return False
    first_row = pose_rows[0]
    terminal_row = pose_rows[-1]
    for column in REQUIRED_COLUMNS:
        if column not in first_row or column not in terminal_row:
            return False
        if abs(float(first_row[column]) - float(terminal_row[column])) > 1e-6:
            return False
    return True


def _validate_closed_winding_terminal_joints(
    first_joints: tuple[float, ...],
    terminal_joints: tuple[float, ...],
    *,
    required_turns: int,
    tolerance_deg: float,
) -> None:
    if len(first_joints) < 6 or len(terminal_joints) < 6:
        raise ValueError("Closed winding handoff requires six joint values per target.")
    for joint_index in range(5):
        delta = float(terminal_joints[joint_index]) - float(first_joints[joint_index])
        if abs(delta) > tolerance_deg:
            raise ValueError(
                "Closed winding terminal violates I1-I5 hard constraint: "
                f"I{joint_index + 1} delta is {delta:.6f} deg."
            )

    joint6_delta = float(terminal_joints[5]) - float(first_joints[5])
    nearest_turns = round(joint6_delta / 360.0)
    residual = joint6_delta - 360.0 * nearest_turns
    if abs(residual) > tolerance_deg or abs(int(nearest_turns)) != int(required_turns):
        raise ValueError(
            "Closed winding terminal violates I6 full-turn hard constraint: "
            f"I6 delta is {joint6_delta:.6f} deg, expected +/-{360 * int(required_turns)} deg."
        )


def write_optimized_pose_csv_from_result(
    csv_path: str | Path,
    result: ProfileEvaluationResult,
) -> Path:
    if result.pose_rows is None:
        raise ValueError("Cannot write optimized pose CSV because result.pose_rows is missing.")
    if len(result.pose_rows) != len(result.row_labels):
        raise ValueError(
            "Cannot write optimized pose CSV because pose_rows and row_labels lengths differ: "
            f"{len(result.pose_rows)} vs {len(result.row_labels)}."
        )

    output_path = Path(csv_path).with_name(f"{Path(csv_path).stem}_optimized.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "row_label",
        "inserted_transition_point",
        "frame_a_origin_dy_mm",
        "frame_a_origin_dz_mm",
        *REQUIRED_COLUMNS,
    ]
    with output_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row_label, inserted_flag, (dy_mm, dz_mm), pose_row in zip(
            result.row_labels,
            result.inserted_flags,
            result.frame_a_origin_yz_profile_mm,
            result.pose_rows,
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
    return output_path


def import_profile_result_to_robodk(
    result: ProfileEvaluationResult,
    *,
    robot_name: str | None = None,
    frame_name: str | None = None,
    program_name: str | None = None,
    prefix: str = "YZNS",
    clear_prefix: bool = True,
    create_program: bool = True,
    create_cartesian_markers: bool = False,
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
                program_token = _sanitize_name(resolved_program_name)
                target_name = (
                    f"{safe_prefix}_{program_token}_J_{index:0{target_index_width}d}_{label_token}"
                )
                _delete_item_if_exists(
                    context.rdk,
                    target_name,
                    context.api["ITEM_TYPE_TARGET"],
                )
                target = context.rdk.AddTarget(target_name, context.frame, context.robot)
                target.setRobot(context.robot)
                target_move_type = (
                    "MoveJ"
                    if _is_joint_space_bridge_label(row_label)
                    else "MoveL"
                    if bool(inserted_flag) or previous_inserted
                    else settings.move_type
                )
                _apply_selected_target(
                    target,
                    pose=pose,
                    joints=tuple(float(value) for value in selected_entry.joints),
                    move_type=target_move_type,
                )
                _append_move_instruction(program, target, target_move_type)
                if settings.hide_targets_after_generation:
                    _set_item_visible(target, False)
                program_target_count += 1

            previous_inserted = bool(inserted_flag) and not _is_joint_space_bridge_label(row_label)
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


def _coerce_pose_rows(pose_rows: Any) -> tuple[dict[str, float], ...]:
    normalized_rows: list[dict[str, float]] = []
    for index, pose_row in enumerate(pose_rows):
        normalized_row: dict[str, float] = {}
        try:
            raw_row = dict(pose_row)
        except Exception as exc:
            raise ValueError(f"Invalid pose row at index {index}: {pose_row!r}") from exc
        for key, value in raw_row.items():
            if value is None:
                continue
            try:
                normalized_row[str(key)] = float(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Invalid numeric pose value at row {index}, field {key!r}: {value!r}"
                ) from exc
        normalized_rows.append(normalized_row)
    return tuple(normalized_rows)


def _selected_profile_value(
    selected_profile: dict[str, Any],
    key: str,
    default: Any,
) -> Any:
    value = selected_profile.get(key)
    return default if value is None else value


def _coerce_handoff_object(payload: Any, *, field_name: str) -> dict[str, Any]:
    if isinstance(payload, dict):
        return dict(payload)
    raise ValueError(f"Handoff package field '{field_name}' must be an object.")


def _coerce_handoff_rows(payload: Any, *, field_name: str) -> tuple[dict[str, Any], ...]:
    if payload in (None, ()):
        return ()
    if not isinstance(payload, (list, tuple)):
        raise ValueError(f"Handoff package field '{field_name}' must be an array.")
    rows: list[dict[str, Any]] = []
    for index, row in enumerate(payload):
        if not isinstance(row, dict):
            raise ValueError(
                f"Handoff package field '{field_name}' row {index} must be an object."
            )
        rows.append(dict(row))
    return tuple(rows)


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
