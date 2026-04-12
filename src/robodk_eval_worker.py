from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import src.global_search as global_search_module
import src.ik_collection as ik_collection_module
import src.local_repair as local_repair_module
import src.path_optimizer as path_optimizer_module
from src.collab_models import (
    EvaluationBatchRequest,
    EvaluationBatchResult,
    FailedSegment,
    ProfileEvaluationRequest,
    ProfileEvaluationResult,
    RemoteSearchRequest,
    SelectedPathEntry,
    load_json_file,
    write_json_file,
)
from src.geometry import _normalize_angle_range, _trim_joint_vector
from src.motion_settings import RoboDKMotionSettings, build_motion_settings_from_dict
from src.robodk_program import (
    _append_move_instruction,
    _apply_motion_settings,
    _apply_selected_target,
    _build_program_waypoints,
    _delete_stale_bridge_targets,
    _ensure_final_path_is_valid_or_raise,
    _import_robodk_api,
    _require_item,
    _write_optimized_pose_csv,
)
from src.runtime_profiler import (
    format_runtime_profile,
    profile_runtime_section,
    reset_runtime_profile,
    runtime_profile_snapshot,
)


@dataclass
class LiveStationContext:
    api: dict[str, object]
    rdk: object
    robot: object
    frame: object
    mat_type: object
    original_joints: tuple[float, ...]
    lower_limits: tuple[float, ...]
    upper_limits: tuple[float, ...]
    joint_count: int
    tool_pose: object
    reference_pose: object
    robot_name: str
    frame_name: str


_PROFILE_HOOKS_INSTALLED = False


def _wrap_profiled(section_name: str, function: Callable[..., Any]) -> Callable[..., Any]:
    def wrapped(*args, **kwargs):
        with profile_runtime_section(section_name):
            return function(*args, **kwargs)

    wrapped.__name__ = getattr(function, "__name__", "wrapped")
    wrapped.__doc__ = getattr(function, "__doc__")
    wrapped.__wrapped__ = function
    return wrapped


def install_runtime_profile_hooks() -> None:
    global _PROFILE_HOOKS_INSTALLED
    if _PROFILE_HOOKS_INSTALLED:
        return

    path_optimizer_module._optimize_joint_path = _wrap_profiled(
        "dp_path_selection",
        path_optimizer_module._optimize_joint_path,
    )
    local_repair_module._refine_path_with_frame_a_origin_profile = _wrap_profiled(
        "window_repair",
        local_repair_module._refine_path_with_frame_a_origin_profile,
    )
    local_repair_module._attempt_inserted_transition_repair = _wrap_profiled(
        "inserted_repair",
        local_repair_module._attempt_inserted_transition_repair,
    )
    path_optimizer_module._joint_limit_penalty = _wrap_profiled(
        "config_singularity",
        path_optimizer_module._joint_limit_penalty,
    )
    path_optimizer_module._singularity_penalty = _wrap_profiled(
        "config_singularity",
        path_optimizer_module._singularity_penalty,
    )
    ik_collection_module._joint_limit_penalty = path_optimizer_module._joint_limit_penalty
    ik_collection_module._singularity_penalty = path_optimizer_module._singularity_penalty
    ik_collection_module._collect_ik_candidates = _wrap_profiled(
        "ik_collection",
        ik_collection_module._collect_ik_candidates,
    )
    global_search_module._collect_ik_candidates = ik_collection_module._collect_ik_candidates
    _PROFILE_HOOKS_INSTALLED = True


def open_live_station_context(
    *,
    robot_name: str,
    frame_name: str,
) -> LiveStationContext:
    api = _import_robodk_api()
    rdk = api["Robolink"]()
    robot = _require_item(rdk, robot_name, api["ITEM_TYPE_ROBOT"], "Robot")
    frame = _require_item(rdk, frame_name, api["ITEM_TYPE_FRAME"], "Reference frame")

    current_joints_list = robot.Joints().list()
    joint_count = len(current_joints_list)
    original_joints = _trim_joint_vector(current_joints_list, joint_count)
    lower_limits_raw, upper_limits_raw, _ = robot.JointLimits()
    lower_limits = _trim_joint_vector(lower_limits_raw.list(), joint_count)
    upper_limits = _trim_joint_vector(upper_limits_raw.list(), joint_count)

    robot.setPoseFrame(frame)
    tool_pose = robot.PoseTool()
    reference_pose = robot.PoseFrame()

    return LiveStationContext(
        api=api,
        rdk=rdk,
        robot=robot,
        frame=frame,
        mat_type=api["Mat"],
        original_joints=original_joints,
        lower_limits=lower_limits,
        upper_limits=upper_limits,
        joint_count=joint_count,
        tool_pose=tool_pose,
        reference_pose=reference_pose,
        robot_name=robot_name,
        frame_name=frame_name,
    )


def _prepare_start_joints(
    request: ProfileEvaluationRequest,
    context: LiveStationContext,
) -> tuple[float, ...]:
    if request.start_joints is None:
        return context.original_joints
    return tuple(float(value) for value in request.start_joints[: context.joint_count])


def materialize_program(
    context: LiveStationContext,
    request: ProfileEvaluationRequest,
    search_result,
    settings: RoboDKMotionSettings,
) -> str:
    program_name = request.program_name or f"Validation_{request.request_id}"
    if request.optimized_csv_path:
        _write_optimized_pose_csv(Path(request.optimized_csv_path), search_result)

    existing_program = context.rdk.Item(program_name, context.api["ITEM_TYPE_PROGRAM"])
    if existing_program.Valid():
        existing_program.Delete()

    _delete_stale_bridge_targets(context.rdk, context.api["ITEM_TYPE_TARGET"])
    program = context.rdk.AddProgram(program_name, context.robot)
    program.setRobot(context.robot)
    if hasattr(program, "setPoseFrame"):
        program.setPoseFrame(context.frame)
    if hasattr(program, "setPoseTool"):
        program.setPoseTool(context.robot.PoseTool())
    _apply_motion_settings(program, settings)

    for waypoint in _build_program_waypoints(search_result, motion_settings=settings):
        existing_target = context.rdk.Item(waypoint.name, context.api["ITEM_TYPE_TARGET"])
        if existing_target.Valid():
            existing_target.Delete()

        target = context.rdk.AddTarget(waypoint.name, context.frame, context.robot)
        target.setRobot(context.robot)
        _apply_selected_target(
            target,
            pose=waypoint.pose,
            joints=waypoint.joints,
            move_type=waypoint.move_type,
        )
        _append_move_instruction(program, target, waypoint.move_type)

    return str(program.Name())


def _collect_failed_segments(
    search_result,
    bridge_trigger_joint_delta_deg: float,
) -> tuple[FailedSegment, ...]:
    return tuple(
        FailedSegment(
            segment_index=int(segment_index),
            left_label=str(search_result.row_labels[segment_index]),
            right_label=str(search_result.row_labels[segment_index + 1]),
            config_changed=bool(config_changed),
            max_joint_delta_deg=float(max_joint_delta),
            mean_joint_delta_deg=float(mean_joint_delta),
        )
        for segment_index, config_changed, max_joint_delta, mean_joint_delta in local_repair_module._collect_problem_segments(
            search_result.selected_path,
            bridge_trigger_joint_delta_deg=bridge_trigger_joint_delta_deg,
        )
    )


def _collect_ik_empty_rows(search_result) -> tuple[str, ...]:
    return tuple(
        str(search_result.row_labels[index])
        for index, layer in enumerate(search_result.ik_layers)
        if not layer.candidates
    )


def _search_result_to_profile_result(
    *,
    request: ProfileEvaluationRequest,
    search_result,
    settings: RoboDKMotionSettings,
    elapsed_seconds: float,
    diagnostics: str | None,
    metadata: dict[str, Any] | None = None,
    error_message: str | None = None,
) -> ProfileEvaluationResult:
    total_candidates = sum(len(layer.candidates) for layer in search_result.ik_layers)
    failed_segments = _collect_failed_segments(
        search_result,
        bridge_trigger_joint_delta_deg=settings.bridge_trigger_joint_delta_deg,
    )
    result_metadata = dict(request.metadata)
    result_metadata.update(metadata or {})
    is_valid = error_message is None and diagnostics is None
    return ProfileEvaluationResult(
        request_id=request.request_id,
        status="valid" if is_valid else "invalid",
        timing_seconds=elapsed_seconds,
        motion_settings=dict(request.motion_settings),
        total_candidates=total_candidates,
        invalid_row_count=int(search_result.invalid_row_count),
        ik_empty_row_count=int(search_result.ik_empty_row_count),
        config_switches=int(search_result.config_switches),
        bridge_like_segments=int(search_result.bridge_like_segments),
        worst_joint_step_deg=float(search_result.worst_joint_step_deg),
        mean_joint_step_deg=float(search_result.mean_joint_step_deg),
        total_cost=float(search_result.total_cost),
        row_labels=tuple(str(label) for label in search_result.row_labels),
        inserted_flags=tuple(bool(flag) for flag in search_result.inserted_flags),
        frame_a_origin_yz_profile_mm=tuple(
            (float(dy_mm), float(dz_mm))
            for dy_mm, dz_mm in search_result.frame_a_origin_yz_profile_mm
        ),
        selected_path=tuple(
            SelectedPathEntry(
                joints=tuple(float(value) for value in candidate.joints),
                config_flags=tuple(int(value) for value in candidate.config_flags),
            )
            for candidate in search_result.selected_path
        ),
        failing_segments=failed_segments,
        ik_empty_rows=_collect_ik_empty_rows(search_result),
        focus_report=local_repair_module._format_focus_segment_report(search_result),
        diagnostics=diagnostics,
        error_message=error_message,
        profiling=runtime_profile_snapshot(),
        metadata=result_metadata,
        pose_rows=tuple(dict(row) for row in search_result.pose_rows)
        if request.include_pose_rows_in_result
        else None,
    )


def evaluate_request(
    request: ProfileEvaluationRequest,
    context: LiveStationContext,
) -> tuple[ProfileEvaluationResult, Any]:
    install_runtime_profile_hooks()
    reset_runtime_profile()
    start_time = time.perf_counter()

    settings = build_motion_settings_from_dict(request.motion_settings)
    optimizer_settings = path_optimizer_module._build_optimizer_settings(context.joint_count, settings)
    a1_lower_deg, a1_upper_deg = _normalize_angle_range(settings.a1_min_deg, settings.a1_max_deg)
    start_joints = _prepare_start_joints(request, context)

    context.robot.setJoints(list(start_joints or context.original_joints))

    if request.strategy == "full_search":
        search_result = global_search_module._search_best_exact_pose_path(
            request.reference_pose_rows,
            robot=context.robot,
            mat_type=context.mat_type,
            move_type=settings.move_type,
            start_joints=start_joints,
            tool_pose=context.tool_pose,
            reference_pose=context.reference_pose,
            joint_count=context.joint_count,
            motion_settings=settings,
            optimizer_settings=optimizer_settings,
            a1_lower_deg=a1_lower_deg,
            a1_upper_deg=a1_upper_deg,
            a2_max_deg=settings.a2_max_deg,
            joint_constraint_tolerance_deg=settings.joint_constraint_tolerance_deg,
        )
    elif request.strategy == "exact_profile":
        seed_joints = ik_collection_module._build_seed_joint_strategies(
            robot=context.robot,
            lower_limits=context.lower_limits,
            upper_limits=context.upper_limits,
            joint_count=context.joint_count,
        )
        search_result = global_search_module._evaluate_frame_a_origin_profile(
            request.reference_pose_rows,
            frame_a_origin_yz_profile_mm=request.frame_a_origin_yz_profile_mm,
            row_labels=request.row_labels,
            inserted_flags=request.inserted_flags,
            robot=context.robot,
            mat_type=context.mat_type,
            move_type=settings.move_type,
            start_joints=start_joints,
            tool_pose=context.tool_pose,
            reference_pose=context.reference_pose,
            joint_count=context.joint_count,
            optimizer_settings=optimizer_settings,
            a1_lower_deg=a1_lower_deg,
            a1_upper_deg=a1_upper_deg,
            a2_max_deg=settings.a2_max_deg,
            joint_constraint_tolerance_deg=settings.joint_constraint_tolerance_deg,
            seed_joints=seed_joints,
            lower_limits=context.lower_limits,
            upper_limits=context.upper_limits,
            bridge_trigger_joint_delta_deg=settings.bridge_trigger_joint_delta_deg,
        )
    else:
        raise ValueError(f"Unsupported evaluation strategy: {request.strategy}")

    if request.run_window_repair:
        search_result = local_repair_module._refine_path_with_frame_a_origin_profile(
            search_result,
            robot=context.robot,
            mat_type=context.mat_type,
            move_type=settings.move_type,
            start_joints=start_joints,
            tool_pose=context.tool_pose,
            reference_pose=context.reference_pose,
            joint_count=context.joint_count,
            motion_settings=settings,
            optimizer_settings=optimizer_settings,
            lower_limits=context.lower_limits,
            upper_limits=context.upper_limits,
            a1_lower_deg=a1_lower_deg,
            a1_upper_deg=a1_upper_deg,
            a2_max_deg=settings.a2_max_deg,
            joint_constraint_tolerance_deg=settings.joint_constraint_tolerance_deg,
        )

    if request.run_inserted_repair:
        inserted_result = local_repair_module._attempt_inserted_transition_repair(
            search_result,
            robot=context.robot,
            mat_type=context.mat_type,
            move_type=settings.move_type,
            start_joints=start_joints,
            tool_pose=context.tool_pose,
            reference_pose=context.reference_pose,
            joint_count=context.joint_count,
            motion_settings=settings,
            optimizer_settings=optimizer_settings,
            lower_limits=context.lower_limits,
            upper_limits=context.upper_limits,
            a1_lower_deg=a1_lower_deg,
            a1_upper_deg=a1_upper_deg,
            a2_max_deg=settings.a2_max_deg,
            joint_constraint_tolerance_deg=settings.joint_constraint_tolerance_deg,
        )
        if inserted_result is not None:
            search_result = inserted_result

    validation_message = None
    extra_metadata: dict[str, Any] = {}
    try:
        _ensure_final_path_is_valid_or_raise(search_result, settings=settings)
        if request.create_program:
            extra_metadata["program_name"] = materialize_program(context, request, search_result, settings)
    except RuntimeError as exc:
        validation_message = str(exc)

    elapsed_seconds = time.perf_counter() - start_time
    result = _search_result_to_profile_result(
        request=request,
        search_result=search_result,
        settings=settings,
        elapsed_seconds=elapsed_seconds,
        diagnostics=validation_message,
        metadata=extra_metadata,
    )
    print(
        f"[{request.request_id}] status={result.status}, "
        f"ik_empty_rows={result.ik_empty_row_count}, "
        f"config_switches={result.config_switches}, "
        f"bridge_like_segments={result.bridge_like_segments}, "
        f"worst_joint_step={result.worst_joint_step_deg:.3f} deg, "
        f"time={result.timing_seconds:.3f}s"
    )
    return result, search_result


def _load_batch_request(path: str | Path) -> EvaluationBatchRequest:
    payload = load_json_file(path)
    if "evaluations" in payload:
        return EvaluationBatchRequest.from_dict(payload)
    if "base_request" in payload:
        remote_request = RemoteSearchRequest.from_dict(payload)
        return EvaluationBatchRequest(evaluations=(remote_request.base_request,))
    return EvaluationBatchRequest(
        evaluations=(ProfileEvaluationRequest.from_dict(payload),)
    )


def evaluate_batch_request(
    batch_request: EvaluationBatchRequest,
) -> EvaluationBatchResult:
    if not batch_request.evaluations:
        raise ValueError("No evaluation requests found.")

    first_request = batch_request.evaluations[0]
    context = open_live_station_context(
        robot_name=first_request.robot_name,
        frame_name=first_request.frame_name,
    )
    context.rdk.Render(False)
    try:
        results = tuple(evaluate_request(request, context)[0] for request in batch_request.evaluations)
    finally:
        context.rdk.Render(True)
        if context.original_joints:
            context.robot.setJoints(list(context.original_joints))

    return EvaluationBatchResult(results=results)


def evaluate_batch_file(request_path: str | Path, result_path: str | Path) -> Path:
    batch_request = _load_batch_request(request_path)
    batch_result = evaluate_batch_request(batch_request)
    output_path = write_json_file(result_path, batch_result.to_dict())
    if batch_result.results:
        print(format_runtime_profile(batch_result.results[0].profiling))
    print(f"Wrote evaluation result batch: {output_path}")
    return output_path


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate one or more Frame-A Y/Z profile requests against the live RoboDK station."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command_name in ("eval", "eval-batch"):
        command_parser = subparsers.add_parser(
            command_name,
            help="Evaluate a request JSON or batch request JSON.",
        )
        command_parser.add_argument("--request", required=True, help="Path to a JSON request or batch request.")
        command_parser.add_argument("--result", required=True, help="Path to write the JSON result batch.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    try:
        evaluate_batch_file(args.request, args.result)
    except Exception as exc:
        print(f"Error: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
