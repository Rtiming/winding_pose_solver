from __future__ import annotations

import argparse
import atexit
import multiprocessing
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

import src.search.global_search as global_search_module
import src.search.ik_collection as ik_collection_module
import src.search.local_repair as local_repair_module
import src.search.path_optimizer as path_optimizer_module
from src.core.collab_models import (
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
from src.core.geometry import _normalize_angle_range, _trim_joint_vector
from src.core.motion_settings import RoboDKMotionSettings, build_motion_settings_from_dict
from src.core.robot_interface import build_robot_interface
from src.robodk_runtime.program import (
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
from src.runtime.profiler import (
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


@dataclass(frozen=True)
class PreparedEvaluationResources:
    settings: RoboDKMotionSettings
    optimizer_settings: Any
    a1_lower_deg: float
    a1_upper_deg: float
    start_joints: tuple[float, ...]
    robot_interface: Any
    seed_joints: tuple[tuple[float, ...], ...]


_PROFILE_HOOKS_INSTALLED = False
_SIX_AXIS_IK_CALIBRATION_CHECK_DONE = False
_OFFLINE_BATCH_EXECUTOR: ProcessPoolExecutor | None = None
_OFFLINE_BATCH_EXECUTOR_WORKERS: int | None = None
_OFFLINE_BATCH_WORKER_CONTEXTS: dict[tuple[str, str], LiveStationContext] = {}


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
    local_repair_module._collect_ik_candidates = ik_collection_module._collect_ik_candidates
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


def _pose_to_numpy(pose: Any) -> np.ndarray:
    return np.array(pose.rows, dtype=float)


def _report_six_axis_ik_calibration_status(context: LiveStationContext) -> None:
    global _SIX_AXIS_IK_CALIBRATION_CHECK_DONE
    if _SIX_AXIS_IK_CALIBRATION_CHECK_DONE:
        return

    from src.six_axis_ik import config as ik_config

    live_tool = _pose_to_numpy(context.tool_pose)
    live_frame = _pose_to_numpy(context.reference_pose)
    configured_tool = ik_config.get_configured_tool_pose()
    configured_frame = ik_config.get_configured_frame_pose()

    tool_diff = float(np.max(np.abs(live_tool - configured_tool)))
    frame_diff = float(np.max(np.abs(live_frame - configured_frame)))

    if tool_diff <= 1e-9 and frame_diff <= 1e-9:
        print(
            "[IK backend] SixAxisIK calibration matches the live RoboDK Tool/Frame "
            f"(tool_diff={tool_diff:.3e}, frame_diff={frame_diff:.3e})."
        )
    else:
        print(
            "[IK backend] WARNING: SixAxisIK calibration differs from the live RoboDK Tool/Frame. "
            f"(tool_diff={tool_diff:.6g}, frame_diff={frame_diff:.6g})"
        )
        print(
            "  Configured frame xyz/rzyx: "
            f"{ik_config.CONFIGURED_FRAME_POSE_XYZ_RZYX_MM_DEG}"
        )
        print(
            "  Configured tool xyz/rzyx: "
            f"{ik_config.CONFIGURED_TOOL_POSE_XYZ_RZYX_MM_DEG}"
        )

    _SIX_AXIS_IK_CALIBRATION_CHECK_DONE = True


def _prepare_start_joints(
    request: ProfileEvaluationRequest,
    context: LiveStationContext,
) -> tuple[float, ...]:
    if request.start_joints is None:
        return context.original_joints
    return tuple(float(value) for value in request.start_joints[: context.joint_count])


def _prepare_evaluation_resources(
    request: ProfileEvaluationRequest,
    context: LiveStationContext,
) -> PreparedEvaluationResources:
    settings = build_motion_settings_from_dict(request.motion_settings)
    optimizer_settings = path_optimizer_module._build_optimizer_settings(
        context.joint_count,
        settings,
    )
    a1_lower_deg, a1_upper_deg = _normalize_angle_range(
        settings.a1_min_deg,
        settings.a1_max_deg,
    )
    start_joints = _prepare_start_joints(request, context)

    robot_interface = build_robot_interface(
        ik_backend=getattr(settings, "ik_backend", "robodk"),
        robodk_robot=context.robot,
    )
    if getattr(settings, "ik_backend", "robodk") == "six_axis_ik" and context.rdk is not None:
        _report_six_axis_ik_calibration_status(context)

    if getattr(robot_interface, "ik_seed_invariant", False):
        seed_joints: tuple[tuple[float, ...], ...] = ()
    else:
        seed_joints = ik_collection_module._build_seed_joint_strategies(
            robot=robot_interface,
            lower_limits=context.lower_limits,
            upper_limits=context.upper_limits,
            joint_count=context.joint_count,
        )

    return PreparedEvaluationResources(
        settings=settings,
        optimizer_settings=optimizer_settings,
        a1_lower_deg=a1_lower_deg,
        a1_upper_deg=a1_upper_deg,
        start_joints=start_joints,
        robot_interface=robot_interface,
        seed_joints=seed_joints,
    )


def _profile_changed_row_indices(
    baseline_profile: Sequence[tuple[float, float]],
    candidate_profile: Sequence[tuple[float, float]],
) -> tuple[int, ...]:
    if len(baseline_profile) != len(candidate_profile):
        return tuple(range(len(candidate_profile)))

    changed_indices: list[int] = []
    for row_index, (baseline_offset, candidate_offset) in enumerate(
        zip(baseline_profile, candidate_profile)
    ):
        if (
            abs(float(baseline_offset[0]) - float(candidate_offset[0])) > 1e-9
            or abs(float(baseline_offset[1]) - float(candidate_offset[1])) > 1e-9
        ):
            changed_indices.append(row_index)
    return tuple(changed_indices)


def _request_profile_cache_key(
    request: ProfileEvaluationRequest,
) -> tuple[tuple[float, float], ...]:
    return tuple(
        (round(float(dy_mm), 6), round(float(dz_mm), 6))
        for dy_mm, dz_mm in request.frame_a_origin_yz_profile_mm
    )


def _evaluate_exact_profile_search(
    request: ProfileEvaluationRequest,
    context: LiveStationContext,
    resources: PreparedEvaluationResources,
    *,
    reused_search_result=None,
):
    recompute_row_indices = None
    reused_ik_layers = None
    if reused_search_result is not None:
        recompute_row_indices = _profile_changed_row_indices(
            reused_search_result.frame_a_origin_yz_profile_mm,
            request.frame_a_origin_yz_profile_mm,
        )
        reused_ik_layers = reused_search_result.ik_layers

    return global_search_module._evaluate_frame_a_origin_profile(
        request.reference_pose_rows,
        frame_a_origin_yz_profile_mm=request.frame_a_origin_yz_profile_mm,
        row_labels=request.row_labels,
        inserted_flags=request.inserted_flags,
        robot=resources.robot_interface,
        mat_type=context.mat_type,
        move_type=resources.settings.move_type,
        start_joints=resources.start_joints,
        tool_pose=context.tool_pose,
        reference_pose=context.reference_pose,
        joint_count=context.joint_count,
        optimizer_settings=resources.optimizer_settings,
        a1_lower_deg=resources.a1_lower_deg,
        a1_upper_deg=resources.a1_upper_deg,
        a2_max_deg=resources.settings.a2_max_deg,
        joint_constraint_tolerance_deg=resources.settings.joint_constraint_tolerance_deg,
        seed_joints=resources.seed_joints,
        lower_limits=context.lower_limits,
        upper_limits=context.upper_limits,
        bridge_trigger_joint_delta_deg=resources.settings.bridge_trigger_joint_delta_deg,
        reused_ik_layers=reused_ik_layers,
        recompute_row_indices=recompute_row_indices,
    )


def _apply_optional_repairs(
    request: ProfileEvaluationRequest,
    context: LiveStationContext,
    resources: PreparedEvaluationResources,
    search_result,
):
    updated_result = search_result
    if request.run_window_repair:
        updated_result = local_repair_module._refine_path_with_frame_a_origin_profile(
            updated_result,
            robot=resources.robot_interface,
            mat_type=context.mat_type,
            move_type=resources.settings.move_type,
            start_joints=resources.start_joints,
            tool_pose=context.tool_pose,
            reference_pose=context.reference_pose,
            joint_count=context.joint_count,
            motion_settings=resources.settings,
            optimizer_settings=resources.optimizer_settings,
            lower_limits=context.lower_limits,
            upper_limits=context.upper_limits,
            a1_lower_deg=resources.a1_lower_deg,
            a1_upper_deg=resources.a1_upper_deg,
            a2_max_deg=resources.settings.a2_max_deg,
            joint_constraint_tolerance_deg=resources.settings.joint_constraint_tolerance_deg,
        )

    if request.run_inserted_repair:
        inserted_result = local_repair_module._attempt_inserted_transition_repair(
            updated_result,
            robot=resources.robot_interface,
            mat_type=context.mat_type,
            move_type=resources.settings.move_type,
            start_joints=resources.start_joints,
            tool_pose=context.tool_pose,
            reference_pose=context.reference_pose,
            joint_count=context.joint_count,
            motion_settings=resources.settings,
            optimizer_settings=resources.optimizer_settings,
            lower_limits=context.lower_limits,
            upper_limits=context.upper_limits,
            a1_lower_deg=resources.a1_lower_deg,
            a1_upper_deg=resources.a1_upper_deg,
            a2_max_deg=resources.settings.a2_max_deg,
            joint_constraint_tolerance_deg=resources.settings.joint_constraint_tolerance_deg,
        )
        if inserted_result is not None:
            updated_result = inserted_result
    return updated_result


def _finalize_request_result(
    request: ProfileEvaluationRequest,
    context: LiveStationContext,
    resources: PreparedEvaluationResources,
    search_result,
    *,
    elapsed_seconds: float,
) -> ProfileEvaluationResult:
    validation_message = None
    extra_metadata: dict[str, Any] = {
        "ik_candidate_cache": ik_collection_module.ik_candidate_collection_cache_stats(),
    }
    try:
        _ensure_final_path_is_valid_or_raise(search_result, settings=resources.settings)
        if request.create_program:
            extra_metadata["program_name"] = materialize_program(
                context,
                request,
                search_result,
                resources.settings,
            )
    except RuntimeError as exc:
        validation_message = str(exc)

    return _search_result_to_profile_result(
        request=request,
        search_result=search_result,
        settings=resources.settings,
        elapsed_seconds=elapsed_seconds,
        diagnostics=validation_message,
        metadata=extra_metadata,
    )


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
    ik_collection_module.reset_ik_candidate_collection_cache()
    start_time = time.perf_counter()
    resources = _prepare_evaluation_resources(request, context)

    context.robot.setJoints(list(resources.start_joints or context.original_joints))

    if request.strategy == "full_search":
        search_result = global_search_module._search_best_exact_pose_path(
            request.reference_pose_rows,
            robot=resources.robot_interface,
            mat_type=context.mat_type,
            move_type=resources.settings.move_type,
            start_joints=resources.start_joints,
            tool_pose=context.tool_pose,
            reference_pose=context.reference_pose,
            joint_count=context.joint_count,
            motion_settings=resources.settings,
            optimizer_settings=resources.optimizer_settings,
            a1_lower_deg=resources.a1_lower_deg,
            a1_upper_deg=resources.a1_upper_deg,
            a2_max_deg=resources.settings.a2_max_deg,
            joint_constraint_tolerance_deg=resources.settings.joint_constraint_tolerance_deg,
        )
    elif request.strategy == "exact_profile":
        search_result = _evaluate_exact_profile_search(
            request,
            context,
            resources,
        )
    else:
        raise ValueError(f"Unsupported evaluation strategy: {request.strategy}")

    search_result = _apply_optional_repairs(
        request,
        context,
        resources,
        search_result,
    )

    elapsed_seconds = time.perf_counter() - start_time
    result = _finalize_request_result(
        request=request,
        context=context,
        resources=resources,
        search_result=search_result,
        elapsed_seconds=elapsed_seconds,
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


def evaluate_single_request(
    request: ProfileEvaluationRequest,
) -> tuple[ProfileEvaluationResult, Any]:
    if request.motion_settings.get("ik_backend") == "six_axis_ik" and not request.create_program:
        context = open_offline_ik_station_context(
            robot_name=request.robot_name,
            frame_name=request.frame_name,
        )
        return evaluate_request(request, context)

    context = open_live_station_context(
        robot_name=request.robot_name,
        frame_name=request.frame_name,
    )
    if context.rdk is not None:
        context.rdk.Render(False)
    try:
        return evaluate_request(request, context)
    finally:
        if context.rdk is not None:
            context.rdk.Render(True)
        if context.original_joints:
            context.robot.setJoints(list(context.original_joints))


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


def _can_run_offline(batch_request: EvaluationBatchRequest) -> bool:
    """True if all evaluation requests can run without a live RoboDK station.

    Offline mode is possible when:
    - every request uses ik_backend="six_axis_ik", AND
    - no request asks to create a RoboDK program (create_program=False).
    """
    return all(
        req.motion_settings.get("ik_backend") == "six_axis_ik" and not req.create_program
        for req in batch_request.evaluations
    )


def open_offline_ik_station_context(
    *,
    robot_name: str,
    frame_name: str,
) -> LiveStationContext:
    """Build a mock station context from SixAxisIK config — no live RoboDK connection needed.

    Used for evaluation-only requests with ik_backend="six_axis_ik". The tool pose and
    reference frame are read from the embedded SixAxisIK configuration, which is calibrated
    to match the real station. For the final program-generation step, the live station is
    still required.
    """
    from src.core.simple_mat import SimpleMat
    from src.six_axis_ik import config as ik_config
    from src.core.robot_interface import SixAxisIKRobotInterface

    model = ik_config.build_local_robot_model()

    def _np_to_mat(arr) -> Any:
        return SimpleMat(arr.tolist())

    robot_interface = SixAxisIKRobotInterface(robodk_robot=None)
    lower = tuple(float(v) for v in model.joint_min_deg)
    upper = tuple(float(v) for v in model.joint_max_deg)
    joint_count = 6

    print("[offline-ik] Running without live RoboDK station — using SixAxisIK configured values.")

    return LiveStationContext(
        api={
            "Mat": SimpleMat,
            "ITEM_TYPE_ROBOT": 1,
            "ITEM_TYPE_FRAME": 2,
            "ITEM_TYPE_TARGET": 3,
            "ITEM_TYPE_PROGRAM": 4,
        },
        rdk=None,
        robot=robot_interface,
        frame=None,
        mat_type=SimpleMat,
        original_joints=tuple([0.0] * joint_count),
        lower_limits=lower,
        upper_limits=upper,
        joint_count=joint_count,
        tool_pose=_np_to_mat(ik_config.get_configured_tool_pose()),
        reference_pose=_np_to_mat(ik_config.get_configured_frame_pose()),
        robot_name=robot_name,
        frame_name=frame_name,
    )


def _resolve_offline_batch_workers(request_count: int) -> int:
    configured_value = os.getenv("WPS_OFFLINE_BATCH_WORKERS")
    if configured_value is not None:
        try:
            parsed_value = int(configured_value)
        except ValueError:
            parsed_value = 0
        if parsed_value > 0:
            return max(1, min(request_count, parsed_value))

    cpu_count = os.cpu_count() or 1
    return max(1, min(request_count, 4, max(1, cpu_count // 2)))


def _can_run_offline_parallel(batch_request: EvaluationBatchRequest) -> bool:
    if not _can_run_offline(batch_request):
        return False
    if len(batch_request.evaluations) <= 1:
        return False
    main_file = getattr(__import__("__main__"), "__file__", None)
    if not main_file or not Path(str(main_file)).exists():
        return False

    for request in batch_request.evaluations:
        # Avoid nested process pools on shared machines. When a batch fans out
        # across processes, keep each request itself single-process.
        if int(request.motion_settings.get("local_parallel_workers", 1)) != 1:
            return False
    return True


def _get_offline_batch_executor(worker_count: int) -> ProcessPoolExecutor:
    global _OFFLINE_BATCH_EXECUTOR, _OFFLINE_BATCH_EXECUTOR_WORKERS
    if (
        _OFFLINE_BATCH_EXECUTOR is None
        or _OFFLINE_BATCH_EXECUTOR_WORKERS != worker_count
    ):
        shutdown_offline_batch_executor()
        _OFFLINE_BATCH_EXECUTOR = ProcessPoolExecutor(
            max_workers=worker_count,
            mp_context=multiprocessing.get_context("spawn"),
        )
        _OFFLINE_BATCH_EXECUTOR_WORKERS = worker_count
    return _OFFLINE_BATCH_EXECUTOR


def shutdown_offline_batch_executor() -> None:
    global _OFFLINE_BATCH_EXECUTOR, _OFFLINE_BATCH_EXECUTOR_WORKERS
    if _OFFLINE_BATCH_EXECUTOR is not None:
        _OFFLINE_BATCH_EXECUTOR.shutdown(wait=True, cancel_futures=False)
        _OFFLINE_BATCH_EXECUTOR = None
        _OFFLINE_BATCH_EXECUTOR_WORKERS = None


def _get_cached_offline_batch_worker_context(
    *,
    robot_name: str,
    frame_name: str,
) -> LiveStationContext:
    cache_key = (str(robot_name), str(frame_name))
    cached_context = _OFFLINE_BATCH_WORKER_CONTEXTS.get(cache_key)
    if cached_context is not None:
        return cached_context
    context = open_offline_ik_station_context(
        robot_name=robot_name,
        frame_name=frame_name,
    )
    _OFFLINE_BATCH_WORKER_CONTEXTS[cache_key] = context
    return context


def _evaluate_request_offline_parallel_worker(
    request_payload: dict[str, Any],
) -> dict[str, Any]:
    request = ProfileEvaluationRequest.from_dict(request_payload)
    context = _get_cached_offline_batch_worker_context(
        robot_name=request.robot_name,
        frame_name=request.frame_name,
    )
    result, _search = evaluate_request(request, context)
    return result.to_dict()


def _evaluate_batch_request_offline_parallel(
    batch_request: EvaluationBatchRequest,
) -> EvaluationBatchResult:
    worker_count = _resolve_offline_batch_workers(len(batch_request.evaluations))
    if worker_count <= 1:
        first_request = batch_request.evaluations[0]
        context = open_offline_ik_station_context(
            robot_name=first_request.robot_name,
            frame_name=first_request.frame_name,
        )
        results = tuple(
            evaluate_request(request, context)[0]
            for request in batch_request.evaluations
        )
        return EvaluationBatchResult(results=results)

    executor = _get_offline_batch_executor(worker_count)
    futures: dict[object, int] = {}
    ordered_results: list[ProfileEvaluationResult | None] = [
        None for _ in batch_request.evaluations
    ]
    for request_index, request in enumerate(batch_request.evaluations):
        future = executor.submit(
            _evaluate_request_offline_parallel_worker,
            request.to_dict(),
        )
        futures[future] = request_index

    for future in as_completed(futures):
        request_index = futures[future]
        ordered_results[request_index] = ProfileEvaluationResult.from_dict(future.result())

    return EvaluationBatchResult(
        results=tuple(result for result in ordered_results if result is not None)
    )


def _can_use_shared_exact_profile_batch(
    batch_request: EvaluationBatchRequest,
) -> bool:
    if not _can_run_offline(batch_request):
        return False
    if not batch_request.evaluations:
        return False

    first_request = batch_request.evaluations[0]
    first_reference_rows = tuple(first_request.reference_pose_rows)
    first_row_labels = tuple(first_request.row_labels)
    first_inserted_flags = tuple(first_request.inserted_flags)
    first_motion_settings = dict(first_request.motion_settings)

    return all(
        request.strategy == "exact_profile"
        and not request.create_program
        and tuple(request.reference_pose_rows) == first_reference_rows
        and tuple(request.row_labels) == first_row_labels
        and tuple(request.inserted_flags) == first_inserted_flags
        and dict(request.motion_settings) == first_motion_settings
        for request in batch_request.evaluations
    )


def _evaluate_shared_exact_profile_batch(
    batch_request: EvaluationBatchRequest,
) -> EvaluationBatchResult:
    first_request = batch_request.evaluations[0]
    context = open_offline_ik_station_context(
        robot_name=first_request.robot_name,
        frame_name=first_request.frame_name,
    )
    resources = _prepare_evaluation_resources(first_request, context)
    install_runtime_profile_hooks()
    ik_collection_module.reset_ik_candidate_collection_cache()

    ordered_results: list[ProfileEvaluationResult] = []
    result_cache: dict[tuple[tuple[float, float], ...], ProfileEvaluationResult] = {}
    search_cache: dict[tuple[tuple[float, float], ...], Any] = {}
    reuse_search_result = None

    for request in batch_request.evaluations:
        profile_key = _request_profile_cache_key(request)
        cached_result = result_cache.get(profile_key)
        cached_search_result = search_cache.get(profile_key)
        if cached_result is not None and cached_search_result is not None:
            ordered_results.append(cached_result)
            reuse_search_result = cached_search_result
            continue

        reset_runtime_profile()
        started = time.perf_counter()
        search_result = _evaluate_exact_profile_search(
            request,
            context,
            resources,
            reused_search_result=reuse_search_result,
        )
        search_result = _apply_optional_repairs(
            request,
            context,
            resources,
            search_result,
        )
        result = _finalize_request_result(
            request=request,
            context=context,
            resources=resources,
            search_result=search_result,
            elapsed_seconds=time.perf_counter() - started,
        )
        result_cache[profile_key] = result
        search_cache[profile_key] = search_result
        ordered_results.append(result)
        reuse_search_result = search_result

    return EvaluationBatchResult(results=tuple(ordered_results))


def evaluate_batch_request(
    batch_request: EvaluationBatchRequest,
) -> EvaluationBatchResult:
    if not batch_request.evaluations:
        raise ValueError("No evaluation requests found.")

    first_request = batch_request.evaluations[0]

    if _can_use_shared_exact_profile_batch(batch_request):
        return _evaluate_shared_exact_profile_batch(batch_request)

    if _can_run_offline_parallel(batch_request):
        try:
            return _evaluate_batch_request_offline_parallel(batch_request)
        except Exception:
            shutdown_offline_batch_executor()

    if _can_run_offline(batch_request):
        context = open_offline_ik_station_context(
            robot_name=first_request.robot_name,
            frame_name=first_request.frame_name,
        )
        results = tuple(evaluate_request(request, context)[0] for request in batch_request.evaluations)
        return EvaluationBatchResult(results=results)

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


atexit.register(shutdown_offline_batch_executor)


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
