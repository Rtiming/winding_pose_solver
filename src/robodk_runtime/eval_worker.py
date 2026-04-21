from __future__ import annotations

import argparse
import atexit
import math
import multiprocessing
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, replace
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
from src.core.geometry import _build_pose, _normalize_angle_range, _trim_joint_vector
from src.core.motion_settings import (
    RoboDKMotionSettings,
    build_motion_settings_from_dict,
    motion_settings_to_dict,
)
from src.core.robot_interface import build_robot_interface
from src.search.continuity_metrics import summarize_branch_jump_metrics
from src.robodk_runtime.program import (
    _append_move_instruction,
    _apply_motion_settings,
    _apply_selected_target,
    _build_program_waypoints,
    _delete_stale_bridge_targets,
    _ensure_final_path_is_valid_or_raise,
    _import_robodk_api,
    _require_item,
    _set_item_visible,
    _write_optimized_pose_csv,
    closed_winding_terminal_report,
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


def _resolve_runtime_profile_level() -> str:
    raw_value = os.getenv("WPS_RUNTIME_PROFILE_LEVEL", "minimal")
    profile_level = raw_value.strip().lower()
    if profile_level in {"off", "none", "0", "false"}:
        return "off"
    if profile_level in {"full", "all", "2"}:
        return "full"
    return "minimal"


def install_runtime_profile_hooks() -> None:
    global _PROFILE_HOOKS_INSTALLED
    if _PROFILE_HOOKS_INSTALLED:
        return

    profile_level = _resolve_runtime_profile_level()
    if profile_level == "off":
        _PROFILE_HOOKS_INSTALLED = True
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

    if profile_level == "full":
        path_optimizer_module._joint_limit_penalty = _wrap_profiled(
            "config_singularity",
            path_optimizer_module._joint_limit_penalty,
        )
        path_optimizer_module._singularity_penalty = _wrap_profiled(
            "config_singularity",
            path_optimizer_module._singularity_penalty,
        )
        ik_collection_module._collect_ik_candidates = _wrap_profiled(
            "ik_collection",
            ik_collection_module._collect_ik_candidates,
        )

    ik_collection_module._joint_limit_penalty = path_optimizer_module._joint_limit_penalty
    ik_collection_module._singularity_penalty = path_optimizer_module._singularity_penalty
    global_search_module._collect_ik_candidates = ik_collection_module._collect_ik_candidates
    local_repair_module._collect_ik_candidates = ik_collection_module._collect_ik_candidates
    _PROFILE_HOOKS_INSTALLED = True


def open_live_station_context(
    *,
    robot_name: str,
    frame_name: str,
) -> LiveStationContext:
    api = _import_robodk_api()
    existing_only = os.getenv("WPS_ROBODK_EXISTING_ONLY", "1").strip().lower() not in {
        "0",
        "false",
        "no",
    }
    if existing_only:
        rdk = api["Robolink"](robodk_path="")
        if rdk.Connect() < 1:
            raise RuntimeError(
                "Unable to connect to an already-running RoboDK instance. "
                "Open the RoboDK project/station first, make sure the API server is enabled, "
                "or set WPS_ROBODK_EXISTING_ONLY=0 if you intentionally want this script to "
                "launch RoboDK."
            )
    else:
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
        lock_profile_endpoints=bool(
            getattr(resources.settings, "lock_frame_a_origin_yz_profile_endpoints", True)
        ),
    )


def _apply_optional_repairs(
    request: ProfileEvaluationRequest,
    context: LiveStationContext,
    resources: PreparedEvaluationResources,
    search_result,
    *,
    shared_profile_result_cache: dict[tuple[tuple[float, float], ...], Any] | None = None,
) -> tuple[object, dict[str, Any]]:
    updated_result = search_result
    repair_metadata: dict[str, Any] = {}
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
            profile_result_cache=shared_profile_result_cache,
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
            run_post_inserted_window_repair=bool(request.run_window_repair),
        )
        if inserted_result is not None:
            updated_result = inserted_result
    if request.run_inserted_repair and getattr(
        resources.settings,
        "enable_joint_space_bridge_repair",
        False,
    ):
        bridged_result = local_repair_module._attempt_joint_space_bridge_repair(
            updated_result,
            robot=resources.robot_interface,
            tool_pose=context.tool_pose,
            reference_pose=context.reference_pose,
            joint_count=context.joint_count,
            motion_settings=resources.settings,
            optimizer_settings=resources.optimizer_settings,
            lower_limits=context.lower_limits,
            upper_limits=context.upper_limits,
        )
        if bridged_result is not None:
            updated_result = bridged_result
    if bool(getattr(resources.settings, "enable_same_family_segment_repair", True)):
        repaired_result, same_family_report = _attempt_same_family_segment_repair(
            updated_result,
            request=request,
            context=context,
            resources=resources,
        )
        if int(same_family_report.get("repaired_segments", 0)) > 0:
            print(
                "[repair] Accepted same-family segment repair: "
                f"repaired_segments={same_family_report.get('repaired_segments')}, "
                f"attempted_segments={same_family_report.get('attempted_segments')}."
            )
        updated_result = repaired_result
        repair_metadata["same_family_segment_repair"] = same_family_report
    return updated_result, repair_metadata


def _finalize_request_result(
    request: ProfileEvaluationRequest,
    context: LiveStationContext,
    resources: PreparedEvaluationResources,
    search_result,
    *,
    elapsed_seconds: float,
    repair_metadata: dict[str, Any] | None = None,
) -> ProfileEvaluationResult:
    validation_message = None
    extra_metadata: dict[str, Any] = {
        "ik_candidate_cache": ik_collection_module.ik_candidate_collection_cache_stats(),
        "closed_winding_terminal": closed_winding_terminal_report(
            search_result,
            settings=resources.settings,
        ),
    }
    if repair_metadata:
        extra_metadata["repair"] = dict(repair_metadata)
    force_debug_program = bool(request.metadata.get("force_debug_program_generation"))
    enforce_continuity_gate = bool(request.create_program)
    try:
        _ensure_final_path_is_valid_or_raise(
            search_result,
            settings=resources.settings,
            enforce_continuity_gate=enforce_continuity_gate,
        )
        if request.create_program:
            extra_metadata["program_name"] = materialize_program(
                context,
                request,
                search_result,
                resources.settings,
            )
    except RuntimeError as exc:
        validation_message = str(exc)
        if force_debug_program and request.create_program and context.rdk is not None:
            debug_program_name = materialize_program(
                context,
                request,
                search_result,
                resources.settings,
                skip_waypoint_validation=True,
            )
            extra_metadata["program_name"] = debug_program_name
            extra_metadata["debug_program_name"] = debug_program_name
            extra_metadata["debug_program_generation_forced"] = True
            extra_metadata["debug_program_warning"] = (
                "Program was generated even though strict path validation failed."
            )

    profile_result = _search_result_to_profile_result(
        request=request,
        search_result=search_result,
        settings=resources.settings,
        elapsed_seconds=elapsed_seconds,
        diagnostics=validation_message,
        metadata=extra_metadata,
    )
    from src.runtime.delivery import result_quality_summary

    quality = result_quality_summary(profile_result)
    return replace(
        profile_result,
        gate_tier=str(quality.get("gate_tier", "diagnostic")),
        block_reasons=tuple(
            dict(item) for item in quality.get("block_reasons", [])
        ),
    )


def materialize_program(
    context: LiveStationContext,
    request: ProfileEvaluationRequest,
    search_result,
    settings: RoboDKMotionSettings,
    *,
    skip_waypoint_validation: bool = False,
) -> str:
    program_name = request.program_name or f"Validation_{request.request_id}"
    if request.optimized_csv_path:
        _write_optimized_pose_csv(Path(request.optimized_csv_path), search_result)

    existing_program = context.rdk.Item(program_name, context.api["ITEM_TYPE_PROGRAM"])
    if existing_program.Valid():
        existing_program.Delete()

    _delete_stale_bridge_targets(
        context.rdk,
        context.api["ITEM_TYPE_TARGET"],
        prefix=program_name,
    )
    program = context.rdk.AddProgram(program_name, context.robot)
    program.setRobot(context.robot)
    if hasattr(program, "setPoseFrame"):
        program.setPoseFrame(context.frame)
    if hasattr(program, "setPoseTool"):
        program.setPoseTool(context.robot.PoseTool())
    _apply_motion_settings(program, settings)

    for waypoint in _build_program_waypoints(
        search_result,
        motion_settings=settings,
        validate_joint_continuity=not skip_waypoint_validation,
        target_prefix=program_name,
    ):
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
        if settings.hide_targets_after_generation:
            _set_item_visible(target, False)

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


def _max_joint_step_between_entries(left_entry: object, right_entry: object) -> float:
    left_joints = tuple(float(value) for value in getattr(left_entry, "joints", ()))
    right_joints = tuple(float(value) for value in getattr(right_entry, "joints", ()))
    return max(
        (
            abs(right_value - left_value)
            for left_value, right_value in zip(left_joints, right_joints)
        ),
        default=0.0,
    )


def _collect_same_family_recomputed_candidates(
    *,
    row_index: int,
    desired_config_flags: tuple[int, ...],
    search_result,
    context: LiveStationContext,
    resources: PreparedEvaluationResources,
    selected_path: Sequence[object],
) -> tuple[object, ...]:
    if row_index < 0 or row_index >= len(search_result.pose_rows):
        return ()
    pose = _build_pose(search_result.pose_rows[row_index], context.mat_type)
    seed_candidates: list[tuple[float, ...]] = []
    for probe_index in (row_index - 1, row_index, row_index + 1):
        if 0 <= probe_index < len(selected_path):
            seed_candidates.append(
                tuple(float(value) for value in getattr(selected_path[probe_index], "joints", ()))
            )
    seed_candidates.extend(resources.seed_joints)
    deduped_seed_candidates: list[tuple[float, ...]] = []
    seen_seeds: set[tuple[float, ...]] = set()
    for seed in seed_candidates:
        if not seed or seed in seen_seeds:
            continue
        seen_seeds.add(seed)
        deduped_seed_candidates.append(seed)
    recomputed_candidates = ik_collection_module._collect_ik_candidates(
        resources.robot_interface,
        pose,
        tool_pose=context.tool_pose,
        reference_pose=context.reference_pose,
        lower_limits=context.lower_limits,
        upper_limits=context.upper_limits,
        seed_joints=tuple(deduped_seed_candidates),
        joint_count=context.joint_count,
        optimizer_settings=resources.optimizer_settings,
        a1_lower_deg=resources.a1_lower_deg,
        a1_upper_deg=resources.a1_upper_deg,
        a2_max_deg=resources.settings.a2_max_deg,
        joint_constraint_tolerance_deg=resources.settings.joint_constraint_tolerance_deg,
    )
    return tuple(
        candidate
        for candidate in recomputed_candidates
        if tuple(int(value) for value in candidate.config_flags) == desired_config_flags
    )


def _attempt_same_family_segment_repair(
    search_result,
    *,
    request: ProfileEvaluationRequest,
    context: LiveStationContext,
    resources: PreparedEvaluationResources,
):
    if not search_result.selected_path or not search_result.pose_rows:
        return search_result, {
            "attempted_segments": 0,
            "repaired_segments": 0,
            "repaired_details": [],
            "failed_details": [],
        }

    big_circle_threshold = float(
        getattr(resources.settings, "big_circle_step_deg_threshold", 170.0)
    )
    ratio_threshold = float(
        getattr(resources.settings, "branch_flip_ratio_threshold", 8.0)
    )
    ratio_eps_mm = float(
        getattr(resources.settings, "branch_flip_ratio_eps_mm", 1e-3)
    )
    max_segments = max(
        0,
        int(getattr(resources.settings, "same_family_repair_max_segments", 8)),
    )
    metrics = summarize_branch_jump_metrics(
        search_result.selected_path,
        row_labels=search_result.row_labels,
        pose_rows=search_result.pose_rows,
        big_circle_step_deg_threshold=big_circle_threshold,
        branch_flip_ratio_threshold=ratio_threshold,
        ratio_eps_mm=ratio_eps_mm,
    )
    violent_segments = tuple(metrics.get("violent_branch_segments", ()))
    if not violent_segments or max_segments <= 0:
        return search_result, {
            "attempted_segments": 0,
            "repaired_segments": 0,
            "repaired_details": [],
            "failed_details": [],
        }

    selected_path = list(search_result.selected_path)
    repaired_details: list[dict[str, Any]] = []
    failed_details: list[dict[str, Any]] = []
    attempted_segments = 0
    for violent_segment in violent_segments[:max_segments]:
        try:
            segment_index = int(violent_segment.get("segment_index", -1))
        except Exception:
            continue
        row_index = segment_index + 1
        if row_index <= 0 or row_index >= len(selected_path):
            continue
        attempted_segments += 1
        previous_entry = selected_path[row_index - 1]
        current_entry = selected_path[row_index]
        next_entry = selected_path[row_index + 1] if row_index + 1 < len(selected_path) else None
        desired_config_flags = tuple(
            int(value) for value in getattr(previous_entry, "config_flags", ())
        )
        current_local_worst = max(
            _max_joint_step_between_entries(previous_entry, current_entry),
            0.0
            if next_entry is None
            else _max_joint_step_between_entries(current_entry, next_entry),
        )
        candidate_pool = [
            candidate
            for candidate in search_result.ik_layers[row_index].candidates
            if tuple(int(value) for value in candidate.config_flags) == desired_config_flags
        ]
        if not candidate_pool:
            candidate_pool = list(
                _collect_same_family_recomputed_candidates(
                    row_index=row_index,
                    desired_config_flags=desired_config_flags,
                    search_result=search_result,
                    context=context,
                    resources=resources,
                    selected_path=selected_path,
                )
            )

        best_candidate = None
        best_candidate_local_worst = math.inf
        for candidate in candidate_pool:
            local_worst = max(
                _max_joint_step_between_entries(previous_entry, candidate),
                0.0
                if next_entry is None
                else _max_joint_step_between_entries(candidate, next_entry),
            )
            if local_worst < best_candidate_local_worst - 1e-9:
                best_candidate = candidate
                best_candidate_local_worst = local_worst

        if (
            best_candidate is None
            or best_candidate_local_worst > big_circle_threshold + 1e-9
            or best_candidate_local_worst >= current_local_worst - 1e-9
        ):
            failed_details.append(
                {
                    "segment_index": segment_index,
                    "left_label": str(search_result.row_labels[segment_index]),
                    "right_label": str(search_result.row_labels[row_index]),
                    "reason": "no_improving_same_family_candidate",
                    "current_local_worst_deg": float(current_local_worst),
                    "best_candidate_local_worst_deg": float(
                        best_candidate_local_worst
                        if best_candidate is not None
                        else current_local_worst
                    ),
                }
            )
            continue

        selected_path[row_index] = best_candidate
        repaired_details.append(
            {
                "segment_index": segment_index,
                "left_label": str(search_result.row_labels[segment_index]),
                "right_label": str(search_result.row_labels[row_index]),
                "before_local_worst_deg": float(current_local_worst),
                "after_local_worst_deg": float(best_candidate_local_worst),
                "config_flags": [int(value) for value in desired_config_flags],
            }
        )

    if not repaired_details:
        return search_result, {
            "attempted_segments": attempted_segments,
            "repaired_segments": 0,
            "repaired_details": repaired_details,
            "failed_details": failed_details,
        }

    rebuilt_selected_path = tuple(selected_path)
    (
        config_switches,
        bridge_like_segments,
        worst_joint_step_deg,
        mean_joint_step_deg,
    ) = path_optimizer_module._summarize_selected_path(
        rebuilt_selected_path,
        bridge_trigger_joint_delta_deg=resources.settings.bridge_trigger_joint_delta_deg,
        config_switch_min_joint_delta_deg=resources.settings.config_switch_min_joint_delta_deg,
    )
    rebuilt_result = replace(
        search_result,
        selected_path=rebuilt_selected_path,
        total_cost=float(
            local_repair_module._joint_bridge_path_cost(
                rebuilt_selected_path,
                resources.optimizer_settings,
            )
        ),
        config_switches=int(config_switches),
        bridge_like_segments=int(bridge_like_segments),
        worst_joint_step_deg=float(worst_joint_step_deg),
        mean_joint_step_deg=float(mean_joint_step_deg),
    )
    return rebuilt_result, {
        "attempted_segments": attempted_segments,
        "repaired_segments": len(repaired_details),
        "repaired_details": repaired_details,
        "failed_details": failed_details,
    }


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
    result_metadata = dict(request.metadata)
    result_metadata.update(metadata or {})
    result_metadata["branch_jump_metrics"] = {
        "big_circle_step_count": int(branch_jump_metrics.get("big_circle_step_count", 0)),
        "max_branch_flip_ratio": float(branch_jump_metrics.get("max_branch_flip_ratio", 0.0)),
        "violent_branch_segments": [
            dict(item)
            for item in branch_jump_metrics.get("violent_branch_segments", ())
        ],
    }
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
        big_circle_step_count=int(branch_jump_metrics.get("big_circle_step_count", 0)),
        branch_flip_ratio=float(branch_jump_metrics.get("max_branch_flip_ratio", 0.0)),
        violent_branch_segments=tuple(
            dict(item)
            for item in branch_jump_metrics.get("violent_branch_segments", ())
        ),
        gate_tier="diagnostic",
        block_reasons=(),
        metadata=result_metadata,
        pose_rows=tuple(dict(row) for row in search_result.pose_rows)
        if request.include_pose_rows_in_result
        else None,
    )


def _metadata_bool(payload: dict[str, Any], name: str, default_value: bool) -> bool:
    value = payload.get(name, default_value)
    if isinstance(value, str):
        return value.strip().lower() not in {"0", "false", "no", "off", ""}
    return bool(value)


def _metadata_int(payload: dict[str, Any], name: str, default_value: int) -> int:
    try:
        return int(payload.get(name, default_value))
    except (TypeError, ValueError):
        return int(default_value)


def _evaluate_profile_search(
    request: ProfileEvaluationRequest,
    context: LiveStationContext,
    resources: PreparedEvaluationResources,
):
    if request.strategy == "full_search":
        return global_search_module._search_best_exact_pose_path(
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
    if request.strategy == "exact_profile":
        return _evaluate_exact_profile_search(
            request,
            context,
            resources,
        )
    raise ValueError(f"Unsupported evaluation strategy: {request.strategy}")


def _resources_with_settings(
    resources: PreparedEvaluationResources,
    context: LiveStationContext,
    settings: RoboDKMotionSettings,
) -> PreparedEvaluationResources:
    return replace(
        resources,
        settings=settings,
        optimizer_settings=path_optimizer_module._build_optimizer_settings(
            context.joint_count,
            settings,
        ),
    )


def _branch_jump_metrics_for_search_result(
    search_result,
    settings: RoboDKMotionSettings,
) -> dict[str, Any]:
    return summarize_branch_jump_metrics(
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


def _search_result_gate_values(
    search_result,
    settings: RoboDKMotionSettings,
) -> dict[str, object]:
    selected_path = tuple(getattr(search_result, "selected_path", ()) or ())
    row_labels = tuple(getattr(search_result, "row_labels", ()) or ())
    has_selected_path = bool(selected_path) and len(selected_path) == len(row_labels)
    terminal_report = closed_winding_terminal_report(search_result, settings=settings)
    terminal_ok = not bool(terminal_report.get("closed_path")) or bool(
        terminal_report.get("passed")
    )
    branch_metrics = _branch_jump_metrics_for_search_result(search_result, settings)
    invalid_row_count = int(getattr(search_result, "invalid_row_count", 0))
    ik_empty_row_count = int(getattr(search_result, "ik_empty_row_count", 0))
    bridge_like_segments = int(getattr(search_result, "bridge_like_segments", 0))
    big_circle_step_count = int(branch_metrics.get("big_circle_step_count", 0))
    worst_joint_step_deg = float(getattr(search_result, "worst_joint_step_deg", math.inf))
    worst_step_limit = float(getattr(settings, "official_worst_joint_step_deg_limit", 60.0))
    objective_reachable = (
        invalid_row_count == 0
        and ik_empty_row_count == 0
        and has_selected_path
        and terminal_ok
    )
    official_ready = (
        objective_reachable
        and bridge_like_segments == 0
        and big_circle_step_count == 0
        and worst_joint_step_deg <= worst_step_limit + 1e-9
    )
    return {
        "objective_reachable": bool(objective_reachable),
        "official_ready": bool(official_ready),
        "has_selected_path": bool(has_selected_path),
        "closed_terminal_ok": bool(terminal_ok),
        "closed_terminal": dict(terminal_report),
        "big_circle_step_count": big_circle_step_count,
        "branch_flip_ratio": float(branch_metrics.get("max_branch_flip_ratio", 0.0)),
    }


def _search_result_rank_key(
    search_result,
    settings: RoboDKMotionSettings,
) -> tuple[float, ...]:
    gate = _search_result_gate_values(search_result, settings)
    continuity_violation_score = _search_result_continuity_violation_score(
        search_result,
        settings,
        gate=gate,
    )
    return (
        0.0 if bool(gate["official_ready"]) else 1.0,
        float(getattr(search_result, "invalid_row_count", 0)),
        float(getattr(search_result, "ik_empty_row_count", 0)),
        0.0 if bool(gate["has_selected_path"]) else 1.0,
        0.0 if bool(gate["closed_terminal_ok"]) else 1.0,
        float(continuity_violation_score),
        float(getattr(search_result, "bridge_like_segments", 0)),
        float(gate["big_circle_step_count"]),
        float(getattr(search_result, "worst_joint_step_deg", math.inf)),
        float(gate["branch_flip_ratio"]),
        float(getattr(search_result, "mean_joint_step_deg", math.inf)),
        float(getattr(search_result, "config_switches", 0)),
        float(getattr(search_result, "total_cost", math.inf)),
    )


def _search_result_continuity_violation_score(
    search_result,
    settings: RoboDKMotionSettings,
    *,
    gate: dict[str, object] | None = None,
) -> float:
    gate_values = gate or _search_result_gate_values(search_result, settings)
    worst_step_limit = float(
        getattr(settings, "official_worst_joint_step_deg_limit", 60.0)
    )
    worst_step_over_limit = max(
        0.0,
        float(getattr(search_result, "worst_joint_step_deg", math.inf))
        - worst_step_limit,
    )
    bridge_count = int(getattr(search_result, "bridge_like_segments", 0))
    big_circle_count = int(gate_values["big_circle_step_count"])
    branch_ratio = max(0.0, float(gate_values["branch_flip_ratio"]))

    # Once no variant is officially deliverable, avoid a brittle choice where
    # one fewer diagnostic segment hides a much larger physical joint jump.
    return (
        worst_step_over_limit
        + worst_step_limit * float(bridge_count + big_circle_count)
        + 0.25 * min(branch_ratio, 100.0)
    )


def _search_result_summary(
    *,
    name: str,
    search_result,
    settings: RoboDKMotionSettings,
    run_window_repair: bool,
    run_inserted_repair: bool,
) -> dict[str, object]:
    gate = _search_result_gate_values(search_result, settings)
    return {
        "name": name,
        "ik_max_candidates_per_config_family": int(
            getattr(settings, "ik_max_candidates_per_config_family", 4)
        ),
        "use_guided_config_path": bool(
            getattr(settings, "use_guided_config_path", True)
        ),
        "run_window_repair": bool(run_window_repair),
        "run_inserted_repair": bool(run_inserted_repair),
        "frame_a_origin_yz_envelope_schedule_mm": [
            float(value)
            for value in getattr(settings, "frame_a_origin_yz_envelope_schedule_mm", ())
        ],
        "frame_a_origin_yz_window_radius": int(
            getattr(settings, "frame_a_origin_yz_window_radius", 0)
        ),
        "frame_a_origin_yz_max_passes": int(
            getattr(settings, "frame_a_origin_yz_max_passes", 0)
        ),
        "objective_reachable": bool(gate["objective_reachable"]),
        "official_ready": bool(gate["official_ready"]),
        "invalid_row_count": int(getattr(search_result, "invalid_row_count", 0)),
        "ik_empty_row_count": int(getattr(search_result, "ik_empty_row_count", 0)),
        "config_switches": int(getattr(search_result, "config_switches", 0)),
        "bridge_like_segments": int(getattr(search_result, "bridge_like_segments", 0)),
        "big_circle_step_count": int(gate["big_circle_step_count"]),
        "continuity_violation_score": float(
            _search_result_continuity_violation_score(
                search_result,
                settings,
                gate=gate,
            )
        ),
        "worst_joint_step_deg": float(
            getattr(search_result, "worst_joint_step_deg", math.inf)
        ),
        "mean_joint_step_deg": float(
            getattr(search_result, "mean_joint_step_deg", math.inf)
        ),
        "total_cost": float(getattr(search_result, "total_cost", math.inf)),
        "selected_path_count": len(tuple(getattr(search_result, "selected_path", ()) or ())),
    }


def _run_fixed_point_variant(
    *,
    name: str,
    request: ProfileEvaluationRequest,
    context: LiveStationContext,
    resources: PreparedEvaluationResources,
    settings: RoboDKMotionSettings,
    run_window_repair: bool,
    run_inserted_repair: bool,
):
    variant_resources = _resources_with_settings(resources, context, settings)
    variant_request = replace(
        request,
        request_id=f"{request.request_id}_{name}",
        motion_settings=motion_settings_to_dict(settings),
        run_window_repair=bool(run_window_repair),
        run_inserted_repair=bool(run_inserted_repair),
        create_program=bool(request.create_program),
        metadata={
            **dict(request.metadata),
            "fixed_point_path_variant": name,
        },
    )
    search_result = _evaluate_profile_search(variant_request, context, variant_resources)
    repaired_result, repair_metadata = _apply_optional_repairs(
        variant_request,
        context,
        variant_resources,
        search_result,
    )
    return variant_request, variant_resources, repaired_result, repair_metadata


def _escalate_profile_repair_settings(
    settings: RoboDKMotionSettings,
) -> RoboDKMotionSettings:
    base_envelopes = tuple(
        float(value)
        for value in getattr(settings, "frame_a_origin_yz_envelope_schedule_mm", (6.0,))
        if float(value) >= 0.0
    )
    max_envelope = max(base_envelopes, default=0.0)
    expanded_envelopes: list[float] = list(base_envelopes)
    seen = {round(float(value), 6) for value in expanded_envelopes}
    if max_envelope > 0.0:
        for scale in (1.5, 2.0):
            candidate = min(60.0, round(max_envelope * scale, 3))
            key = round(candidate, 6)
            if candidate > max_envelope + 1e-9 and key not in seen:
                seen.add(key)
                expanded_envelopes.append(candidate)
    expanded_envelopes_tuple = tuple(sorted(expanded_envelopes))
    return replace(
        settings,
        frame_a_origin_yz_envelope_schedule_mm=expanded_envelopes_tuple,
        frame_a_origin_yz_window_radius=max(
            int(getattr(settings, "frame_a_origin_yz_window_radius", 8)),
            12,
        ),
        frame_a_origin_yz_max_passes=max(
            int(getattr(settings, "frame_a_origin_yz_max_passes", 4)),
            6,
        ),
    )


def _apply_fixed_point_path_fallback(
    request: ProfileEvaluationRequest,
    context: LiveStationContext,
    resources: PreparedEvaluationResources,
    search_result,
    repair_metadata: dict[str, Any],
):
    metadata = dict(request.metadata)
    if not _metadata_bool(metadata, "enable_fixed_point_path_fallback", False):
        return request, resources, search_result, repair_metadata

    attempts: list[dict[str, object]] = []
    selected_name = "baseline"
    selected_request = request
    selected_resources = resources
    selected_result = search_result
    selected_repair_metadata = dict(repair_metadata)
    selected_rank = _search_result_rank_key(search_result, resources.settings)
    attempts.append(
        _search_result_summary(
            name="baseline",
            search_result=search_result,
            settings=resources.settings,
            run_window_repair=request.run_window_repair,
            run_inserted_repair=request.run_inserted_repair,
        )
    )

    baseline_official = bool(
        _search_result_gate_values(search_result, resources.settings)["official_ready"]
    )
    triggered = not baseline_official
    if triggered:
        current_limit = int(
            getattr(resources.settings, "ik_max_candidates_per_config_family", 4)
        )
        fallback_limit = max(
            current_limit,
            _metadata_int(metadata, "fixed_point_fallback_max_candidates_per_config", 12),
        )
        disable_guided = _metadata_bool(
            metadata,
            "fixed_point_fallback_disable_guided_path",
            True,
        )
        run_window_repair = _metadata_bool(
            metadata,
            "fixed_point_fallback_run_window_repair",
            False,
        )
        run_inserted_repair = _metadata_bool(
            metadata,
            "fixed_point_fallback_run_inserted_repair",
            False,
        )
        attempted_keys = {
            (
                current_limit,
                bool(getattr(resources.settings, "use_guided_config_path", True)),
                bool(request.run_window_repair),
                bool(request.run_inserted_repair),
            )
        }
        variant_specs: list[tuple[str, RoboDKMotionSettings, bool, bool]] = [
            (
                "expanded_guided",
                replace(
                    resources.settings,
                    ik_max_candidates_per_config_family=fallback_limit,
                    use_guided_config_path=True,
                ),
                False,
                False,
            )
        ]
        if disable_guided:
            variant_specs.append(
                (
                    "expanded_unguided",
                    replace(
                        resources.settings,
                        ik_max_candidates_per_config_family=fallback_limit,
                        use_guided_config_path=False,
                    ),
                    False,
                    False,
                )
            )

        for name, settings, window_repair, inserted_repair in variant_specs:
            key = (
                int(getattr(settings, "ik_max_candidates_per_config_family", 4)),
                bool(getattr(settings, "use_guided_config_path", True)),
                bool(window_repair),
                bool(inserted_repair),
            )
            if key in attempted_keys:
                continue
            attempted_keys.add(key)
            candidate_request, candidate_resources, candidate_result, candidate_repair = (
                _run_fixed_point_variant(
                    name=name,
                    request=request,
                    context=context,
                    resources=resources,
                    settings=settings,
                    run_window_repair=window_repair,
                    run_inserted_repair=inserted_repair,
                )
            )
            attempts.append(
                _search_result_summary(
                    name=name,
                    search_result=candidate_result,
                    settings=settings,
                    run_window_repair=window_repair,
                    run_inserted_repair=inserted_repair,
                )
            )
            candidate_rank = _search_result_rank_key(candidate_result, settings)
            if candidate_rank < selected_rank:
                selected_name = name
                selected_request = candidate_request
                selected_resources = candidate_resources
                selected_result = candidate_result
                selected_repair_metadata = dict(candidate_repair)
                selected_rank = candidate_rank

        official_found = any(bool(item.get("official_ready")) for item in attempts)
        if not official_found and (run_window_repair or run_inserted_repair):
            repair_settings = _escalate_profile_repair_settings(
                replace(
                    resources.settings,
                    ik_max_candidates_per_config_family=fallback_limit,
                    use_guided_config_path=not disable_guided,
                )
            )
            key = (
                int(getattr(repair_settings, "ik_max_candidates_per_config_family", 4)),
                bool(getattr(repair_settings, "use_guided_config_path", True)),
                bool(run_window_repair),
                bool(run_inserted_repair),
            )
            if key not in attempted_keys:
                candidate_request, candidate_resources, candidate_result, candidate_repair = (
                    _run_fixed_point_variant(
                        name="expanded_repair",
                        request=request,
                        context=context,
                        resources=resources,
                        settings=repair_settings,
                        run_window_repair=run_window_repair,
                        run_inserted_repair=run_inserted_repair,
                    )
                )
                attempts.append(
                    _search_result_summary(
                        name="expanded_repair",
                        search_result=candidate_result,
                        settings=repair_settings,
                        run_window_repair=run_window_repair,
                        run_inserted_repair=run_inserted_repair,
                    )
                )
                candidate_rank = _search_result_rank_key(candidate_result, repair_settings)
                if candidate_rank < selected_rank:
                    selected_name = "expanded_repair"
                    selected_request = candidate_request
                    selected_resources = candidate_resources
                    selected_result = candidate_result
                    selected_repair_metadata = dict(candidate_repair)
                    selected_rank = candidate_rank

    selected_repair_metadata = dict(selected_repair_metadata)
    selected_repair_metadata["fixed_point_path_fallback"] = {
        "enabled": True,
        "triggered": bool(triggered),
        "selected_variant": selected_name,
        "attempt_count": len(attempts),
        "attempts": attempts,
    }
    return selected_request, selected_resources, selected_result, selected_repair_metadata


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

    search_result = _evaluate_profile_search(request, context, resources)
    search_result, repair_metadata = _apply_optional_repairs(
        request,
        context,
        resources,
        search_result,
    )
    request, resources, search_result, repair_metadata = _apply_fixed_point_path_fallback(
        request,
        context,
        resources,
        search_result,
        repair_metadata,
    )

    elapsed_seconds = time.perf_counter() - start_time
    result = _finalize_request_result(
        request=request,
        context=context,
        resources=resources,
        search_result=search_result,
        elapsed_seconds=elapsed_seconds,
        repair_metadata=repair_metadata,
    )
    print(
        f"[{request.request_id}] status={result.status}, "
        f"ik_empty_rows={result.ik_empty_row_count}, "
        f"config_switches={result.config_switches}, "
        f"bridge_like_segments={result.bridge_like_segments}, "
        f"big_circle_step_count={result.big_circle_step_count}, "
        f"worst_joint_step={result.worst_joint_step_deg:.3f} deg, "
        f"branch_flip_ratio={result.branch_flip_ratio:.3f}, "
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
    """Build a mock station context from SixAxisIK config - no live RoboDK connection needed.

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

    print("[offline-ik] Running without live RoboDK station - using SixAxisIK configured values.")

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
        search_result, repair_metadata = _apply_optional_repairs(
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
            repair_metadata=repair_metadata,
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
