from __future__ import annotations

import atexit
import math
import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from src.core.motion_settings import RoboDKMotionSettings
from src.core.types import _IKLayer, _PathSearchResult


@dataclass(frozen=True)
class _OfflineExactEvalContext:
    robot: object
    mat_type: object
    original_joints: tuple[float, ...]
    lower_limits: tuple[float, ...]
    upper_limits: tuple[float, ...]
    joint_count: int
    tool_pose: object
    reference_pose: object


@dataclass(frozen=True)
class _ExactProfileBatchSpec:
    reference_pose_rows: tuple[dict[str, float], ...]
    base_frame_a_origin_yz_profile_mm: tuple[tuple[float, float], ...]
    reused_ik_layers: tuple[_IKLayer, ...] | None
    frame_a_origin_yz_profiles_mm: tuple[tuple[tuple[float, float], ...], ...]
    row_labels: tuple[str, ...]
    inserted_flags: tuple[bool, ...]
    motion_settings: RoboDKMotionSettings
    start_joints: tuple[float, ...] | None


_EXACT_PROFILE_EXECUTOR: ProcessPoolExecutor | None = None
_EXACT_PROFILE_EXECUTOR_WORKERS: int | None = None
_WORKER_OFFLINE_CONTEXT: _OfflineExactEvalContext | None = None


def resolve_local_parallel_workers(configured_workers: int) -> int:
    if configured_workers > 0:
        return int(configured_workers)

    cpu_count = os.cpu_count() or 1
    if cpu_count <= 1:
        return 1
    return max(1, min(8, cpu_count - 1))


def maybe_parallel_evaluate_exact_profiles(
    *,
    reference_pose_rows: Sequence[dict[str, float]],
    base_frame_a_origin_yz_profile_mm: Sequence[tuple[float, float]],
    reused_ik_layers: Sequence[_IKLayer] | None,
    frame_a_origin_yz_profiles_mm: Sequence[Sequence[tuple[float, float]]],
    row_labels: Sequence[str],
    inserted_flags: Sequence[bool],
    motion_settings: RoboDKMotionSettings,
    start_joints: Sequence[float] | None,
) -> tuple[_PathSearchResult, ...] | None:
    if motion_settings.ik_backend != "six_axis_ik":
        return None

    worker_count = resolve_local_parallel_workers(motion_settings.local_parallel_workers)
    if worker_count <= 1:
        return None
    if not _can_spawn_local_profile_workers():
        return None

    profile_batches = tuple(
        tuple((float(dy_mm), float(dz_mm)) for dy_mm, dz_mm in profile)
        for profile in frame_a_origin_yz_profiles_mm
    )
    if len(profile_batches) < motion_settings.local_parallel_min_batch_size:
        return None

    base_profile = tuple(
        (float(dy_mm), float(dz_mm))
        for dy_mm, dz_mm in base_frame_a_origin_yz_profile_mm
    )
    reused_ik_layers_tuple = (
        None
        if reused_ik_layers is None
        else tuple(reused_ik_layers)
    )

    chunk_count = min(worker_count, len(profile_batches))
    chunk_size = max(1, math.ceil(len(profile_batches) / max(1, chunk_count)))
    reference_pose_rows_tuple = tuple(dict(row) for row in reference_pose_rows)
    row_labels_tuple = tuple(str(label) for label in row_labels)
    inserted_flags_tuple = tuple(bool(flag) for flag in inserted_flags)
    batch_specs = [
        _ExactProfileBatchSpec(
            reference_pose_rows=reference_pose_rows_tuple,
            base_frame_a_origin_yz_profile_mm=base_profile,
            reused_ik_layers=reused_ik_layers_tuple,
            frame_a_origin_yz_profiles_mm=profile_batches[start_index : start_index + chunk_size],
            row_labels=row_labels_tuple,
            inserted_flags=inserted_flags_tuple,
            motion_settings=motion_settings,
            start_joints=None
            if start_joints is None
            else tuple(float(value) for value in start_joints),
        )
        for start_index in range(0, len(profile_batches), chunk_size)
    ]
    executor = _get_exact_profile_executor(worker_count)
    try:
        results_by_chunk = [
            future.result()
            for future in [
                executor.submit(_evaluate_exact_profile_batch_worker, batch_spec)
                for batch_spec in batch_specs
            ]
        ]
    except Exception:
        shutdown_exact_profile_executor()
        return None
    return tuple(
        result
        for chunk_results in results_by_chunk
        for result in chunk_results
    )


def _can_spawn_local_profile_workers() -> bool:
    main_file = getattr(__import__("__main__"), "__file__", None)
    return bool(main_file and Path(str(main_file)).exists())


def shutdown_exact_profile_executor() -> None:
    global _EXACT_PROFILE_EXECUTOR, _EXACT_PROFILE_EXECUTOR_WORKERS
    if _EXACT_PROFILE_EXECUTOR is not None:
        _EXACT_PROFILE_EXECUTOR.shutdown(wait=True, cancel_futures=False)
        _EXACT_PROFILE_EXECUTOR = None
        _EXACT_PROFILE_EXECUTOR_WORKERS = None


def _get_exact_profile_executor(worker_count: int) -> ProcessPoolExecutor:
    global _EXACT_PROFILE_EXECUTOR, _EXACT_PROFILE_EXECUTOR_WORKERS
    if (
        _EXACT_PROFILE_EXECUTOR is None
        or _EXACT_PROFILE_EXECUTOR_WORKERS != worker_count
    ):
        shutdown_exact_profile_executor()
        _EXACT_PROFILE_EXECUTOR = ProcessPoolExecutor(
            max_workers=worker_count,
            mp_context=multiprocessing.get_context("spawn"),
        )
        _EXACT_PROFILE_EXECUTOR_WORKERS = worker_count
    return _EXACT_PROFILE_EXECUTOR


def _get_worker_offline_context() -> _OfflineExactEvalContext:
    global _WORKER_OFFLINE_CONTEXT
    if _WORKER_OFFLINE_CONTEXT is None:
        from src.core.robot_interface import SixAxisIKRobotInterface
        from src.core.simple_mat import SimpleMat
        from src.six_axis_ik import config as ik_config

        robot_interface = SixAxisIKRobotInterface(robodk_robot=None)
        lower_limits_raw, upper_limits_raw, _ = robot_interface.JointLimits()
        lower_limits = tuple(float(value) for value in lower_limits_raw.list())
        upper_limits = tuple(float(value) for value in upper_limits_raw.list())
        _WORKER_OFFLINE_CONTEXT = _OfflineExactEvalContext(
            robot=robot_interface,
            mat_type=SimpleMat,
            original_joints=tuple(0.0 for _ in lower_limits),
            lower_limits=lower_limits,
            upper_limits=upper_limits,
            joint_count=len(lower_limits),
            tool_pose=SimpleMat(ik_config.get_configured_tool_pose().tolist()),
            reference_pose=SimpleMat(ik_config.get_configured_frame_pose().tolist()),
        )
    return _WORKER_OFFLINE_CONTEXT


def _evaluate_exact_profile_batch_worker(
    batch_spec: _ExactProfileBatchSpec,
) -> tuple[_PathSearchResult, ...]:
    from src.core.geometry import _normalize_angle_range
    from src.search.global_search import _evaluate_frame_a_origin_profile
    from src.search.path_optimizer import _build_optimizer_settings

    context = _get_worker_offline_context()
    motion_settings = batch_spec.motion_settings
    optimizer_settings = _build_optimizer_settings(context.joint_count, motion_settings)
    a1_lower_deg, a1_upper_deg = _normalize_angle_range(
        motion_settings.a1_min_deg,
        motion_settings.a1_max_deg,
    )
    start_joints = (
        context.original_joints
        if batch_spec.start_joints is None
        else tuple(float(value) for value in batch_spec.start_joints)
    )
    reused_ik_layers = batch_spec.reused_ik_layers
    if reused_ik_layers is not None and len(reused_ik_layers) != len(batch_spec.reference_pose_rows):
        reused_ik_layers = None

    return tuple(
        _evaluate_frame_a_origin_profile(
            batch_spec.reference_pose_rows,
            frame_a_origin_yz_profile_mm=profile,
            row_labels=batch_spec.row_labels,
            inserted_flags=batch_spec.inserted_flags,
            robot=context.robot,
            mat_type=context.mat_type,
            move_type=motion_settings.move_type,
            start_joints=start_joints,
            tool_pose=context.tool_pose,
            reference_pose=context.reference_pose,
            joint_count=context.joint_count,
            optimizer_settings=optimizer_settings,
            a1_lower_deg=a1_lower_deg,
            a1_upper_deg=a1_upper_deg,
            a2_max_deg=motion_settings.a2_max_deg,
            joint_constraint_tolerance_deg=motion_settings.joint_constraint_tolerance_deg,
            seed_joints=(),
            lower_limits=context.lower_limits,
            upper_limits=context.upper_limits,
            bridge_trigger_joint_delta_deg=motion_settings.bridge_trigger_joint_delta_deg,
            reused_ik_layers=reused_ik_layers,
            recompute_row_indices=_profile_changed_row_indices(
                batch_spec.base_frame_a_origin_yz_profile_mm,
                profile,
            ),
            lock_profile_endpoints=bool(
                getattr(motion_settings, "lock_frame_a_origin_yz_profile_endpoints", True)
            ),
        )
        for profile in batch_spec.frame_a_origin_yz_profiles_mm
    )


def _profile_changed_row_indices(
    base_profile: Sequence[tuple[float, float]],
    candidate_profile: Sequence[tuple[float, float]],
) -> tuple[int, ...]:
    if len(base_profile) != len(candidate_profile):
        return tuple(range(len(candidate_profile)))

    changed_indices: list[int] = []
    for row_index, (base_offset, candidate_offset) in enumerate(
        zip(base_profile, candidate_profile)
    ):
        if (
            abs(float(base_offset[0]) - float(candidate_offset[0])) > 1e-9
            or abs(float(base_offset[1]) - float(candidate_offset[1])) > 1e-9
        ):
            changed_indices.append(row_index)
    return tuple(changed_indices)


atexit.register(shutdown_exact_profile_executor)
