from __future__ import annotations

from pathlib import Path
from typing import Any

from src.app_runner import AppRuntimeSettings
from src.global_search import _extract_row_labels
from src.motion_settings import motion_settings_to_dict
from src.pose_csv import load_pose_rows
from src.pose_solver import solve_tool_poses
from src.runtime_profiler import (
    profile_runtime_section,
    reset_runtime_profile,
    runtime_profile_snapshot,
)
from .collab_models import ProfileEvaluationRequest, RemoteSearchRequest


def refresh_pose_csv(settings: AppRuntimeSettings) -> Path:
    reset_runtime_profile()
    with profile_runtime_section("pose_solver"):
        solve_tool_poses(
            settings.validation_centerline_csv,
            settings.tool_poses_frame2_csv,
            build_options=settings.frame_build_options,
            target_frame_origin_mm=settings.target_frame_origin_mm,
            target_frame_rotation_xyz_deg=settings.target_frame_rotation_xyz_deg,
            verify_solution=settings.enable_solver_verification,
            verification_row_ids=list(settings.verification_row_ids)
            if settings.verification_row_ids is not None
            else None,
            verification_tolerance=settings.verification_tolerance,
        )
    return settings.tool_poses_frame2_csv


def build_profile_evaluation_request(
    settings: AppRuntimeSettings,
    *,
    request_id: str,
    strategy: str = "full_search",
    refresh_csv: bool = True,
    frame_a_origin_yz_profile_mm: tuple[tuple[float, float], ...] | None = None,
    run_window_repair: bool = True,
    run_inserted_repair: bool = True,
    include_pose_rows_in_result: bool = False,
    create_program: bool = False,
    program_name: str | None = None,
    optimized_csv_path: str | None = None,
    start_joints: tuple[float, ...] | None = None,
    metadata: dict[str, Any] | None = None,
) -> ProfileEvaluationRequest:
    if refresh_csv:
        refresh_pose_csv(settings)

    pose_rows = tuple(load_pose_rows(settings.tool_poses_frame2_csv))
    row_labels = _extract_row_labels(pose_rows)
    zero_profile = tuple((0.0, 0.0) for _ in pose_rows)
    request_metadata = dict(metadata or {})
    request_metadata.setdefault("request_build_profile", runtime_profile_snapshot())
    request_metadata.setdefault("tool_poses_frame2_csv", str(settings.tool_poses_frame2_csv))
    request_metadata.setdefault("validation_centerline_csv", str(settings.validation_centerline_csv))

    return ProfileEvaluationRequest(
        request_id=request_id,
        robot_name=settings.robot_name,
        frame_name=settings.frame_name,
        motion_settings=motion_settings_to_dict(settings.motion_settings),
        reference_pose_rows=pose_rows,
        frame_a_origin_yz_profile_mm=frame_a_origin_yz_profile_mm or zero_profile,
        row_labels=row_labels,
        inserted_flags=tuple(False for _ in pose_rows),
        strategy=strategy,
        start_joints=start_joints,
        run_window_repair=run_window_repair,
        run_inserted_repair=run_inserted_repair,
        include_pose_rows_in_result=include_pose_rows_in_result,
        create_program=create_program,
        program_name=program_name,
        optimized_csv_path=optimized_csv_path,
        metadata=request_metadata,
    )


def build_remote_search_request(
    base_request: ProfileEvaluationRequest,
    *,
    round_index: int = 1,
    candidate_limit: int = 24,
    baseline_result=None,
    metadata: dict[str, Any] | None = None,
) -> RemoteSearchRequest:
    return RemoteSearchRequest(
        base_request=base_request,
        baseline_result=baseline_result,
        round_index=round_index,
        candidate_limit=candidate_limit,
        metadata=dict(metadata or {}),
    )
