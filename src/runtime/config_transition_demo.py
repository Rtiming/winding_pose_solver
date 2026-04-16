from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

from src.core.collab_models import ProfileEvaluationRequest, ProfileEvaluationResult, write_json_file
from src.core.motion_settings import motion_settings_to_dict
from src.core.pose_csv import load_pose_rows
from src.runtime.app import AppRuntimeSettings
from src.runtime.request_builder import refresh_pose_csv
from src.runtime.yz_nullspace_demo import (
    _write_result_pose_csv,
    _write_selected_joint_path_csv,
)
from src.robodk_runtime.eval_worker import (
    _prepare_evaluation_resources,
    _search_result_to_profile_result,
    open_offline_ik_station_context,
)
from src.search.global_search import (
    _apply_frame_a_origin_yz_profile,
    _build_ik_layers_with_diagnostics,
    _finalize_frame_a_origin_profile_result,
)
from src.core.types import _IKLayer


@dataclass(frozen=True)
class ConfigTransitionDemoArtifacts:
    run_id: str
    output_dir: Path
    result_path: Path
    request_path: Path
    joint_path_csv: Path
    pose_csv: Path
    summary_path: Path
    result: ProfileEvaluationResult
    window_start_index: int
    window_end_index: int
    switch_after_index: int
    left_family: tuple[int, int, int]
    right_family: tuple[int, int, int]


def default_config_transition_demo_run_id() -> str:
    return "config_transition_" + datetime.now().strftime("%Y%m%d_%H%M%S")


def parse_config_family(raw: str | Sequence[int]) -> tuple[int, int, int]:
    if isinstance(raw, str):
        values = [int(token.strip()) for token in raw.split(",") if token.strip()]
    else:
        values = [int(value) for value in raw]
    if len(values) != 3:
        raise ValueError(f"Expected three config flags, got: {raw!r}")
    return (values[0], values[1], values[2])


def run_forced_config_transition_demo(
    settings: AppRuntimeSettings,
    *,
    window_start_index: int = 365,
    window_end_index: int = 381,
    switch_after_index: int = 372,
    left_family: tuple[int, int, int] = (0, 0, 1),
    right_family: tuple[int, int, int] = (0, 0, 0),
    output_root: str | Path = Path("artifacts/diagnostics/config_transition_demo"),
    run_id: str | None = None,
    refresh_csv: bool = True,
) -> ConfigTransitionDemoArtifacts:
    """Force a visible IK config-family transition for RoboDK inspection.

    This is a diagnostic/demo path.  It preserves the centerline-derived target
    pose rows but constrains the candidate family before and after a selected
    cut so the resulting program visibly changes robot configuration.
    """

    run_id = run_id or default_config_transition_demo_run_id()
    output_dir = Path(output_root) / run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    if refresh_csv:
        refresh_pose_csv(settings)

    pose_rows = tuple(load_pose_rows(settings.tool_poses_frame2_csv))
    if window_start_index < 0 or window_end_index >= len(pose_rows):
        raise ValueError(
            f"Window [{window_start_index}, {window_end_index}] is outside "
            f"pose row count {len(pose_rows)}."
        )
    if not (window_start_index <= switch_after_index < window_end_index):
        raise ValueError(
            "switch_after_index must lie inside the window and leave at least "
            "one row on the right side."
        )

    window_rows = tuple(
        dict(row) for row in pose_rows[window_start_index : window_end_index + 1]
    )
    row_labels = tuple(str(window_start_index + offset) for offset in range(len(window_rows)))
    inserted_flags = tuple(False for _ in window_rows)
    zero_profile = tuple((0.0, 0.0) for _ in window_rows)

    motion_settings = motion_settings_to_dict(settings.motion_settings)
    motion_settings["ik_backend"] = "six_axis_ik"
    request = ProfileEvaluationRequest(
        request_id=f"{run_id}_forced_{_family_token(left_family)}_to_{_family_token(right_family)}",
        robot_name=settings.robot_name,
        frame_name=settings.frame_name,
        motion_settings=motion_settings,
        reference_pose_rows=window_rows,
        frame_a_origin_yz_profile_mm=zero_profile,
        row_labels=row_labels,
        inserted_flags=inserted_flags,
        strategy="forced_config_transition",
        start_joints=None,
        run_window_repair=False,
        run_inserted_repair=False,
        include_pose_rows_in_result=True,
        create_program=False,
        program_name=None,
        optimized_csv_path=None,
        metadata={
            "entrypoint": "config_transition_demo",
            "target_frame_a_origin_source": "main.py APP_RUNTIME_SETTINGS",
            "window_start_index": int(window_start_index),
            "window_end_index": int(window_end_index),
            "switch_after_index": int(switch_after_index),
            "left_family": list(left_family),
            "right_family": list(right_family),
        },
    )

    start_time = time.perf_counter()
    context = open_offline_ik_station_context(
        robot_name=request.robot_name,
        frame_name=request.frame_name,
    )
    resources = _prepare_evaluation_resources(request, context)
    adjusted_rows = _apply_frame_a_origin_yz_profile(
        window_rows,
        frame_a_origin_yz_profile_mm=zero_profile,
    )
    ik_layers, _ik_empty = _build_ik_layers_with_diagnostics(
        adjusted_rows,
        robot=resources.robot_interface,
        mat_type=context.mat_type,
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
    )
    forced_layers = _force_config_families(
        ik_layers,
        absolute_start_index=window_start_index,
        switch_after_index=switch_after_index,
        left_family=left_family,
        right_family=right_family,
    )
    search_result = _finalize_frame_a_origin_profile_result(
        reference_pose_rows=window_rows,
        adjusted_pose_rows=adjusted_rows,
        ik_layers=forced_layers,
        frame_a_origin_yz_profile_mm=zero_profile,
        row_labels=row_labels,
        inserted_flags=inserted_flags,
        robot=resources.robot_interface,
        move_type=resources.settings.move_type,
        start_joints=resources.start_joints,
        optimizer_settings=resources.optimizer_settings,
        bridge_trigger_joint_delta_deg=resources.settings.bridge_trigger_joint_delta_deg,
    )
    result = _search_result_to_profile_result(
        request=request,
        search_result=search_result,
        settings=resources.settings,
        elapsed_seconds=time.perf_counter() - start_time,
        diagnostics=None,
        metadata={"forced_config_transition": True},
    )

    request_path = write_json_file(output_dir / "request.json", request.to_dict())
    result_path = write_json_file(output_dir / "result.json", result.to_dict())
    joint_path_csv = _write_selected_joint_path_csv(
        result,
        output_dir / "selected_joint_path.csv",
    )
    pose_csv = _write_result_pose_csv(result, output_dir / "pose_path.csv")
    summary_path = write_json_file(
        output_dir / "summary.json",
        _summary_payload(
            result=result,
            window_start_index=window_start_index,
            window_end_index=window_end_index,
            switch_after_index=switch_after_index,
            left_family=left_family,
            right_family=right_family,
        ),
    )

    return ConfigTransitionDemoArtifacts(
        run_id=run_id,
        output_dir=output_dir,
        result_path=result_path,
        request_path=request_path,
        joint_path_csv=joint_path_csv,
        pose_csv=pose_csv,
        summary_path=summary_path,
        result=result,
        window_start_index=window_start_index,
        window_end_index=window_end_index,
        switch_after_index=switch_after_index,
        left_family=left_family,
        right_family=right_family,
    )


def _force_config_families(
    ik_layers: Sequence[_IKLayer],
    *,
    absolute_start_index: int,
    switch_after_index: int,
    left_family: tuple[int, int, int],
    right_family: tuple[int, int, int],
) -> tuple[_IKLayer, ...]:
    forced_layers: list[_IKLayer] = []
    for local_index, layer in enumerate(ik_layers):
        absolute_index = absolute_start_index + local_index
        required_family = left_family if absolute_index <= switch_after_index else right_family
        candidates = tuple(
            candidate for candidate in layer.candidates if candidate.config_flags == required_family
        )
        if not candidates:
            raise RuntimeError(
                f"No IK candidates with config {required_family} at row {absolute_index}."
            )
        forced_layers.append(_IKLayer(pose=layer.pose, candidates=candidates))
    return tuple(forced_layers)


def _summary_payload(
    *,
    result: ProfileEvaluationResult,
    window_start_index: int,
    window_end_index: int,
    switch_after_index: int,
    left_family: tuple[int, int, int],
    right_family: tuple[int, int, int],
) -> dict[str, Any]:
    transitions: list[dict[str, Any]] = []
    for index, (previous, current) in enumerate(zip(result.selected_path, result.selected_path[1:])):
        if previous.config_flags == current.config_flags:
            continue
        deltas = [
            abs(float(current_joint) - float(previous_joint))
            for previous_joint, current_joint in zip(previous.joints, current.joints)
        ]
        transitions.append(
            {
                "segment_index": index,
                "left_label": result.row_labels[index],
                "right_label": result.row_labels[index + 1],
                "from_config": list(previous.config_flags),
                "to_config": list(current.config_flags),
                "max_joint_delta_deg": max(deltas, default=0.0),
                "joint_deltas_deg": deltas,
            }
        )

    return {
        "method": "forced IK config-family transition demo",
        "note": (
            "This is a visual diagnostic path, not an automatic production "
            "selection policy. It keeps target pose rows fixed and constrains "
            "IK families around the selected cut."
        ),
        "window_start_index": int(window_start_index),
        "window_end_index": int(window_end_index),
        "switch_after_index": int(switch_after_index),
        "left_family": list(left_family),
        "right_family": list(right_family),
        "status": result.status,
        "selected_path_count": len(result.selected_path),
        "row_count": len(result.row_labels),
        "config_switches": int(result.config_switches),
        "bridge_like_segments": int(result.bridge_like_segments),
        "worst_joint_step_deg": float(result.worst_joint_step_deg),
        "mean_joint_step_deg": float(result.mean_joint_step_deg),
        "family_transitions": transitions,
    }


def _family_token(family: Sequence[int]) -> str:
    return "".join(str(int(value)) for value in family)
