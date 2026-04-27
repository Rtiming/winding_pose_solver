from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

from src.core.collab_models import (
    ProfileEvaluationRequest,
    ProfileEvaluationResult,
    write_json_file,
)
from src.core.motion_settings import motion_settings_to_dict
from src.core.pose_csv import REQUIRED_COLUMNS, load_pose_rows
from src.runtime.app import AppRuntimeSettings
from src.runtime.request_builder import refresh_pose_csv
from src.robodk_runtime.eval_worker import evaluate_request, open_offline_ik_station_context
from src.search.global_search import _extract_row_labels


@dataclass(frozen=True)
class YZNullspaceDemoArtifacts:
    run_id: str
    output_dir: Path
    pose_csv_path: Path
    baseline_request_path: Path
    baseline_result_path: Path
    baseline_joint_path_csv: Path
    optimized_request_path: Path
    optimized_result_path: Path
    optimized_joint_path_csv: Path
    optimized_pose_csv: Path
    summary_path: Path
    baseline_result: ProfileEvaluationResult
    optimized_result: ProfileEvaluationResult
    window_start_index: int
    window_end_index: int
    focus_start_label: str
    focus_end_label: str


def default_yz_nullspace_demo_run_id() -> str:
    token = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"yz_nullspace_{token}"


def run_yz_nullspace_demo(
    settings: AppRuntimeSettings,
    *,
    start_index: int,
    end_index: int,
    padding: int = 8,
    output_root: str | Path = Path("artifacts/diagnostics/yz_nullspace_demo"),
    run_id: str | None = None,
    refresh_csv: bool = True,
    bridge_trigger_joint_delta_deg: float | None = None,
) -> YZNullspaceDemoArtifacts:
    """Run a focused Frame-A Y/Z two-DOF validation.

    The baseline keeps the Frame-A Y/Z profile fixed at zero.  The optimized run
    lets the existing local repair search use per-row Y/Z offsets, which is the
    project-level analogue of using a two-dimensional null space while preserving
    the hard target-row constraints.
    """

    run_id = run_id or default_yz_nullspace_demo_run_id()
    output_dir = Path(output_root) / run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    if refresh_csv:
        refresh_pose_csv(settings)

    pose_rows = tuple(load_pose_rows(settings.tool_poses_frame2_csv))
    window_start, window_end = _resolve_window(
        row_count=len(pose_rows),
        start_index=start_index,
        end_index=end_index,
        padding=padding,
    )
    window_pose_rows = tuple(dict(row) for row in pose_rows[window_start : window_end + 1])
    row_labels = _extract_row_labels(window_pose_rows)
    zero_profile = tuple((0.0, 0.0) for _ in window_pose_rows)
    inserted_flags = tuple(False for _ in window_pose_rows)

    focus_start_label = _label_for_index(pose_rows, start_index)
    focus_end_label = _label_for_index(pose_rows, end_index)

    common_metadata: dict[str, Any] = {
        "entrypoint": "yz_nullspace_demo",
        "method": "4 hard task constraints with Frame-A Y/Z as two free variables",
        "window": {
            "requested_start_index": int(start_index),
            "requested_end_index": int(end_index),
            "padding": int(padding),
            "window_start_index": int(window_start),
            "window_end_index": int(window_end),
            "focus_start_label": focus_start_label,
            "focus_end_label": focus_end_label,
            "row_labels": list(row_labels),
        },
    }
    if bridge_trigger_joint_delta_deg is not None:
        common_metadata["bridge_trigger_joint_delta_deg_override"] = float(
            bridge_trigger_joint_delta_deg
        )

    baseline_request = _build_demo_request(
        settings,
        request_id=f"{run_id}_baseline_fixed_yz",
        pose_rows=window_pose_rows,
        row_labels=row_labels,
        inserted_flags=inserted_flags,
        frame_a_origin_yz_profile_mm=zero_profile,
        strategy="exact_profile",
        run_window_repair=False,
        run_inserted_repair=False,
        bridge_trigger_joint_delta_deg=bridge_trigger_joint_delta_deg,
        metadata={**common_metadata, "variant": "baseline_fixed_yz"},
    )
    optimized_request = _build_demo_request(
        settings,
        request_id=f"{run_id}_optimized_yz_dof",
        pose_rows=window_pose_rows,
        row_labels=row_labels,
        inserted_flags=inserted_flags,
        frame_a_origin_yz_profile_mm=zero_profile,
        strategy="full_search",
        run_window_repair=True,
        run_inserted_repair=True,
        bridge_trigger_joint_delta_deg=bridge_trigger_joint_delta_deg,
        metadata={**common_metadata, "variant": "optimized_yz_dof"},
    )

    context = open_offline_ik_station_context(
        robot_name=settings.robot_name,
        frame_name=settings.frame_name,
    )

    baseline_result, _baseline_search = evaluate_request(baseline_request, context)
    optimized_result, _optimized_search = evaluate_request(optimized_request, context)

    baseline_request_path = write_json_file(
        output_dir / "baseline_request.json",
        baseline_request.to_dict(),
    )
    baseline_result_path = write_json_file(
        output_dir / "baseline_result.json",
        baseline_result.to_dict(),
    )
    optimized_request_path = write_json_file(
        output_dir / "optimized_request.json",
        optimized_request.to_dict(),
    )
    optimized_result_path = write_json_file(
        output_dir / "optimized_result.json",
        optimized_result.to_dict(),
    )

    baseline_joint_path_csv = _write_selected_joint_path_csv(
        baseline_result,
        output_dir / "baseline_selected_joint_path.csv",
    )
    optimized_joint_path_csv = _write_selected_joint_path_csv(
        optimized_result,
        output_dir / "optimized_selected_joint_path.csv",
    )
    optimized_pose_csv = _write_result_pose_csv(
        optimized_result,
        output_dir / "optimized_pose_path.csv",
    )
    summary_path = write_json_file(
        output_dir / "summary.json",
        _build_summary_payload(
            baseline_result=baseline_result,
            optimized_result=optimized_result,
            window_start_index=window_start,
            window_end_index=window_end,
            focus_start_label=focus_start_label,
            focus_end_label=focus_end_label,
            pose_csv_path=settings.tool_poses_frame2_csv,
        ),
    )

    return YZNullspaceDemoArtifacts(
        run_id=run_id,
        output_dir=output_dir,
        pose_csv_path=Path(settings.tool_poses_frame2_csv),
        baseline_request_path=baseline_request_path,
        baseline_result_path=baseline_result_path,
        baseline_joint_path_csv=baseline_joint_path_csv,
        optimized_request_path=optimized_request_path,
        optimized_result_path=optimized_result_path,
        optimized_joint_path_csv=optimized_joint_path_csv,
        optimized_pose_csv=optimized_pose_csv,
        summary_path=summary_path,
        baseline_result=baseline_result,
        optimized_result=optimized_result,
        window_start_index=window_start,
        window_end_index=window_end,
        focus_start_label=focus_start_label,
        focus_end_label=focus_end_label,
    )


def _resolve_window(
    *,
    row_count: int,
    start_index: int,
    end_index: int,
    padding: int,
) -> tuple[int, int]:
    if row_count <= 0:
        raise ValueError("No pose rows are available.")
    if start_index < 0 or end_index < 0 or start_index > end_index:
        raise ValueError(f"Invalid focus window [{start_index}, {end_index}].")
    if end_index >= row_count:
        raise ValueError(
            f"Invalid focus end index {end_index}; pose row count is {row_count}."
        )

    safe_padding = max(0, int(padding))
    return (
        max(0, int(start_index) - safe_padding),
        min(row_count - 1, int(end_index) + safe_padding),
    )


def _label_for_index(pose_rows: Sequence[dict[str, float]], row_index: int) -> str:
    if row_index < 0 or row_index >= len(pose_rows):
        return str(row_index)
    label = _extract_row_labels((pose_rows[row_index],))[0]
    return str(label)


def _build_demo_request(
    settings: AppRuntimeSettings,
    *,
    request_id: str,
    pose_rows: tuple[dict[str, float], ...],
    row_labels: tuple[str, ...],
    inserted_flags: tuple[bool, ...],
    frame_a_origin_yz_profile_mm: tuple[tuple[float, float], ...],
    strategy: str,
    run_window_repair: bool,
    run_inserted_repair: bool,
    bridge_trigger_joint_delta_deg: float | None,
    metadata: dict[str, Any],
) -> ProfileEvaluationRequest:
    motion_settings = motion_settings_to_dict(settings.motion_settings)
    motion_settings["ik_backend"] = "six_axis_ik"
    if bridge_trigger_joint_delta_deg is not None:
        motion_settings["bridge_trigger_joint_delta_deg"] = float(
            bridge_trigger_joint_delta_deg
        )
    return ProfileEvaluationRequest(
        request_id=request_id,
        robot_name=settings.robot_name,
        frame_name=settings.frame_name,
        motion_settings=motion_settings,
        reference_pose_rows=pose_rows,
        frame_a_origin_yz_profile_mm=frame_a_origin_yz_profile_mm,
        row_labels=row_labels,
        inserted_flags=inserted_flags,
        strategy=strategy,
        start_joints=None,
        run_window_repair=run_window_repair,
        run_inserted_repair=run_inserted_repair,
        include_pose_rows_in_result=True,
        create_program=False,
        program_name=None,
        optimized_csv_path=None,
        metadata=metadata,
    )


def _write_selected_joint_path_csv(
    result: ProfileEvaluationResult,
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8-sig", newline="") as handle:
        fieldnames = [
            "row_label",
            "inserted_transition_point",
            "config_flags",
            "j1_deg",
            "j2_deg",
            "j3_deg",
            "j4_deg",
            "j5_deg",
            "j6_deg",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row_label, inserted_flag, selected_entry in zip(
            result.row_labels,
            result.inserted_flags,
            result.selected_path,
        ):
            joints = tuple(float(value) for value in selected_entry.joints)
            writer.writerow(
                {
                    "row_label": str(row_label),
                    "inserted_transition_point": int(bool(inserted_flag)),
                    "config_flags": ",".join(
                        str(int(value)) for value in selected_entry.config_flags
                    ),
                    "j1_deg": joints[0] if len(joints) > 0 else "",
                    "j2_deg": joints[1] if len(joints) > 1 else "",
                    "j3_deg": joints[2] if len(joints) > 2 else "",
                    "j4_deg": joints[3] if len(joints) > 3 else "",
                    "j5_deg": joints[4] if len(joints) > 4 else "",
                    "j6_deg": joints[5] if len(joints) > 5 else "",
                }
            )
    return output_path


def _write_result_pose_csv(
    result: ProfileEvaluationResult,
    output_path: Path,
) -> Path:
    if result.pose_rows is None:
        raise ValueError("Result does not include pose_rows; cannot write pose CSV.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8-sig", newline="") as handle:
        fieldnames = [
            "row_label",
            "inserted_transition_point",
            "frame_a_origin_dy_mm",
            "frame_a_origin_dz_mm",
            *REQUIRED_COLUMNS,
        ]
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
                    "row_label": str(row_label),
                    "inserted_transition_point": int(bool(inserted_flag)),
                    "frame_a_origin_dy_mm": float(dy_mm),
                    "frame_a_origin_dz_mm": float(dz_mm),
                    **{column: float(pose_row[column]) for column in REQUIRED_COLUMNS},
                }
            )
    return output_path


def _build_summary_payload(
    *,
    baseline_result: ProfileEvaluationResult,
    optimized_result: ProfileEvaluationResult,
    window_start_index: int,
    window_end_index: int,
    focus_start_label: str,
    focus_end_label: str,
    pose_csv_path: Path,
) -> dict[str, Any]:
    baseline_worst = float(baseline_result.worst_joint_step_deg)
    optimized_worst = float(optimized_result.worst_joint_step_deg)
    worst_delta = (
        baseline_worst - optimized_worst
        if math.isfinite(baseline_worst) and math.isfinite(optimized_worst)
        else None
    )
    return {
        "method": "Frame-A Y/Z two-DOF profile search",
        "constraint_interpretation": {
            "hard_constraints": [
                "target row order is preserved",
                "target pose rows come from the centerline-derived task geometry",
                "IK reachability must be non-empty for every row",
                "selected path must contain one joint vector per row",
            ],
            "free_variables": [
                "frame_a_origin_dy_mm",
                "frame_a_origin_dz_mm",
            ],
        },
        "pose_csv_path": str(pose_csv_path),
        "window": {
            "window_start_index": int(window_start_index),
            "window_end_index": int(window_end_index),
            "focus_start_label": str(focus_start_label),
            "focus_end_label": str(focus_end_label),
        },
        "baseline": _result_summary(baseline_result),
        "optimized": _result_summary(optimized_result),
        "comparison": {
            "optimized_reachable": _has_complete_selected_path(optimized_result),
            "baseline_reachable": _has_complete_selected_path(baseline_result),
            "worst_joint_step_delta_deg": worst_delta,
            "config_switch_delta": int(baseline_result.config_switches)
            - int(optimized_result.config_switches),
            "bridge_like_segment_delta": int(baseline_result.bridge_like_segments)
            - int(optimized_result.bridge_like_segments),
        },
    }


def _result_summary(result: ProfileEvaluationResult) -> dict[str, Any]:
    nonzero_offsets = sum(
        1
        for dy_mm, dz_mm in result.frame_a_origin_yz_profile_mm
        if abs(float(dy_mm)) > 1e-9 or abs(float(dz_mm)) > 1e-9
    )
    y_values = [float(item[0]) for item in result.frame_a_origin_yz_profile_mm]
    z_values = [float(item[1]) for item in result.frame_a_origin_yz_profile_mm]
    return {
        "request_id": result.request_id,
        "status": result.status,
        "invalid_row_count": int(result.invalid_row_count),
        "ik_empty_row_count": int(result.ik_empty_row_count),
        "selected_path_count": len(result.selected_path),
        "row_count": len(result.row_labels),
        "config_switches": int(result.config_switches),
        "bridge_like_segments": int(result.bridge_like_segments),
        "worst_joint_step_deg": _finite_float_or_none(result.worst_joint_step_deg),
        "mean_joint_step_deg": _finite_float_or_none(result.mean_joint_step_deg),
        "total_cost": _finite_float_or_none(result.total_cost),
        "timing_seconds": float(result.timing_seconds),
        "nonzero_offset_count": int(nonzero_offsets),
        "dy_range_mm": [
            min(y_values, default=0.0),
            max(y_values, default=0.0),
        ],
        "dz_range_mm": [
            min(z_values, default=0.0),
            max(z_values, default=0.0),
        ],
        "failing_segments": [
            {
                "segment_index": segment.segment_index,
                "left_label": segment.left_label,
                "right_label": segment.right_label,
                "max_joint_delta_deg": segment.max_joint_delta_deg,
                "config_changed": segment.config_changed,
            }
            for segment in result.failing_segments
        ],
    }


def _finite_float_or_none(value: float) -> float | None:
    value = float(value)
    if not math.isfinite(value):
        return None
    return value


def _has_complete_selected_path(result: ProfileEvaluationResult) -> bool:
    return (
        int(result.invalid_row_count) == 0
        and int(result.ik_empty_row_count) == 0
        and bool(result.selected_path)
        and len(result.selected_path) == len(result.row_labels)
    )
