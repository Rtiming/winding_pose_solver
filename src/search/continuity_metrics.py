from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence


@dataclass(frozen=True)
class SegmentContinuityDiagnostic:
    segment_index: int
    left_label: str
    right_label: str
    config_changed: bool
    max_joint_delta_deg: float
    mean_joint_delta_deg: float
    tcp_step_mm: float
    branch_flip_ratio: float


def _joint_tuple(path_entry: object) -> tuple[float, ...]:
    joints = getattr(path_entry, "joints", path_entry)
    if isinstance(joints, tuple):
        return tuple(float(value) for value in joints)
    if isinstance(joints, list):
        return tuple(float(value) for value in joints)
    return ()


def _config_tuple(path_entry: object) -> tuple[int, ...]:
    config_flags = getattr(path_entry, "config_flags", ())
    if isinstance(config_flags, tuple):
        return tuple(int(value) for value in config_flags)
    if isinstance(config_flags, list):
        return tuple(int(value) for value in config_flags)
    return ()


def _safe_row_label(
    row_labels: Sequence[str] | None,
    row_index: int,
) -> str:
    if row_labels is None or row_index >= len(row_labels):
        return str(row_index)
    return str(row_labels[row_index])


def _tcp_step_mm(
    pose_rows: Sequence[dict[str, float]] | None,
    left_index: int,
    right_index: int,
) -> float:
    if pose_rows is None:
        return 0.0
    if left_index < 0 or right_index < 0:
        return 0.0
    if left_index >= len(pose_rows) or right_index >= len(pose_rows):
        return 0.0
    left_row = pose_rows[left_index]
    right_row = pose_rows[right_index]
    dx_mm = float(right_row.get("x_mm", 0.0)) - float(left_row.get("x_mm", 0.0))
    dy_mm = float(right_row.get("y_mm", 0.0)) - float(left_row.get("y_mm", 0.0))
    dz_mm = float(right_row.get("z_mm", 0.0)) - float(left_row.get("z_mm", 0.0))
    return math.sqrt(dx_mm * dx_mm + dy_mm * dy_mm + dz_mm * dz_mm)


def _segment_as_dict(segment: SegmentContinuityDiagnostic) -> dict[str, Any]:
    return {
        "segment_index": int(segment.segment_index),
        "left_label": segment.left_label,
        "right_label": segment.right_label,
        "config_changed": bool(segment.config_changed),
        "max_joint_delta_deg": float(segment.max_joint_delta_deg),
        "mean_joint_delta_deg": float(segment.mean_joint_delta_deg),
        "tcp_step_mm": float(segment.tcp_step_mm),
        "branch_flip_ratio": float(segment.branch_flip_ratio),
    }


def compute_segment_continuity_diagnostics(
    selected_path: Sequence[object],
    *,
    row_labels: Sequence[str] | None = None,
    pose_rows: Sequence[dict[str, float]] | None = None,
    ratio_eps_mm: float = 1e-3,
) -> tuple[SegmentContinuityDiagnostic, ...]:
    diagnostics: list[SegmentContinuityDiagnostic] = []
    if len(selected_path) <= 1:
        return ()

    epsilon_mm = max(1e-9, float(ratio_eps_mm))
    for segment_index, (previous_entry, current_entry) in enumerate(
        zip(selected_path, selected_path[1:])
    ):
        previous_joints = _joint_tuple(previous_entry)
        current_joints = _joint_tuple(current_entry)
        joint_deltas = [
            abs(current - previous)
            for previous, current in zip(previous_joints, current_joints)
        ]
        max_joint_delta_deg = max(joint_deltas, default=0.0)
        mean_joint_delta_deg = (
            sum(joint_deltas) / len(joint_deltas) if joint_deltas else 0.0
        )
        tcp_step_mm = _tcp_step_mm(
            pose_rows,
            segment_index,
            segment_index + 1,
        )
        branch_flip_ratio = max_joint_delta_deg / (tcp_step_mm + epsilon_mm)
        diagnostics.append(
            SegmentContinuityDiagnostic(
                segment_index=int(segment_index),
                left_label=_safe_row_label(row_labels, segment_index),
                right_label=_safe_row_label(row_labels, segment_index + 1),
                config_changed=(
                    _config_tuple(previous_entry) != _config_tuple(current_entry)
                ),
                max_joint_delta_deg=float(max_joint_delta_deg),
                mean_joint_delta_deg=float(mean_joint_delta_deg),
                tcp_step_mm=float(tcp_step_mm),
                branch_flip_ratio=float(branch_flip_ratio),
            )
        )
    return tuple(diagnostics)


def summarize_branch_jump_metrics(
    selected_path: Sequence[object],
    *,
    row_labels: Sequence[str] | None = None,
    pose_rows: Sequence[dict[str, float]] | None = None,
    big_circle_step_deg_threshold: float,
    branch_flip_ratio_threshold: float,
    ratio_eps_mm: float = 1e-3,
) -> dict[str, Any]:
    segment_diagnostics = compute_segment_continuity_diagnostics(
        selected_path,
        row_labels=row_labels,
        pose_rows=pose_rows,
        ratio_eps_mm=ratio_eps_mm,
    )
    big_circle_threshold = float(big_circle_step_deg_threshold)
    ratio_threshold = float(branch_flip_ratio_threshold)
    big_circle_cutoff = big_circle_threshold - 1e-9
    violent_cutoff = ratio_threshold - 1e-9
    big_circle_step_count = 0
    violent_segments: list[dict[str, Any]] = []
    max_branch_flip_ratio = 0.0
    for segment in segment_diagnostics:
        branch_flip_ratio = float(segment.branch_flip_ratio)
        if branch_flip_ratio > max_branch_flip_ratio:
            max_branch_flip_ratio = branch_flip_ratio
        if segment.max_joint_delta_deg >= big_circle_cutoff:
            big_circle_step_count += 1
            if branch_flip_ratio >= violent_cutoff:
                violent_segments.append(_segment_as_dict(segment))
    return {
        "segment_diagnostics": segment_diagnostics,
        "big_circle_step_count": int(big_circle_step_count),
        "max_branch_flip_ratio": float(max_branch_flip_ratio),
        "violent_branch_segments": tuple(violent_segments),
    }
