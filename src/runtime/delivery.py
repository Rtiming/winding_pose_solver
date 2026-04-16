from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

from src.core.collab_models import (
    ProfileEvaluationRequest,
    ProfileEvaluationResult,
    load_json_file,
    write_json_file,
)


STRICT_DELIVERY_GATE: dict[str, object] = {
    "policy": "objective_reachable",
    "required_invalid_row_count": 0,
    "required_ik_empty_row_count": 0,
    "requires_selected_joint_path": True,
    "treats_status_as_diagnostic": True,
    "treats_config_switches_as_warning": True,
    "treats_bridge_like_segments_as_warning": True,
    "treats_worst_joint_step_as_warning": True,
}


def _result_has_selected_path(result: object) -> bool:
    selected_path = getattr(result, "selected_path", None)
    row_labels = getattr(result, "row_labels", None)
    if selected_path is None:
        return False
    try:
        selected_count = len(selected_path)
    except TypeError:
        return False
    if selected_count <= 0:
        return False
    if row_labels is None:
        return True
    try:
        return selected_count == len(row_labels)
    except TypeError:
        return True


def result_is_strictly_valid(result: object | None) -> bool:
    """Return whether a result is deliverable for the user's target objective.

    Historical callers use the "strict" name, but the current delivery policy
    intentionally treats continuity diagnostics as warnings. The hard gate is
    whether every target row has an IK solution and a complete selected path.
    """
    if result is None:
        return False
    return (
        int(getattr(result, "invalid_row_count", 0)) == 0
        and int(getattr(result, "ik_empty_row_count", 0)) == 0
        and _result_has_selected_path(result)
    )


def _payload_has_selected_path(payload: dict[str, object]) -> bool:
    selected_path = payload.get("selected_path")
    row_labels = payload.get("row_labels")
    if not isinstance(selected_path, list) or not selected_path:
        return False
    if isinstance(row_labels, list):
        return len(selected_path) == len(row_labels)
    return True


def result_payload_is_strictly_valid(payload: dict[str, object] | None) -> bool:
    if payload is None:
        return False
    return (
        int(payload.get("invalid_row_count", 0)) == 0
        and int(payload.get("ik_empty_row_count", 0)) == 0
        and _payload_has_selected_path(payload)
    )


def result_semantic_status(result: object | None) -> str:
    if result is None:
        return "success"
    return "valid" if result_is_strictly_valid(result) else "invalid"


def result_has_continuity_warnings(result: object | None) -> bool:
    if result is None:
        return False
    failing_segments = getattr(result, "failing_segments", ())
    return (
        int(getattr(result, "config_switches", 0)) > 0
        or int(getattr(result, "bridge_like_segments", 0)) > 0
        or bool(failing_segments)
    )


def result_quality_summary(result: ProfileEvaluationResult) -> dict[str, object]:
    return {
        "request_id": str(result.request_id),
        "status": str(result.status),
        "ik_empty_row_count": int(result.ik_empty_row_count),
        "config_switches": int(result.config_switches),
        "bridge_like_segments": int(result.bridge_like_segments),
        "invalid_row_count": int(result.invalid_row_count),
        "worst_joint_step_deg": float(result.worst_joint_step_deg),
        "mean_joint_step_deg": float(result.mean_joint_step_deg),
        "total_cost": float(result.total_cost),
        "objective_reachable": bool(result_is_strictly_valid(result)),
        "official_delivery_allowed": bool(result_is_strictly_valid(result)),
        "strictly_valid": bool(result_is_strictly_valid(result)),
        "continuity_warnings": {
            "status": str(result.status),
            "config_switches": int(result.config_switches),
            "bridge_like_segments": int(result.bridge_like_segments),
            "worst_joint_step_deg": float(result.worst_joint_step_deg),
        },
    }


def selected_joint_path_rows(result: ProfileEvaluationResult) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for row_label, inserted_flag, selected_entry in zip(
        result.row_labels,
        result.inserted_flags,
        result.selected_path,
    ):
        rows.append(
            {
                "row_label": str(row_label),
                "inserted_transition_point": bool(inserted_flag),
                "joints_deg": [float(value) for value in selected_entry.joints],
                "config_flags": [int(value) for value in selected_entry.config_flags],
            }
        )
    return rows


def _stringify_artifact_paths(
    artifacts: dict[str, str | Path | None] | None,
) -> dict[str, str | None]:
    return {
        str(label): (str(path) if path is not None else None)
        for label, path in (artifacts or {}).items()
    }


def build_handoff_payload(
    *,
    run_id: str,
    receiver_request: ProfileEvaluationRequest,
    selected_result: ProfileEvaluationResult,
    artifacts: dict[str, str | Path | None] | None = None,
    diagnostics_artifacts: dict[str, str | Path | None] | None = None,
    producer_role: str = "online/server",
    consumer_role: str = "online/receiver",
    allow_invalid: bool = False,
    package_kind: str = "official_handoff",
) -> dict[str, object]:
    delivery_allowed = result_is_strictly_valid(selected_result)
    if not delivery_allowed and not allow_invalid:
        raise ValueError("Refusing to build a handoff package for a non-deliverable result.")

    return {
        "schema_version": 1,
        "produced_at": datetime.now().isoformat(timespec="seconds"),
        "run_id": str(run_id),
        "package_kind": package_kind,
        "producer_role": producer_role,
        "consumer_role": consumer_role,
        "delivery_gate": {
            **STRICT_DELIVERY_GATE,
            "objective_reachable": bool(delivery_allowed),
            "official_delivery_allowed": bool(delivery_allowed),
            "debug_invalid_output": bool(not delivery_allowed),
        },
        "strict_quality_gate": {
            **STRICT_DELIVERY_GATE,
            "strictly_valid": bool(delivery_allowed),
            "objective_reachable": bool(delivery_allowed),
            "official_delivery_allowed": bool(delivery_allowed),
            "debug_invalid_output": bool(not delivery_allowed),
        },
        "selection": result_quality_summary(selected_result),
        "selected_profile": {
            "frame_a_origin_yz_profile_mm": [
                [float(dy_mm), float(dz_mm)]
                for dy_mm, dz_mm in selected_result.frame_a_origin_yz_profile_mm
            ],
            "row_labels": [str(label) for label in selected_result.row_labels],
            "inserted_flags": [bool(flag) for flag in selected_result.inserted_flags],
        },
        "selected_joint_path": selected_joint_path_rows(selected_result),
        "reference_pose_rows": [dict(row) for row in receiver_request.reference_pose_rows],
        "optimized_pose_rows": (
            None
            if selected_result.pose_rows is None
            else [dict(row) for row in selected_result.pose_rows]
        ),
        "server_diagnostics": _stringify_artifact_paths(diagnostics_artifacts),
        "artifacts": _stringify_artifact_paths(artifacts),
        "receiver_request": receiver_request.to_dict(),
    }


def write_handoff_package(
    output_path: str | Path,
    *,
    run_id: str,
    receiver_request: ProfileEvaluationRequest,
    selected_result: ProfileEvaluationResult,
    artifacts: dict[str, str | Path | None] | None = None,
    diagnostics_artifacts: dict[str, str | Path | None] | None = None,
    producer_role: str = "online/server",
    consumer_role: str = "online/receiver",
    allow_invalid: bool = False,
    package_kind: str = "official_handoff",
) -> Path:
    payload = build_handoff_payload(
        run_id=run_id,
        receiver_request=receiver_request,
        selected_result=selected_result,
        artifacts=artifacts,
        diagnostics_artifacts=diagnostics_artifacts,
        producer_role=producer_role,
        consumer_role=consumer_role,
        allow_invalid=allow_invalid,
        package_kind=package_kind,
    )
    return write_json_file(output_path, payload)


def load_handoff_package(path: str | Path, *, allow_invalid: bool = False) -> dict[str, Any]:
    payload = load_json_file(path)
    if int(payload.get("schema_version", 0)) != 1:
        raise ValueError(f"Unsupported handoff schema_version: {payload.get('schema_version')}")
    if (
        not allow_invalid
        and not bool(dict(payload.get("strict_quality_gate", {})).get("official_delivery_allowed"))
    ):
        raise ValueError("Handoff package is not marked deliverable.")
    if "receiver_request" not in payload:
        raise ValueError("Handoff package is missing receiver_request.")
    return payload
