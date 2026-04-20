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
    "requires_status_valid": True,
    "requires_closed_winding_terminal_if_closed_path": True,
    "required_bridge_like_segments": 0,
    "required_big_circle_step_count": 0,
    "required_worst_joint_step_deg_limit": 60.0,
    "treats_config_switches_as_warning": True,
    "treats_bridge_like_segments_as_warning": False,
    "treats_big_circle_step_as_warning": False,
    "treats_worst_joint_step_as_warning": False,
}


def _safe_int(value: object, default_value: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default_value)


def _safe_float(value: object, default_value: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default_value)


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


def _payload_has_selected_path(payload: dict[str, object]) -> bool:
    selected_path = payload.get("selected_path")
    row_labels = payload.get("row_labels")
    if not isinstance(selected_path, list) or not selected_path:
        return False
    if isinstance(row_labels, list):
        return len(selected_path) == len(row_labels)
    return True


def _result_passes_closed_winding_terminal_if_needed(result: object) -> bool:
    metadata = getattr(result, "metadata", None)
    if not isinstance(metadata, dict):
        return True
    report = metadata.get("closed_winding_terminal")
    if not isinstance(report, dict):
        return True
    if not bool(report.get("closed_path", False)):
        return True
    return bool(report.get("passed", False))


def _payload_passes_closed_winding_terminal_if_needed(payload: dict[str, object]) -> bool:
    metadata = payload.get("metadata", {})
    if not isinstance(metadata, dict):
        return True
    report = metadata.get("closed_winding_terminal")
    if not isinstance(report, dict):
        return True
    if not bool(report.get("closed_path", False)):
        return True
    return bool(report.get("passed", False))


def _delivery_gate_details_from_metrics(
    *,
    status_value: object,
    invalid_row_count: int,
    ik_empty_row_count: int,
    has_selected_path: bool,
    passes_closed_terminal: bool,
    bridge_like_segments: int,
    big_circle_step_count: int,
    worst_joint_step_deg: float,
    worst_step_limit: float,
) -> dict[str, object]:
    status_text = str(status_value)
    status_valid = status_text == "valid"
    objective_reachable = (
        status_valid
        and invalid_row_count == int(STRICT_DELIVERY_GATE["required_invalid_row_count"])
        and ik_empty_row_count == int(STRICT_DELIVERY_GATE["required_ik_empty_row_count"])
        and has_selected_path
        and passes_closed_terminal
    )

    strictly_valid = (
        objective_reachable
        and bridge_like_segments == int(STRICT_DELIVERY_GATE["required_bridge_like_segments"])
        and big_circle_step_count == int(STRICT_DELIVERY_GATE["required_big_circle_step_count"])
        and worst_joint_step_deg <= float(worst_step_limit) + 1e-9
    )

    block_reasons: list[dict[str, object]] = []
    if not status_valid:
        block_reasons.append(
            {
                "code": "status_not_valid",
                "message": "result.status is not 'valid'",
                "value": status_text,
            }
        )
    if invalid_row_count != int(STRICT_DELIVERY_GATE["required_invalid_row_count"]):
        block_reasons.append(
            {
                "code": "invalid_row_count_nonzero",
                "message": "invalid_row_count must be zero",
                "value": int(invalid_row_count),
            }
        )
    if ik_empty_row_count != int(STRICT_DELIVERY_GATE["required_ik_empty_row_count"]):
        block_reasons.append(
            {
                "code": "ik_empty_row_count_nonzero",
                "message": "ik_empty_row_count must be zero",
                "value": int(ik_empty_row_count),
            }
        )
    if not has_selected_path:
        block_reasons.append(
            {
                "code": "selected_path_missing_or_incomplete",
                "message": "selected_path must exist and match row_labels length",
            }
        )
    if not passes_closed_terminal:
        block_reasons.append(
            {
                "code": "closed_winding_terminal_failed",
                "message": "closed winding terminal hard rule failed",
            }
        )
    if bridge_like_segments > int(STRICT_DELIVERY_GATE["required_bridge_like_segments"]):
        block_reasons.append(
            {
                "code": "bridge_like_segments_nonzero",
                "message": "bridge_like_segments must be zero for official delivery",
                "value": int(bridge_like_segments),
            }
        )
    if big_circle_step_count > int(STRICT_DELIVERY_GATE["required_big_circle_step_count"]):
        block_reasons.append(
            {
                "code": "big_circle_step_detected",
                "message": "big_circle_step_count must be zero for official delivery",
                "value": int(big_circle_step_count),
            }
        )
    if worst_joint_step_deg > float(worst_step_limit) + 1e-9:
        block_reasons.append(
            {
                "code": "worst_joint_step_exceeds_limit",
                "message": "worst_joint_step_deg exceeds official limit",
                "value": float(worst_joint_step_deg),
                "limit": float(worst_step_limit),
            }
        )

    gate_tier = "official" if strictly_valid else ("debug" if objective_reachable else "diagnostic")
    return {
        "strictly_valid": bool(strictly_valid),
        "objective_reachable": bool(objective_reachable),
        "official_delivery_allowed": bool(strictly_valid),
        "gate_tier": gate_tier,
        "block_reasons": block_reasons,
    }


def _result_delivery_gate_details(result: object | None) -> dict[str, object]:
    if result is None:
        return {
            "strictly_valid": False,
            "objective_reachable": False,
            "official_delivery_allowed": False,
            "gate_tier": "diagnostic",
            "block_reasons": [
                {
                    "code": "missing_result",
                    "message": "result is None",
                }
            ],
        }

    invalid_row_count = _safe_int(getattr(result, "invalid_row_count", 0))
    ik_empty_row_count = _safe_int(getattr(result, "ik_empty_row_count", 0))
    has_selected_path = _result_has_selected_path(result)
    passes_closed_terminal = _result_passes_closed_winding_terminal_if_needed(result)
    bridge_like_segments = _safe_int(getattr(result, "bridge_like_segments", 0))
    big_circle_step_count = _safe_int(getattr(result, "big_circle_step_count", 0))
    worst_joint_step_deg = _safe_float(getattr(result, "worst_joint_step_deg", 0.0))
    worst_step_limit = _safe_float(
        getattr(
            result,
            "motion_settings",
            {},
        ).get(
            "official_worst_joint_step_deg_limit",
            STRICT_DELIVERY_GATE["required_worst_joint_step_deg_limit"],
        )
        if isinstance(getattr(result, "motion_settings", {}), dict)
        else STRICT_DELIVERY_GATE["required_worst_joint_step_deg_limit"]
    )

    return _delivery_gate_details_from_metrics(
        status_value=getattr(result, "status", "unknown"),
        invalid_row_count=invalid_row_count,
        ik_empty_row_count=ik_empty_row_count,
        has_selected_path=has_selected_path,
        passes_closed_terminal=passes_closed_terminal,
        bridge_like_segments=bridge_like_segments,
        big_circle_step_count=big_circle_step_count,
        worst_joint_step_deg=worst_joint_step_deg,
        worst_step_limit=worst_step_limit,
    )


def summary_metrics_delivery_gate_details(row: dict[str, object]) -> dict[str, object]:
    """Apply the official quality gate to compact metric-only summary rows.

    Origin-search result rows intentionally do not carry selected_path or the
    closed-terminal report because they are compact search summaries.  The
    full handoff path is still validated later through result_quality_summary()
    and load_handoff_package().
    """
    worst_step_limit = _safe_float(
        row.get(
            "official_worst_joint_step_deg_limit",
            STRICT_DELIVERY_GATE["required_worst_joint_step_deg_limit"],
        )
    )
    return _delivery_gate_details_from_metrics(
        status_value=row.get("status", "invalid"),
        invalid_row_count=_safe_int(row.get("invalid_row_count", 0)),
        ik_empty_row_count=_safe_int(row.get("ik_empty_row_count", 0)),
        has_selected_path=True,
        passes_closed_terminal=True,
        bridge_like_segments=_safe_int(row.get("bridge_like_segments", 0)),
        big_circle_step_count=_safe_int(row.get("big_circle_step_count", 0)),
        worst_joint_step_deg=_safe_float(row.get("worst_joint_step_deg", 0.0)),
        worst_step_limit=worst_step_limit,
    )


def _payload_delivery_gate_details(payload: dict[str, object] | None) -> dict[str, object]:
    if payload is None:
        return {
            "strictly_valid": False,
            "objective_reachable": False,
            "official_delivery_allowed": False,
            "gate_tier": "diagnostic",
            "block_reasons": [
                {
                    "code": "missing_payload",
                    "message": "payload is None",
                }
            ],
        }

    pseudo_result = type("PayloadResult", (), {})()
    setattr(pseudo_result, "status", payload.get("status"))
    setattr(pseudo_result, "invalid_row_count", payload.get("invalid_row_count"))
    setattr(pseudo_result, "ik_empty_row_count", payload.get("ik_empty_row_count"))
    setattr(pseudo_result, "selected_path", payload.get("selected_path"))
    setattr(pseudo_result, "row_labels", payload.get("row_labels"))
    setattr(pseudo_result, "metadata", payload.get("metadata", {}))
    setattr(pseudo_result, "bridge_like_segments", payload.get("bridge_like_segments"))
    setattr(pseudo_result, "big_circle_step_count", payload.get("big_circle_step_count"))
    setattr(pseudo_result, "worst_joint_step_deg", payload.get("worst_joint_step_deg"))
    setattr(
        pseudo_result,
        "motion_settings",
        payload.get("motion_settings", {}),
    )
    return _result_delivery_gate_details(pseudo_result)


def result_is_strictly_valid(result: object | None) -> bool:
    return bool(_result_delivery_gate_details(result).get("strictly_valid"))


def result_payload_is_strictly_valid(payload: dict[str, object] | None) -> bool:
    return bool(_payload_delivery_gate_details(payload).get("strictly_valid"))


def result_semantic_status(result: object | None) -> str:
    if result is None:
        return "success"
    return "valid" if result_is_strictly_valid(result) else "invalid"


def result_has_continuity_warnings(result: object | None) -> bool:
    if result is None:
        return False
    failing_segments = getattr(result, "failing_segments", ())
    worst_joint_step_deg = _safe_float(getattr(result, "worst_joint_step_deg", 0.0))
    worst_step_limit = _safe_float(
        getattr(result, "motion_settings", {}).get(
            "official_worst_joint_step_deg_limit",
            STRICT_DELIVERY_GATE["required_worst_joint_step_deg_limit"],
        )
        if isinstance(getattr(result, "motion_settings", {}), dict)
        else STRICT_DELIVERY_GATE["required_worst_joint_step_deg_limit"]
    )
    return (
        _safe_int(getattr(result, "bridge_like_segments", 0)) > 0
        or _safe_int(getattr(result, "big_circle_step_count", 0)) > 0
        or worst_joint_step_deg > worst_step_limit + 1e-9
        or bool(failing_segments)
    )


def result_quality_summary(result: ProfileEvaluationResult) -> dict[str, object]:
    gate = _result_delivery_gate_details(result)
    summary = {
        "request_id": str(result.request_id),
        "status": str(result.status),
        "ik_empty_row_count": int(result.ik_empty_row_count),
        "config_switches": int(result.config_switches),
        "bridge_like_segments": int(result.bridge_like_segments),
        "big_circle_step_count": int(result.big_circle_step_count),
        "branch_flip_ratio": float(result.branch_flip_ratio),
        "violent_branch_segments": [dict(item) for item in result.violent_branch_segments],
        "invalid_row_count": int(result.invalid_row_count),
        "worst_joint_step_deg": float(result.worst_joint_step_deg),
        "mean_joint_step_deg": float(result.mean_joint_step_deg),
        "total_cost": float(result.total_cost),
        "objective_reachable": bool(gate["objective_reachable"]),
        "official_delivery_allowed": bool(gate["official_delivery_allowed"]),
        "strictly_valid": bool(gate["strictly_valid"]),
        "gate_tier": str(gate["gate_tier"]),
        "block_reasons": [dict(item) for item in gate["block_reasons"]],
        "continuity_warnings": {
            "status": str(result.status),
            "config_switches": int(result.config_switches),
            "bridge_like_segments": int(result.bridge_like_segments),
            "big_circle_step_count": int(result.big_circle_step_count),
            "worst_joint_step_deg": float(result.worst_joint_step_deg),
            "branch_flip_ratio": float(result.branch_flip_ratio),
        },
    }
    return summary


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
    gate = _result_delivery_gate_details(selected_result)
    delivery_allowed = bool(gate["official_delivery_allowed"])
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
            "objective_reachable": bool(gate["objective_reachable"]),
            "official_delivery_allowed": bool(gate["official_delivery_allowed"]),
            "debug_invalid_output": bool(not delivery_allowed),
            "gate_tier": str(gate["gate_tier"]),
            "block_reasons": [dict(item) for item in gate["block_reasons"]],
        },
        "strict_quality_gate": {
            **STRICT_DELIVERY_GATE,
            "strictly_valid": bool(gate["strictly_valid"]),
            "objective_reachable": bool(gate["objective_reachable"]),
            "official_delivery_allowed": bool(gate["official_delivery_allowed"]),
            "debug_invalid_output": bool(not delivery_allowed),
            "gate_tier": str(gate["gate_tier"]),
            "block_reasons": [dict(item) for item in gate["block_reasons"]],
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
