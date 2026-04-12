from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


def _normalize_pose_rows(
    pose_rows: list[dict[str, Any]] | tuple[dict[str, Any], ...],
) -> tuple[dict[str, float], ...]:
    normalized_rows: list[dict[str, float]] = []
    for pose_row in pose_rows:
        normalized_row: dict[str, float] = {}
        for key, value in pose_row.items():
            if isinstance(value, bool):
                normalized_row[str(key)] = float(value)
            elif value is None:
                continue
            else:
                normalized_row[str(key)] = float(value)
        normalized_rows.append(normalized_row)
    return tuple(normalized_rows)


def _normalize_profile(
    profile: list[list[float]] | list[tuple[float, float]] | tuple[tuple[float, float], ...] | None,
    *,
    row_count: int,
) -> tuple[tuple[float, float], ...]:
    if profile is None:
        return tuple((0.0, 0.0) for _ in range(row_count))
    return tuple((float(item[0]), float(item[1])) for item in profile)


def _normalize_row_labels(
    row_labels: list[str] | tuple[str, ...] | None,
    *,
    row_count: int,
) -> tuple[str, ...]:
    if row_labels is None:
        return tuple(str(index) for index in range(row_count))
    return tuple(str(label) for label in row_labels)


def _normalize_inserted_flags(
    inserted_flags: list[bool] | tuple[bool, ...] | None,
    *,
    row_count: int,
) -> tuple[bool, ...]:
    if inserted_flags is None:
        return tuple(False for _ in range(row_count))
    return tuple(bool(flag) for flag in inserted_flags)


@dataclass(frozen=True)
class SelectedPathEntry:
    joints: tuple[float, ...]
    config_flags: tuple[int, ...]

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "SelectedPathEntry":
        return cls(
            joints=tuple(float(value) for value in payload.get("joints", ())),
            config_flags=tuple(int(value) for value in payload.get("config_flags", ())),
        )


@dataclass(frozen=True)
class FailedSegment:
    segment_index: int
    left_label: str
    right_label: str
    config_changed: bool
    max_joint_delta_deg: float
    mean_joint_delta_deg: float

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "FailedSegment":
        return cls(
            segment_index=int(payload["segment_index"]),
            left_label=str(payload["left_label"]),
            right_label=str(payload["right_label"]),
            config_changed=bool(payload["config_changed"]),
            max_joint_delta_deg=float(payload["max_joint_delta_deg"]),
            mean_joint_delta_deg=float(payload["mean_joint_delta_deg"]),
        )


@dataclass(frozen=True)
class ProfileEvaluationRequest:
    request_id: str
    robot_name: str
    frame_name: str
    motion_settings: dict[str, Any]
    reference_pose_rows: tuple[dict[str, float], ...]
    frame_a_origin_yz_profile_mm: tuple[tuple[float, float], ...]
    row_labels: tuple[str, ...]
    inserted_flags: tuple[bool, ...]
    strategy: str = "exact_profile"
    start_joints: tuple[float, ...] | None = None
    run_window_repair: bool = True
    run_inserted_repair: bool = True
    include_pose_rows_in_result: bool = False
    create_program: bool = False
    program_name: str | None = None
    optimized_csv_path: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ProfileEvaluationRequest":
        reference_pose_rows = _normalize_pose_rows(payload.get("reference_pose_rows", ()))
        row_count = len(reference_pose_rows)
        start_joints = payload.get("start_joints")
        return cls(
            request_id=str(payload["request_id"]),
            robot_name=str(payload["robot_name"]),
            frame_name=str(payload["frame_name"]),
            motion_settings=dict(payload.get("motion_settings", {})),
            reference_pose_rows=reference_pose_rows,
            frame_a_origin_yz_profile_mm=_normalize_profile(
                payload.get("frame_a_origin_yz_profile_mm"),
                row_count=row_count,
            ),
            row_labels=_normalize_row_labels(payload.get("row_labels"), row_count=row_count),
            inserted_flags=_normalize_inserted_flags(
                payload.get("inserted_flags"),
                row_count=row_count,
            ),
            strategy=str(payload.get("strategy", "exact_profile")),
            start_joints=None
            if start_joints in (None, [])
            else tuple(float(value) for value in start_joints),
            run_window_repair=bool(payload.get("run_window_repair", True)),
            run_inserted_repair=bool(payload.get("run_inserted_repair", True)),
            include_pose_rows_in_result=bool(payload.get("include_pose_rows_in_result", False)),
            create_program=bool(payload.get("create_program", False)),
            program_name=payload.get("program_name"),
            optimized_csv_path=payload.get("optimized_csv_path"),
            metadata=dict(payload.get("metadata", {})),
        )

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["reference_pose_rows"] = [dict(row) for row in self.reference_pose_rows]
        payload["frame_a_origin_yz_profile_mm"] = [
            [float(dy_mm), float(dz_mm)]
            for dy_mm, dz_mm in self.frame_a_origin_yz_profile_mm
        ]
        payload["row_labels"] = list(self.row_labels)
        payload["inserted_flags"] = [bool(flag) for flag in self.inserted_flags]
        if self.start_joints is not None:
            payload["start_joints"] = [float(value) for value in self.start_joints]
        return payload


@dataclass(frozen=True)
class ProfileEvaluationResult:
    request_id: str
    status: str
    timing_seconds: float
    motion_settings: dict[str, Any]
    total_candidates: int
    invalid_row_count: int
    ik_empty_row_count: int
    config_switches: int
    bridge_like_segments: int
    worst_joint_step_deg: float
    mean_joint_step_deg: float
    total_cost: float
    row_labels: tuple[str, ...]
    inserted_flags: tuple[bool, ...]
    frame_a_origin_yz_profile_mm: tuple[tuple[float, float], ...]
    selected_path: tuple[SelectedPathEntry, ...]
    failing_segments: tuple[FailedSegment, ...]
    ik_empty_rows: tuple[str, ...]
    focus_report: str
    diagnostics: str | None
    error_message: str | None
    profiling: dict[str, dict[str, float | int]]
    metadata: dict[str, Any] = field(default_factory=dict)
    pose_rows: tuple[dict[str, float], ...] | None = None

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ProfileEvaluationResult":
        pose_rows_payload = payload.get("pose_rows")
        return cls(
            request_id=str(payload["request_id"]),
            status=str(payload["status"]),
            timing_seconds=float(payload.get("timing_seconds", 0.0)),
            motion_settings=dict(payload.get("motion_settings", {})),
            total_candidates=int(payload.get("total_candidates", 0)),
            invalid_row_count=int(payload.get("invalid_row_count", 0)),
            ik_empty_row_count=int(payload.get("ik_empty_row_count", 0)),
            config_switches=int(payload.get("config_switches", 0)),
            bridge_like_segments=int(payload.get("bridge_like_segments", 0)),
            worst_joint_step_deg=float(payload.get("worst_joint_step_deg", 0.0)),
            mean_joint_step_deg=float(payload.get("mean_joint_step_deg", 0.0)),
            total_cost=float(payload.get("total_cost", 0.0)),
            row_labels=tuple(str(label) for label in payload.get("row_labels", ())),
            inserted_flags=tuple(bool(flag) for flag in payload.get("inserted_flags", ())),
            frame_a_origin_yz_profile_mm=tuple(
                (float(item[0]), float(item[1]))
                for item in payload.get("frame_a_origin_yz_profile_mm", ())
            ),
            selected_path=tuple(
                SelectedPathEntry.from_dict(item) for item in payload.get("selected_path", ())
            ),
            failing_segments=tuple(
                FailedSegment.from_dict(item) for item in payload.get("failing_segments", ())
            ),
            ik_empty_rows=tuple(str(label) for label in payload.get("ik_empty_rows", ())),
            focus_report=str(payload.get("focus_report", "")),
            diagnostics=payload.get("diagnostics"),
            error_message=payload.get("error_message"),
            profiling={
                str(name): {
                    "seconds": float(stats.get("seconds", 0.0)),
                    "count": int(stats.get("count", 0)),
                }
                for name, stats in dict(payload.get("profiling", {})).items()
            },
            metadata=dict(payload.get("metadata", {})),
            pose_rows=None if pose_rows_payload is None else _normalize_pose_rows(pose_rows_payload),
        )

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["row_labels"] = list(self.row_labels)
        payload["inserted_flags"] = [bool(flag) for flag in self.inserted_flags]
        payload["frame_a_origin_yz_profile_mm"] = [
            [float(dy_mm), float(dz_mm)]
            for dy_mm, dz_mm in self.frame_a_origin_yz_profile_mm
        ]
        payload["selected_path"] = [asdict(item) for item in self.selected_path]
        payload["failing_segments"] = [asdict(item) for item in self.failing_segments]
        payload["ik_empty_rows"] = list(self.ik_empty_rows)
        if self.pose_rows is not None:
            payload["pose_rows"] = [dict(row) for row in self.pose_rows]
        return payload


@dataclass(frozen=True)
class EvaluationBatchRequest:
    evaluations: tuple[ProfileEvaluationRequest, ...]

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "EvaluationBatchRequest":
        return cls(
            evaluations=tuple(
                ProfileEvaluationRequest.from_dict(item)
                for item in payload.get("evaluations", ())
            )
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "evaluations": [evaluation.to_dict() for evaluation in self.evaluations],
        }


@dataclass(frozen=True)
class EvaluationBatchResult:
    results: tuple[ProfileEvaluationResult, ...]

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "EvaluationBatchResult":
        return cls(
            results=tuple(
                ProfileEvaluationResult.from_dict(item)
                for item in payload.get("results", ())
            )
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "results": [result.to_dict() for result in self.results],
        }


@dataclass(frozen=True)
class RemoteSearchRequest:
    base_request: ProfileEvaluationRequest
    baseline_result: ProfileEvaluationResult | None = None
    round_index: int = 1
    candidate_limit: int = 24
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "RemoteSearchRequest":
        baseline_payload = payload.get("baseline_result")
        return cls(
            base_request=ProfileEvaluationRequest.from_dict(payload["base_request"]),
            baseline_result=None
            if baseline_payload is None
            else ProfileEvaluationResult.from_dict(baseline_payload),
            round_index=int(payload.get("round_index", 1)),
            candidate_limit=int(payload.get("candidate_limit", 24)),
            metadata=dict(payload.get("metadata", {})),
        )

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "base_request": self.base_request.to_dict(),
            "round_index": self.round_index,
            "candidate_limit": self.candidate_limit,
            "metadata": dict(self.metadata),
        }
        if self.baseline_result is not None:
            payload["baseline_result"] = self.baseline_result.to_dict()
        return payload


@dataclass(frozen=True)
class RemoteSearchSummary:
    best_request_id: str | None
    result_count: int
    sorted_request_ids: tuple[str, ...]
    failing_segment_counts: dict[str, int]
    ik_empty_row_counts: dict[str, int]
    conclusion: str
    notes: tuple[str, ...] = ()

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "RemoteSearchSummary":
        return cls(
            best_request_id=payload.get("best_request_id"),
            result_count=int(payload.get("result_count", 0)),
            sorted_request_ids=tuple(str(value) for value in payload.get("sorted_request_ids", ())),
            failing_segment_counts={
                str(key): int(value)
                for key, value in dict(payload.get("failing_segment_counts", {})).items()
            },
            ik_empty_row_counts={
                str(key): int(value)
                for key, value in dict(payload.get("ik_empty_row_counts", {})).items()
            },
            conclusion=str(payload.get("conclusion", "")),
            notes=tuple(str(value) for value in payload.get("notes", ())),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "best_request_id": self.best_request_id,
            "result_count": self.result_count,
            "sorted_request_ids": list(self.sorted_request_ids),
            "failing_segment_counts": dict(self.failing_segment_counts),
            "ik_empty_row_counts": dict(self.ik_empty_row_counts),
            "conclusion": self.conclusion,
            "notes": list(self.notes),
        }


def load_json_file(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json_file(path: str | Path, payload: dict[str, Any]) -> Path:
    target_path = Path(path)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    with target_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
    return target_path
