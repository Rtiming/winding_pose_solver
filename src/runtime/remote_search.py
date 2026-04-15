from __future__ import annotations

import argparse
import math
from collections import Counter
from typing import Iterable

from src.core.collab_models import (
    EvaluationBatchRequest,
    EvaluationBatchResult,
    ProfileEvaluationRequest,
    ProfileEvaluationResult,
    RemoteSearchRequest,
    RemoteSearchSummary,
    load_json_file,
    write_json_file,
)


def _iter_uniform_profile_offsets(
    *,
    max_abs_offset_mm: float,
    step_mm: float,
) -> tuple[tuple[float, float], ...]:
    if step_mm <= 0.0 or max_abs_offset_mm < 0.0:
        return ()

    shell_limit = int(math.floor(max_abs_offset_mm / step_mm + 1e-9))
    offsets: list[tuple[float, float]] = []
    for dy_index in range(-shell_limit, shell_limit + 1):
        for dz_index in range(-shell_limit, shell_limit + 1):
            offset = (dy_index * step_mm, dz_index * step_mm)
            if any(abs(value) > max_abs_offset_mm + 1e-9 for value in offset):
                continue
            offsets.append(offset)

    offsets.sort(
        key=lambda offset: (
            0.0 if abs(offset[0]) <= 1e-9 and abs(offset[1]) <= 1e-9 else 1.0,
            abs(offset[0]) + abs(offset[1]),
            max(abs(offset[0]), abs(offset[1])),
            offset,
        )
    )
    return tuple(offsets)


def _result_sort_key(result: ProfileEvaluationResult) -> tuple[float, ...]:
    return (
        float(result.invalid_row_count),
        float(result.ik_empty_row_count),
        float(result.config_switches),
        float(result.bridge_like_segments),
        float(result.worst_joint_step_deg),
        float(result.mean_joint_step_deg),
        float(result.total_cost),
        float(result.timing_seconds),
    )


def _clone_request(
    base_request: ProfileEvaluationRequest,
    *,
    request_id: str,
    profile: tuple[tuple[float, float], ...],
    metadata: dict[str, object],
) -> ProfileEvaluationRequest:
    # Batch candidates only evaluate profiles quickly — repairs are expensive and should
    # only run on the final best candidate during program generation.
    return ProfileEvaluationRequest(
        request_id=request_id,
        robot_name=base_request.robot_name,
        frame_name=base_request.frame_name,
        motion_settings=dict(base_request.motion_settings),
        reference_pose_rows=tuple(dict(row) for row in base_request.reference_pose_rows),
        frame_a_origin_yz_profile_mm=profile,
        row_labels=tuple(base_request.row_labels),
        inserted_flags=tuple(base_request.inserted_flags),
        strategy="exact_profile",
        start_joints=base_request.start_joints,
        run_window_repair=False,
        run_inserted_repair=False,
        include_pose_rows_in_result=False,
        create_program=False,
        program_name=base_request.program_name,
        optimized_csv_path=base_request.optimized_csv_path,
        metadata=metadata,
    )


def _max_abs_offset_mm(base_request: ProfileEvaluationRequest) -> float:
    motion_settings = base_request.motion_settings
    envelope_values = tuple(
        float(value)
        for value in motion_settings.get("frame_a_origin_yz_envelope_schedule_mm", (6.0,))
    )
    return max((value for value in envelope_values if value >= 0.0), default=0.0)


def _clamp_profile(
    profile: tuple[tuple[float, float], ...],
    *,
    max_abs_offset_mm: float,
) -> tuple[tuple[float, float], ...]:
    return tuple(
        (
            round(max(-max_abs_offset_mm, min(max_abs_offset_mm, float(dy_mm))), 6),
            round(max(-max_abs_offset_mm, min(max_abs_offset_mm, float(dz_mm))), 6),
        )
        for dy_mm, dz_mm in profile
    )


def _candidate_sort_key(candidate: ProfileEvaluationRequest) -> tuple[float, ...]:
    strategy = str(candidate.metadata.get("strategy", ""))
    strategy_priority = {
        "local_window": 0.0,
        "local_corridor_asym": 1.0,
        "local_basis": 2.0,
        "uniform": 3.0,
    }.get(strategy, 9.0)
    return (
        strategy_priority,
        abs(float(candidate.metadata.get("dy_mm", 0.0)))
        + abs(float(candidate.metadata.get("dz_mm", 0.0))),
        abs(float(candidate.metadata.get("left_dy_mm", 0.0)))
        + abs(float(candidate.metadata.get("left_dz_mm", 0.0)))
        + abs(float(candidate.metadata.get("right_dy_mm", 0.0)))
        + abs(float(candidate.metadata.get("right_dz_mm", 0.0))),
        candidate.request_id,
    )


def _candidate_group_key(candidate: ProfileEvaluationRequest) -> str:
    strategy = str(candidate.metadata.get("strategy", ""))
    if strategy != "local_window":
        return strategy or "other"

    dy_mm = float(candidate.metadata.get("dy_mm", 0.0))
    dz_mm = float(candidate.metadata.get("dz_mm", 0.0))
    if abs(dy_mm) > 1e-9 and abs(dz_mm) <= 1e-9:
        return "local_window_y"
    if abs(dy_mm) <= 1e-9 and abs(dz_mm) > 1e-9:
        return "local_window_z"
    return "local_window_diag"


def _choose_focus_segments(
    request: RemoteSearchRequest,
) -> tuple[int, ...]:
    baseline = request.baseline_result
    row_labels = list(request.base_request.row_labels)
    label_to_index = {label: index for index, label in enumerate(row_labels)}

    focus_segments: list[int] = []
    if baseline is not None and baseline.failing_segments:
        focus_segments.extend(segment.segment_index for segment in baseline.failing_segments[:3])

    if baseline is not None and baseline.ik_empty_rows:
        for label in baseline.ik_empty_rows[:6]:
            index = label_to_index.get(label)
            if index is None:
                continue
            focus_segments.append(max(0, min(len(row_labels) - 2, index - 1)))

    for preferred_pair in ("381", "382", "380"):
        index = label_to_index.get(preferred_pair)
        if index is not None:
            focus_segments.append(max(0, min(len(row_labels) - 2, index)))

    if not focus_segments and len(row_labels) >= 2:
        focus_segments.append(max(0, len(row_labels) // 2 - 1))

    unique_segments: list[int] = []
    seen: set[int] = set()
    for segment in focus_segments:
        if segment in seen:
            continue
        seen.add(segment)
        unique_segments.append(segment)
    return tuple(unique_segments[:4])


def _build_uniform_candidates(request: RemoteSearchRequest) -> list[ProfileEvaluationRequest]:
    motion_settings = request.base_request.motion_settings
    row_count = len(request.base_request.reference_pose_rows)
    candidates: list[ProfileEvaluationRequest] = []
    ordinal = 0

    for envelope_mm in motion_settings.get("frame_a_origin_yz_envelope_schedule_mm", (6.0,)):
        for step_mm in motion_settings.get("frame_a_origin_yz_step_schedule_mm", (4.0, 2.0, 1.0)):
            if float(step_mm) <= 0.0 or float(envelope_mm) < 0.0:
                continue
            for dy_mm, dz_mm in _iter_uniform_profile_offsets(
                max_abs_offset_mm=float(envelope_mm),
                step_mm=float(step_mm),
            ):
                if abs(dy_mm) <= 1e-9 and abs(dz_mm) <= 1e-9:
                    continue
                ordinal += 1
                candidates.append(
                    _clone_request(
                        request.base_request,
                        request_id=f"round{request.round_index}_uniform_{ordinal:03d}",
                        profile=tuple((dy_mm, dz_mm) for _ in range(row_count)),
                        metadata={
                            "strategy": "uniform",
                            "dy_mm": dy_mm,
                            "dz_mm": dz_mm,
                            "envelope_mm": float(envelope_mm),
                            "step_mm": float(step_mm),
                        },
                    )
                )
    return candidates


def _apply_local_window_delta(
    profile: tuple[tuple[float, float], ...],
    *,
    segment_index: int,
    window_radius: int,
    dy_mm: float,
    dz_mm: float,
    ) -> tuple[tuple[float, float], ...]:
    updated_profile = [list(offset) for offset in profile]
    center = min(len(updated_profile) - 1, segment_index + 1)
    for row_index in range(
        max(0, center - window_radius),
        min(len(updated_profile), center + window_radius + 1),
    ):
        distance = abs(row_index - center)
        weight = max(0.0, 1.0 - distance / max(1, window_radius + 1))
        updated_profile[row_index][0] += dy_mm * weight
        updated_profile[row_index][1] += dz_mm * weight
    return tuple((round(offset[0], 6), round(offset[1], 6)) for offset in updated_profile)


def _apply_asymmetric_corridor_delta(
    profile: tuple[tuple[float, float], ...],
    *,
    segment_index: int,
    window_radius: int,
    left_dy_mm: float,
    left_dz_mm: float,
    right_dy_mm: float,
    right_dz_mm: float,
) -> tuple[tuple[float, float], ...]:
    updated_profile = [list(offset) for offset in profile]
    center = min(len(updated_profile) - 1, segment_index + 1)
    corridor_start = max(0, center - window_radius)
    corridor_end = min(len(updated_profile) - 1, center + window_radius)
    for row_index in range(corridor_start, corridor_end + 1):
        if row_index <= center:
            distance = center - row_index
            side_radius = max(1, center - corridor_start + 1)
            weight = max(0.0, 1.0 - distance / side_radius)
            updated_profile[row_index][0] += left_dy_mm * weight
            updated_profile[row_index][1] += left_dz_mm * weight
        else:
            distance = row_index - center
            side_radius = max(1, corridor_end - center + 1)
            weight = max(0.0, 1.0 - distance / side_radius)
            updated_profile[row_index][0] += right_dy_mm * weight
            updated_profile[row_index][1] += right_dz_mm * weight
    return tuple((round(offset[0], 6), round(offset[1], 6)) for offset in updated_profile)


def _apply_two_node_basis_delta(
    profile: tuple[tuple[float, float], ...],
    *,
    segment_index: int,
    window_radius: int,
    start_dy_mm: float,
    start_dz_mm: float,
    end_dy_mm: float,
    end_dz_mm: float,
) -> tuple[tuple[float, float], ...]:
    updated_profile = [list(offset) for offset in profile]
    center = min(len(updated_profile) - 1, segment_index + 1)
    corridor_start = max(0, center - window_radius)
    corridor_end = min(len(updated_profile) - 1, center + window_radius)
    corridor_length = max(1, corridor_end - corridor_start)
    for row_index in range(corridor_start, corridor_end + 1):
        alpha = (row_index - corridor_start) / corridor_length
        updated_profile[row_index][0] += ((1.0 - alpha) * start_dy_mm) + (alpha * end_dy_mm)
        updated_profile[row_index][1] += ((1.0 - alpha) * start_dz_mm) + (alpha * end_dz_mm)
    return tuple((round(offset[0], 6), round(offset[1], 6)) for offset in updated_profile)


def _build_local_candidates(request: RemoteSearchRequest) -> list[ProfileEvaluationRequest]:
    motion_settings = request.base_request.motion_settings
    baseline_result = request.baseline_result
    base_profile = (
        baseline_result.frame_a_origin_yz_profile_mm
        if baseline_result is not None
        else request.base_request.frame_a_origin_yz_profile_mm
    )
    if not base_profile:
        base_profile = tuple((0.0, 0.0) for _ in request.base_request.reference_pose_rows)

    max_abs_offset_mm = _max_abs_offset_mm(request.base_request)
    focus_segments = _choose_focus_segments(request)
    step_values = tuple(
        float(value)
        for value in motion_settings.get("frame_a_origin_yz_step_schedule_mm", (2.0, 1.0))
    )
    window_radius = int(motion_settings.get("frame_a_origin_yz_window_radius", 8))
    candidates: list[ProfileEvaluationRequest] = []
    seen_profiles: set[tuple[tuple[float, float], ...]] = set()
    ordinal = 0

    for segment_index in focus_segments:
        for step_mm in step_values[:2]:
            for dy_mm, dz_mm in (
                (step_mm, 0.0),
                (-step_mm, 0.0),
                (0.0, step_mm),
                (0.0, -step_mm),
                (step_mm, step_mm),
                (step_mm, -step_mm),
                (-step_mm, step_mm),
                (-step_mm, -step_mm),
            ):
                profile = _apply_local_window_delta(
                    base_profile,
                    segment_index=segment_index,
                    window_radius=window_radius,
                    dy_mm=dy_mm,
                    dz_mm=dz_mm,
                )
                profile = _clamp_profile(profile, max_abs_offset_mm=max_abs_offset_mm)
                if profile in seen_profiles:
                    continue
                seen_profiles.add(profile)
                ordinal += 1
                left_label = request.base_request.row_labels[segment_index]
                right_label = request.base_request.row_labels[
                    min(len(request.base_request.row_labels) - 1, segment_index + 1)
                ]
                candidates.append(
                    _clone_request(
                        request.base_request,
                        request_id=f"round{request.round_index}_local_{ordinal:03d}",
                        profile=profile,
                        metadata={
                            "strategy": "local_window",
                            "segment_index": segment_index,
                            "segment_label": f"{left_label}->{right_label}",
                            "window_radius": window_radius,
                            "dy_mm": dy_mm,
                            "dz_mm": dz_mm,
                            "step_mm": step_mm,
                        },
                    )
                )

            for left_dy_mm, left_dz_mm, right_dy_mm, right_dz_mm in (
                (step_mm, 0.0, 0.0, 0.0),
                (-step_mm, 0.0, 0.0, 0.0),
                (0.0, step_mm, 0.0, 0.0),
                (0.0, -step_mm, 0.0, 0.0),
                (0.0, 0.0, step_mm, 0.0),
                (0.0, 0.0, -step_mm, 0.0),
                (0.0, 0.0, 0.0, step_mm),
                (0.0, 0.0, 0.0, -step_mm),
                (step_mm, 0.0, -step_mm, 0.0),
                (-step_mm, 0.0, step_mm, 0.0),
                (0.0, step_mm, 0.0, -step_mm),
                (0.0, -step_mm, 0.0, step_mm),
            ):
                profile = _apply_asymmetric_corridor_delta(
                    base_profile,
                    segment_index=segment_index,
                    window_radius=window_radius,
                    left_dy_mm=left_dy_mm,
                    left_dz_mm=left_dz_mm,
                    right_dy_mm=right_dy_mm,
                    right_dz_mm=right_dz_mm,
                )
                profile = _clamp_profile(profile, max_abs_offset_mm=max_abs_offset_mm)
                if profile in seen_profiles:
                    continue
                seen_profiles.add(profile)
                ordinal += 1
                left_label = request.base_request.row_labels[segment_index]
                right_label = request.base_request.row_labels[
                    min(len(request.base_request.row_labels) - 1, segment_index + 1)
                ]
                candidates.append(
                    _clone_request(
                        request.base_request,
                        request_id=f"round{request.round_index}_corridor_{ordinal:03d}",
                        profile=profile,
                        metadata={
                            "strategy": "local_corridor_asym",
                            "segment_index": segment_index,
                            "segment_label": f"{left_label}->{right_label}",
                            "window_radius": window_radius,
                            "left_dy_mm": left_dy_mm,
                            "left_dz_mm": left_dz_mm,
                            "right_dy_mm": right_dy_mm,
                            "right_dz_mm": right_dz_mm,
                            "step_mm": step_mm,
                        },
                    )
                )

            for start_dy_mm, start_dz_mm, end_dy_mm, end_dz_mm in (
                (step_mm, 0.0, 0.0, 0.0),
                (-step_mm, 0.0, 0.0, 0.0),
                (0.0, step_mm, 0.0, 0.0),
                (0.0, -step_mm, 0.0, 0.0),
                (0.0, 0.0, step_mm, 0.0),
                (0.0, 0.0, -step_mm, 0.0),
                (0.0, 0.0, 0.0, step_mm),
                (0.0, 0.0, 0.0, -step_mm),
                (step_mm, 0.0, -step_mm, 0.0),
                (0.0, step_mm, 0.0, -step_mm),
            ):
                profile = _apply_two_node_basis_delta(
                    base_profile,
                    segment_index=segment_index,
                    window_radius=window_radius,
                    start_dy_mm=start_dy_mm,
                    start_dz_mm=start_dz_mm,
                    end_dy_mm=end_dy_mm,
                    end_dz_mm=end_dz_mm,
                )
                profile = _clamp_profile(profile, max_abs_offset_mm=max_abs_offset_mm)
                if profile in seen_profiles:
                    continue
                seen_profiles.add(profile)
                ordinal += 1
                left_label = request.base_request.row_labels[segment_index]
                right_label = request.base_request.row_labels[
                    min(len(request.base_request.row_labels) - 1, segment_index + 1)
                ]
                candidates.append(
                    _clone_request(
                        request.base_request,
                        request_id=f"round{request.round_index}_basis_{ordinal:03d}",
                        profile=profile,
                        metadata={
                            "strategy": "local_basis",
                            "segment_index": segment_index,
                            "segment_label": f"{left_label}->{right_label}",
                            "window_radius": window_radius,
                            "left_dy_mm": start_dy_mm,
                            "left_dz_mm": start_dz_mm,
                            "right_dy_mm": end_dy_mm,
                            "right_dz_mm": end_dz_mm,
                            "step_mm": step_mm,
                        },
                    )
                )
    return candidates


def _limit_candidates_with_diversity(
    candidates: list[ProfileEvaluationRequest],
    *,
    candidate_limit: int,
    prefer_local_candidates: bool,
) -> tuple[ProfileEvaluationRequest, ...]:
    sorted_candidates = sorted(candidates, key=_candidate_sort_key)
    if not sorted_candidates:
        return ()

    if not prefer_local_candidates:
        return tuple(sorted_candidates[: max(1, candidate_limit)])

    strategy_order = (
        "local_window_y",
        "local_window_z",
        "local_corridor_asym",
        "local_basis",
        "local_window_diag",
        "uniform",
    )
    grouped_candidates: dict[str, list[ProfileEvaluationRequest]] = {
        strategy: [] for strategy in {
            "local_window_y",
            "local_window_z",
            "local_window_diag",
            "local_corridor_asym",
            "local_basis",
            "uniform",
        }
    }
    extra_candidates: list[ProfileEvaluationRequest] = []
    for candidate in sorted_candidates:
        strategy = _candidate_group_key(candidate)
        if strategy in grouped_candidates:
            grouped_candidates[strategy].append(candidate)
        else:
            extra_candidates.append(candidate)

    selected: list[ProfileEvaluationRequest] = []
    selected_ids: set[str] = set()
    strategy_positions = {
        strategy: 0 for strategy in grouped_candidates
    }
    while len(selected) < max(1, candidate_limit):
        added_any = False
        for strategy in strategy_order:
            strategy_candidates = grouped_candidates[strategy]
            strategy_index = strategy_positions[strategy]
            if strategy_index >= len(strategy_candidates):
                continue
            candidate = strategy_candidates[strategy_index]
            strategy_positions[strategy] = strategy_index + 1
            if candidate.request_id in selected_ids:
                continue
            selected.append(candidate)
            selected_ids.add(candidate.request_id)
            added_any = True
            if len(selected) >= max(1, candidate_limit):
                break
        if not added_any:
            break

    if len(selected) < max(1, candidate_limit):
        for candidate in sorted_candidates:
            if candidate.request_id in selected_ids:
                continue
            selected.append(candidate)
            selected_ids.add(candidate.request_id)
            if len(selected) >= max(1, candidate_limit):
                break

    return tuple(selected)


def propose_candidates(request: RemoteSearchRequest) -> EvaluationBatchRequest:
    prefer_local_candidates = bool(request.metadata.get("prefer_local_candidates"))
    candidates = _build_uniform_candidates(request)
    if request.round_index >= 2 or request.baseline_result is not None:
        candidates.extend(_build_local_candidates(request))

    local_first = prefer_local_candidates or request.round_index >= 2
    limited = _limit_candidates_with_diversity(
        candidates,
        candidate_limit=max(1, request.candidate_limit),
        prefer_local_candidates=local_first,
    )
    return EvaluationBatchRequest(evaluations=limited)


def summarize_results(results: Iterable[ProfileEvaluationResult]) -> RemoteSearchSummary:
    result_list = list(results)
    sorted_results = sorted(result_list, key=_result_sort_key)
    failing_segments = Counter()
    ik_empty_rows = Counter()
    for result in result_list:
        for segment in result.failing_segments:
            failing_segments[f"{segment.left_label}->{segment.right_label}"] += 1
        for label in result.ik_empty_rows:
            ik_empty_rows[label] += 1

    if not sorted_results:
        return RemoteSearchSummary(
            best_request_id=None,
            result_count=0,
            sorted_request_ids=(),
            failing_segment_counts={},
            ik_empty_row_counts={},
            conclusion="No evaluation results were provided.",
        )

    best_result = sorted_results[0]
    if best_result.status == "valid":
        conclusion = (
            f"Found a clean candidate: {best_result.request_id} "
            f"(worst_joint_step={best_result.worst_joint_step_deg:.3f} deg)."
        )
    else:
        conclusion = (
            f"No clean candidate yet; best request is {best_result.request_id} with "
            f"ik_empty_rows={best_result.ik_empty_row_count}, "
            f"config_switches={best_result.config_switches}, "
            f"bridge_like_segments={best_result.bridge_like_segments}."
        )

    notes = []
    if failing_segments:
        hot_segment, hot_count = failing_segments.most_common(1)[0]
        notes.append(f"Most frequent failing segment: {hot_segment} ({hot_count} occurrences).")
    if ik_empty_rows:
        hot_row, hot_count = ik_empty_rows.most_common(1)[0]
        notes.append(f"Most frequent IK-empty row: {hot_row} ({hot_count} occurrences).")

    return RemoteSearchSummary(
        best_request_id=best_result.request_id,
        result_count=len(result_list),
        sorted_request_ids=tuple(result.request_id for result in sorted_results),
        failing_segment_counts=dict(failing_segments),
        ik_empty_row_counts=dict(ik_empty_rows),
        conclusion=conclusion,
        notes=tuple(notes),
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate remote profile candidates or summarize evaluated batches."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    propose_parser = subparsers.add_parser("propose", help="Generate candidate profile evaluations.")
    propose_parser.add_argument("--request", required=True, help="Path to a RemoteSearchRequest JSON file.")
    propose_parser.add_argument("--candidates", required=True, help="Path to write the candidate batch JSON.")

    summarize_parser = subparsers.add_parser("summarize", help="Summarize evaluated candidate results.")
    summarize_parser.add_argument("--results", required=True, help="Path to an EvaluationBatchResult JSON file.")
    summarize_parser.add_argument("--summary", required=True, help="Path to write the summary JSON.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    try:
        if args.command == "propose":
            remote_request = RemoteSearchRequest.from_dict(load_json_file(args.request))
            batch = propose_candidates(remote_request)
            output_path = write_json_file(args.candidates, batch.to_dict())
            print(f"Wrote candidate batch: {output_path}")
            return 0

        results_payload = load_json_file(args.results)
        batch_result = EvaluationBatchResult.from_dict(results_payload)
        summary = summarize_results(batch_result.results)
        output_path = write_json_file(args.summary, summary.to_dict())
        print(summary.conclusion)
        for note in summary.notes:
            print(note)
        print(f"Wrote summary: {output_path}")
        return 0
    except Exception as exc:
        print(f"Error: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
