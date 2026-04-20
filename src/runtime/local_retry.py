from __future__ import annotations

import os
from dataclasses import dataclass
from time import perf_counter
from typing import Any

import src.search.ik_collection as ik_collection_module
from src.core.collab_models import (
    EvaluationBatchRequest,
    ProfileEvaluationRequest,
    ProfileEvaluationResult,
)
from src.robodk_runtime.eval_worker import (
    _apply_optional_repairs,
    _evaluate_exact_profile_search,
    _finalize_request_result,
    _prepare_evaluation_resources,
    evaluate_batch_request,
    install_runtime_profile_hooks,
    open_offline_ik_station_context,
)
from src.runtime.profiler import reset_runtime_profile
from src.runtime.remote_search import propose_candidates, summarize_results
from src.runtime.request_builder import build_remote_search_request
from src.runtime.delivery import result_has_continuity_warnings


@dataclass(frozen=True)
class LocalProfileRetryOutcome:
    best_result: ProfileEvaluationResult
    payload: dict[str, Any]


def _profile_cache_key(
    request: ProfileEvaluationRequest,
) -> tuple[tuple[float, float], ...]:
    return tuple(
        (round(float(dy_mm), 6), round(float(dz_mm), 6))
        for dy_mm, dz_mm in request.frame_a_origin_yz_profile_mm
    )


def _result_sort_key(result: ProfileEvaluationResult) -> tuple[float, ...]:
    return (
        float(result.invalid_row_count),
        float(result.ik_empty_row_count),
        float(result.bridge_like_segments),
        float(getattr(result, "big_circle_step_count", 0)),
        float(result.worst_joint_step_deg),
        float(result.mean_joint_step_deg),
        float(result.config_switches),
        float(result.total_cost),
        float(result.timing_seconds),
    )


def _positive_int_from_env(name: str) -> int | None:
    raw_value = os.getenv(name)
    if raw_value is None:
        return None
    try:
        parsed_value = int(raw_value)
    except ValueError:
        return None
    if parsed_value <= 0:
        return None
    return parsed_value


def _force_single_process_request(
    request: ProfileEvaluationRequest,
) -> ProfileEvaluationRequest:
    motion_settings = dict(request.motion_settings)
    motion_settings["local_parallel_workers"] = 1
    motion_settings["local_parallel_min_batch_size"] = 999999
    metadata = dict(request.metadata)
    profile_worker_override = _positive_int_from_env("WPS_SERVER_PROFILE_WORKERS")
    if profile_worker_override is not None:
        min_batch_override = _positive_int_from_env("WPS_SERVER_PROFILE_MIN_BATCH_SIZE") or 4
        motion_settings["local_parallel_workers"] = profile_worker_override
        motion_settings["local_parallel_min_batch_size"] = min_batch_override
        metadata["server_repair_parallel"] = {
            "local_parallel_workers": profile_worker_override,
            "local_parallel_min_batch_size": min_batch_override,
        }
    return ProfileEvaluationRequest(
        request_id=request.request_id,
        robot_name=request.robot_name,
        frame_name=request.frame_name,
        motion_settings=motion_settings,
        reference_pose_rows=tuple(dict(row) for row in request.reference_pose_rows),
        frame_a_origin_yz_profile_mm=tuple(
            (float(dy_mm), float(dz_mm))
            for dy_mm, dz_mm in request.frame_a_origin_yz_profile_mm
        ),
        row_labels=tuple(request.row_labels),
        inserted_flags=tuple(request.inserted_flags),
        strategy=request.strategy,
        start_joints=request.start_joints,
        run_window_repair=request.run_window_repair,
        run_inserted_repair=request.run_inserted_repair,
        include_pose_rows_in_result=request.include_pose_rows_in_result,
        create_program=request.create_program,
        program_name=request.program_name,
        optimized_csv_path=request.optimized_csv_path,
        metadata=metadata,
    )


def _build_repair_request(
    *,
    candidate_request: ProfileEvaluationRequest,
    base_request: ProfileEvaluationRequest,
    retry_round: int,
    retry_rank: int,
) -> ProfileEvaluationRequest:
    candidate_metadata = dict(candidate_request.metadata)
    candidate_metadata["local_profile_retry"] = {
        "round_index": retry_round,
        "candidate_rank": retry_rank,
        "source_request_id": candidate_request.request_id,
    }
    return ProfileEvaluationRequest(
        request_id=f"{candidate_request.request_id}_repair",
        robot_name=base_request.robot_name,
        frame_name=base_request.frame_name,
        motion_settings=dict(base_request.motion_settings),
        reference_pose_rows=tuple(dict(row) for row in base_request.reference_pose_rows),
        frame_a_origin_yz_profile_mm=tuple(
            (float(dy_mm), float(dz_mm))
            for dy_mm, dz_mm in candidate_request.frame_a_origin_yz_profile_mm
        ),
        row_labels=tuple(base_request.row_labels),
        inserted_flags=tuple(base_request.inserted_flags),
        strategy="exact_profile",
        start_joints=base_request.start_joints,
        run_window_repair=True,
        run_inserted_repair=True,
        include_pose_rows_in_result=True,
        create_program=False,
        program_name=base_request.program_name,
        optimized_csv_path=base_request.optimized_csv_path,
        metadata=candidate_metadata,
    )


def _top_result_summary(
    result: ProfileEvaluationResult,
) -> dict[str, Any]:
    return {
        "request_id": str(result.request_id),
        "status": str(result.status),
        "invalid_row_count": int(result.invalid_row_count),
        "ik_empty_row_count": int(result.ik_empty_row_count),
        "config_switches": int(result.config_switches),
        "bridge_like_segments": int(result.bridge_like_segments),
        "big_circle_step_count": int(getattr(result, "big_circle_step_count", 0)),
        "branch_flip_ratio": float(getattr(result, "branch_flip_ratio", 0.0)),
        "worst_joint_step_deg": float(result.worst_joint_step_deg),
        "mean_joint_step_deg": float(result.mean_joint_step_deg),
        "total_cost": float(result.total_cost),
    }


def _request_summary(
    request: ProfileEvaluationRequest,
) -> dict[str, Any]:
    metadata = dict(request.metadata)
    summary = {
        "request_id": str(request.request_id),
        "strategy": str(metadata.get("strategy", request.strategy)),
    }
    for field_name in (
        "segment_index",
        "segment_label",
        "window_radius",
        "step_mm",
        "dy_mm",
        "dz_mm",
        "left_dy_mm",
        "left_dz_mm",
        "right_dy_mm",
        "right_dz_mm",
    ):
        if field_name in metadata:
            field_value = metadata[field_name]
            if isinstance(field_value, (int, float)):
                summary[field_name] = float(field_value)
            else:
                summary[field_name] = field_value
    return summary


def _focus_segment_indices(
    result: ProfileEvaluationResult,
) -> tuple[int, ...]:
    ordered_segments = sorted(
        result.failing_segments,
        key=lambda segment: (
            0 if segment.config_changed else 1,
            -float(segment.max_joint_delta_deg),
            -float(segment.mean_joint_delta_deg),
            int(segment.segment_index),
        ),
    )
    unique_indices: list[int] = []
    seen: set[int] = set()
    for segment in ordered_segments:
        segment_index = int(segment.segment_index)
        if segment_index in seen:
            continue
        seen.add(segment_index)
        unique_indices.append(segment_index)
        if len(unique_indices) >= 3:
            break
    return tuple(unique_indices)


def _focus_segment_metrics(
    result: ProfileEvaluationResult,
    *,
    focus_segments: tuple[int, ...],
) -> tuple[float, ...]:
    if not focus_segments:
        return ()
    segment_map = {
        int(segment.segment_index): float(segment.max_joint_delta_deg)
        for segment in result.failing_segments
    }
    fallback = float(result.worst_joint_step_deg) + 1_000.0
    return tuple(
        segment_map.get(
            segment_index,
            0.0 if result.status == "valid" else fallback,
        )
        for segment_index in focus_segments
    )


def _repair_candidate_sort_key(
    result: ProfileEvaluationResult,
    *,
    focus_segments: tuple[int, ...],
) -> tuple[float, ...]:
    focus_metrics = _focus_segment_metrics(result, focus_segments=focus_segments)
    unresolved_focus_count = sum(1 for value in focus_metrics if value > 1e-9)
    return (
        float(unresolved_focus_count),
        *(float(value) for value in focus_metrics),
        *_result_sort_key(result),
    )


def _round_progress_sort_key(
    result: ProfileEvaluationResult,
    *,
    focus_segments: tuple[int, ...],
) -> tuple[float, ...]:
    focus_metrics = _focus_segment_metrics(result, focus_segments=focus_segments)
    unresolved_focus_count = sum(1 for value in focus_metrics if value > 1e-9)
    return (
        float(unresolved_focus_count),
        *(float(value) for value in focus_metrics),
        float(result.bridge_like_segments),
        float(getattr(result, "big_circle_step_count", 0)),
        float(result.config_switches),
        *_result_sort_key(result),
    )


def _focus_segments_resolved(
    result: ProfileEvaluationResult,
    *,
    focus_segments: tuple[int, ...],
) -> bool:
    if not focus_segments:
        return False
    return all(value <= 1e-9 for value in _focus_segment_metrics(result, focus_segments=focus_segments))


def _can_use_internal_exact_retry(
    base_request: ProfileEvaluationRequest,
    baseline_search_result,
) -> bool:
    return (
        baseline_search_result is not None
        and base_request.motion_settings.get("ik_backend") == "six_axis_ik"
        and not base_request.create_program
    )


def _evaluate_exact_candidates_with_reuse(
    requests: tuple[ProfileEvaluationRequest, ...],
    *,
    base_request: ProfileEvaluationRequest,
    baseline_search_result,
    result_cache: dict[tuple[tuple[float, float], ...], ProfileEvaluationResult],
    search_cache: dict[tuple[tuple[float, float], ...], Any],
    cache_metrics: dict[str, int] | None = None,
) -> tuple[tuple[ProfileEvaluationResult, Any], ...]:
    install_runtime_profile_hooks()
    ik_collection_module.reset_ik_candidate_collection_cache()
    context = open_offline_ik_station_context(
        robot_name=base_request.robot_name,
        frame_name=base_request.frame_name,
    )
    resource_cache: dict[tuple[tuple[str, str], ...], Any] = {}

    def _resources_for(request: ProfileEvaluationRequest):
        settings_key = tuple(
            sorted((str(key), repr(value)) for key, value in dict(request.motion_settings).items())
        )
        cached_resources = resource_cache.get(settings_key)
        if cached_resources is None:
            cached_resources = _prepare_evaluation_resources(request, context)
            resource_cache[settings_key] = cached_resources
        return cached_resources

    ordered_pairs: list[tuple[ProfileEvaluationResult, Any] | None] = [None for _ in requests]
    for request_index, request in enumerate(requests):
        profile_key = _profile_cache_key(request)
        cached_result = result_cache.get(profile_key)
        cached_search_result = search_cache.get(profile_key)
        if cached_result is not None and cached_search_result is not None:
            if cache_metrics is not None:
                cache_metrics["exact_hits"] = cache_metrics.get("exact_hits", 0) + 1
            ordered_pairs[request_index] = (cached_result, cached_search_result)
            continue

        if cache_metrics is not None:
            cache_metrics["exact_misses"] = cache_metrics.get("exact_misses", 0) + 1
        reset_runtime_profile()
        started = perf_counter()
        resources = _resources_for(request)
        search_result = _evaluate_exact_profile_search(
            request,
            context,
            resources,
            reused_search_result=baseline_search_result,
        )
        result = _finalize_request_result(
            request=request,
            context=context,
            resources=resources,
            search_result=search_result,
            elapsed_seconds=perf_counter() - started,
        )
        result_cache[profile_key] = result
        search_cache[profile_key] = search_result
        ordered_pairs[request_index] = (result, search_result)

    return tuple(pair for pair in ordered_pairs if pair is not None)


def _evaluate_repair_requests_from_exact_results(
    requests: tuple[ProfileEvaluationRequest, ...],
    *,
    base_request: ProfileEvaluationRequest,
    exact_search_cache: dict[tuple[tuple[float, float], ...], Any],
    result_cache: dict[tuple[tuple[float, float], ...], ProfileEvaluationResult],
    search_cache: dict[tuple[tuple[float, float], ...], Any],
    cache_metrics: dict[str, int] | None = None,
) -> tuple[tuple[ProfileEvaluationResult, Any], ...]:
    install_runtime_profile_hooks()
    context = open_offline_ik_station_context(
        robot_name=base_request.robot_name,
        frame_name=base_request.frame_name,
    )
    resource_cache: dict[tuple[tuple[str, str], ...], Any] = {}

    def _resources_for(request: ProfileEvaluationRequest):
        settings_key = tuple(
            sorted((str(key), repr(value)) for key, value in dict(request.motion_settings).items())
        )
        cached_resources = resource_cache.get(settings_key)
        if cached_resources is None:
            cached_resources = _prepare_evaluation_resources(request, context)
            resource_cache[settings_key] = cached_resources
        return cached_resources

    ordered_pairs: list[tuple[ProfileEvaluationResult, Any] | None] = [None for _ in requests]
    for request_index, request in enumerate(requests):
        profile_key = _profile_cache_key(request)
        cached_result = result_cache.get(profile_key)
        cached_search_result = search_cache.get(profile_key)
        if cached_result is not None and cached_search_result is not None:
            if cache_metrics is not None:
                cache_metrics["repair_hits"] = cache_metrics.get("repair_hits", 0) + 1
            ordered_pairs[request_index] = (cached_result, cached_search_result)
            continue

        exact_search_result = exact_search_cache.get(profile_key)
        if exact_search_result is None:
            if cache_metrics is not None:
                cache_metrics["repair_fallbacks"] = cache_metrics.get("repair_fallbacks", 0) + 1
            fallback_results = evaluate_batch_request(EvaluationBatchRequest(evaluations=(request,)))
            fallback_result = fallback_results.results[0]
            result_cache[profile_key] = fallback_result
            ordered_pairs[request_index] = (fallback_result, None)
            continue

        if cache_metrics is not None:
            cache_metrics["repair_misses"] = cache_metrics.get("repair_misses", 0) + 1
        reset_runtime_profile()
        started = perf_counter()
        resources = _resources_for(request)
        repaired_search_result, repair_metadata = _apply_optional_repairs(
            request,
            context,
            resources,
            exact_search_result,
        )
        repaired_result = _finalize_request_result(
            request=request,
            context=context,
            resources=resources,
            search_result=repaired_search_result,
            elapsed_seconds=perf_counter() - started,
            repair_metadata=repair_metadata,
        )
        result_cache[profile_key] = repaired_result
        search_cache[profile_key] = repaired_search_result
        ordered_pairs[request_index] = (repaired_result, repaired_search_result)

    return tuple(pair for pair in ordered_pairs if pair is not None)


def _evaluate_requests_with_batch_fallback(
    requests: tuple[ProfileEvaluationRequest, ...],
    *,
    result_cache: dict[tuple[tuple[float, float], ...], ProfileEvaluationResult],
    cache_metrics: dict[str, int] | None = None,
) -> tuple[ProfileEvaluationResult, ...]:
    ordered_results: list[ProfileEvaluationResult | None] = [None for _ in requests]
    uncached_requests: list[ProfileEvaluationRequest] = []
    uncached_indices: list[int] = []

    for request_index, request in enumerate(requests):
        cached_result = result_cache.get(_profile_cache_key(request))
        if cached_result is not None:
            if cache_metrics is not None:
                cache_metrics["fallback_hits"] = cache_metrics.get("fallback_hits", 0) + 1
            ordered_results[request_index] = cached_result
            continue
        uncached_requests.append(request)
        uncached_indices.append(request_index)

    if uncached_requests:
        if cache_metrics is not None:
            cache_metrics["fallback_misses"] = cache_metrics.get("fallback_misses", 0) + len(uncached_requests)
        uncached_batch = EvaluationBatchRequest(evaluations=tuple(uncached_requests))
        uncached_results = evaluate_batch_request(uncached_batch)
        for request_index, request, result in zip(
            uncached_indices,
            uncached_requests,
            uncached_results.results,
        ):
            result_cache[_profile_cache_key(request)] = result
            ordered_results[request_index] = result

    return tuple(result for result in ordered_results if result is not None)


def run_local_profile_retry(
    base_request: ProfileEvaluationRequest,
    baseline_result: ProfileEvaluationResult,
    *,
    candidate_limit: int,
    repair_retry_limit: int,
    max_rounds: int,
    baseline_search_result=None,
) -> LocalProfileRetryOutcome:
    retry_started = perf_counter()
    if baseline_result.status == "valid" and not result_has_continuity_warnings(baseline_result):
        return LocalProfileRetryOutcome(
            best_result=baseline_result,
            payload={
                "entrypoint": "local_profile_retry",
                "initial_result": _top_result_summary(baseline_result),
                "best_result": _top_result_summary(baseline_result),
                "rounds": [],
                "skipped": "baseline already target-reachable without continuity warnings",
            },
        )

    best_result = baseline_result
    working_baseline = baseline_result
    working_baseline_search_result = baseline_search_result
    rounds_payload: list[dict[str, Any]] = []
    exact_result_cache: dict[tuple[tuple[float, float], ...], ProfileEvaluationResult] = {}
    exact_search_cache: dict[tuple[tuple[float, float], ...], Any] = {}
    repair_result_cache: dict[tuple[tuple[float, float], ...], ProfileEvaluationResult] = {}
    repair_search_cache: dict[tuple[tuple[float, float], ...], Any] = {}
    cache_metrics: dict[str, int] = {
        "exact_hits": 0,
        "exact_misses": 0,
        "repair_hits": 0,
        "repair_misses": 0,
        "repair_fallbacks": 0,
        "fallback_hits": 0,
        "fallback_misses": 0,
    }

    retry_round_count = max(0, int(max_rounds))
    repair_retry_budget = max(0, int(repair_retry_limit))
    for round_index in range(1, retry_round_count + 1):
        round_started = perf_counter()
        focus_segments = _focus_segment_indices(working_baseline)
        remote_request = build_remote_search_request(
            base_request,
            round_index=round_index,
            candidate_limit=max(1, int(candidate_limit)),
            baseline_result=working_baseline,
            metadata={
                "entrypoint": "local_profile_retry",
                "round_index": round_index,
                "prefer_local_candidates": True,
            },
        )
        candidate_proposal_started = perf_counter()
        candidate_batch = propose_candidates(remote_request)
        candidate_proposal_seconds = perf_counter() - candidate_proposal_started
        if not candidate_batch.evaluations:
            break

        prepared_requests = tuple(
            _force_single_process_request(request)
            for request in candidate_batch.evaluations
        )

        candidate_eval_started = perf_counter()
        candidate_pairs_with_search: tuple[tuple[ProfileEvaluationResult, Any], ...] = ()
        if _can_use_internal_exact_retry(base_request, working_baseline_search_result):
            candidate_pairs_with_search = _evaluate_exact_candidates_with_reuse(
                prepared_requests,
                base_request=base_request,
                baseline_search_result=working_baseline_search_result,
                result_cache=exact_result_cache,
                search_cache=exact_search_cache,
                cache_metrics=cache_metrics,
            )
            evaluated_candidate_results = tuple(result for result, _search in candidate_pairs_with_search)
            candidate_search_by_request_id = {
                request.request_id: search_result
                for request, (_result, search_result) in zip(
                    prepared_requests,
                    candidate_pairs_with_search,
                )
            }
        else:
            evaluated_candidate_results = _evaluate_requests_with_batch_fallback(
                prepared_requests,
                result_cache=exact_result_cache,
                cache_metrics=cache_metrics,
            )
            candidate_search_by_request_id = {}
        candidate_eval_seconds = perf_counter() - candidate_eval_started

        candidate_pairs = sorted(
            zip(prepared_requests, evaluated_candidate_results),
            key=lambda pair: _result_sort_key(pair[1]),
        )
        best_exact_request, best_exact_result = candidate_pairs[0]
        repair_candidate_pairs = sorted(
            candidate_pairs,
            key=lambda pair: _repair_candidate_sort_key(
                pair[1],
                focus_segments=focus_segments,
            ),
        )
        best_focus_request, best_focus_result = repair_candidate_pairs[0]
        round_summary: dict[str, Any] = {
            "round_index": round_index,
            "candidate_count": len(candidate_pairs),
            "focus_segments": list(focus_segments),
            "candidate_summary": summarize_results(evaluated_candidate_results).to_dict(),
            "best_exact_candidate": _top_result_summary(best_exact_result),
            "best_exact_candidate_request": _request_summary(best_exact_request),
            "best_focus_exact_candidate": _top_result_summary(best_focus_result),
            "best_focus_exact_candidate_request": _request_summary(best_focus_request),
        }

        repair_requests: tuple[ProfileEvaluationRequest, ...] = ()
        if (
            result_has_continuity_warnings(best_focus_result)
            and repair_retry_budget > 0
            and not _focus_segments_resolved(
                best_focus_result,
                focus_segments=focus_segments,
            )
        ):
            selected_repair_pairs = tuple(
                repair_candidate_pairs[:repair_retry_budget]
            )
            round_summary["selected_repair_sources"] = [
                {
                    **_request_summary(pair[0]),
                    "status": pair[1].status,
                    "worst_joint_step_deg": float(pair[1].worst_joint_step_deg),
                    "focus_metrics": list(
                        _focus_segment_metrics(
                            pair[1],
                            focus_segments=focus_segments,
                        )
                    ),
                }
                for pair in selected_repair_pairs
            ]
            repair_requests = tuple(
                _force_single_process_request(
                    _build_repair_request(
                        candidate_request=candidate_request,
                        base_request=base_request,
                        retry_round=round_index,
                        retry_rank=retry_rank,
                    )
                )
                for retry_rank, (candidate_request, _candidate_result) in enumerate(
                    selected_repair_pairs,
                    start=1,
                )
            )

        best_repair_result = None
        best_repair_request_id = None
        repair_search_by_request_id: dict[str, Any] = {}
        repair_eval_seconds = 0.0
        if repair_requests:
            repair_eval_started = perf_counter()
            if _can_use_internal_exact_retry(base_request, working_baseline_search_result):
                repair_pairs_with_search = _evaluate_repair_requests_from_exact_results(
                    repair_requests,
                    base_request=base_request,
                    exact_search_cache=exact_search_cache,
                    result_cache=repair_result_cache,
                    search_cache=repair_search_cache,
                    cache_metrics=cache_metrics,
                )
                repair_results = tuple(result for result, _search in repair_pairs_with_search)
                repair_search_by_request_id = {
                    request.request_id: search_result
                    for request, (_result, search_result) in zip(
                        repair_requests,
                        repair_pairs_with_search,
                    )
                    if search_result is not None
                }
            else:
                repair_results = _evaluate_requests_with_batch_fallback(
                    repair_requests,
                    result_cache=repair_result_cache,
                    cache_metrics=cache_metrics,
                )
            repair_pairs = sorted(
                zip(repair_requests, repair_results),
                key=lambda pair: _result_sort_key(pair[1]),
            )
            best_repair_request_id, best_repair_result = repair_pairs[0]
            round_summary["best_repair_candidate"] = _top_result_summary(best_repair_result)
            if best_repair_request_id is not None:
                round_summary["best_repair_candidate_request"] = _request_summary(
                    best_repair_request_id
                )
            repair_eval_seconds = perf_counter() - repair_eval_started

        current_candidates_by_request_id: dict[str, tuple[ProfileEvaluationResult, Any]] = {
            str(working_baseline.request_id): (
                working_baseline,
                working_baseline_search_result,
            ),
            str(best_exact_request.request_id): (
                best_exact_result,
                candidate_search_by_request_id.get(best_exact_request.request_id),
            ),
            str(best_focus_request.request_id): (
                best_focus_result,
                candidate_search_by_request_id.get(best_focus_request.request_id),
            ),
        }
        if best_repair_result is not None:
            assert best_repair_request_id is not None
            current_candidates_by_request_id[str(best_repair_request_id.request_id)] = (
                best_repair_result,
                repair_search_by_request_id.get(best_repair_request_id.request_id),
            )

        current_round_best_result, current_round_best_search_result = min(
            current_candidates_by_request_id.values(),
            key=lambda pair: _round_progress_sort_key(
                pair[0],
                focus_segments=focus_segments,
            ),
        )
        round_summary["round_best"] = _top_result_summary(current_round_best_result)
        round_summary["round_cache_metrics"] = dict(cache_metrics)
        round_summary["timing_seconds"] = {
            "candidate_proposal": candidate_proposal_seconds,
            "candidate_eval": candidate_eval_seconds,
            "repair_eval": repair_eval_seconds,
            "round_total": perf_counter() - round_started,
        }
        rounds_payload.append(round_summary)

        if _round_progress_sort_key(
            current_round_best_result,
            focus_segments=focus_segments,
        ) < _round_progress_sort_key(best_result, focus_segments=focus_segments):
            best_result = current_round_best_result

        if _round_progress_sort_key(
            current_round_best_result,
            focus_segments=focus_segments,
        ) < _round_progress_sort_key(working_baseline, focus_segments=focus_segments):
            working_baseline = current_round_best_result
            if current_round_best_search_result is not None:
                working_baseline_search_result = current_round_best_search_result
        else:
            break

        if best_result.status == "valid" and not result_has_continuity_warnings(best_result):
            break

    return LocalProfileRetryOutcome(
        best_result=best_result,
        payload={
            "entrypoint": "local_profile_retry",
            "initial_result": _top_result_summary(baseline_result),
            "best_result": _top_result_summary(best_result),
            "rounds": rounds_payload,
            "timing_seconds": {
                "retry_total": perf_counter() - retry_started,
            },
            "cache_metrics": dict(cache_metrics),
            "cache_sizes": {
                "exact_result_cache": len(exact_result_cache),
                "exact_search_cache": len(exact_search_cache),
                "repair_result_cache": len(repair_result_cache),
                "repair_search_cache": len(repair_search_cache),
            },
        },
    )
