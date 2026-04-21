from __future__ import annotations

import heapq
import itertools
import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy.optimize import linprog, lsq_linear

from src.search.bridge_builder import _insert_interpolated_transition_rows
from src.core.geometry import _build_pose, _mean_abs_joint_delta
from src.search.global_search import (
    _evaluate_frame_a_origin_profile,
    _normalize_frame_a_origin_yz_profile,
    _summarize_profile_metrics,
    _profile_cache_key,
    _path_search_sort_key,
)
from src.search.ik_collection import _collect_ik_candidates
from src.search.parallel_profile_eval import maybe_parallel_evaluate_exact_profiles
from src.search.path_optimizer import (
    _candidate_lineage_key,
    _candidate_transition_penalty,
    _is_benign_wrist_singularity_config_change,
    _joint_limit_penalty,
    _joint_transition_penalty,
    _singularity_penalty,
    _summarize_selected_path,
)
from src.core.types import _IKCandidate, _IKLayer, _PathOptimizerSettings, _PathSearchResult


@dataclass(frozen=True)
class _WindowState:
    dy_mm: float
    dz_mm: float
    candidate: _IKCandidate


@dataclass(frozen=True)
class _GlobalRowLinearization:
    joints: tuple[float, ...]
    gradient_y: tuple[float, ...]
    gradient_z: tuple[float, ...]


@dataclass(frozen=True)
class _FamilyLinearizedCandidate:
    candidate: _IKCandidate
    gradient_y: tuple[float, ...]
    gradient_z: tuple[float, ...]


@dataclass(frozen=True)
class _HandoverCorridorModel:
    corridor_start: int
    corridor_end: int
    left_family: tuple[int, ...]
    right_family: tuple[int, ...]
    left_lineage: tuple[int, ...]
    right_lineage: tuple[int, ...]
    left_chain: tuple[_FamilyLinearizedCandidate | None, ...]
    right_chain: tuple[_FamilyLinearizedCandidate | None, ...]


@dataclass(frozen=True)
class _LinearizedDifferenceTerm:
    left_state: _FamilyLinearizedCandidate
    left_row_index: int
    right_state: _FamilyLinearizedCandidate
    right_row_index: int
    left_is_fixed: bool = False
    right_is_fixed: bool = False


@dataclass(frozen=True)
class _HandoverTargetDescriptor:
    target_segment: int
    predicted_corridor_max_step_deg: float
    predicted_target_cut_max_step_deg: float


@dataclass(frozen=True)
class _HandoverSolveResult:
    corridor_delta: tuple[tuple[float, float], ...]
    predicted_corridor_max_step_deg: float
    predicted_target_cut_max_step_deg: float
    solver_name: str


@dataclass(frozen=True)
class _HandoverCandidateMetadata:
    segment_index: int
    corridor_start: int
    corridor_end: int
    target_segment: int
    predicted_corridor_max_step_deg: float
    predicted_target_cut_max_step_deg: float
    solver_name: str


_GLOBAL_CONTINUOUS_REFINEMENT_MAX_ITERS = 2
_GLOBAL_CONTINUOUS_REFINEMENT_FD_STEP_MM = 1.0
_GLOBAL_CONTINUOUS_REFINEMENT_TRUST_SCALES = (1.0, 0.5, 0.25, 0.125, 0.0625)
_GLOBAL_CONTINUOUS_UPDATE_MAGNITUDE_WEIGHT = 0.12
_GLOBAL_CONTINUOUS_UPDATE_SMOOTHNESS_WEIGHT = 2.50
_GLOBAL_CONTINUOUS_UPDATE_CURVATURE_WEIGHT = 6.00
_GLOBAL_CONTINUOUS_MINMAX_TOLERANCE_DEG = 1e-3
# Experimental global min-max stage. Keep disabled by default because some
# single-switch cases regress after downstream repair despite better local metrics.
_GLOBAL_CONTINUOUS_ENABLE_MINMAX = False

_HANDOVER_CORRIDOR_MAX_ITERS = 2
_HANDOVER_CORRIDOR_RADIUS_SCALE = 2
_HANDOVER_CORRIDOR_MAX_TARGET_SEGMENTS = 4
_HANDOVER_CORRIDOR_HANDOVER_WEIGHT = 8.00
_HANDOVER_CORRIDOR_OVERLAP_WEIGHT = 3.50
_HANDOVER_CORRIDOR_CONTINUITY_WEIGHT = 1.00
_HANDOVER_CORRIDOR_BOUNDARY_WEIGHT = 2.50
_HANDOVER_CORRIDOR_UPDATE_MAGNITUDE_WEIGHT = 0.08
_HANDOVER_CORRIDOR_UPDATE_SMOOTHNESS_WEIGHT = 2.00
_HANDOVER_CORRIDOR_UPDATE_CURVATURE_WEIGHT = 4.50
_HANDOVER_CORRIDOR_TRUST_SCALES = (2.0, 1.5, 1.0, 0.5, 0.25, 0.125)
_HANDOVER_CORRIDOR_MINMAX_TOLERANCE_DEG = 1e-3
_HANDOVER_CORRIDOR_STAGE2_GROUP_WEIGHTS = (
    1_000_000.0,
    10_000.0,
    1_000.0,
    10.0,
    1.0,
    0.1,
)
_INSCRIBED_OCTAGON_COS_22P5 = 0.9238795325112867
_WINDOW_STATE_MAX_CANDIDATES_PER_OFFSET = 16
_WINDOW_STATE_MAX_CANDIDATES_PER_FAMILY = 3


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


def _evaluate_exact_profile_with_cache(
    candidate_profile: Sequence[tuple[float, float]],
    *,
    base_result: _PathSearchResult,
    lock_profile_endpoints: bool,
    profile_result_cache: dict[tuple[tuple[float, float], ...], _PathSearchResult],
    robot,
    mat_type,
    move_type: str,
    start_joints: tuple[float, ...],
    tool_pose,
    reference_pose,
    joint_count: int,
    optimizer_settings: _PathOptimizerSettings,
    a1_lower_deg: float,
    a1_upper_deg: float,
    a2_max_deg: float,
    joint_constraint_tolerance_deg: float,
    seed_joints: Sequence[tuple[float, ...]],
    lower_limits: Sequence[float],
    upper_limits: Sequence[float],
    bridge_trigger_joint_delta_deg: float,
) -> _PathSearchResult:
    candidate_profile = _normalize_frame_a_origin_yz_profile(
        base_result.reference_pose_rows,
        candidate_profile,
        lock_profile_endpoints=lock_profile_endpoints,
    )
    profile_key = _profile_cache_key(candidate_profile)
    cached_result = profile_result_cache.get(profile_key)
    if cached_result is not None:
        return cached_result

    recompute_row_indices = _profile_changed_row_indices(
        base_result.frame_a_origin_yz_profile_mm,
        candidate_profile,
    )
    evaluated_result = _evaluate_frame_a_origin_profile(
        base_result.reference_pose_rows,
        frame_a_origin_yz_profile_mm=candidate_profile,
        row_labels=base_result.row_labels,
        inserted_flags=base_result.inserted_flags,
        robot=robot,
        mat_type=mat_type,
        move_type=move_type,
        start_joints=start_joints,
        tool_pose=tool_pose,
        reference_pose=reference_pose,
        joint_count=joint_count,
        optimizer_settings=optimizer_settings,
        a1_lower_deg=a1_lower_deg,
        a1_upper_deg=a1_upper_deg,
        a2_max_deg=a2_max_deg,
        joint_constraint_tolerance_deg=joint_constraint_tolerance_deg,
        seed_joints=seed_joints,
        lower_limits=lower_limits,
        upper_limits=upper_limits,
        bridge_trigger_joint_delta_deg=bridge_trigger_joint_delta_deg,
        reused_ik_layers=base_result.ik_layers,
        recompute_row_indices=recompute_row_indices,
        lock_profile_endpoints=lock_profile_endpoints,
    )
    profile_result_cache[profile_key] = evaluated_result
    return evaluated_result


def _lock_profile_endpoints_if_needed(
    profile: Sequence[tuple[float, float]],
    *,
    reference_pose_rows: Sequence[dict[str, float]],
    motion_settings,
) -> tuple[tuple[float, float], ...]:
    return _normalize_frame_a_origin_yz_profile(
        reference_pose_rows,
        profile,
        lock_profile_endpoints=bool(
            getattr(motion_settings, "lock_frame_a_origin_yz_profile_endpoints", True)
        ),
    )


def _lineage_matches_family(
    candidate: _IKCandidate,
    *,
    family_flags: tuple[int, ...],
    lineage_key: tuple[int, ...] | None,
) -> bool:
    if candidate.config_flags != family_flags:
        return False
    if lineage_key is None or lineage_key == family_flags:
        return True
    return _candidate_lineage_key(candidate) == lineage_key


def _refine_path_with_frame_a_origin_profile(
    search_result: _PathSearchResult,
    *,
    robot,
    mat_type,
    move_type: str,
    start_joints: tuple[float, ...],
    tool_pose,
    reference_pose,
    joint_count: int,
    motion_settings,
    optimizer_settings: _PathOptimizerSettings,
    lower_limits: Sequence[float],
    upper_limits: Sequence[float],
    a1_lower_deg: float,
    a1_upper_deg: float,
    a2_max_deg: float,
    joint_constraint_tolerance_deg: float,
    profile_result_cache: dict[tuple[tuple[float, float], ...], _PathSearchResult] | None = None,
) -> _PathSearchResult:
    best_result = search_result
    if not best_result.selected_path:
        return best_result

    lower_limits_tuple = tuple(float(value) for value in lower_limits[:joint_count])
    upper_limits_tuple = tuple(float(value) for value in upper_limits[:joint_count])
    continuity_step_limit_mm = _resolve_profile_continuity_step_limit(motion_settings)
    if profile_result_cache is None:
        profile_result_cache = {}
    profile_result_cache.setdefault(
        _profile_cache_key(best_result.frame_a_origin_yz_profile_mm),
        best_result,
    )
    best_result = _refine_path_with_global_continuous_profile(
        best_result,
        robot=robot,
        mat_type=mat_type,
        move_type=move_type,
        start_joints=start_joints,
        tool_pose=tool_pose,
        reference_pose=reference_pose,
        joint_count=joint_count,
        motion_settings=motion_settings,
        optimizer_settings=optimizer_settings,
        lower_limits=lower_limits_tuple,
        upper_limits=upper_limits_tuple,
        a1_lower_deg=a1_lower_deg,
        a1_upper_deg=a1_upper_deg,
        a2_max_deg=a2_max_deg,
        joint_constraint_tolerance_deg=joint_constraint_tolerance_deg,
        profile_result_cache=profile_result_cache,
    )
    best_result = _refine_path_with_handover_corridors(
        best_result,
        robot=robot,
        mat_type=mat_type,
        move_type=move_type,
        start_joints=start_joints,
        tool_pose=tool_pose,
        reference_pose=reference_pose,
        joint_count=joint_count,
        motion_settings=motion_settings,
        optimizer_settings=optimizer_settings,
        lower_limits=lower_limits_tuple,
        upper_limits=upper_limits_tuple,
        a1_lower_deg=a1_lower_deg,
        a1_upper_deg=a1_upper_deg,
        a2_max_deg=a2_max_deg,
        joint_constraint_tolerance_deg=joint_constraint_tolerance_deg,
        profile_result_cache=profile_result_cache,
    )

    for pass_index in range(max(0, motion_settings.frame_a_origin_yz_max_passes)):
        problem_segments = _collect_problem_segments(
            best_result.selected_path,
            bridge_trigger_joint_delta_deg=motion_settings.bridge_trigger_joint_delta_deg,
            max_segments=5,
        )
        if not problem_segments:
            break

        accepted_result: _PathSearchResult | None = None
        accepted_metadata: tuple[int, int, int, float, float] | None = None
        candidate_profiles: list[
            tuple[
                tuple[tuple[float, float], ...],
                tuple[int, int, int, float, float],
            ]
        ] = []
        seen_profiles: set[tuple[tuple[float, float], ...]] = set()
        for segment_index, _config_changed, _max_joint_delta, _mean_joint_delta in problem_segments[:5]:
            window_start = max(0, segment_index - motion_settings.frame_a_origin_yz_window_radius)
            window_end = min(
                len(best_result.reference_pose_rows) - 1,
                segment_index + 1 + motion_settings.frame_a_origin_yz_window_radius,
            )

            # For config-switch segments add wider envelope passes so the DP can reach
            # bridge points that lie farther from the nominal winding path.
            envelope_schedule = tuple(
                float(value)
                for value in motion_settings.frame_a_origin_yz_envelope_schedule_mm
            )
            normal_max_envelope = max(envelope_schedule, default=0.0)
            if _config_changed:
                widened_envelopes: list[float] = []
                seen_envelopes = set(envelope_schedule)
                for multiplier in (2.0, 3.0):
                    wider = round(normal_max_envelope * multiplier, 1)
                    if wider not in seen_envelopes:
                        seen_envelopes.add(wider)
                        widened_envelopes.append(wider)
                envelope_schedule = (*envelope_schedule, *widened_envelopes)

            base_step_schedule = tuple(
                float(value) for value in motion_settings.frame_a_origin_yz_step_schedule_mm
            )
            coarse_step_schedule = base_step_schedule[:2] or base_step_schedule
            for envelope_mm in envelope_schedule:
                # Coarser steps for extra-wide passes to keep cost bounded.
                step_schedule = (
                    coarse_step_schedule
                    if float(envelope_mm) > normal_max_envelope
                    else base_step_schedule
                )
                for step_mm in step_schedule:
                    candidate_profile = _solve_window_profile_dp(
                        best_result,
                        window_start=window_start,
                        window_end=window_end,
                        max_abs_offset_mm=envelope_mm,
                        step_mm=step_mm,
                        robot=robot,
                        mat_type=mat_type,
                        tool_pose=tool_pose,
                        reference_pose=reference_pose,
                        joint_count=joint_count,
                        optimizer_settings=optimizer_settings,
                        a1_lower_deg=a1_lower_deg,
                        a1_upper_deg=a1_upper_deg,
                        a2_max_deg=a2_max_deg,
                        joint_constraint_tolerance_deg=joint_constraint_tolerance_deg,
                        lower_limits=lower_limits_tuple,
                        upper_limits=upper_limits_tuple,
                        start_joints=start_joints,
                    )
                    if candidate_profile is None:
                        continue
                    candidate_profile = _lock_profile_endpoints_if_needed(
                        candidate_profile,
                        reference_pose_rows=best_result.reference_pose_rows,
                        motion_settings=motion_settings,
                    )
                    profile_key = tuple(
                        (round(float(dy_mm), 6), round(float(dz_mm), 6))
                        for dy_mm, dz_mm in candidate_profile
                    )
                    if profile_key in seen_profiles:
                        continue
                    seen_profiles.add(profile_key)
                    candidate_profiles.append(
                        (
                            candidate_profile,
                            (
                                segment_index,
                                window_start,
                                window_end,
                                float(envelope_mm),
                                float(step_mm),
                            ),
                        )
                    )

        cached_evaluated_candidates: list[
            tuple[
                tuple[tuple[tuple[float, float], ...], tuple[int, int, int, float, float]],
                _PathSearchResult,
            ]
        ] = []
        uncached_candidate_profiles: list[
            tuple[tuple[tuple[float, float], ...], tuple[int, int, int, float, float]]
        ] = []
        for candidate_profile, metadata in candidate_profiles:
            cached_result = profile_result_cache.get(_profile_cache_key(candidate_profile))
            if cached_result is None:
                uncached_candidate_profiles.append((candidate_profile, metadata))
            else:
                cached_evaluated_candidates.append(
                    ((candidate_profile, metadata), cached_result)
                )

        evaluated_uncached_candidates: list[
            tuple[
                tuple[tuple[tuple[float, float], ...], tuple[int, int, int, float, float]],
                _PathSearchResult,
            ]
        ] = []
        if uncached_candidate_profiles:
            parallel_results = maybe_parallel_evaluate_exact_profiles(
                reference_pose_rows=best_result.reference_pose_rows,
                base_frame_a_origin_yz_profile_mm=best_result.frame_a_origin_yz_profile_mm,
                reused_ik_layers=best_result.ik_layers,
                frame_a_origin_yz_profiles_mm=[
                    candidate_profile
                    for candidate_profile, _metadata in uncached_candidate_profiles
                ],
                row_labels=best_result.row_labels,
                inserted_flags=best_result.inserted_flags,
                motion_settings=motion_settings,
                start_joints=start_joints,
            )
        else:
            parallel_results = None

        if parallel_results is not None:
            for (candidate_profile, metadata), parallel_result in zip(
                uncached_candidate_profiles,
                parallel_results,
            ):
                profile_result_cache[_profile_cache_key(candidate_profile)] = parallel_result
                evaluated_uncached_candidates.append(
                    ((candidate_profile, metadata), parallel_result)
                )
        else:
            for candidate_profile, metadata in uncached_candidate_profiles:
                evaluated_uncached_candidates.append(
                    (
                        (candidate_profile, metadata),
                        _evaluate_exact_profile_with_cache(
                            candidate_profile,
                            base_result=best_result,
                            lock_profile_endpoints=bool(
                                getattr(
                                    motion_settings,
                                    "lock_frame_a_origin_yz_profile_endpoints",
                                    True,
                                )
                            ),
                            profile_result_cache=profile_result_cache,
                            robot=robot,
                            mat_type=mat_type,
                            move_type=move_type,
                            start_joints=start_joints,
                            tool_pose=tool_pose,
                            reference_pose=reference_pose,
                            joint_count=joint_count,
                            optimizer_settings=optimizer_settings,
                            a1_lower_deg=a1_lower_deg,
                            a1_upper_deg=a1_upper_deg,
                            a2_max_deg=a2_max_deg,
                            joint_constraint_tolerance_deg=joint_constraint_tolerance_deg,
                            seed_joints=(
                                ()
                                if getattr(robot, "ik_seed_invariant", False)
                                else _build_local_seed_joints(
                                    best_result,
                                    metadata[1],
                                    metadata[2],
                                )
                            ),
                            lower_limits=lower_limits_tuple,
                            upper_limits=upper_limits_tuple,
                            bridge_trigger_joint_delta_deg=(
                                motion_settings.bridge_trigger_joint_delta_deg
                            ),
                        ),
                    )
                )

        evaluated_candidates = itertools.chain(
            cached_evaluated_candidates,
            evaluated_uncached_candidates,
        )

        for (_candidate_profile, metadata), candidate_result in evaluated_candidates:
            if _path_search_sort_key(candidate_result) < _path_search_sort_key(best_result):
                if (
                    accepted_result is None
                    or _path_search_sort_key(candidate_result)
                    < _path_search_sort_key(accepted_result)
                ):
                    accepted_result = candidate_result
                    accepted_metadata = metadata

        if accepted_result is None:
            break
        assert accepted_metadata is not None
        segment_index, window_start, window_end, envelope_mm, step_mm = accepted_metadata
        print(
            "Accepted Frame-2 Y/Z window refinement: "
            f"pass={pass_index + 1}, "
            f"segment={best_result.row_labels[segment_index]}->{best_result.row_labels[segment_index + 1]}, "
            f"window={best_result.row_labels[window_start]}->{best_result.row_labels[window_end]}, "
            f"envelope={envelope_mm:.1f} mm, step={step_mm:.1f} mm, "
            f"bridge_like_segments={accepted_result.bridge_like_segments}, "
            f"worst_joint_step={accepted_result.worst_joint_step_deg:.3f} deg."
        )
        best_result = accepted_result

    return best_result


def _refine_path_with_global_continuous_profile(
    search_result: _PathSearchResult,
    *,
    robot,
    mat_type,
    move_type: str,
    start_joints: tuple[float, ...],
    tool_pose,
    reference_pose,
    joint_count: int,
    motion_settings,
    optimizer_settings: _PathOptimizerSettings,
    lower_limits: Sequence[float],
    upper_limits: Sequence[float],
    a1_lower_deg: float,
    a1_upper_deg: float,
    a2_max_deg: float,
    joint_constraint_tolerance_deg: float,
    profile_result_cache: dict[tuple[tuple[float, float], ...], _PathSearchResult],
) -> _PathSearchResult:
    best_result = search_result
    if not best_result.selected_path or not best_result.ik_layers:
        return best_result

    max_abs_offset_mm = max(
        (float(value) for value in motion_settings.frame_a_origin_yz_envelope_schedule_mm),
        default=0.0,
    )
    if max_abs_offset_mm <= 0.0:
        return best_result

    fd_step_mm = min(
        _GLOBAL_CONTINUOUS_REFINEMENT_FD_STEP_MM,
        max_abs_offset_mm,
    )
    if fd_step_mm <= 0.0:
        return best_result
    continuity_step_limit_mm = _resolve_profile_continuity_step_limit(motion_settings)

    for iteration_index in range(_GLOBAL_CONTINUOUS_REFINEMENT_MAX_ITERS):
        problem_segments = _collect_problem_segments(
            best_result.selected_path,
            bridge_trigger_joint_delta_deg=motion_settings.bridge_trigger_joint_delta_deg,
        )
        if not problem_segments:
            break

        linearizations = _build_global_profile_linearizations(
            best_result,
            robot=robot,
            mat_type=mat_type,
            tool_pose=tool_pose,
            reference_pose=reference_pose,
            joint_count=joint_count,
            optimizer_settings=optimizer_settings,
            a1_lower_deg=a1_lower_deg,
            a1_upper_deg=a1_upper_deg,
            a2_max_deg=a2_max_deg,
            joint_constraint_tolerance_deg=joint_constraint_tolerance_deg,
            lower_limits=lower_limits,
            upper_limits=upper_limits,
            max_abs_offset_mm=max_abs_offset_mm,
            fd_step_mm=fd_step_mm,
        )
        if linearizations is None:
            break

        max_profile_step_mm = continuity_step_limit_mm
        update_proposals: list[tuple[str, tuple[tuple[float, float], ...]]] = []
        has_config_switch_problem = any(
            config_changed for _seg, config_changed, _max, _mean in problem_segments
        )
        use_minmax = bool(_GLOBAL_CONTINUOUS_ENABLE_MINMAX)
        if (
            not use_minmax
            and has_config_switch_problem
            and int(getattr(best_result, "bridge_like_segments", 0)) >= 2
        ):
            # Multi-bridge windows are dominated by the worst transition.
            # Enable min-max stage to target peak joint jump directly.
            use_minmax = True
        if use_minmax:
            minmax_update = _solve_global_profile_minmax_update(
                best_result,
                linearizations,
                motion_settings=motion_settings,
                max_abs_offset_mm=max_abs_offset_mm,
                max_profile_step_mm=max_profile_step_mm,
                problem_segments=problem_segments,
            )
            if minmax_update is not None:
                profile_delta, _predicted_max_step_deg = minmax_update
                update_proposals.append(("minmax", profile_delta))

        lsq_profile_delta = _solve_global_profile_update(
            best_result,
            linearizations,
            optimizer_settings=optimizer_settings,
            motion_settings=motion_settings,
            max_abs_offset_mm=max_abs_offset_mm,
            problem_segments=problem_segments,
        )
        if lsq_profile_delta is not None:
            update_proposals.append(("lsq", lsq_profile_delta))

        if not update_proposals:
            break

        candidate_profile_specs: list[tuple[tuple[tuple[float, float], ...], str]] = []
        seen_profile_keys: set[tuple[tuple[float, float], ...]] = set()
        for solver_name, profile_delta in update_proposals:
            for trust_scale in _GLOBAL_CONTINUOUS_REFINEMENT_TRUST_SCALES:
                candidate_profile = _apply_profile_delta(
                    best_result.frame_a_origin_yz_profile_mm,
                    profile_delta,
                    trust_scale=trust_scale,
                    max_abs_offset_mm=max_abs_offset_mm,
                )
                candidate_profile = _lock_profile_endpoints_if_needed(
                    candidate_profile,
                    reference_pose_rows=best_result.reference_pose_rows,
                    motion_settings=motion_settings,
                )
                if candidate_profile == best_result.frame_a_origin_yz_profile_mm:
                    continue
                if not _profile_is_continuous_enough(
                    candidate_profile,
                    motion_settings=motion_settings,
                    max_step_mm=max_profile_step_mm,
                ):
                    continue
                profile_key = tuple(
                    (
                        round(float(offset[0]), 6),
                        round(float(offset[1]), 6),
                    )
                    for offset in candidate_profile
                )
                if profile_key in seen_profile_keys:
                    continue
                seen_profile_keys.add(profile_key)
                candidate_profile_specs.append((candidate_profile, solver_name))

        if not candidate_profile_specs:
            break

        cached_evaluated_candidates: list[
            tuple[tuple[tuple[float, float], ...], str, _PathSearchResult]
        ] = []
        uncached_candidate_profile_specs: list[tuple[tuple[tuple[float, float], ...], str]] = []
        for candidate_profile, solver_name in candidate_profile_specs:
            cached_result = profile_result_cache.get(_profile_cache_key(candidate_profile))
            if cached_result is None:
                uncached_candidate_profile_specs.append((candidate_profile, solver_name))
            else:
                cached_evaluated_candidates.append(
                    (candidate_profile, solver_name, cached_result)
                )

        evaluated_uncached_candidates: list[
            tuple[tuple[tuple[float, float], ...], str, _PathSearchResult]
        ] = []
        if uncached_candidate_profile_specs:
            candidate_profiles = [
                profile for profile, _solver_name in uncached_candidate_profile_specs
            ]
            parallel_results = maybe_parallel_evaluate_exact_profiles(
                reference_pose_rows=best_result.reference_pose_rows,
                base_frame_a_origin_yz_profile_mm=best_result.frame_a_origin_yz_profile_mm,
                reused_ik_layers=best_result.ik_layers,
                frame_a_origin_yz_profiles_mm=candidate_profiles,
                row_labels=best_result.row_labels,
                inserted_flags=best_result.inserted_flags,
                motion_settings=motion_settings,
                start_joints=start_joints,
            )
        else:
            parallel_results = None

        if parallel_results is not None:
            for (candidate_profile, solver_name), parallel_result in zip(
                uncached_candidate_profile_specs,
                parallel_results,
            ):
                profile_result_cache[_profile_cache_key(candidate_profile)] = parallel_result
                evaluated_uncached_candidates.append(
                    (candidate_profile, solver_name, parallel_result)
                )
        else:
            for candidate_profile, solver_name in uncached_candidate_profile_specs:
                evaluated_uncached_candidates.append(
                    (
                        candidate_profile,
                        solver_name,
                        _evaluate_exact_profile_with_cache(
                            candidate_profile,
                            base_result=best_result,
                            lock_profile_endpoints=bool(
                                getattr(
                                    motion_settings,
                                    "lock_frame_a_origin_yz_profile_endpoints",
                                    True,
                                )
                            ),
                            profile_result_cache=profile_result_cache,
                            robot=robot,
                            mat_type=mat_type,
                            move_type=move_type,
                            start_joints=start_joints,
                            tool_pose=tool_pose,
                            reference_pose=reference_pose,
                            joint_count=joint_count,
                            optimizer_settings=optimizer_settings,
                            a1_lower_deg=a1_lower_deg,
                            a1_upper_deg=a1_upper_deg,
                            a2_max_deg=a2_max_deg,
                            joint_constraint_tolerance_deg=joint_constraint_tolerance_deg,
                            seed_joints=(
                                ()
                                if getattr(robot, "ik_seed_invariant", False)
                                else _build_local_seed_joints(
                                    best_result,
                                    0,
                                    len(best_result.reference_pose_rows) - 1,
                                )
                            ),
                            lower_limits=lower_limits,
                            upper_limits=upper_limits,
                            bridge_trigger_joint_delta_deg=(
                                motion_settings.bridge_trigger_joint_delta_deg
                            ),
                        ),
                    )
                )

        evaluated_candidates = itertools.chain(
            cached_evaluated_candidates,
            evaluated_uncached_candidates,
        )

        accepted_result: _PathSearchResult | None = None
        accepted_profile: tuple[tuple[float, float], ...] | None = None
        accepted_solver_name = "lsq"
        for candidate_profile, solver_name, candidate_result in evaluated_candidates:
            if not _profile_is_continuous_enough(
                candidate_profile,
                motion_settings=motion_settings,
                max_step_mm=continuity_step_limit_mm,
            ):
                continue
            if _path_search_sort_key(candidate_result) < _path_search_sort_key(best_result):
                if (
                    accepted_result is None
                    or _path_search_sort_key(candidate_result)
                    < _path_search_sort_key(accepted_result)
                ):
                    accepted_result = candidate_result
                    accepted_profile = candidate_profile
                    accepted_solver_name = solver_name

        if accepted_result is None or accepted_profile is None:
            break

        max_profile_shift_mm = max(
            math.hypot(
                float(new_value[0]) - float(old_value[0]),
                float(new_value[1]) - float(old_value[1]),
            )
            for old_value, new_value in zip(
                best_result.frame_a_origin_yz_profile_mm,
                accepted_profile,
            )
        )
        print(
            "Accepted global continuous Frame-2 Y/Z refinement: "
            f"iteration={iteration_index + 1}, "
            f"solver={accepted_solver_name}, "
            f"max_profile_shift={max_profile_shift_mm:.3f} mm, "
            f"config_switches={accepted_result.config_switches}, "
            f"bridge_like_segments={accepted_result.bridge_like_segments}, "
            f"worst_joint_step={accepted_result.worst_joint_step_deg:.3f} deg."
        )
        best_result = accepted_result

    return best_result


def _build_global_profile_linearizations(
    search_result: _PathSearchResult,
    *,
    robot,
    mat_type,
    tool_pose,
    reference_pose,
    joint_count: int,
    optimizer_settings: _PathOptimizerSettings,
    a1_lower_deg: float,
    a1_upper_deg: float,
    a2_max_deg: float,
    joint_constraint_tolerance_deg: float,
    lower_limits: Sequence[float],
    upper_limits: Sequence[float],
    max_abs_offset_mm: float,
    fd_step_mm: float,
) -> tuple[_GlobalRowLinearization, ...] | None:
    linearizations: list[_GlobalRowLinearization] = []
    for row_index, (reference_row, current_offset, selected_candidate) in enumerate(
        zip(
            search_result.reference_pose_rows,
            search_result.frame_a_origin_yz_profile_mm,
            search_result.selected_path,
        )
    ):
        gradient_y = _estimate_profile_joint_gradient(
            reference_row=reference_row,
            base_offset=current_offset,
            axis_index=0,
            robot=robot,
            mat_type=mat_type,
            tool_pose=tool_pose,
            reference_pose=reference_pose,
            joint_count=joint_count,
            optimizer_settings=optimizer_settings,
            a1_lower_deg=a1_lower_deg,
            a1_upper_deg=a1_upper_deg,
            a2_max_deg=a2_max_deg,
            joint_constraint_tolerance_deg=joint_constraint_tolerance_deg,
            lower_limits=lower_limits,
            upper_limits=upper_limits,
            selected_candidate=selected_candidate,
            max_abs_offset_mm=max_abs_offset_mm,
            fd_step_mm=fd_step_mm,
        )
        gradient_z = _estimate_profile_joint_gradient(
            reference_row=reference_row,
            base_offset=current_offset,
            axis_index=1,
            robot=robot,
            mat_type=mat_type,
            tool_pose=tool_pose,
            reference_pose=reference_pose,
            joint_count=joint_count,
            optimizer_settings=optimizer_settings,
            a1_lower_deg=a1_lower_deg,
            a1_upper_deg=a1_upper_deg,
            a2_max_deg=a2_max_deg,
            joint_constraint_tolerance_deg=joint_constraint_tolerance_deg,
            lower_limits=lower_limits,
            upper_limits=upper_limits,
            selected_candidate=selected_candidate,
            max_abs_offset_mm=max_abs_offset_mm,
            fd_step_mm=fd_step_mm,
        )
        linearizations.append(
            _GlobalRowLinearization(
                joints=tuple(float(value) for value in selected_candidate.joints),
                gradient_y=gradient_y,
                gradient_z=gradient_z,
            )
        )

    if not linearizations:
        return None
    return tuple(linearizations)


def _estimate_profile_joint_gradient(
    *,
    reference_row: dict[str, float],
    base_offset: tuple[float, float],
    axis_index: int,
    robot,
    mat_type,
    tool_pose,
    reference_pose,
    joint_count: int,
    optimizer_settings: _PathOptimizerSettings,
    a1_lower_deg: float,
    a1_upper_deg: float,
    a2_max_deg: float,
    joint_constraint_tolerance_deg: float,
    lower_limits: Sequence[float],
    upper_limits: Sequence[float],
    selected_candidate: _IKCandidate,
    max_abs_offset_mm: float,
    fd_step_mm: float,
) -> tuple[float, ...]:
    positive_offset = list(base_offset)
    negative_offset = list(base_offset)
    positive_offset[axis_index] += fd_step_mm
    negative_offset[axis_index] -= fd_step_mm

    positive_candidate = None
    negative_candidate = None
    if abs(positive_offset[axis_index]) <= max_abs_offset_mm + 1e-9:
        positive_candidate = _collect_profile_axis_candidate(
            reference_row=reference_row,
            offset=(float(positive_offset[0]), float(positive_offset[1])),
            selected_candidate=selected_candidate,
            robot=robot,
            mat_type=mat_type,
            tool_pose=tool_pose,
            reference_pose=reference_pose,
            joint_count=joint_count,
            optimizer_settings=optimizer_settings,
            a1_lower_deg=a1_lower_deg,
            a1_upper_deg=a1_upper_deg,
            a2_max_deg=a2_max_deg,
            joint_constraint_tolerance_deg=joint_constraint_tolerance_deg,
            lower_limits=lower_limits,
            upper_limits=upper_limits,
        )
    if abs(negative_offset[axis_index]) <= max_abs_offset_mm + 1e-9:
        negative_candidate = _collect_profile_axis_candidate(
            reference_row=reference_row,
            offset=(float(negative_offset[0]), float(negative_offset[1])),
            selected_candidate=selected_candidate,
            robot=robot,
            mat_type=mat_type,
            tool_pose=tool_pose,
            reference_pose=reference_pose,
            joint_count=joint_count,
            optimizer_settings=optimizer_settings,
            a1_lower_deg=a1_lower_deg,
            a1_upper_deg=a1_upper_deg,
            a2_max_deg=a2_max_deg,
            joint_constraint_tolerance_deg=joint_constraint_tolerance_deg,
            lower_limits=lower_limits,
            upper_limits=upper_limits,
        )

    if positive_candidate is not None and negative_candidate is not None:
        return tuple(
            (float(positive_value) - float(negative_value)) / (2.0 * fd_step_mm)
            for positive_value, negative_value in zip(
                positive_candidate.joints,
                negative_candidate.joints,
            )
        )
    if positive_candidate is not None:
        return tuple(
            (float(positive_value) - float(base_value)) / fd_step_mm
            for positive_value, base_value in zip(
                positive_candidate.joints,
                selected_candidate.joints,
            )
        )
    if negative_candidate is not None:
        return tuple(
            (float(base_value) - float(negative_value)) / fd_step_mm
            for base_value, negative_value in zip(
                selected_candidate.joints,
                negative_candidate.joints,
            )
        )
    return tuple(0.0 for _ in range(len(selected_candidate.joints)))


def _collect_profile_axis_candidate(
    *,
    reference_row: dict[str, float],
    offset: tuple[float, float],
    selected_candidate: _IKCandidate,
    robot,
    mat_type,
    tool_pose,
    reference_pose,
    joint_count: int,
    optimizer_settings: _PathOptimizerSettings,
    a1_lower_deg: float,
    a1_upper_deg: float,
    a2_max_deg: float,
    joint_constraint_tolerance_deg: float,
    lower_limits: Sequence[float],
    upper_limits: Sequence[float],
) -> _IKCandidate | None:
    return _collect_profile_family_candidate(
        reference_row=reference_row,
        offset=offset,
        reference_candidate=selected_candidate,
        family_flags=selected_candidate.config_flags,
        lineage_key=_candidate_lineage_key(selected_candidate),
        robot=robot,
        mat_type=mat_type,
        tool_pose=tool_pose,
        reference_pose=reference_pose,
        joint_count=joint_count,
        optimizer_settings=optimizer_settings,
        a1_lower_deg=a1_lower_deg,
        a1_upper_deg=a1_upper_deg,
        a2_max_deg=a2_max_deg,
        joint_constraint_tolerance_deg=joint_constraint_tolerance_deg,
        lower_limits=lower_limits,
        upper_limits=upper_limits,
    )


def _collect_profile_family_candidate(
    *,
    reference_row: dict[str, float],
    offset: tuple[float, float],
    reference_candidate: _IKCandidate,
    family_flags: tuple[int, ...],
    lineage_key: tuple[int, ...] | None,
    robot,
    mat_type,
    tool_pose,
    reference_pose,
    joint_count: int,
    optimizer_settings: _PathOptimizerSettings,
    a1_lower_deg: float,
    a1_upper_deg: float,
    a2_max_deg: float,
    joint_constraint_tolerance_deg: float,
    lower_limits: Sequence[float],
    upper_limits: Sequence[float],
) -> _IKCandidate | None:
    adjusted_row = dict(reference_row)
    adjusted_row["x_mm"] = float(reference_row["x_mm"])
    adjusted_row["y_mm"] = float(reference_row["y_mm"]) + float(offset[0])
    adjusted_row["z_mm"] = float(reference_row["z_mm"]) + float(offset[1])
    pose = _build_pose(adjusted_row, mat_type)
    candidates = _collect_ik_candidates(
        robot,
        pose,
        tool_pose=tool_pose,
        reference_pose=reference_pose,
        lower_limits=tuple(float(value) for value in lower_limits[:joint_count]),
        upper_limits=tuple(float(value) for value in upper_limits[:joint_count]),
        seed_joints=(
            ()
            if getattr(robot, "ik_seed_invariant", False)
            else (tuple(float(value) for value in reference_candidate.joints),)
        ),
        joint_count=joint_count,
        optimizer_settings=optimizer_settings,
        a1_lower_deg=a1_lower_deg,
        a1_upper_deg=a1_upper_deg,
        a2_max_deg=a2_max_deg,
        joint_constraint_tolerance_deg=joint_constraint_tolerance_deg,
    )
    same_family_candidates = [
        candidate
        for candidate in candidates
        if candidate.config_flags == family_flags
    ]
    if not same_family_candidates:
        return None
    preferred_candidates = [
        candidate
        for candidate in same_family_candidates
        if _lineage_matches_family(
            candidate,
            family_flags=family_flags,
            lineage_key=lineage_key,
        )
    ]
    candidate_pool = preferred_candidates or same_family_candidates
    return min(
        candidate_pool,
        key=lambda candidate: _joint_transition_penalty(
            reference_candidate.joints,
            candidate.joints,
            optimizer_settings,
        ),
    )


def _build_global_joint_step_expression(
    *,
    linearizations: Sequence[_GlobalRowLinearization],
    segment_index: int,
    joint_index: int,
) -> tuple[np.ndarray, float]:
    row_count = len(linearizations)
    coefficient = np.zeros(row_count * 2, dtype=float)
    previous_row = linearizations[segment_index]
    current_row = linearizations[segment_index + 1]
    constant = float(current_row.joints[joint_index]) - float(previous_row.joints[joint_index])

    coefficient[segment_index * 2] -= float(previous_row.gradient_y[joint_index])
    coefficient[(segment_index * 2) + 1] -= float(previous_row.gradient_z[joint_index])
    coefficient[(segment_index + 1) * 2] += float(current_row.gradient_y[joint_index])
    coefficient[((segment_index + 1) * 2) + 1] += float(current_row.gradient_z[joint_index])
    return coefficient, constant


def _solve_global_profile_minmax_update(
    search_result: _PathSearchResult,
    linearizations: Sequence[_GlobalRowLinearization],
    *,
    motion_settings,
    max_abs_offset_mm: float,
    max_profile_step_mm: float,
    problem_segments: Sequence[tuple[int, bool, float, float]],
) -> tuple[tuple[tuple[float, float], ...], float] | None:
    row_count = len(linearizations)
    if row_count <= 1:
        return None

    x_count = row_count * 2
    t_index = x_count
    base_profile = search_result.frame_a_origin_yz_profile_mm

    lower_bounds = np.zeros(x_count, dtype=float)
    upper_bounds = np.zeros(x_count, dtype=float)
    for row_index, offset in enumerate(base_profile):
        current_dy_mm = float(offset[0])
        current_dz_mm = float(offset[1])
        lower_bounds[row_index * 2] = -max_abs_offset_mm - current_dy_mm
        upper_bounds[row_index * 2] = max_abs_offset_mm - current_dy_mm
        lower_bounds[(row_index * 2) + 1] = -max_abs_offset_mm - current_dz_mm
        upper_bounds[(row_index * 2) + 1] = max_abs_offset_mm - current_dz_mm

    primary_expressions: list[tuple[int, int, np.ndarray, float]] = []
    for segment_index in range(row_count - 1):
        joint_count = min(
            len(linearizations[segment_index].joints),
            len(linearizations[segment_index + 1].joints),
        )
        for joint_index in range(joint_count):
            coefficient, constant = _build_global_joint_step_expression(
                linearizations=linearizations,
                segment_index=segment_index,
                joint_index=joint_index,
            )
            primary_expressions.append((segment_index, joint_index, coefficient, constant))
    if not primary_expressions:
        return None

    stage1_variable_count = x_count + 1
    stage1_objective = np.zeros(stage1_variable_count, dtype=float)
    stage1_objective[t_index] = 1.0
    stage1_rows: list[np.ndarray] = []
    stage1_rhs: list[float] = []
    for _segment_index, _joint_index, coefficient, constant in primary_expressions:
        _append_abs_epigraph_constraints(
            stage1_rows,
            stage1_rhs,
            coefficient=coefficient,
            constant=constant,
            t_index=t_index,
            variable_count=stage1_variable_count,
        )

    _append_profile_continuity_constraints(
        stage1_rows,
        stage1_rhs,
        corridor_row_count=row_count,
        base_profile=base_profile,
        max_profile_step_mm=max_profile_step_mm,
        variable_count=stage1_variable_count,
    )

    stage1_bounds = [
        (float(lower_bounds[index]), float(upper_bounds[index]))
        for index in range(x_count)
    ]
    stage1_bounds.append((0.0, None))

    stage1_solution = linprog(
        c=stage1_objective,
        A_ub=np.vstack(stage1_rows),
        b_ub=np.asarray(stage1_rhs, dtype=float),
        bounds=stage1_bounds,
        method="highs",
    )
    if not stage1_solution.success or stage1_solution.x is None:
        return None

    stage1_t_max_deg = float(stage1_solution.x[t_index])
    problem_segment_indices = {
        int(segment_index)
        for segment_index, _config_changed, _max_joint_delta, _mean_joint_delta in problem_segments
        if 0 <= int(segment_index) < (row_count - 1)
    }
    problem_expressions = [
        (coefficient, constant)
        for segment_index, _joint_index, coefficient, constant in primary_expressions
        if segment_index in problem_segment_indices
    ]
    if not problem_expressions:
        problem_expressions = [
            (coefficient, constant)
            for _segment_index, _joint_index, coefficient, constant in primary_expressions
        ]

    all_step_expressions = [
        (coefficient, constant)
        for _segment_index, _joint_index, coefficient, constant in primary_expressions
    ]
    magnitude_expressions: list[tuple[np.ndarray, float]] = []
    for row_index in range(row_count):
        for axis_index in range(2):
            coefficient = np.zeros(x_count, dtype=float)
            coefficient[(row_index * 2) + axis_index] = 1.0
            magnitude_expressions.append((coefficient, 0.0))

    smoothness_expressions: list[tuple[np.ndarray, float]] = []
    for row_index in range(1, row_count):
        for axis_index in range(2):
            smoothness_expressions.append(
                _build_axis_difference_expression(
                    corridor_row_count=row_count,
                    left_row_index=row_index - 1,
                    right_row_index=row_index,
                    axis_index=axis_index,
                )
            )

    curvature_expressions: list[tuple[np.ndarray, float]] = []
    for row_index in range(1, row_count - 1):
        for axis_index in range(2):
            curvature_expressions.append(
                _build_axis_curvature_expression(
                    corridor_row_count=row_count,
                    center_row_index=row_index,
                    axis_index=axis_index,
                )
            )

    stage2_groups: tuple[tuple[tuple[np.ndarray, float], ...], ...] = (
        tuple(problem_expressions),
        tuple(all_step_expressions),
        tuple(magnitude_expressions),
        tuple(smoothness_expressions),
        tuple(curvature_expressions),
    )
    group_upper_bounds = [
        sum(
            _expression_abs_upper_bound(
                coefficient,
                constant,
                lower_bounds=lower_bounds,
                upper_bounds=upper_bounds,
            )
            for coefficient, constant in group
        )
        for group in stage2_groups
    ]
    group_weights = _build_lexicographic_group_weights(group_upper_bounds)

    stage2_variable_count = x_count + 1 + sum(len(group) for group in stage2_groups)
    stage2_objective = np.zeros(stage2_variable_count, dtype=float)
    stage2_bounds: list[tuple[float | None, float | None]] = [
        (float(lower_bounds[index]), float(upper_bounds[index]))
        for index in range(x_count)
    ]
    stage2_bounds.append((0.0, float(stage1_t_max_deg + _GLOBAL_CONTINUOUS_MINMAX_TOLERANCE_DEG)))

    stage2_rows: list[np.ndarray] = []
    stage2_rhs: list[float] = []
    for _segment_index, _joint_index, coefficient, constant in primary_expressions:
        _append_abs_epigraph_constraints(
            stage2_rows,
            stage2_rhs,
            coefficient=coefficient,
            constant=constant,
            t_index=t_index,
            variable_count=stage2_variable_count,
        )

    _append_profile_continuity_constraints(
        stage2_rows,
        stage2_rhs,
        corridor_row_count=row_count,
        base_profile=base_profile,
        max_profile_step_mm=max_profile_step_mm,
        variable_count=stage2_variable_count,
    )

    slack_index = x_count + 1
    for group_weight, group in zip(group_weights, stage2_groups):
        for coefficient, constant in group:
            stage2_objective[slack_index] = float(group_weight)
            stage2_bounds.append((0.0, None))
            _append_abs_slack_constraints(
                stage2_rows,
                stage2_rhs,
                coefficient=coefficient,
                constant=constant,
                slack_index=slack_index,
                variable_count=stage2_variable_count,
            )
            slack_index += 1

    stage2_solution = linprog(
        c=stage2_objective,
        A_ub=np.vstack(stage2_rows),
        b_ub=np.asarray(stage2_rhs, dtype=float),
        bounds=stage2_bounds,
        method="highs",
    )
    if not stage2_solution.success or stage2_solution.x is None:
        corridor_delta = tuple(
            (
                float(stage1_solution.x[row_index * 2]),
                float(stage1_solution.x[(row_index * 2) + 1]),
            )
            for row_index in range(row_count)
        )
        return corridor_delta, stage1_t_max_deg

    corridor_delta = tuple(
        (
            float(stage2_solution.x[row_index * 2]),
            float(stage2_solution.x[(row_index * 2) + 1]),
        )
        for row_index in range(row_count)
    )
    return corridor_delta, float(stage2_solution.x[t_index])


def _solve_global_profile_update(
    search_result: _PathSearchResult,
    linearizations: Sequence[_GlobalRowLinearization],
    *,
    optimizer_settings: _PathOptimizerSettings,
    motion_settings,
    max_abs_offset_mm: float,
    problem_segments: Sequence[tuple[int, bool, float, float]],
) -> tuple[tuple[float, float], ...] | None:
    row_count = len(linearizations)
    variable_count = row_count * 2
    if row_count == 0:
        return None

    problem_segment_weights = {
        int(segment_index): 1.0 + min(24.0, float(max_joint_delta) / 6.0)
        for segment_index, _config_changed, max_joint_delta, _mean_joint_delta in problem_segments
    }
    preferred_limits = tuple(float(value) for value in optimizer_settings.preferred_joint_step_deg)
    rows: list[np.ndarray] = []
    rhs_values: list[float] = []

    for segment_index in range(row_count - 1):
        previous_row = linearizations[segment_index]
        current_row = linearizations[segment_index + 1]
        joint_deltas = [
            abs(current_value - previous_value)
            for previous_value, current_value in zip(previous_row.joints, current_row.joints)
        ]
        max_joint_delta = max(joint_deltas, default=0.0)
        segment_weight = 0.20 + min(
            12.0,
            (max_joint_delta / max(1.0, max(preferred_limits, default=1.0))) ** 2,
        )
        segment_weight *= problem_segment_weights.get(segment_index, 1.0)
        segment_scale = math.sqrt(segment_weight)

        for joint_index, (previous_joint, current_joint) in enumerate(
            zip(previous_row.joints, current_row.joints)
        ):
            preferred_limit = preferred_limits[joint_index] if joint_index < len(preferred_limits) else 1.0
            joint_scale = (
                segment_scale
                * optimizer_settings.joint_delta_weights[joint_index]
                / max(1.0, preferred_limit)
            )
            row = np.zeros(variable_count, dtype=float)
            row[(segment_index * 2)] = -joint_scale * previous_row.gradient_y[joint_index]
            row[(segment_index * 2) + 1] = -joint_scale * previous_row.gradient_z[joint_index]
            row[((segment_index + 1) * 2)] = joint_scale * current_row.gradient_y[joint_index]
            row[((segment_index + 1) * 2) + 1] = joint_scale * current_row.gradient_z[joint_index]
            rows.append(row)
            rhs_values.append(-joint_scale * (current_joint - previous_joint))

    update_magnitude_weight = math.sqrt(_GLOBAL_CONTINUOUS_UPDATE_MAGNITUDE_WEIGHT)
    update_smoothness_weight = math.sqrt(_GLOBAL_CONTINUOUS_UPDATE_SMOOTHNESS_WEIGHT)
    update_curvature_weight = math.sqrt(_GLOBAL_CONTINUOUS_UPDATE_CURVATURE_WEIGHT)

    for row_index in range(row_count):
        for axis_index in range(2):
            row = np.zeros(variable_count, dtype=float)
            row[(row_index * 2) + axis_index] = update_magnitude_weight
            rows.append(row)
            rhs_values.append(0.0)

    for row_index in range(1, row_count):
        for axis_index in range(2):
            row = np.zeros(variable_count, dtype=float)
            row[((row_index - 1) * 2) + axis_index] = -update_smoothness_weight
            row[(row_index * 2) + axis_index] = update_smoothness_weight
            rows.append(row)
            rhs_values.append(0.0)

    for row_index in range(1, row_count - 1):
        for axis_index in range(2):
            row = np.zeros(variable_count, dtype=float)
            row[((row_index - 1) * 2) + axis_index] = update_curvature_weight
            row[(row_index * 2) + axis_index] = -2.0 * update_curvature_weight
            row[((row_index + 1) * 2) + axis_index] = update_curvature_weight
            rows.append(row)
            rhs_values.append(0.0)

    if not rows:
        return None

    matrix = np.vstack(rows)
    rhs = np.asarray(rhs_values, dtype=float)

    lower_bounds = np.zeros(variable_count, dtype=float)
    upper_bounds = np.zeros(variable_count, dtype=float)
    for row_index, offset in enumerate(search_result.frame_a_origin_yz_profile_mm):
        current_dy_mm = float(offset[0])
        current_dz_mm = float(offset[1])
        lower_bounds[row_index * 2] = -max_abs_offset_mm - current_dy_mm
        upper_bounds[row_index * 2] = max_abs_offset_mm - current_dy_mm
        lower_bounds[(row_index * 2) + 1] = -max_abs_offset_mm - current_dz_mm
        upper_bounds[(row_index * 2) + 1] = max_abs_offset_mm - current_dz_mm

    solution = lsq_linear(
        matrix,
        rhs,
        bounds=(lower_bounds, upper_bounds),
        lsmr_tol="auto",
        verbose=0,
        max_iter=200,
    )
    if not solution.success:
        return None

    update_norm = float(np.linalg.norm(solution.x, ord=np.inf))
    if not math.isfinite(update_norm) or update_norm <= 1e-6:
        return None

    return tuple(
        (float(solution.x[row_index * 2]), float(solution.x[(row_index * 2) + 1]))
        for row_index in range(row_count)
    )


def _apply_profile_delta(
    profile: Sequence[tuple[float, float]],
    profile_delta: Sequence[tuple[float, float]],
    *,
    trust_scale: float,
    max_abs_offset_mm: float,
) -> tuple[tuple[float, float], ...]:
    return tuple(
        (
            float(
                max(
                    -max_abs_offset_mm,
                    min(
                        max_abs_offset_mm,
                        float(offset[0]) + trust_scale * float(delta[0]),
                    ),
                )
            ),
            float(
                max(
                    -max_abs_offset_mm,
                    min(
                        max_abs_offset_mm,
                        float(offset[1]) + trust_scale * float(delta[1]),
                    ),
                )
            ),
        )
        for offset, delta in zip(profile, profile_delta)
    )


def _profile_is_continuous_enough(
    profile: Sequence[tuple[float, float]],
    *,
    motion_settings,
    max_step_mm: float | None = None,
) -> bool:
    if len(profile) <= 1:
        return True

    if max_step_mm is None:
        max_step_mm = _resolve_profile_continuity_step_limit(motion_settings)

    return all(
        math.hypot(
            float(current_offset[0]) - float(previous_offset[0]),
            float(current_offset[1]) - float(previous_offset[1]),
        ) <= max_step_mm + 1e-9
        for previous_offset, current_offset in zip(profile, profile[1:])
    )


def _resolve_profile_continuity_step_limit(
    motion_settings,
) -> float:
    positive_steps = sorted(
        float(value)
        for value in motion_settings.frame_a_origin_yz_step_schedule_mm
        if float(value) > 0.0
    )
    reference_step_mm = positive_steps[min(1, len(positive_steps) - 1)] if positive_steps else 1.0
    return max(2.0, 2.0 * reference_step_mm)


def _refine_path_with_handover_corridors(
    search_result: _PathSearchResult,
    *,
    robot,
    mat_type,
    move_type: str,
    start_joints: tuple[float, ...],
    tool_pose,
    reference_pose,
    joint_count: int,
    motion_settings,
    optimizer_settings: _PathOptimizerSettings,
    lower_limits: Sequence[float],
    upper_limits: Sequence[float],
    a1_lower_deg: float,
    a1_upper_deg: float,
    a2_max_deg: float,
    joint_constraint_tolerance_deg: float,
    profile_result_cache: dict[tuple[tuple[float, float], ...], _PathSearchResult],
) -> _PathSearchResult:
    best_result = search_result
    if len(best_result.selected_path) <= 1 or not best_result.ik_layers:
        return best_result

    max_abs_offset_mm = max(
        (float(value) for value in motion_settings.frame_a_origin_yz_envelope_schedule_mm),
        default=0.0,
    )
    if max_abs_offset_mm <= 0.0:
        return best_result

    fd_step_mm = min(_GLOBAL_CONTINUOUS_REFINEMENT_FD_STEP_MM, max_abs_offset_mm)
    if fd_step_mm <= 0.0:
        return best_result
    continuity_step_limit_mm = _resolve_profile_continuity_step_limit(motion_settings)

    for iteration_index in range(_HANDOVER_CORRIDOR_MAX_ITERS):
        config_problem_segments = [
            segment
            for segment in _collect_problem_segments(
                best_result.selected_path,
                bridge_trigger_joint_delta_deg=motion_settings.bridge_trigger_joint_delta_deg,
                max_segments=3,
            )
            if segment[1]
        ]
        if not config_problem_segments:
            break

        candidate_profiles: list[
            tuple[
                tuple[tuple[float, float], ...],
                _HandoverCandidateMetadata,
            ]
        ] = []
        seen_profiles: set[tuple[tuple[float, float], ...]] = set()

        for segment_index, _config_changed, _max_joint_delta, _mean_joint_delta in config_problem_segments[:3]:
            corridor_model = _build_handover_corridor_model(
                best_result,
                segment_index=segment_index,
                robot=robot,
                mat_type=mat_type,
                tool_pose=tool_pose,
                reference_pose=reference_pose,
                joint_count=joint_count,
                optimizer_settings=optimizer_settings,
                a1_lower_deg=a1_lower_deg,
                a1_upper_deg=a1_upper_deg,
                a2_max_deg=a2_max_deg,
                joint_constraint_tolerance_deg=joint_constraint_tolerance_deg,
                lower_limits=lower_limits,
                upper_limits=upper_limits,
                motion_settings=motion_settings,
                max_abs_offset_mm=max_abs_offset_mm,
                fd_step_mm=fd_step_mm,
            )
            if corridor_model is None:
                continue

            for target_descriptor in _select_handover_target_segments(
                corridor_model,
                search_result=best_result,
                focus_segment_index=segment_index,
                optimizer_settings=optimizer_settings,
            ):
                solve_results: list[_HandoverSolveResult] = []
                primary_solve_result = _solve_handover_corridor_update(
                    best_result,
                    corridor_model=corridor_model,
                    target_segment=target_descriptor.target_segment,
                    optimizer_settings=optimizer_settings,
                    max_abs_offset_mm=max_abs_offset_mm,
                    max_profile_step_mm=continuity_step_limit_mm,
                )
                if primary_solve_result is not None:
                    solve_results.append(primary_solve_result)
                    if primary_solve_result.solver_name == "linprog":
                        l2_sidecar_result = _solve_handover_corridor_update_l2(
                            best_result,
                            corridor_model=corridor_model,
                            target_segment=target_descriptor.target_segment,
                            optimizer_settings=optimizer_settings,
                            max_abs_offset_mm=max_abs_offset_mm,
                            max_profile_step_mm=continuity_step_limit_mm,
                            reason="sidecar",
                        )
                        if l2_sidecar_result is not None:
                            solve_results.append(l2_sidecar_result)
                if not solve_results:
                    continue

                for solve_result in solve_results:
                    for trust_scale in _HANDOVER_CORRIDOR_TRUST_SCALES:
                        candidate_profile = _apply_corridor_profile_delta(
                            best_result.frame_a_origin_yz_profile_mm,
                            corridor_start=corridor_model.corridor_start,
                            corridor_delta=solve_result.corridor_delta,
                            trust_scale=trust_scale,
                            max_abs_offset_mm=max_abs_offset_mm,
                        )
                        candidate_profile = _lock_profile_endpoints_if_needed(
                            candidate_profile,
                            reference_pose_rows=best_result.reference_pose_rows,
                            motion_settings=motion_settings,
                        )
                        if candidate_profile == best_result.frame_a_origin_yz_profile_mm:
                            continue
                        if not _profile_is_continuous_enough(
                            candidate_profile,
                            motion_settings=motion_settings,
                            max_step_mm=continuity_step_limit_mm,
                        ):
                            continue
                        corridor_delta_scaled = tuple(
                            (
                                float(delta[0]) * trust_scale,
                                float(delta[1]) * trust_scale,
                            )
                            for delta in solve_result.corridor_delta
                        )
                        predicted_corridor_max_step_deg, predicted_target_cut_max_step_deg = (
                            _predict_handover_corridor_metrics(
                                corridor_model,
                                target_segment=target_descriptor.target_segment,
                                corridor_delta=corridor_delta_scaled,
                                search_result=best_result,
                            )
                        )
                        profile_key = tuple(
                            (round(float(dy_mm), 6), round(float(dz_mm), 6))
                            for dy_mm, dz_mm in candidate_profile
                        )
                        if profile_key in seen_profiles:
                            continue
                        seen_profiles.add(profile_key)
                        candidate_profiles.append(
                            (
                                candidate_profile,
                                _HandoverCandidateMetadata(
                                    segment_index,
                                    corridor_model.corridor_start,
                                    corridor_model.corridor_end,
                                    target_descriptor.target_segment,
                                    predicted_corridor_max_step_deg,
                                    predicted_target_cut_max_step_deg,
                                    solve_result.solver_name,
                                ),
                            )
                        )

        if not candidate_profiles:
            break

        cached_evaluated_candidates: list[
            tuple[
                tuple[tuple[tuple[float, float], ...], _HandoverCandidateMetadata],
                _PathSearchResult,
            ]
        ] = []
        uncached_candidate_profiles: list[
            tuple[tuple[tuple[float, float], ...], _HandoverCandidateMetadata]
        ] = []
        for candidate_profile, metadata in candidate_profiles:
            cached_result = profile_result_cache.get(_profile_cache_key(candidate_profile))
            if cached_result is None:
                uncached_candidate_profiles.append((candidate_profile, metadata))
            else:
                cached_evaluated_candidates.append(
                    ((candidate_profile, metadata), cached_result)
                )

        evaluated_uncached_candidates: list[
            tuple[
                tuple[tuple[tuple[float, float], ...], _HandoverCandidateMetadata],
                _PathSearchResult,
            ]
        ] = []
        if uncached_candidate_profiles:
            parallel_results = maybe_parallel_evaluate_exact_profiles(
                reference_pose_rows=best_result.reference_pose_rows,
                base_frame_a_origin_yz_profile_mm=best_result.frame_a_origin_yz_profile_mm,
                reused_ik_layers=best_result.ik_layers,
                frame_a_origin_yz_profiles_mm=[
                    candidate_profile
                    for candidate_profile, _metadata in uncached_candidate_profiles
                ],
                row_labels=best_result.row_labels,
                inserted_flags=best_result.inserted_flags,
                motion_settings=motion_settings,
                start_joints=start_joints,
            )
        else:
            parallel_results = None

        if parallel_results is not None:
            for (candidate_profile, metadata), parallel_result in zip(
                uncached_candidate_profiles,
                parallel_results,
            ):
                profile_result_cache[_profile_cache_key(candidate_profile)] = parallel_result
                evaluated_uncached_candidates.append(
                    ((candidate_profile, metadata), parallel_result)
                )
        else:
            for candidate_profile, metadata in uncached_candidate_profiles:
                evaluated_uncached_candidates.append(
                    (
                        (candidate_profile, metadata),
                        _evaluate_exact_profile_with_cache(
                            candidate_profile,
                            base_result=best_result,
                            lock_profile_endpoints=bool(
                                getattr(
                                    motion_settings,
                                    "lock_frame_a_origin_yz_profile_endpoints",
                                    True,
                                )
                            ),
                            profile_result_cache=profile_result_cache,
                            robot=robot,
                            mat_type=mat_type,
                            move_type=move_type,
                            start_joints=start_joints,
                            tool_pose=tool_pose,
                            reference_pose=reference_pose,
                            joint_count=joint_count,
                            optimizer_settings=optimizer_settings,
                            a1_lower_deg=a1_lower_deg,
                            a1_upper_deg=a1_upper_deg,
                            a2_max_deg=a2_max_deg,
                            joint_constraint_tolerance_deg=joint_constraint_tolerance_deg,
                            seed_joints=(
                                ()
                                if getattr(robot, "ik_seed_invariant", False)
                                else _build_local_seed_joints(
                                    best_result,
                                    metadata[1],
                                    metadata[2],
                                )
                            ),
                            lower_limits=lower_limits,
                            upper_limits=upper_limits,
                            bridge_trigger_joint_delta_deg=(
                                motion_settings.bridge_trigger_joint_delta_deg
                            ),
                        ),
                    )
                )

        evaluated_candidates = itertools.chain(
            cached_evaluated_candidates,
            evaluated_uncached_candidates,
        )

        accepted_result: _PathSearchResult | None = None
        accepted_metadata: _HandoverCandidateMetadata | None = None
        for (candidate_profile, metadata), candidate_result in evaluated_candidates:
            if not _profile_is_continuous_enough(
                candidate_profile,
                motion_settings=motion_settings,
                max_step_mm=continuity_step_limit_mm,
            ):
                continue
            if _path_search_sort_key(candidate_result) < _path_search_sort_key(best_result):
                if (
                    accepted_result is None
                    or _path_search_sort_key(candidate_result)
                    < _path_search_sort_key(accepted_result)
                ):
                    accepted_result = candidate_result
                    accepted_metadata = metadata

        if accepted_result is None or accepted_metadata is None:
            break

        segment_index = accepted_metadata.segment_index
        corridor_start = accepted_metadata.corridor_start
        corridor_end = accepted_metadata.corridor_end
        target_segment = accepted_metadata.target_segment
        print(
            "Accepted handover corridor refinement: "
            f"iteration={iteration_index + 1}, "
            f"segment={best_result.row_labels[segment_index]}->{best_result.row_labels[segment_index + 1]}, "
            f"corridor={best_result.row_labels[corridor_start]}->{best_result.row_labels[corridor_end]}, "
            f"target_cut={best_result.row_labels[target_segment]}->{best_result.row_labels[target_segment + 1]}, "
            f"solver={accepted_metadata.solver_name}, "
            f"predicted_corridor_max_step={accepted_metadata.predicted_corridor_max_step_deg:.3f} deg, "
            f"predicted_target_cut_max_step={accepted_metadata.predicted_target_cut_max_step_deg:.3f} deg, "
            f"exact_worst_joint_step={accepted_result.worst_joint_step_deg:.3f} deg, "
            f"config_switches={accepted_result.config_switches}, "
            f"bridge_like_segments={accepted_result.bridge_like_segments}."
        )
        best_result = accepted_result

    return best_result


def _build_handover_corridor_model(
    search_result: _PathSearchResult,
    *,
    segment_index: int,
    robot,
    mat_type,
    tool_pose,
    reference_pose,
    joint_count: int,
    optimizer_settings: _PathOptimizerSettings,
    a1_lower_deg: float,
    a1_upper_deg: float,
    a2_max_deg: float,
    joint_constraint_tolerance_deg: float,
    lower_limits: Sequence[float],
    upper_limits: Sequence[float],
    motion_settings,
    max_abs_offset_mm: float,
    fd_step_mm: float,
) -> _HandoverCorridorModel | None:
    if segment_index < 0 or segment_index + 1 >= len(search_result.selected_path):
        return None

    left_candidate = search_result.selected_path[segment_index]
    right_candidate = search_result.selected_path[segment_index + 1]
    if left_candidate.config_flags == right_candidate.config_flags:
        return None
    left_lineage = _candidate_lineage_key(left_candidate)
    right_lineage = _candidate_lineage_key(right_candidate)

    base_radius = max(4, motion_settings.frame_a_origin_yz_window_radius)
    row_count = len(search_result.reference_pose_rows)
    radius_schedule = (
        min(row_count - 1, max(base_radius, base_radius * _HANDOVER_CORRIDOR_RADIUS_SCALE)),
        min(row_count - 1, max(base_radius * 3, base_radius * (_HANDOVER_CORRIDOR_RADIUS_SCALE + 1))),
    )
    for radius in radius_schedule:
        corridor_start = max(0, segment_index - radius)
        corridor_end = min(row_count - 1, segment_index + 1 + radius)
        left_chain = _build_family_chain_linearization(
            search_result,
            corridor_start=corridor_start,
            corridor_end=corridor_end,
            anchor_index=segment_index,
            family_flags=left_candidate.config_flags,
            lineage_key=left_lineage,
            anchor_candidate=left_candidate,
            robot=robot,
            mat_type=mat_type,
            tool_pose=tool_pose,
            reference_pose=reference_pose,
            joint_count=joint_count,
            optimizer_settings=optimizer_settings,
            a1_lower_deg=a1_lower_deg,
            a1_upper_deg=a1_upper_deg,
            a2_max_deg=a2_max_deg,
            joint_constraint_tolerance_deg=joint_constraint_tolerance_deg,
            lower_limits=lower_limits,
            upper_limits=upper_limits,
            max_abs_offset_mm=max_abs_offset_mm,
            fd_step_mm=fd_step_mm,
        )
        right_chain = _build_family_chain_linearization(
            search_result,
            corridor_start=corridor_start,
            corridor_end=corridor_end,
            anchor_index=segment_index + 1,
            family_flags=right_candidate.config_flags,
            lineage_key=right_lineage,
            anchor_candidate=right_candidate,
            robot=robot,
            mat_type=mat_type,
            tool_pose=tool_pose,
            reference_pose=reference_pose,
            joint_count=joint_count,
            optimizer_settings=optimizer_settings,
            a1_lower_deg=a1_lower_deg,
            a1_upper_deg=a1_upper_deg,
            a2_max_deg=a2_max_deg,
            joint_constraint_tolerance_deg=joint_constraint_tolerance_deg,
            lower_limits=lower_limits,
            upper_limits=upper_limits,
            max_abs_offset_mm=max_abs_offset_mm,
            fd_step_mm=fd_step_mm,
        )
        if _handover_corridor_has_overlap(left_chain, right_chain):
            return _HandoverCorridorModel(
                corridor_start=corridor_start,
                corridor_end=corridor_end,
                left_family=left_candidate.config_flags,
                right_family=right_candidate.config_flags,
                left_lineage=left_lineage,
                right_lineage=right_lineage,
                left_chain=left_chain,
                right_chain=right_chain,
            )

    return None


def _build_family_chain_linearization(
    search_result: _PathSearchResult,
    *,
    corridor_start: int,
    corridor_end: int,
    anchor_index: int,
    family_flags: tuple[int, ...],
    lineage_key: tuple[int, ...] | None,
    anchor_candidate: _IKCandidate,
    robot,
    mat_type,
    tool_pose,
    reference_pose,
    joint_count: int,
    optimizer_settings: _PathOptimizerSettings,
    a1_lower_deg: float,
    a1_upper_deg: float,
    a2_max_deg: float,
    joint_constraint_tolerance_deg: float,
    lower_limits: Sequence[float],
    upper_limits: Sequence[float],
    max_abs_offset_mm: float,
    fd_step_mm: float,
) -> tuple[_FamilyLinearizedCandidate | None, ...]:
    corridor_length = corridor_end - corridor_start + 1
    chain: list[_FamilyLinearizedCandidate | None] = [None] * corridor_length
    anchor_local_index = anchor_index - corridor_start
    chain[anchor_local_index] = _build_family_linearized_candidate(
        reference_row=search_result.reference_pose_rows[anchor_index],
        offset=search_result.frame_a_origin_yz_profile_mm[anchor_index],
        base_candidate=anchor_candidate,
        family_flags=family_flags,
        lineage_key=lineage_key,
        robot=robot,
        mat_type=mat_type,
        tool_pose=tool_pose,
        reference_pose=reference_pose,
        joint_count=joint_count,
        optimizer_settings=optimizer_settings,
        a1_lower_deg=a1_lower_deg,
        a1_upper_deg=a1_upper_deg,
        a2_max_deg=a2_max_deg,
        joint_constraint_tolerance_deg=joint_constraint_tolerance_deg,
        lower_limits=lower_limits,
        upper_limits=upper_limits,
        max_abs_offset_mm=max_abs_offset_mm,
        fd_step_mm=fd_step_mm,
    )

    previous_candidate = anchor_candidate
    for row_index in range(anchor_index - 1, corridor_start - 1, -1):
        linearized_candidate = _collect_family_linearized_candidate(
            reference_row=search_result.reference_pose_rows[row_index],
            offset=search_result.frame_a_origin_yz_profile_mm[row_index],
            reference_candidate=previous_candidate,
            family_flags=family_flags,
            lineage_key=lineage_key,
            robot=robot,
            mat_type=mat_type,
            tool_pose=tool_pose,
            reference_pose=reference_pose,
            joint_count=joint_count,
            optimizer_settings=optimizer_settings,
            a1_lower_deg=a1_lower_deg,
            a1_upper_deg=a1_upper_deg,
            a2_max_deg=a2_max_deg,
            joint_constraint_tolerance_deg=joint_constraint_tolerance_deg,
            lower_limits=lower_limits,
            upper_limits=upper_limits,
            max_abs_offset_mm=max_abs_offset_mm,
            fd_step_mm=fd_step_mm,
        )
        if linearized_candidate is None:
            break
        chain[row_index - corridor_start] = linearized_candidate
        previous_candidate = linearized_candidate.candidate

    previous_candidate = anchor_candidate
    for row_index in range(anchor_index + 1, corridor_end + 1):
        linearized_candidate = _collect_family_linearized_candidate(
            reference_row=search_result.reference_pose_rows[row_index],
            offset=search_result.frame_a_origin_yz_profile_mm[row_index],
            reference_candidate=previous_candidate,
            family_flags=family_flags,
            lineage_key=lineage_key,
            robot=robot,
            mat_type=mat_type,
            tool_pose=tool_pose,
            reference_pose=reference_pose,
            joint_count=joint_count,
            optimizer_settings=optimizer_settings,
            a1_lower_deg=a1_lower_deg,
            a1_upper_deg=a1_upper_deg,
            a2_max_deg=a2_max_deg,
            joint_constraint_tolerance_deg=joint_constraint_tolerance_deg,
            lower_limits=lower_limits,
            upper_limits=upper_limits,
            max_abs_offset_mm=max_abs_offset_mm,
            fd_step_mm=fd_step_mm,
        )
        if linearized_candidate is None:
            break
        chain[row_index - corridor_start] = linearized_candidate
        previous_candidate = linearized_candidate.candidate

    return tuple(chain)


def _build_family_linearized_candidate(
    *,
    reference_row: dict[str, float],
    offset: tuple[float, float],
    base_candidate: _IKCandidate,
    family_flags: tuple[int, ...],
    lineage_key: tuple[int, ...] | None,
    robot,
    mat_type,
    tool_pose,
    reference_pose,
    joint_count: int,
    optimizer_settings: _PathOptimizerSettings,
    a1_lower_deg: float,
    a1_upper_deg: float,
    a2_max_deg: float,
    joint_constraint_tolerance_deg: float,
    lower_limits: Sequence[float],
    upper_limits: Sequence[float],
    max_abs_offset_mm: float,
    fd_step_mm: float,
) -> _FamilyLinearizedCandidate:
    gradient_y = _estimate_family_joint_gradient(
        reference_row=reference_row,
        base_offset=offset,
        axis_index=0,
        base_candidate=base_candidate,
        family_flags=family_flags,
        lineage_key=lineage_key,
        robot=robot,
        mat_type=mat_type,
        tool_pose=tool_pose,
        reference_pose=reference_pose,
        joint_count=joint_count,
        optimizer_settings=optimizer_settings,
        a1_lower_deg=a1_lower_deg,
        a1_upper_deg=a1_upper_deg,
        a2_max_deg=a2_max_deg,
        joint_constraint_tolerance_deg=joint_constraint_tolerance_deg,
        lower_limits=lower_limits,
        upper_limits=upper_limits,
        max_abs_offset_mm=max_abs_offset_mm,
        fd_step_mm=fd_step_mm,
    )
    gradient_z = _estimate_family_joint_gradient(
        reference_row=reference_row,
        base_offset=offset,
        axis_index=1,
        base_candidate=base_candidate,
        family_flags=family_flags,
        lineage_key=lineage_key,
        robot=robot,
        mat_type=mat_type,
        tool_pose=tool_pose,
        reference_pose=reference_pose,
        joint_count=joint_count,
        optimizer_settings=optimizer_settings,
        a1_lower_deg=a1_lower_deg,
        a1_upper_deg=a1_upper_deg,
        a2_max_deg=a2_max_deg,
        joint_constraint_tolerance_deg=joint_constraint_tolerance_deg,
        lower_limits=lower_limits,
        upper_limits=upper_limits,
        max_abs_offset_mm=max_abs_offset_mm,
        fd_step_mm=fd_step_mm,
    )
    return _FamilyLinearizedCandidate(
        candidate=base_candidate,
        gradient_y=gradient_y,
        gradient_z=gradient_z,
    )


def _collect_family_linearized_candidate(
    *,
    reference_row: dict[str, float],
    offset: tuple[float, float],
    reference_candidate: _IKCandidate,
    family_flags: tuple[int, ...],
    lineage_key: tuple[int, ...] | None,
    robot,
    mat_type,
    tool_pose,
    reference_pose,
    joint_count: int,
    optimizer_settings: _PathOptimizerSettings,
    a1_lower_deg: float,
    a1_upper_deg: float,
    a2_max_deg: float,
    joint_constraint_tolerance_deg: float,
    lower_limits: Sequence[float],
    upper_limits: Sequence[float],
    max_abs_offset_mm: float,
    fd_step_mm: float,
) -> _FamilyLinearizedCandidate | None:
    candidate = _collect_profile_family_candidate(
        reference_row=reference_row,
        offset=offset,
        reference_candidate=reference_candidate,
        family_flags=family_flags,
        lineage_key=lineage_key,
        robot=robot,
        mat_type=mat_type,
        tool_pose=tool_pose,
        reference_pose=reference_pose,
        joint_count=joint_count,
        optimizer_settings=optimizer_settings,
        a1_lower_deg=a1_lower_deg,
        a1_upper_deg=a1_upper_deg,
        a2_max_deg=a2_max_deg,
        joint_constraint_tolerance_deg=joint_constraint_tolerance_deg,
        lower_limits=lower_limits,
        upper_limits=upper_limits,
    )
    if candidate is None:
        return None
    return _build_family_linearized_candidate(
        reference_row=reference_row,
        offset=offset,
        base_candidate=candidate,
        family_flags=family_flags,
        lineage_key=lineage_key,
        robot=robot,
        mat_type=mat_type,
        tool_pose=tool_pose,
        reference_pose=reference_pose,
        joint_count=joint_count,
        optimizer_settings=optimizer_settings,
        a1_lower_deg=a1_lower_deg,
        a1_upper_deg=a1_upper_deg,
        a2_max_deg=a2_max_deg,
        joint_constraint_tolerance_deg=joint_constraint_tolerance_deg,
        lower_limits=lower_limits,
        upper_limits=upper_limits,
        max_abs_offset_mm=max_abs_offset_mm,
        fd_step_mm=fd_step_mm,
    )


def _estimate_family_joint_gradient(
    *,
    reference_row: dict[str, float],
    base_offset: tuple[float, float],
    axis_index: int,
    base_candidate: _IKCandidate,
    family_flags: tuple[int, ...],
    lineage_key: tuple[int, ...] | None,
    robot,
    mat_type,
    tool_pose,
    reference_pose,
    joint_count: int,
    optimizer_settings: _PathOptimizerSettings,
    a1_lower_deg: float,
    a1_upper_deg: float,
    a2_max_deg: float,
    joint_constraint_tolerance_deg: float,
    lower_limits: Sequence[float],
    upper_limits: Sequence[float],
    max_abs_offset_mm: float,
    fd_step_mm: float,
) -> tuple[float, ...]:
    positive_offset = list(base_offset)
    negative_offset = list(base_offset)
    positive_offset[axis_index] += fd_step_mm
    negative_offset[axis_index] -= fd_step_mm

    positive_candidate = None
    negative_candidate = None
    if abs(positive_offset[axis_index]) <= max_abs_offset_mm + 1e-9:
        positive_candidate = _collect_profile_family_candidate(
            reference_row=reference_row,
            offset=(float(positive_offset[0]), float(positive_offset[1])),
            reference_candidate=base_candidate,
            family_flags=family_flags,
            lineage_key=lineage_key,
            robot=robot,
            mat_type=mat_type,
            tool_pose=tool_pose,
            reference_pose=reference_pose,
            joint_count=joint_count,
            optimizer_settings=optimizer_settings,
            a1_lower_deg=a1_lower_deg,
            a1_upper_deg=a1_upper_deg,
            a2_max_deg=a2_max_deg,
            joint_constraint_tolerance_deg=joint_constraint_tolerance_deg,
            lower_limits=lower_limits,
            upper_limits=upper_limits,
        )
    if abs(negative_offset[axis_index]) <= max_abs_offset_mm + 1e-9:
        negative_candidate = _collect_profile_family_candidate(
            reference_row=reference_row,
            offset=(float(negative_offset[0]), float(negative_offset[1])),
            reference_candidate=base_candidate,
            family_flags=family_flags,
            lineage_key=lineage_key,
            robot=robot,
            mat_type=mat_type,
            tool_pose=tool_pose,
            reference_pose=reference_pose,
            joint_count=joint_count,
            optimizer_settings=optimizer_settings,
            a1_lower_deg=a1_lower_deg,
            a1_upper_deg=a1_upper_deg,
            a2_max_deg=a2_max_deg,
            joint_constraint_tolerance_deg=joint_constraint_tolerance_deg,
            lower_limits=lower_limits,
            upper_limits=upper_limits,
        )

    if positive_candidate is not None and negative_candidate is not None:
        return tuple(
            (float(positive_value) - float(negative_value)) / (2.0 * fd_step_mm)
            for positive_value, negative_value in zip(
                positive_candidate.joints,
                negative_candidate.joints,
            )
        )
    if positive_candidate is not None:
        return tuple(
            (float(positive_value) - float(base_value)) / fd_step_mm
            for positive_value, base_value in zip(
                positive_candidate.joints,
                base_candidate.joints,
            )
        )
    if negative_candidate is not None:
        return tuple(
            (float(base_value) - float(negative_value)) / fd_step_mm
            for base_value, negative_value in zip(
                base_candidate.joints,
                negative_candidate.joints,
            )
        )
    return tuple(0.0 for _ in range(len(base_candidate.joints)))


def _handover_corridor_has_overlap(
    left_chain: Sequence[_FamilyLinearizedCandidate | None],
    right_chain: Sequence[_FamilyLinearizedCandidate | None],
) -> bool:
    return any(
        left_chain[local_index] is not None and right_chain[local_index + 1] is not None
        for local_index in range(len(left_chain) - 1)
    )


def _select_handover_target_segments(
    corridor_model: _HandoverCorridorModel,
    *,
    search_result: _PathSearchResult,
    focus_segment_index: int,
    optimizer_settings: _PathOptimizerSettings,
) -> tuple[_HandoverTargetDescriptor, ...]:
    scored_segments: list[tuple[float, float, float, int, _HandoverTargetDescriptor]] = []
    for local_index in range(len(corridor_model.left_chain) - 1):
        left_state = corridor_model.left_chain[local_index]
        right_state = corridor_model.right_chain[local_index + 1]
        if left_state is None or right_state is None:
            continue
        segment_index = corridor_model.corridor_start + local_index
        predicted_corridor_max_step_deg, predicted_target_cut_max_step_deg = (
            _predict_handover_corridor_metrics(
                corridor_model,
                target_segment=segment_index,
                search_result=search_result,
            )
        )
        descriptor = _HandoverTargetDescriptor(
            target_segment=segment_index,
            predicted_corridor_max_step_deg=predicted_corridor_max_step_deg,
            predicted_target_cut_max_step_deg=predicted_target_cut_max_step_deg,
        )
        scored_segments.append(
            (
                predicted_corridor_max_step_deg,
                predicted_target_cut_max_step_deg,
                _joint_transition_penalty(
                    left_state.candidate.joints,
                    right_state.candidate.joints,
                    optimizer_settings,
                ),
                abs(segment_index - focus_segment_index),
                descriptor,
            )
        )

    if not scored_segments:
        return ()

    scored_segments.sort()
    target_segments: list[_HandoverTargetDescriptor] = []
    seen_segments: set[int] = set()
    for _predicted_corridor, _predicted_target, _score, _distance, descriptor in scored_segments:
        if descriptor.target_segment in seen_segments:
            continue
        seen_segments.add(descriptor.target_segment)
        target_segments.append(descriptor)
        if len(target_segments) >= _HANDOVER_CORRIDOR_MAX_TARGET_SEGMENTS:
            break
    return tuple(target_segments)


def _solve_handover_corridor_update(
    search_result: _PathSearchResult,
    *,
    corridor_model: _HandoverCorridorModel,
    target_segment: int,
    optimizer_settings: _PathOptimizerSettings,
    max_abs_offset_mm: float,
    max_profile_step_mm: float,
) -> _HandoverSolveResult | None:
    corridor_row_count = corridor_model.corridor_end - corridor_model.corridor_start + 1
    if corridor_row_count <= 1:
        return None

    target_local_segment = target_segment - corridor_model.corridor_start
    if target_local_segment < 0 or target_local_segment >= corridor_row_count - 1:
        return None

    primary_terms, target_term, overlap_terms, boundary_terms = _build_handover_problem_terms(
        search_result,
        corridor_model=corridor_model,
        target_segment=target_segment,
    )
    if target_term is None or not primary_terms:
        return None

    lower_bounds, upper_bounds = _build_corridor_delta_bounds(
        search_result,
        corridor_model=corridor_model,
        max_abs_offset_mm=max_abs_offset_mm,
    )

    stage1_result = _solve_handover_corridor_minmax_lp(
        corridor_row_count=corridor_row_count,
        primary_terms=primary_terms,
        base_profile=search_result.frame_a_origin_yz_profile_mm[
            corridor_model.corridor_start : corridor_model.corridor_end + 1
        ],
        max_profile_step_mm=max_profile_step_mm,
        lower_bounds=lower_bounds,
        upper_bounds=upper_bounds,
    )
    if stage1_result is None:
        return _solve_handover_corridor_update_l2(
            search_result,
            corridor_model=corridor_model,
            target_segment=target_segment,
            optimizer_settings=optimizer_settings,
            max_abs_offset_mm=max_abs_offset_mm,
            max_profile_step_mm=max_profile_step_mm,
            reason="stage1_failed",
        )

    _stage1_delta, best_t_max_deg = stage1_result
    stage2_delta = _solve_handover_corridor_secondary_lp(
        corridor_row_count=corridor_row_count,
        primary_terms=primary_terms,
        target_term=target_term,
        overlap_terms=overlap_terms,
        boundary_terms=boundary_terms,
        base_profile=search_result.frame_a_origin_yz_profile_mm[
            corridor_model.corridor_start : corridor_model.corridor_end + 1
        ],
        max_profile_step_mm=max_profile_step_mm,
        lower_bounds=lower_bounds,
        upper_bounds=upper_bounds,
        t_upper=best_t_max_deg + _HANDOVER_CORRIDOR_MINMAX_TOLERANCE_DEG,
    )
    if stage2_delta is None:
        return _solve_handover_corridor_update_l2(
            search_result,
            corridor_model=corridor_model,
            target_segment=target_segment,
            optimizer_settings=optimizer_settings,
            max_abs_offset_mm=max_abs_offset_mm,
            max_profile_step_mm=max_profile_step_mm,
            reason="stage2_failed",
        )

    update_norm = max(
        (
            max(abs(float(delta[0])), abs(float(delta[1])))
            for delta in stage2_delta
        ),
        default=0.0,
    )
    if not math.isfinite(update_norm) or update_norm <= 1e-6:
        return _solve_handover_corridor_update_l2(
            search_result,
            corridor_model=corridor_model,
            target_segment=target_segment,
            optimizer_settings=optimizer_settings,
            max_abs_offset_mm=max_abs_offset_mm,
            max_profile_step_mm=max_profile_step_mm,
            reason="stage2_empty",
        )

    predicted_corridor_max_step_deg, predicted_target_cut_max_step_deg = (
        _predict_handover_corridor_metrics(
            corridor_model,
            target_segment=target_segment,
            search_result=search_result,
            corridor_delta=stage2_delta,
        )
    )
    return _HandoverSolveResult(
        corridor_delta=stage2_delta,
        predicted_corridor_max_step_deg=predicted_corridor_max_step_deg,
        predicted_target_cut_max_step_deg=predicted_target_cut_max_step_deg,
        solver_name="linprog",
    )


def _build_handover_problem_terms(
    search_result: _PathSearchResult,
    *,
    corridor_model: _HandoverCorridorModel,
    target_segment: int,
) -> tuple[
    tuple[_LinearizedDifferenceTerm, ...],
    _LinearizedDifferenceTerm | None,
    tuple[_LinearizedDifferenceTerm, ...],
    tuple[_LinearizedDifferenceTerm, ...],
]:
    corridor_row_count = corridor_model.corridor_end - corridor_model.corridor_start + 1
    target_local_segment = target_segment - corridor_model.corridor_start
    if target_local_segment < 0 or target_local_segment >= corridor_row_count - 1:
        return (), None, (), ()

    primary_terms: list[_LinearizedDifferenceTerm] = []
    overlap_terms: list[_LinearizedDifferenceTerm] = []
    boundary_terms: list[_LinearizedDifferenceTerm] = []

    for local_segment_index in range(target_local_segment):
        previous_state = corridor_model.left_chain[local_segment_index]
        current_state = corridor_model.left_chain[local_segment_index + 1]
        if previous_state is None or current_state is None:
            continue
        primary_terms.append(
            _LinearizedDifferenceTerm(
                left_state=previous_state,
                left_row_index=local_segment_index,
                right_state=current_state,
                right_row_index=local_segment_index + 1,
            )
        )

    for local_segment_index in range(target_local_segment + 1, corridor_row_count - 1):
        previous_state = corridor_model.right_chain[local_segment_index]
        current_state = corridor_model.right_chain[local_segment_index + 1]
        if previous_state is None or current_state is None:
            continue
        primary_terms.append(
            _LinearizedDifferenceTerm(
                left_state=previous_state,
                left_row_index=local_segment_index,
                right_state=current_state,
                right_row_index=local_segment_index + 1,
            )
        )

    left_handover_state = corridor_model.left_chain[target_local_segment]
    right_handover_state = corridor_model.right_chain[target_local_segment + 1]
    target_term: _LinearizedDifferenceTerm | None = None
    if left_handover_state is not None and right_handover_state is not None:
        target_term = _LinearizedDifferenceTerm(
            left_state=left_handover_state,
            left_row_index=target_local_segment,
            right_state=right_handover_state,
            right_row_index=target_local_segment + 1,
        )
        primary_terms.append(target_term)

    for local_row_index in range(
        max(0, target_local_segment - 1),
        min(corridor_row_count, target_local_segment + 3),
    ):
        left_state = corridor_model.left_chain[local_row_index]
        right_state = corridor_model.right_chain[local_row_index]
        if left_state is None or right_state is None:
            continue
        overlap_terms.append(
            _LinearizedDifferenceTerm(
                left_state=left_state,
                left_row_index=local_row_index,
                right_state=right_state,
                right_row_index=local_row_index,
            )
        )

    if corridor_model.corridor_start > 0 and corridor_model.left_chain[0] is not None:
        outside_candidate = search_result.selected_path[corridor_model.corridor_start - 1]
        if _lineage_matches_family(
            outside_candidate,
            family_flags=corridor_model.left_family,
            lineage_key=corridor_model.left_lineage,
        ):
            boundary_terms.append(
                _LinearizedDifferenceTerm(
                    left_state=_make_fixed_family_linearized_candidate(outside_candidate),
                    left_row_index=0,
                    right_state=corridor_model.left_chain[0],
                    right_row_index=0,
                    left_is_fixed=True,
                )
            )

    if (
        corridor_model.corridor_end + 1 < len(search_result.selected_path)
        and corridor_model.right_chain[-1] is not None
    ):
        outside_candidate = search_result.selected_path[corridor_model.corridor_end + 1]
        if _lineage_matches_family(
            outside_candidate,
            family_flags=corridor_model.right_family,
            lineage_key=corridor_model.right_lineage,
        ):
            boundary_terms.append(
                _LinearizedDifferenceTerm(
                    left_state=corridor_model.right_chain[-1],
                    left_row_index=corridor_row_count - 1,
                    right_state=_make_fixed_family_linearized_candidate(outside_candidate),
                    right_row_index=corridor_row_count - 1,
                    right_is_fixed=True,
                )
            )

    primary_terms.extend(boundary_terms)
    return tuple(primary_terms), target_term, tuple(overlap_terms), tuple(boundary_terms)


def _make_fixed_family_linearized_candidate(
    candidate: _IKCandidate,
) -> _FamilyLinearizedCandidate:
    return _FamilyLinearizedCandidate(
        candidate=candidate,
        gradient_y=tuple(0.0 for _ in candidate.joints),
        gradient_z=tuple(0.0 for _ in candidate.joints),
    )


def _build_corridor_delta_bounds(
    search_result: _PathSearchResult,
    *,
    corridor_model: _HandoverCorridorModel,
    max_abs_offset_mm: float,
) -> tuple[np.ndarray, np.ndarray]:
    corridor_row_count = corridor_model.corridor_end - corridor_model.corridor_start + 1
    lower_bounds = np.zeros(corridor_row_count * 2, dtype=float)
    upper_bounds = np.zeros(corridor_row_count * 2, dtype=float)
    for local_row_index, row_index in enumerate(
        range(corridor_model.corridor_start, corridor_model.corridor_end + 1)
    ):
        current_offset = search_result.frame_a_origin_yz_profile_mm[row_index]
        current_dy_mm = float(current_offset[0])
        current_dz_mm = float(current_offset[1])
        lower_bounds[local_row_index * 2] = -max_abs_offset_mm - current_dy_mm
        upper_bounds[local_row_index * 2] = max_abs_offset_mm - current_dy_mm
        lower_bounds[(local_row_index * 2) + 1] = -max_abs_offset_mm - current_dz_mm
        upper_bounds[(local_row_index * 2) + 1] = max_abs_offset_mm - current_dz_mm
    return lower_bounds, upper_bounds


def _predict_handover_corridor_metrics(
    corridor_model: _HandoverCorridorModel,
    *,
    target_segment: int,
    search_result: _PathSearchResult,
    corridor_delta: Sequence[tuple[float, float]] | None = None,
) -> tuple[float, float]:
    primary_terms, target_term, _overlap_terms, _boundary_terms = _build_handover_problem_terms(
        search_result,
        corridor_model=corridor_model,
        target_segment=target_segment,
    )
    if target_term is None or not primary_terms:
        return math.inf, math.inf

    predicted_corridor_max_step_deg = max(
        (
            max(abs(value) for value in _predict_term_joint_deltas(term, corridor_delta=corridor_delta))
            for term in primary_terms
        ),
        default=math.inf,
    )
    predicted_target_cut_max_step_deg = max(
        abs(value)
        for value in _predict_term_joint_deltas(target_term, corridor_delta=corridor_delta)
    )
    return float(predicted_corridor_max_step_deg), float(predicted_target_cut_max_step_deg)


def _predict_term_joint_deltas(
    term: _LinearizedDifferenceTerm,
    *,
    corridor_delta: Sequence[tuple[float, float]] | None = None,
) -> tuple[float, ...]:
    joint_deltas: list[float] = []
    for joint_index, (left_joint, right_joint) in enumerate(
        zip(term.left_state.candidate.joints, term.right_state.candidate.joints)
    ):
        delta_value = float(right_joint) - float(left_joint)
        if not term.left_is_fixed and corridor_delta is not None:
            delta_value -= (
                float(term.left_state.gradient_y[joint_index]) * float(corridor_delta[term.left_row_index][0])
                + float(term.left_state.gradient_z[joint_index]) * float(corridor_delta[term.left_row_index][1])
            )
        if not term.right_is_fixed and corridor_delta is not None:
            delta_value += (
                float(term.right_state.gradient_y[joint_index]) * float(corridor_delta[term.right_row_index][0])
                + float(term.right_state.gradient_z[joint_index]) * float(corridor_delta[term.right_row_index][1])
            )
        joint_deltas.append(float(delta_value))
    return tuple(joint_deltas)


def _build_term_expression(
    term: _LinearizedDifferenceTerm,
    *,
    joint_index: int,
    corridor_row_count: int,
) -> tuple[np.ndarray, float]:
    coefficient = np.zeros(corridor_row_count * 2, dtype=float)
    constant = float(term.right_state.candidate.joints[joint_index]) - float(
        term.left_state.candidate.joints[joint_index]
    )
    if not term.left_is_fixed:
        coefficient[term.left_row_index * 2] -= float(term.left_state.gradient_y[joint_index])
        coefficient[(term.left_row_index * 2) + 1] -= float(term.left_state.gradient_z[joint_index])
    if not term.right_is_fixed:
        coefficient[term.right_row_index * 2] += float(term.right_state.gradient_y[joint_index])
        coefficient[(term.right_row_index * 2) + 1] += float(term.right_state.gradient_z[joint_index])
    return coefficient, constant


def _solve_handover_corridor_minmax_lp(
    *,
    corridor_row_count: int,
    primary_terms: Sequence[_LinearizedDifferenceTerm],
    base_profile: Sequence[tuple[float, float]],
    max_profile_step_mm: float,
    lower_bounds: np.ndarray,
    upper_bounds: np.ndarray,
) -> tuple[tuple[tuple[float, float], ...], float] | None:
    x_count = corridor_row_count * 2
    t_index = x_count
    variable_count = x_count + 1
    objective = np.zeros(variable_count, dtype=float)
    objective[t_index] = 1.0

    a_rows: list[np.ndarray] = []
    b_values: list[float] = []
    for term in primary_terms:
        for joint_index in range(len(term.left_state.candidate.joints)):
            coefficient, constant = _build_term_expression(
                term,
                joint_index=joint_index,
                corridor_row_count=corridor_row_count,
            )
            _append_abs_epigraph_constraints(
                a_rows,
                b_values,
                coefficient=coefficient,
                constant=constant,
                t_index=t_index,
                variable_count=variable_count,
            )

    if not a_rows:
        return None

    _append_profile_continuity_constraints(
        a_rows,
        b_values,
        corridor_row_count=corridor_row_count,
        base_profile=base_profile,
        max_profile_step_mm=max_profile_step_mm,
        variable_count=variable_count,
    )

    bounds = [
        (float(lower_bounds[index]), float(upper_bounds[index]))
        for index in range(x_count)
    ]
    bounds.append((0.0, None))

    solution = linprog(
        c=objective,
        A_ub=np.vstack(a_rows),
        b_ub=np.asarray(b_values, dtype=float),
        bounds=bounds,
        method="highs",
    )
    if not solution.success or solution.x is None:
        return None

    corridor_delta = tuple(
        (float(solution.x[local_row_index * 2]), float(solution.x[(local_row_index * 2) + 1]))
        for local_row_index in range(corridor_row_count)
    )
    return corridor_delta, float(solution.x[t_index])


def _solve_handover_corridor_secondary_lp(
    *,
    corridor_row_count: int,
    primary_terms: Sequence[_LinearizedDifferenceTerm],
    target_term: _LinearizedDifferenceTerm,
    overlap_terms: Sequence[_LinearizedDifferenceTerm],
    boundary_terms: Sequence[_LinearizedDifferenceTerm],
    base_profile: Sequence[tuple[float, float]],
    max_profile_step_mm: float,
    lower_bounds: np.ndarray,
    upper_bounds: np.ndarray,
    t_upper: float,
) -> tuple[tuple[float, float], ...] | None:
    x_count = corridor_row_count * 2
    t_index = x_count

    slack_groups = _build_handover_stage2_slack_groups(
        corridor_row_count=corridor_row_count,
        target_term=target_term,
        overlap_terms=overlap_terms,
        boundary_terms=boundary_terms,
    )
    group_upper_bounds = [
        sum(
            _expression_abs_upper_bound(
                coefficient,
                constant,
                lower_bounds=lower_bounds,
                upper_bounds=upper_bounds,
            )
            for coefficient, constant in group
        )
        for group in slack_groups
    ]
    group_weights = _build_lexicographic_group_weights(group_upper_bounds)

    variable_count = x_count + 1 + sum(len(group) for group in slack_groups)
    objective = np.zeros(variable_count, dtype=float)
    bounds: list[tuple[float | None, float | None]] = [
        (float(lower_bounds[index]), float(upper_bounds[index]))
        for index in range(x_count)
    ]
    bounds.append((0.0, float(t_upper)))

    a_rows: list[np.ndarray] = []
    b_values: list[float] = []
    for term in primary_terms:
        for joint_index in range(len(term.left_state.candidate.joints)):
            coefficient, constant = _build_term_expression(
                term,
                joint_index=joint_index,
                corridor_row_count=corridor_row_count,
            )
            _append_abs_epigraph_constraints(
                a_rows,
                b_values,
                coefficient=coefficient,
                constant=constant,
                t_index=t_index,
                variable_count=variable_count,
            )

    _append_profile_continuity_constraints(
        a_rows,
        b_values,
        corridor_row_count=corridor_row_count,
        base_profile=base_profile,
        max_profile_step_mm=max_profile_step_mm,
        variable_count=variable_count,
    )

    next_variable_index = x_count + 1
    for group_weight, group in zip(group_weights, slack_groups):
        for coefficient, constant in group:
            slack_index = next_variable_index
            next_variable_index += 1
            bounds.append((0.0, None))
            objective[slack_index] = float(group_weight)
            _append_abs_slack_constraints(
                a_rows,
                b_values,
                coefficient=coefficient,
                constant=constant,
                slack_index=slack_index,
                variable_count=variable_count,
            )

    solution = linprog(
        c=objective,
        A_ub=np.vstack(a_rows),
        b_ub=np.asarray(b_values, dtype=float),
        bounds=bounds,
        method="highs",
    )
    if not solution.success or solution.x is None:
        return None

    return tuple(
        (float(solution.x[local_row_index * 2]), float(solution.x[(local_row_index * 2) + 1]))
        for local_row_index in range(corridor_row_count)
    )


def _solve_handover_corridor_update_l2(
    search_result: _PathSearchResult,
    *,
    corridor_model: _HandoverCorridorModel,
    target_segment: int,
    optimizer_settings: _PathOptimizerSettings,
    max_abs_offset_mm: float,
    max_profile_step_mm: float,
    reason: str = "fallback",
) -> _HandoverSolveResult | None:
    corridor_row_count = corridor_model.corridor_end - corridor_model.corridor_start + 1
    variable_count = corridor_row_count * 2
    if corridor_row_count <= 1:
        return None

    target_local_segment = target_segment - corridor_model.corridor_start
    if target_local_segment < 0 or target_local_segment >= corridor_row_count - 1:
        return None

    preferred_limits = tuple(float(value) for value in optimizer_settings.preferred_joint_step_deg)
    rows: list[np.ndarray] = []
    rhs_values: list[float] = []

    for local_segment_index in range(target_local_segment):
        previous_state = corridor_model.left_chain[local_segment_index]
        current_state = corridor_model.left_chain[local_segment_index + 1]
        if previous_state is None or current_state is None:
            continue
        _append_linearized_difference_rows(
            rows,
            rhs_values,
            variable_count=variable_count,
            left_state=previous_state,
            left_row_index=local_segment_index,
            right_state=current_state,
            right_row_index=local_segment_index + 1,
            preferred_limits=preferred_limits,
            joint_weights=optimizer_settings.joint_delta_weights,
            base_scale=math.sqrt(_HANDOVER_CORRIDOR_CONTINUITY_WEIGHT),
        )

    for local_segment_index in range(target_local_segment + 1, corridor_row_count - 1):
        previous_state = corridor_model.right_chain[local_segment_index]
        current_state = corridor_model.right_chain[local_segment_index + 1]
        if previous_state is None or current_state is None:
            continue
        _append_linearized_difference_rows(
            rows,
            rhs_values,
            variable_count=variable_count,
            left_state=previous_state,
            left_row_index=local_segment_index,
            right_state=current_state,
            right_row_index=local_segment_index + 1,
            preferred_limits=preferred_limits,
            joint_weights=optimizer_settings.joint_delta_weights,
            base_scale=math.sqrt(_HANDOVER_CORRIDOR_CONTINUITY_WEIGHT),
        )

    left_handover_state = corridor_model.left_chain[target_local_segment]
    right_handover_state = corridor_model.right_chain[target_local_segment + 1]
    if left_handover_state is None or right_handover_state is None:
        return None
    handover_joint_emphasis = _build_handover_joint_emphasis(
        left_handover_state,
        right_handover_state,
    )
    _append_linearized_difference_rows(
        rows,
        rhs_values,
        variable_count=variable_count,
        left_state=left_handover_state,
        left_row_index=target_local_segment,
        right_state=right_handover_state,
        right_row_index=target_local_segment + 1,
        preferred_limits=preferred_limits,
        joint_weights=optimizer_settings.joint_delta_weights,
        base_scale=math.sqrt(_HANDOVER_CORRIDOR_HANDOVER_WEIGHT),
        joint_emphasis=handover_joint_emphasis,
    )

    for local_row_index in range(
        max(0, target_local_segment - 1),
        min(corridor_row_count, target_local_segment + 3),
    ):
        left_state = corridor_model.left_chain[local_row_index]
        right_state = corridor_model.right_chain[local_row_index]
        if left_state is None or right_state is None:
            continue
        _append_linearized_difference_rows(
            rows,
            rhs_values,
            variable_count=variable_count,
            left_state=left_state,
            left_row_index=local_row_index,
            right_state=right_state,
            right_row_index=local_row_index,
            preferred_limits=preferred_limits,
            joint_weights=optimizer_settings.joint_delta_weights,
            base_scale=math.sqrt(_HANDOVER_CORRIDOR_OVERLAP_WEIGHT),
            joint_emphasis=handover_joint_emphasis,
        )

    if corridor_model.corridor_start > 0 and corridor_model.left_chain[0] is not None:
        boundary_state = corridor_model.left_chain[0]
        outside_candidate = search_result.selected_path[corridor_model.corridor_start - 1]
        if _lineage_matches_family(
            outside_candidate,
            family_flags=corridor_model.left_family,
            lineage_key=corridor_model.left_lineage,
        ):
            _append_linearized_difference_rows(
                rows,
                rhs_values,
                variable_count=variable_count,
                left_state=_FamilyLinearizedCandidate(
                    candidate=outside_candidate,
                    gradient_y=tuple(0.0 for _ in outside_candidate.joints),
                    gradient_z=tuple(0.0 for _ in outside_candidate.joints),
                ),
                left_row_index=0,
                right_state=boundary_state,
                right_row_index=0,
                preferred_limits=preferred_limits,
                joint_weights=optimizer_settings.joint_delta_weights,
                base_scale=math.sqrt(_HANDOVER_CORRIDOR_BOUNDARY_WEIGHT),
                left_is_fixed=True,
            )

    if (
        corridor_model.corridor_end + 1 < len(search_result.selected_path)
        and corridor_model.right_chain[-1] is not None
    ):
        boundary_state = corridor_model.right_chain[-1]
        outside_candidate = search_result.selected_path[corridor_model.corridor_end + 1]
        if _lineage_matches_family(
            outside_candidate,
            family_flags=corridor_model.right_family,
            lineage_key=corridor_model.right_lineage,
        ):
            _append_linearized_difference_rows(
                rows,
                rhs_values,
                variable_count=variable_count,
                left_state=boundary_state,
                left_row_index=corridor_row_count - 1,
                right_state=_FamilyLinearizedCandidate(
                    candidate=outside_candidate,
                    gradient_y=tuple(0.0 for _ in outside_candidate.joints),
                    gradient_z=tuple(0.0 for _ in outside_candidate.joints),
                ),
                right_row_index=corridor_row_count - 1,
                preferred_limits=preferred_limits,
                joint_weights=optimizer_settings.joint_delta_weights,
                base_scale=math.sqrt(_HANDOVER_CORRIDOR_BOUNDARY_WEIGHT),
                right_is_fixed=True,
            )

    magnitude_weight = math.sqrt(_HANDOVER_CORRIDOR_UPDATE_MAGNITUDE_WEIGHT)
    smoothness_weight = math.sqrt(_HANDOVER_CORRIDOR_UPDATE_SMOOTHNESS_WEIGHT)
    curvature_weight = math.sqrt(_HANDOVER_CORRIDOR_UPDATE_CURVATURE_WEIGHT)

    for local_row_index in range(corridor_row_count):
        for axis_index in range(2):
            row = np.zeros(variable_count, dtype=float)
            row[(local_row_index * 2) + axis_index] = magnitude_weight
            rows.append(row)
            rhs_values.append(0.0)

    for local_row_index in range(1, corridor_row_count):
        for axis_index in range(2):
            row = np.zeros(variable_count, dtype=float)
            row[((local_row_index - 1) * 2) + axis_index] = -smoothness_weight
            row[(local_row_index * 2) + axis_index] = smoothness_weight
            rows.append(row)
            rhs_values.append(0.0)

    for local_row_index in range(1, corridor_row_count - 1):
        for axis_index in range(2):
            row = np.zeros(variable_count, dtype=float)
            row[((local_row_index - 1) * 2) + axis_index] = curvature_weight
            row[(local_row_index * 2) + axis_index] = -2.0 * curvature_weight
            row[((local_row_index + 1) * 2) + axis_index] = curvature_weight
            rows.append(row)
            rhs_values.append(0.0)

    if not rows:
        return None

    matrix = np.vstack(rows)
    rhs = np.asarray(rhs_values, dtype=float)

    lower_bounds, upper_bounds = _build_corridor_delta_bounds(
        search_result,
        corridor_model=corridor_model,
        max_abs_offset_mm=max_abs_offset_mm,
    )

    solution = lsq_linear(
        matrix,
        rhs,
        bounds=(lower_bounds, upper_bounds),
        lsmr_tol="auto",
        verbose=0,
        max_iter=200,
    )
    if not solution.success:
        return None

    update_norm = float(np.linalg.norm(solution.x, ord=np.inf))
    if not math.isfinite(update_norm) or update_norm <= 1e-6:
        return None

    corridor_delta = tuple(
        (float(solution.x[local_row_index * 2]), float(solution.x[(local_row_index * 2) + 1]))
        for local_row_index in range(corridor_row_count)
    )
    predicted_corridor_max_step_deg, predicted_target_cut_max_step_deg = (
        _predict_handover_corridor_metrics(
            corridor_model,
            target_segment=target_segment,
            search_result=search_result,
            corridor_delta=corridor_delta,
        )
    )
    return _HandoverSolveResult(
        corridor_delta=corridor_delta,
        predicted_corridor_max_step_deg=predicted_corridor_max_step_deg,
        predicted_target_cut_max_step_deg=predicted_target_cut_max_step_deg,
        solver_name=f"lsq_fallback:{reason}",
    )


def _append_linearized_difference_rows(
    rows: list[np.ndarray],
    rhs_values: list[float],
    *,
    variable_count: int,
    left_state: _FamilyLinearizedCandidate,
    left_row_index: int,
    right_state: _FamilyLinearizedCandidate,
    right_row_index: int,
    preferred_limits: Sequence[float],
    joint_weights: Sequence[float],
    base_scale: float,
    joint_emphasis: Sequence[float] | None = None,
    left_is_fixed: bool = False,
    right_is_fixed: bool = False,
) -> None:
    for joint_index, (left_joint, right_joint) in enumerate(
        zip(left_state.candidate.joints, right_state.candidate.joints)
    ):
        preferred_limit = preferred_limits[joint_index] if joint_index < len(preferred_limits) else 1.0
        joint_weight = joint_weights[joint_index] if joint_index < len(joint_weights) else 1.0
        if joint_emphasis is not None and joint_index < len(joint_emphasis):
            joint_weight *= float(joint_emphasis[joint_index])
        joint_scale = base_scale * joint_weight / max(1.0, preferred_limit)
        row = np.zeros(variable_count, dtype=float)
        if not left_is_fixed:
            row[left_row_index * 2] = -joint_scale * left_state.gradient_y[joint_index]
            row[(left_row_index * 2) + 1] = -joint_scale * left_state.gradient_z[joint_index]
        if not right_is_fixed:
            row[right_row_index * 2] = joint_scale * right_state.gradient_y[joint_index]
            row[(right_row_index * 2) + 1] = joint_scale * right_state.gradient_z[joint_index]
        rows.append(row)
        rhs_values.append(-joint_scale * (right_joint - left_joint))


def _build_handover_joint_emphasis(
    left_state: _FamilyLinearizedCandidate,
    right_state: _FamilyLinearizedCandidate,
) -> tuple[float, ...]:
    deltas = [
        abs(current - previous)
        for previous, current in zip(
            left_state.candidate.joints,
            right_state.candidate.joints,
        )
    ]
    max_delta = max(deltas, default=0.0)
    if max_delta <= 1e-9:
        return tuple(1.0 for _ in deltas)
    return tuple(
        1.0 + 6.0 * (delta / max_delta) * (delta / max_delta)
        for delta in deltas
    )


def _build_handover_stage2_slack_groups(
    *,
    corridor_row_count: int,
    target_term: _LinearizedDifferenceTerm,
    overlap_terms: Sequence[_LinearizedDifferenceTerm],
    boundary_terms: Sequence[_LinearizedDifferenceTerm],
) -> tuple[
    tuple[tuple[np.ndarray, float], ...],
    tuple[tuple[np.ndarray, float], ...],
    tuple[tuple[np.ndarray, float], ...],
    tuple[tuple[np.ndarray, float], ...],
    tuple[tuple[np.ndarray, float], ...],
    tuple[tuple[np.ndarray, float], ...],
]:
    target_cut_expressions = tuple(
        _build_term_expression(
            target_term,
            joint_index=joint_index,
            corridor_row_count=corridor_row_count,
        )
        for joint_index in range(len(target_term.left_state.candidate.joints))
    )
    overlap_expressions = tuple(
        _build_term_expression(
            term,
            joint_index=joint_index,
            corridor_row_count=corridor_row_count,
        )
        for term in overlap_terms
        for joint_index in range(len(term.left_state.candidate.joints))
    )
    boundary_expressions = tuple(
        _build_term_expression(
            term,
            joint_index=joint_index,
            corridor_row_count=corridor_row_count,
        )
        for term in boundary_terms
        for joint_index in range(len(term.left_state.candidate.joints))
    )
    update_expressions = tuple(
        _build_axis_expression(
            corridor_row_count=corridor_row_count,
            local_row_index=local_row_index,
            axis_index=axis_index,
        )
        for local_row_index in range(corridor_row_count)
        for axis_index in range(2)
    )
    first_diff_expressions = tuple(
        _build_axis_difference_expression(
            corridor_row_count=corridor_row_count,
            left_row_index=local_row_index - 1,
            right_row_index=local_row_index,
            axis_index=axis_index,
        )
        for local_row_index in range(1, corridor_row_count)
        for axis_index in range(2)
    )
    second_diff_expressions = tuple(
        _build_axis_curvature_expression(
            corridor_row_count=corridor_row_count,
            center_row_index=local_row_index,
            axis_index=axis_index,
        )
        for local_row_index in range(1, corridor_row_count - 1)
        for axis_index in range(2)
    )
    return (
        target_cut_expressions,
        overlap_expressions,
        boundary_expressions,
        update_expressions,
        first_diff_expressions,
        second_diff_expressions,
    )


def _build_axis_expression(
    *,
    corridor_row_count: int,
    local_row_index: int,
    axis_index: int,
) -> tuple[np.ndarray, float]:
    coefficient = np.zeros(corridor_row_count * 2, dtype=float)
    coefficient[(local_row_index * 2) + axis_index] = 1.0
    return coefficient, 0.0


def _build_axis_difference_expression(
    *,
    corridor_row_count: int,
    left_row_index: int,
    right_row_index: int,
    axis_index: int,
) -> tuple[np.ndarray, float]:
    coefficient = np.zeros(corridor_row_count * 2, dtype=float)
    coefficient[(left_row_index * 2) + axis_index] = -1.0
    coefficient[(right_row_index * 2) + axis_index] = 1.0
    return coefficient, 0.0


def _build_axis_curvature_expression(
    *,
    corridor_row_count: int,
    center_row_index: int,
    axis_index: int,
) -> tuple[np.ndarray, float]:
    coefficient = np.zeros(corridor_row_count * 2, dtype=float)
    coefficient[((center_row_index - 1) * 2) + axis_index] = 1.0
    coefficient[(center_row_index * 2) + axis_index] = -2.0
    coefficient[((center_row_index + 1) * 2) + axis_index] = 1.0
    return coefficient, 0.0


def _build_lexicographic_group_weights(
    group_upper_bounds: Sequence[float],
) -> tuple[float, ...]:
    if not group_upper_bounds:
        return ()
    weights = [0.0] * len(group_upper_bounds)
    running_capacity = 1.0
    for group_index in range(len(group_upper_bounds) - 1, -1, -1):
        minimum_weight = _HANDOVER_CORRIDOR_STAGE2_GROUP_WEIGHTS[group_index]
        weights[group_index] = max(minimum_weight, running_capacity)
        running_capacity = weights[group_index] * max(1.0, group_upper_bounds[group_index]) + running_capacity
    return tuple(weights)


def _expression_abs_upper_bound(
    coefficient: np.ndarray,
    constant: float,
    *,
    lower_bounds: np.ndarray,
    upper_bounds: np.ndarray,
) -> float:
    max_value = float(constant)
    min_value = float(constant)
    for index, coeff_value in enumerate(coefficient):
        if coeff_value >= 0.0:
            max_value += float(coeff_value) * float(upper_bounds[index])
            min_value += float(coeff_value) * float(lower_bounds[index])
        else:
            max_value += float(coeff_value) * float(lower_bounds[index])
            min_value += float(coeff_value) * float(upper_bounds[index])
    return max(abs(max_value), abs(min_value))


def _append_abs_epigraph_constraints(
    a_rows: list[np.ndarray],
    b_values: list[float],
    *,
    coefficient: np.ndarray,
    constant: float,
    t_index: int,
    variable_count: int,
) -> None:
    positive_row = np.zeros(variable_count, dtype=float)
    positive_row[: len(coefficient)] = coefficient
    positive_row[t_index] = -1.0
    a_rows.append(positive_row)
    b_values.append(-float(constant))

    negative_row = np.zeros(variable_count, dtype=float)
    negative_row[: len(coefficient)] = -coefficient
    negative_row[t_index] = -1.0
    a_rows.append(negative_row)
    b_values.append(float(constant))


def _append_abs_slack_constraints(
    a_rows: list[np.ndarray],
    b_values: list[float],
    *,
    coefficient: np.ndarray,
    constant: float,
    slack_index: int,
    variable_count: int,
) -> None:
    positive_row = np.zeros(variable_count, dtype=float)
    positive_row[: len(coefficient)] = coefficient
    positive_row[slack_index] = -1.0
    a_rows.append(positive_row)
    b_values.append(-float(constant))

    negative_row = np.zeros(variable_count, dtype=float)
    negative_row[: len(coefficient)] = -coefficient
    negative_row[slack_index] = -1.0
    a_rows.append(negative_row)
    b_values.append(float(constant))


def _append_profile_continuity_constraints(
    a_rows: list[np.ndarray],
    b_values: list[float],
    *,
    corridor_row_count: int,
    base_profile: Sequence[tuple[float, float]],
    max_profile_step_mm: float,
    variable_count: int,
) -> None:
    if corridor_row_count <= 1:
        return

    diagonal_limit = max_profile_step_mm * math.sqrt(2.0) * _INSCRIBED_OCTAGON_COS_22P5
    for local_row_index in range(corridor_row_count - 1):
        base_dy = float(base_profile[local_row_index + 1][0]) - float(base_profile[local_row_index][0])
        base_dz = float(base_profile[local_row_index + 1][1]) - float(base_profile[local_row_index][1])

        dy_expression = _build_axis_difference_expression(
            corridor_row_count=corridor_row_count,
            left_row_index=local_row_index,
            right_row_index=local_row_index + 1,
            axis_index=0,
        )[0]
        dz_expression = _build_axis_difference_expression(
            corridor_row_count=corridor_row_count,
            left_row_index=local_row_index,
            right_row_index=local_row_index + 1,
            axis_index=1,
        )[0]

        _append_linear_inequality_pair(
            a_rows,
            b_values,
            coefficient=dy_expression,
            constant=base_dy,
            limit=max_profile_step_mm,
            variable_count=variable_count,
        )
        _append_linear_inequality_pair(
            a_rows,
            b_values,
            coefficient=dz_expression,
            constant=base_dz,
            limit=max_profile_step_mm,
            variable_count=variable_count,
        )
        _append_linear_inequality_pair(
            a_rows,
            b_values,
            coefficient=dy_expression + dz_expression,
            constant=base_dy + base_dz,
            limit=diagonal_limit,
            variable_count=variable_count,
        )
        _append_linear_inequality_pair(
            a_rows,
            b_values,
            coefficient=dy_expression - dz_expression,
            constant=base_dy - base_dz,
            limit=diagonal_limit,
            variable_count=variable_count,
        )


def _append_linear_inequality_pair(
    a_rows: list[np.ndarray],
    b_values: list[float],
    *,
    coefficient: np.ndarray,
    constant: float,
    limit: float,
    variable_count: int,
) -> None:
    positive_row = np.zeros(variable_count, dtype=float)
    positive_row[: len(coefficient)] = coefficient
    a_rows.append(positive_row)
    b_values.append(float(limit) - float(constant))

    negative_row = np.zeros(variable_count, dtype=float)
    negative_row[: len(coefficient)] = -coefficient
    a_rows.append(negative_row)
    b_values.append(float(limit) + float(constant))


def _apply_corridor_profile_delta(
    profile: Sequence[tuple[float, float]],
    *,
    corridor_start: int,
    corridor_delta: Sequence[tuple[float, float]],
    trust_scale: float,
    max_abs_offset_mm: float,
) -> tuple[tuple[float, float], ...]:
    updated_profile = list(profile)
    for local_row_index, delta in enumerate(corridor_delta):
        row_index = corridor_start + local_row_index
        current_offset = updated_profile[row_index]
        updated_profile[row_index] = (
            float(
                max(
                    -max_abs_offset_mm,
                    min(
                        max_abs_offset_mm,
                        float(current_offset[0]) + trust_scale * float(delta[0]),
                    ),
                )
            ),
            float(
                max(
                    -max_abs_offset_mm,
                    min(
                        max_abs_offset_mm,
                        float(current_offset[1]) + trust_scale * float(delta[1]),
                    ),
                )
            ),
        )
    return tuple(updated_profile)


def _attempt_inserted_transition_repair(
    search_result: _PathSearchResult,
    *,
    robot,
    mat_type,
    move_type: str,
    start_joints: tuple[float, ...],
    tool_pose,
    reference_pose,
    joint_count: int,
    motion_settings,
    optimizer_settings: _PathOptimizerSettings,
    lower_limits: Sequence[float],
    upper_limits: Sequence[float],
    a1_lower_deg: float,
    a1_upper_deg: float,
    a2_max_deg: float,
    joint_constraint_tolerance_deg: float,
    run_post_inserted_window_repair: bool = True,
) -> _PathSearchResult | None:
    if not search_result.selected_path:
        return None

    lower_limits_tuple = tuple(float(value) for value in lower_limits[:joint_count])
    upper_limits_tuple = tuple(float(value) for value in upper_limits[:joint_count])
    current_result = search_result
    accepted_any = False
    attempted_repairs: set[tuple[str, str, int]] = set()
    max_inserted_repair_passes = max(
        1,
        min(4, len(tuple(motion_settings.frame_a_origin_yz_insertion_counts)) + 2),
    )

    for pass_index in range(max_inserted_repair_passes):
        problem_segments = _collect_problem_segments(
            current_result.selected_path,
            bridge_trigger_joint_delta_deg=motion_settings.bridge_trigger_joint_delta_deg,
        )
        if not problem_segments:
            return current_result

        best_repair_result: _PathSearchResult | None = None
        best_repair_key: tuple[float, ...] | None = None
        best_repair_note: tuple[str, str, int, int] | None = None
        base_worst_joint_step = float(current_result.worst_joint_step_deg)
        base_problem_count = len(problem_segments)

        for segment_index, _config_changed, _max_joint_delta, _mean_joint_delta in problem_segments[:4]:
            left_label = str(current_result.row_labels[segment_index])
            right_label = str(current_result.row_labels[segment_index + 1])
            for insertion_count in motion_settings.frame_a_origin_yz_insertion_counts:
                repair_signature = (left_label, right_label, int(insertion_count))
                if repair_signature in attempted_repairs:
                    continue
                attempted_repairs.add(repair_signature)
                (
                    augmented_reference_rows,
                    augmented_profile,
                    augmented_labels,
                    augmented_flags,
                ) = _insert_interpolated_transition_rows(
                    current_result.reference_pose_rows,
                    current_result.frame_a_origin_yz_profile_mm,
                    current_result.row_labels,
                    current_result.inserted_flags,
                    segment_index=segment_index,
                    insertion_count=insertion_count,
                )
                candidate_result = _evaluate_frame_a_origin_profile(
                    augmented_reference_rows,
                    frame_a_origin_yz_profile_mm=augmented_profile,
                    row_labels=augmented_labels,
                    inserted_flags=augmented_flags,
                    robot=robot,
                    mat_type=mat_type,
                    move_type=move_type,
                    start_joints=start_joints,
                    tool_pose=tool_pose,
                    reference_pose=reference_pose,
                    joint_count=joint_count,
                    optimizer_settings=optimizer_settings,
                    a1_lower_deg=a1_lower_deg,
                    a1_upper_deg=a1_upper_deg,
                    a2_max_deg=a2_max_deg,
                    joint_constraint_tolerance_deg=joint_constraint_tolerance_deg,
                    seed_joints=(
                        ()
                        if getattr(robot, "ik_seed_invariant", False)
                        else _build_local_seed_joints(current_result, segment_index, segment_index + 1)
                    ),
                    lower_limits=lower_limits_tuple,
                    upper_limits=upper_limits_tuple,
                    bridge_trigger_joint_delta_deg=motion_settings.bridge_trigger_joint_delta_deg,
                    lock_profile_endpoints=bool(
                        getattr(
                            motion_settings,
                            "lock_frame_a_origin_yz_profile_endpoints",
                            True,
                        )
                    ),
                )
                residual_problems = _collect_problem_segments(
                    candidate_result.selected_path,
                    bridge_trigger_joint_delta_deg=motion_settings.bridge_trigger_joint_delta_deg,
                )
                if candidate_result.invalid_row_count != 0 or candidate_result.ik_empty_row_count != 0:
                    continue

                is_clean_repair = not residual_problems
                improves_global_sort = _path_search_sort_key(candidate_result) < _path_search_sort_key(current_result)
                reduces_problem_count = len(residual_problems) < base_problem_count
                reduces_worst_jump = (
                    float(candidate_result.worst_joint_step_deg)
                    < base_worst_joint_step - 1e-6
                )
                does_not_worsen_large_jump = (
                    float(candidate_result.worst_joint_step_deg)
                    <= base_worst_joint_step + 1e-6
                )
                if not (
                    is_clean_repair
                    or improves_global_sort
                    or reduces_problem_count
                    or reduces_worst_jump
                    or does_not_worsen_large_jump
                ):
                    continue

                repair_key = (
                    0.0 if is_clean_repair else 1.0,
                    float(len(residual_problems)),
                    float(candidate_result.worst_joint_step_deg),
                    float(candidate_result.mean_joint_step_deg),
                    float(candidate_result.config_switches),
                    float(candidate_result.bridge_like_segments),
                    float(candidate_result.offset_step_jitter_mm),
                    float(candidate_result.offset_jerk_mm),
                    float(candidate_result.total_cost),
                )
                if best_repair_key is None or repair_key < best_repair_key:
                    best_repair_key = repair_key
                    best_repair_result = candidate_result
                    best_repair_note = (
                        left_label,
                        right_label,
                        int(insertion_count),
                        len(residual_problems),
                    )

        if best_repair_result is None or best_repair_note is None:
            break

        if run_post_inserted_window_repair:
            refined_repair_result = _refine_path_with_frame_a_origin_profile(
                best_repair_result,
                robot=robot,
                mat_type=mat_type,
                move_type=move_type,
                start_joints=start_joints,
                tool_pose=tool_pose,
                reference_pose=reference_pose,
                joint_count=joint_count,
                motion_settings=motion_settings,
                optimizer_settings=optimizer_settings,
                lower_limits=lower_limits_tuple,
                upper_limits=upper_limits_tuple,
                a1_lower_deg=a1_lower_deg,
                a1_upper_deg=a1_upper_deg,
                a2_max_deg=a2_max_deg,
                joint_constraint_tolerance_deg=joint_constraint_tolerance_deg,
            )
            if _path_search_sort_key(refined_repair_result) < _path_search_sort_key(
                best_repair_result
            ):
                refined_residual_problems = _collect_problem_segments(
                    refined_repair_result.selected_path,
                    bridge_trigger_joint_delta_deg=(
                        motion_settings.bridge_trigger_joint_delta_deg
                    ),
                )
                best_repair_result = refined_repair_result
                best_repair_note = (
                    best_repair_note[0],
                    best_repair_note[1],
                    best_repair_note[2],
                    len(refined_residual_problems),
                )

        current_result = best_repair_result
        accepted_any = True
        left_label, right_label, insertion_count, residual_count = best_repair_note
        residual_note = "clean" if residual_count == 0 else f"residual_warnings={residual_count}"
        print(
            "Accepted inserted transition samples: "
            f"pass={pass_index + 1}, "
            f"segment={left_label}->{right_label}, "
            f"inserted_points={insertion_count}, "
            f"waypoint_count={len(current_result.pose_rows)}, "
            f"worst_joint_step={current_result.worst_joint_step_deg:.3f} deg, "
            f"{residual_note}."
        )

    return current_result if accepted_any else None


def _attempt_joint_space_bridge_repair(
    search_result: _PathSearchResult,
    *,
    robot,
    tool_pose,
    reference_pose,
    joint_count: int,
    motion_settings,
    optimizer_settings: _PathOptimizerSettings,
    lower_limits: Sequence[float],
    upper_limits: Sequence[float],
) -> _PathSearchResult | None:
    """Split residual wrist/config jumps with explicit joint-space bridge waypoints.

    This is deliberately different from the closed-path terminal A6 equivalence:
    each inserted bridge has one concrete joint vector and all middle-path
    continuity metrics still use real absolute joint differences.  The bridge
    exists as a motion-instruction fallback for residual wrist flips that Y/Z
    pose repair cannot remove without a large physical detour.
    """

    if not bool(getattr(motion_settings, "enable_joint_space_bridge_repair", False)):
        return None
    if not search_result.selected_path:
        return None

    max_insertions = int(
        getattr(motion_settings, "joint_space_bridge_max_insertions_per_segment", 0)
    )
    if max_insertions <= 0:
        return None

    problem_segments = _collect_problem_segments(
        search_result.selected_path,
        bridge_trigger_joint_delta_deg=motion_settings.bridge_trigger_joint_delta_deg,
    )
    if not problem_segments:
        return None

    problem_segment_indices = {int(segment_index) for segment_index, *_ in problem_segments}
    step_limits = _normalize_joint_bridge_step_limits(
        getattr(motion_settings, "bridge_step_deg", (20.0,)),
        joint_count=joint_count,
    )
    lower_limits_tuple = tuple(float(value) for value in lower_limits[:joint_count])
    upper_limits_tuple = tuple(float(value) for value in upper_limits[:joint_count])

    new_reference_rows: list[dict[str, float]] = []
    new_pose_rows: list[dict[str, float]] = []
    new_ik_layers: list[_IKLayer] = []
    new_selected_path: list[_IKCandidate] = []
    new_profile: list[tuple[float, float]] = []
    new_labels: list[str] = []
    new_flags: list[bool] = []
    inserted_total = 0
    bridged_segments: list[tuple[str, str, int]] = []
    skipped_segments: list[tuple[str, str, float, float]] = []

    def append_existing(row_index: int) -> None:
        new_reference_rows.append(dict(search_result.reference_pose_rows[row_index]))
        new_pose_rows.append(dict(search_result.pose_rows[row_index]))
        new_ik_layers.append(search_result.ik_layers[row_index])
        new_selected_path.append(search_result.selected_path[row_index])
        new_profile.append(
            (
                float(search_result.frame_a_origin_yz_profile_mm[row_index][0]),
                float(search_result.frame_a_origin_yz_profile_mm[row_index][1]),
            )
        )
        new_labels.append(str(search_result.row_labels[row_index]))
        new_flags.append(bool(search_result.inserted_flags[row_index]))

    for row_index in range(len(search_result.selected_path)):
        append_existing(row_index)
        if row_index + 1 >= len(search_result.selected_path):
            continue
        if row_index not in problem_segment_indices:
            continue

        previous_candidate = search_result.selected_path[row_index]
        current_candidate = search_result.selected_path[row_index + 1]
        insertion_count = _joint_bridge_insertion_count(
            previous_candidate.joints,
            current_candidate.joints,
            step_limits=step_limits,
        )
        insertion_count = min(insertion_count, max_insertions)
        if insertion_count <= 0:
            continue

        left_label = str(search_result.row_labels[row_index])
        right_label = str(search_result.row_labels[row_index + 1])
        left_profile = search_result.frame_a_origin_yz_profile_mm[row_index]
        right_profile = search_result.frame_a_origin_yz_profile_mm[row_index + 1]
        bridge_entries: list[tuple[dict[str, float], dict[str, float], object, _IKCandidate, tuple[float, float], str]] = []
        for insertion_index in range(1, insertion_count + 1):
            ratio = insertion_index / (insertion_count + 1)
            bridge_joints = tuple(
                float(previous + (current - previous) * ratio)
                for previous, current in zip(previous_candidate.joints, current_candidate.joints)
            )
            bridge_pose = _pose_from_joints_in_frame(
                robot,
                bridge_joints,
                tool_pose=tool_pose,
                reference_pose=reference_pose,
            )
            bridge_pose_row = _pose_row_from_pose_object(bridge_pose)
            bridge_profile = (
                float(left_profile[0] + (right_profile[0] - left_profile[0]) * ratio),
                float(left_profile[1] + (right_profile[1] - left_profile[1]) * ratio),
            )
            bridge_reference_row = dict(bridge_pose_row)
            bridge_reference_row["y_mm"] = float(bridge_reference_row["y_mm"]) - bridge_profile[0]
            bridge_reference_row["z_mm"] = float(bridge_reference_row["z_mm"]) - bridge_profile[1]
            bridge_candidate = _make_joint_bridge_candidate(
                robot,
                bridge_joints,
                lower_limits=lower_limits_tuple,
                upper_limits=upper_limits_tuple,
                optimizer_settings=optimizer_settings,
            )
            bridge_entries.append(
                (
                    bridge_reference_row,
                    bridge_pose_row,
                    bridge_pose,
                    bridge_candidate,
                    bridge_profile,
                    f"{left_label}_JBR_{insertion_index:02d}",
                )
            )

        detour_metrics = _joint_bridge_tcp_detour_metrics(
            search_result.pose_rows[row_index],
            search_result.pose_rows[row_index + 1],
            [entry[1] for entry in bridge_entries],
        )
        max_deviation_mm = float(
            getattr(motion_settings, "joint_space_bridge_max_tcp_deviation_mm", 20.0)
        )
        max_path_ratio = float(
            getattr(motion_settings, "joint_space_bridge_max_tcp_path_ratio", 4.0)
        )
        if (
            detour_metrics["max_deviation_mm"] > max_deviation_mm + 1e-9
            or detour_metrics["path_ratio"] > max_path_ratio + 1e-9
        ):
            skipped_segments.append(
                (
                    left_label,
                    right_label,
                    detour_metrics["path_ratio"],
                    detour_metrics["max_deviation_mm"],
                )
            )
            continue

        for bridge_reference_row, bridge_pose_row, bridge_pose, bridge_candidate, bridge_profile, bridge_label in bridge_entries:
            new_reference_rows.append(bridge_reference_row)
            new_pose_rows.append(bridge_pose_row)
            new_ik_layers.append(_IKLayer(pose=bridge_pose, candidates=(bridge_candidate,)))
            new_selected_path.append(bridge_candidate)
            new_profile.append(bridge_profile)
            new_labels.append(bridge_label)
            new_flags.append(True)
            inserted_total += 1
        bridged_segments.append((left_label, right_label, insertion_count))

    if inserted_total <= 0:
        if skipped_segments:
            skipped_note = ", ".join(
                f"{left}->{right}:ratio={ratio:.2f},dev={deviation:.1f}mm"
                for left, right, ratio, deviation in skipped_segments[:4]
            )
            print(
                "Rejected joint-space bridge repair because TCP detour is too large: "
                f"{skipped_note}."
            )
        return None

    (
        config_switches,
        bridge_like_segments,
        worst_joint_step_deg,
        mean_joint_step_deg,
    ) = _summarize_selected_path(
        tuple(new_selected_path),
        bridge_trigger_joint_delta_deg=motion_settings.bridge_trigger_joint_delta_deg,
    )
    (
        offset_step_jitter_mm,
        offset_jerk_mm,
        max_abs_offset_mm,
        total_abs_offset_mm,
    ) = _summarize_profile_metrics(tuple(new_profile))

    bridged_result = _PathSearchResult(
        reference_pose_rows=tuple(new_reference_rows),
        pose_rows=tuple(new_pose_rows),
        ik_layers=tuple(new_ik_layers),
        selected_path=tuple(new_selected_path),
        total_cost=_joint_bridge_path_cost(tuple(new_selected_path), optimizer_settings),
        frame_a_origin_yz_profile_mm=tuple(new_profile),
        row_labels=tuple(new_labels),
        inserted_flags=tuple(new_flags),
        invalid_row_count=search_result.invalid_row_count,
        ik_empty_row_count=search_result.ik_empty_row_count,
        config_switches=config_switches,
        bridge_like_segments=bridge_like_segments,
        worst_joint_step_deg=worst_joint_step_deg,
        mean_joint_step_deg=mean_joint_step_deg,
        offset_step_jitter_mm=offset_step_jitter_mm,
        offset_jerk_mm=offset_jerk_mm,
        max_abs_offset_mm=max_abs_offset_mm,
        total_abs_offset_mm=total_abs_offset_mm,
    )

    if _path_search_sort_key(bridged_result) >= _path_search_sort_key(search_result):
        return None

    segment_note = ", ".join(
        f"{left}->{right}:{count}" for left, right, count in bridged_segments[:6]
    )
    if len(bridged_segments) > 6:
        segment_note += f", +{len(bridged_segments) - 6} more"
    print(
        "Accepted joint-space bridge repair: "
        f"inserted_points={inserted_total}, "
        f"segments=[{segment_note}], "
        f"config_switches={bridged_result.config_switches}, "
        f"bridge_like_segments={bridged_result.bridge_like_segments}, "
        f"worst_joint_step={bridged_result.worst_joint_step_deg:.3f} deg."
    )
    return bridged_result


def _normalize_joint_bridge_step_limits(
    step_limits: Sequence[float],
    *,
    joint_count: int,
) -> tuple[float, ...]:
    normalized = [float(value) for value in step_limits]
    if not normalized:
        normalized = [20.0]
    if len(normalized) < joint_count:
        normalized.extend([normalized[-1]] * (joint_count - len(normalized)))
    return tuple(max(1e-6, value) for value in normalized[:joint_count])


def _joint_bridge_insertion_count(
    previous_joints: Sequence[float],
    current_joints: Sequence[float],
    *,
    step_limits: Sequence[float],
) -> int:
    required_steps = 1
    for previous, current, limit in zip(previous_joints, current_joints, step_limits):
        required_steps = max(required_steps, int(math.ceil(abs(current - previous) / limit)))
    return max(0, required_steps - 1)


def _joint_bridge_tcp_detour_metrics(
    left_pose_row: dict[str, float],
    right_pose_row: dict[str, float],
    bridge_pose_rows: Sequence[dict[str, float]],
) -> dict[str, float]:
    points = [
        _pose_row_translation(left_pose_row),
        *(_pose_row_translation(row) for row in bridge_pose_rows),
        _pose_row_translation(right_pose_row),
    ]
    direct_distance = _point_distance(points[0], points[-1])
    path_distance = sum(_point_distance(first, second) for first, second in zip(points, points[1:]))
    max_deviation = max(
        _point_to_segment_distance(point, points[0], points[-1])
        for point in points
    )
    path_ratio = path_distance / max(direct_distance, 1e-9)
    return {
        "direct_distance_mm": float(direct_distance),
        "path_distance_mm": float(path_distance),
        "path_ratio": float(path_ratio),
        "max_deviation_mm": float(max_deviation),
    }


def _pose_row_translation(row: dict[str, float]) -> tuple[float, float, float]:
    return (
        float(row["x_mm"]),
        float(row["y_mm"]),
        float(row["z_mm"]),
    )


def _point_distance(
    first: Sequence[float],
    second: Sequence[float],
) -> float:
    return math.sqrt(
        sum((float(current) - float(previous)) ** 2 for previous, current in zip(first, second))
    )


def _point_to_segment_distance(
    point: Sequence[float],
    segment_start: Sequence[float],
    segment_end: Sequence[float],
) -> float:
    start = tuple(float(value) for value in segment_start)
    end = tuple(float(value) for value in segment_end)
    query = tuple(float(value) for value in point)
    direction = tuple(end_value - start_value for start_value, end_value in zip(start, end))
    length_sq = sum(value * value for value in direction)
    if length_sq <= 1e-12:
        return _point_distance(query, start)
    ratio = sum(
        (query_value - start_value) * direction_value
        for query_value, start_value, direction_value in zip(query, start, direction)
    ) / length_sq
    ratio = max(0.0, min(1.0, ratio))
    projection = tuple(
        start_value + ratio * direction_value
        for start_value, direction_value in zip(start, direction)
    )
    return _point_distance(query, projection)


def _pose_from_joints_in_frame(
    robot,
    joints: Sequence[float],
    *,
    tool_pose,
    reference_pose,
):
    pose_from_joints = getattr(robot, "PoseFromJointsInFrame", None)
    if not callable(pose_from_joints):
        raise RuntimeError("The IK backend does not expose PoseFromJointsInFrame.")
    return pose_from_joints(tuple(float(value) for value in joints), tool_pose, reference_pose)


def _pose_row_from_pose_object(pose) -> dict[str, float]:
    return {
        "x_mm": float(pose[0, 3]),
        "y_mm": float(pose[1, 3]),
        "z_mm": float(pose[2, 3]),
        "r11": float(pose[0, 0]),
        "r12": float(pose[0, 1]),
        "r13": float(pose[0, 2]),
        "r21": float(pose[1, 0]),
        "r22": float(pose[1, 1]),
        "r23": float(pose[1, 2]),
        "r31": float(pose[2, 0]),
        "r32": float(pose[2, 1]),
        "r33": float(pose[2, 2]),
    }


def _make_joint_bridge_candidate(
    robot,
    joints: Sequence[float],
    *,
    lower_limits: Sequence[float],
    upper_limits: Sequence[float],
    optimizer_settings: _PathOptimizerSettings,
) -> _IKCandidate:
    joints_tuple = tuple(float(value) for value in joints)
    config_flags = _config_flags_for_joints(robot, joints_tuple)
    return _IKCandidate(
        joints=joints_tuple,
        config_flags=config_flags,
        joint_limit_penalty=_joint_limit_penalty(
            joints_tuple,
            lower_limits,
            upper_limits,
            optimizer_settings,
        ),
        singularity_penalty=_singularity_penalty(robot, joints_tuple, optimizer_settings),
        branch_id=None,
    )


def _config_flags_for_joints(robot, joints: tuple[float, ...]) -> tuple[int, ...]:
    try:
        config_values = robot.JointsConfig(list(joints)).list()
    except Exception:
        return ()
    return tuple(int(round(value)) for value in config_values[:3])


def _joint_bridge_path_cost(
    selected_path: Sequence[_IKCandidate],
    optimizer_settings: _PathOptimizerSettings,
) -> float:
    return float(
        sum(
            _candidate_transition_penalty(previous_candidate, current_candidate, optimizer_settings)
            for previous_candidate, current_candidate in zip(selected_path, selected_path[1:])
        )
    )


def _add_boundary_scan_states(
    *,
    current_offset: tuple[float, float],
    max_abs_offset_mm: float,
    probe_fn,  # callable(dy_mm: float, dz_mm: float) -> None
) -> None:
    """Probe along axis directions from the origin and current_offset.

    For windows that span a config-family transition, this supplements the
    current-offset-centered scan with origin-anchored axis probes that may
    reach IK configurations the offset-centered scan misses.  The probe_fn
    handles deduplication and result collection.
    """
    scan_step_mm = max(2.0, max_abs_offset_mm / 4.0)
    steps = max(1, int(math.ceil(max_abs_offset_mm / scan_step_mm)))
    axes = [(1.0, 0.0), (-1.0, 0.0), (0.0, 1.0), (0.0, -1.0)]
    origins = [(0.0, 0.0), (float(current_offset[0]), float(current_offset[1]))]

    for origin_dy, origin_dz in origins:
        for axis_dy, axis_dz in axes:
            for step in range(1, steps + 1):
                dy = origin_dy + step * scan_step_mm * axis_dy
                dz = origin_dz + step * scan_step_mm * axis_dz
                if abs(dy) > max_abs_offset_mm + 1e-9 or abs(dz) > max_abs_offset_mm + 1e-9:
                    break
                probe_fn(dy, dz)


def _select_window_state_candidates(
    candidates: Sequence[_IKCandidate],
) -> tuple[_IKCandidate, ...]:
    """Keep local Y/Z DP candidate states family-diverse.

    SixAxisIK can return enough candidates that a plain ``candidates[:N]`` cut
    drops an entire wrist-flip family.  That is especially bad in the exact
    places this repair is meant to handle, so retain a small quota per config
    family and explicitly keep the nearest-J5 candidate in each family.
    """

    if len(candidates) <= _WINDOW_STATE_MAX_CANDIDATES_PER_OFFSET:
        return tuple(candidates)

    grouped: dict[tuple[int, ...], list[_IKCandidate]] = {}
    for candidate in candidates:
        grouped.setdefault(candidate.config_flags, []).append(candidate)

    selected: list[_IKCandidate] = []
    seen: set[tuple[float, ...]] = set()

    def append_candidate(candidate: _IKCandidate) -> None:
        key = tuple(round(float(value), 9) for value in candidate.joints)
        if key in seen:
            return
        seen.add(key)
        selected.append(candidate)

    def quality_key(candidate: _IKCandidate) -> tuple[float, float, tuple[float, ...]]:
        abs_j5 = abs(float(candidate.joints[4])) if len(candidate.joints) >= 5 else math.inf
        return (
            float(candidate.joint_limit_penalty + candidate.singularity_penalty),
            abs_j5,
            tuple(float(value) for value in candidate.joints),
        )

    def wrist_key(candidate: _IKCandidate) -> tuple[float, float, tuple[float, ...]]:
        abs_j5 = abs(float(candidate.joints[4])) if len(candidate.joints) >= 5 else math.inf
        return (
            abs_j5,
            float(candidate.joint_limit_penalty + candidate.singularity_penalty),
            tuple(float(value) for value in candidate.joints),
        )

    for family in sorted(grouped):
        family_candidates = sorted(grouped[family], key=quality_key)
        for candidate in family_candidates[:_WINDOW_STATE_MAX_CANDIDATES_PER_FAMILY]:
            append_candidate(candidate)
        append_candidate(min(family_candidates, key=wrist_key))

    for candidate in sorted(candidates, key=lambda item: (quality_key(item), item.config_flags)):
        if len(selected) >= _WINDOW_STATE_MAX_CANDIDATES_PER_OFFSET:
            break
        append_candidate(candidate)

    return tuple(selected[:_WINDOW_STATE_MAX_CANDIDATES_PER_OFFSET])


def _solve_window_profile_dp(
    search_result: _PathSearchResult,
    *,
    window_start: int,
    window_end: int,
    max_abs_offset_mm: float,
    step_mm: float,
    robot,
    mat_type,
    tool_pose,
    reference_pose,
    joint_count: int,
    optimizer_settings: _PathOptimizerSettings,
    a1_lower_deg: float,
    a1_upper_deg: float,
    a2_max_deg: float,
    joint_constraint_tolerance_deg: float,
    lower_limits: Sequence[float],
    upper_limits: Sequence[float],
    start_joints: Sequence[float],
) -> tuple[tuple[float, float], ...] | None:
    if step_mm <= 0.0 or max_abs_offset_mm < 0.0 or window_end < window_start:
        return None

    offset_step_limit_mm = max(1.0, step_mm * 2.0)
    window_layers: list[tuple[_WindowState, ...]] = []
    dedup_rounding = 6
    seed_joints = (
        ()
        if getattr(robot, "ik_seed_invariant", False)
        else _build_local_seed_joints(search_result, window_start, window_end)
    )
    lower_limits_tuple = tuple(float(v) for v in lower_limits)
    upper_limits_tuple = tuple(float(v) for v in upper_limits)

    left_boundary_candidates = (
        search_result.ik_layers[window_start - 1].candidates
        if window_start > 0 and window_start - 1 < len(search_result.ik_layers)
        else ()
    )
    right_boundary_candidates = (
        search_result.ik_layers[window_end + 1].candidates
        if window_end + 1 < len(search_result.ik_layers)
        else ()
    )

    # Determine whether left and right boundary config families differ (config switch).
    # If so, we run a targeted 1D boundary scan for the central rows of the window so
    # that the DP has access to "bridge" candidates that straddle both config families.
    left_configs_fs = frozenset(c.config_flags for c in left_boundary_candidates)
    right_configs_fs = frozenset(c.config_flags for c in right_boundary_candidates)
    is_config_switch_window = (
        bool(left_configs_fs)
        and bool(right_configs_fs)
        and left_configs_fs.isdisjoint(right_configs_fs)
    )
    # Scan rows: middle third of the window
    if is_config_switch_window:
        span = window_end - window_start
        mid = window_start + span // 2
        half_span = max(1, span // 4)
        bridge_scan_rows: frozenset[int] = frozenset(
            range(max(window_start, mid - half_span), min(window_end + 1, mid + half_span + 1))
        )
    else:
        bridge_scan_rows = frozenset()

    # Eagerly compute boundary candidates once; injected for all relevant rows below.
    _bridge_candidate_cache: dict[tuple[float, float], list[_IKCandidate]] = {}

    def _probe_offset(dy_mm: float, dz_mm: float, reference_row: dict) -> list[_IKCandidate]:
        key = (round(dy_mm, dedup_rounding), round(dz_mm, dedup_rounding))
        if key not in _bridge_candidate_cache:
            adj = dict(reference_row)
            adj["x_mm"] = float(reference_row["x_mm"])
            adj["y_mm"] = float(reference_row["y_mm"]) + dy_mm
            adj["z_mm"] = float(reference_row["z_mm"]) + dz_mm
            pose_adj = _build_pose(adj, mat_type)
            _bridge_candidate_cache[key] = _collect_ik_candidates(
                robot,
                pose_adj,
                tool_pose=tool_pose,
                reference_pose=reference_pose,
                lower_limits=lower_limits_tuple,
                upper_limits=upper_limits_tuple,
                seed_joints=seed_joints,
                joint_count=joint_count,
                optimizer_settings=optimizer_settings,
                a1_lower_deg=a1_lower_deg,
                a1_upper_deg=a1_upper_deg,
                a2_max_deg=a2_max_deg,
                joint_constraint_tolerance_deg=joint_constraint_tolerance_deg,
            )
        return _bridge_candidate_cache[key]

    for row_index in range(window_start, window_end + 1):
        reference_row = search_result.reference_pose_rows[row_index]
        current_offset = search_result.frame_a_origin_yz_profile_mm[row_index]
        row_states: list[_WindowState] = []
        seen: set[tuple[float, float, tuple[float, ...]]] = set()
        _bridge_candidate_cache.clear()

        def _append_state(dy_mm: float, dz_mm: float) -> None:
            for cand in _select_window_state_candidates(
                _probe_offset(dy_mm, dz_mm, reference_row)
            ):
                dk = (
                    round(dy_mm, dedup_rounding),
                    round(dz_mm, dedup_rounding),
                    tuple(round(v, dedup_rounding) for v in cand.joints),
                )
                if dk not in seen:
                    seen.add(dk)
                    row_states.append(_WindowState(dy_mm=dy_mm, dz_mm=dz_mm, candidate=cand))

        for dy_mm, dz_mm in _build_window_offset_states(
            current_offset,
            max_abs_offset_mm=max_abs_offset_mm,
            step_mm=step_mm,
        ):
            adjusted_row = dict(reference_row)
            adjusted_row["x_mm"] = float(reference_row["x_mm"])
            adjusted_row["y_mm"] = float(reference_row["y_mm"]) + dy_mm
            adjusted_row["z_mm"] = float(reference_row["z_mm"]) + dz_mm
            pose = _build_pose(adjusted_row, mat_type)
            candidates = _collect_ik_candidates(
                robot,
                pose,
                tool_pose=tool_pose,
                reference_pose=reference_pose,
                lower_limits=lower_limits_tuple,
                upper_limits=upper_limits_tuple,
                seed_joints=seed_joints,
                joint_count=joint_count,
                optimizer_settings=optimizer_settings,
                a1_lower_deg=a1_lower_deg,
                a1_upper_deg=a1_upper_deg,
                a2_max_deg=a2_max_deg,
                joint_constraint_tolerance_deg=joint_constraint_tolerance_deg,
            )
            for candidate in _select_window_state_candidates(candidates):
                dedup_key = (
                    round(dy_mm, dedup_rounding),
                    round(dz_mm, dedup_rounding),
                    tuple(round(value, dedup_rounding) for value in candidate.joints),
                )
                if dedup_key in seen:
                    continue
                seen.add(dedup_key)
                row_states.append(_WindowState(dy_mm=dy_mm, dz_mm=dz_mm, candidate=candidate))

        # Analytic branch-boundary scan for config-switch transitions.
        # Scan along the dy and dz axes to find where the available IK config family
        # changes — these boundary points are likely "bridge points" that allow the DP
        # to route through the config transition without a large joint jump.
        if row_index in bridge_scan_rows:
            _add_boundary_scan_states(
                current_offset=current_offset,
                max_abs_offset_mm=max_abs_offset_mm,
                probe_fn=_append_state,
            )

        if not row_states:
            return None

        row_states.sort(
            key=lambda state: (
                state.candidate.config_flags,
                state.candidate.joint_limit_penalty + state.candidate.singularity_penalty,
                abs(state.dy_mm) + abs(state.dz_mm),
                state.candidate.joints,
            )
        )
        window_layers.append(tuple(row_states))

    previous_costs: list[float] = []
    backpointers: list[list[int]] = []

    left_boundary_offset = (
        search_result.frame_a_origin_yz_profile_mm[window_start - 1]
        if window_start > 0
        else (0.0, 0.0)
    )
    right_boundary_offset = (
        search_result.frame_a_origin_yz_profile_mm[window_end + 1]
        if window_end + 1 < len(search_result.frame_a_origin_yz_profile_mm)
        else (0.0, 0.0)
    )
    for state in window_layers[0]:
        if not _passes_offset_step_constraint(
            left_boundary_offset,
            (state.dy_mm, state.dz_mm),
            limit_mm=offset_step_limit_mm,
        ):
            previous_costs.append(math.inf)
            continue
        if left_boundary_candidates:
            transition_cost = min(
                _candidate_transition_penalty(
                    boundary_candidate,
                    state.candidate,
                    optimizer_settings,
                )
                for boundary_candidate in left_boundary_candidates
            )
        else:
            transition_cost = optimizer_settings.start_transition_weight * _joint_transition_penalty(
                start_joints,
                state.candidate.joints,
                optimizer_settings,
            )
        previous_costs.append(
            transition_cost
            + _window_state_node_cost(state)
            + _offset_transition_penalty(left_boundary_offset, (state.dy_mm, state.dz_mm))
        )

    for layer_index in range(1, len(window_layers)):
        current_layer = window_layers[layer_index]
        previous_layer = window_layers[layer_index - 1]
        current_costs = [math.inf] * len(current_layer)
        current_backpointers = [-1] * len(current_layer)

        for current_index, current_state in enumerate(current_layer):
            best_cost = math.inf
            best_previous_index = -1
            for previous_index, previous_state in enumerate(previous_layer):
                if not math.isfinite(previous_costs[previous_index]):
                    continue
                if not _passes_offset_step_constraint(
                    (previous_state.dy_mm, previous_state.dz_mm),
                    (current_state.dy_mm, current_state.dz_mm),
                    limit_mm=offset_step_limit_mm,
                ):
                    continue
                transition_cost = _candidate_transition_penalty(
                    previous_state.candidate,
                    current_state.candidate,
                    optimizer_settings,
                )
                transition_cost += _offset_transition_penalty(
                    (previous_state.dy_mm, previous_state.dz_mm),
                    (current_state.dy_mm, current_state.dz_mm),
                )
                total_cost = (
                    previous_costs[previous_index]
                    + transition_cost
                    + _window_state_node_cost(current_state)
                )
                if total_cost < best_cost:
                    best_cost = total_cost
                    best_previous_index = previous_index

            current_costs[current_index] = best_cost
            current_backpointers[current_index] = best_previous_index

        if not any(math.isfinite(cost) for cost in current_costs):
            return None

        previous_costs = current_costs
        backpointers.append(current_backpointers)

    best_total_cost = math.inf
    best_last_index = -1
    for state_index, state in enumerate(window_layers[-1]):
        if not math.isfinite(previous_costs[state_index]):
            continue
        if not _passes_offset_step_constraint(
            (state.dy_mm, state.dz_mm),
            right_boundary_offset,
            limit_mm=offset_step_limit_mm,
        ):
            continue
        transition_cost = _offset_transition_penalty(
            (state.dy_mm, state.dz_mm),
            right_boundary_offset,
        )
        if right_boundary_candidates:
            transition_cost += min(
                _candidate_transition_penalty(
                    state.candidate,
                    boundary_candidate,
                    optimizer_settings,
                )
                for boundary_candidate in right_boundary_candidates
            )
        total_cost = previous_costs[state_index] + transition_cost
        if total_cost < best_total_cost:
            best_total_cost = total_cost
            best_last_index = state_index

    if best_last_index < 0:
        return None

    chosen_states = [window_layers[-1][best_last_index]]
    for layer_index in range(len(window_layers) - 2, -1, -1):
        best_last_index = backpointers[layer_index][best_last_index]
        chosen_states.append(window_layers[layer_index][best_last_index])
    chosen_states.reverse()

    updated_profile = list(search_result.frame_a_origin_yz_profile_mm)
    for row_index, state in zip(range(window_start, window_end + 1), chosen_states):
        updated_profile[row_index] = (float(state.dy_mm), float(state.dz_mm))
    return tuple(updated_profile)


_CONFIG_SWITCH_MIN_JOINT_DELTA_DEG = 5.0  # mirrors path_optimizer default


def _collect_problem_segments(
    selected_path: Sequence[_IKCandidate],
    *,
    bridge_trigger_joint_delta_deg: float,
    max_segments: int | None = None,
) -> tuple[tuple[int, bool, float, float], ...]:
    segments: list[tuple[int, bool, float, float]] = []
    for segment_index, (previous_candidate, current_candidate) in enumerate(
        zip(selected_path, selected_path[1:])
    ):
        max_joint_delta = max(
            (
                abs(current - previous)
                for previous, current in zip(previous_candidate.joints, current_candidate.joints)
            ),
            default=0.0,
        )
        mean_joint_delta = _mean_abs_joint_delta(
            previous_candidate.joints,
            current_candidate.joints,
        )
        config_changed = previous_candidate.config_flags != current_candidate.config_flags
        # Only treat config changes as problems when the joint step is meaningful
        # (>= _CONFIG_SWITCH_MIN_JOINT_DELTA_DEG).  Near-zero config "flips"
        # caused by a bit crossing zero (e.g. wrist flip near J5≈0°) are benign
        # and should not trigger expensive window-repair passes.
        benign_wrist_flip = _is_benign_wrist_singularity_config_change(
            previous_candidate,
            current_candidate,
            max_joint_delta_deg=max_joint_delta,
            bridge_trigger_joint_delta_deg=bridge_trigger_joint_delta_deg,
        )
        meaningful_config_problem = (
            config_changed
            and not benign_wrist_flip
            and max_joint_delta >= _CONFIG_SWITCH_MIN_JOINT_DELTA_DEG
        )
        if meaningful_config_problem or max_joint_delta > bridge_trigger_joint_delta_deg:
            segments.append((segment_index, config_changed, max_joint_delta, mean_joint_delta))

    if not segments:
        return ()

    def sort_key(item: tuple[int, bool, float, float]) -> tuple[float, ...]:
        # Keep a deterministic tie-breaker on segment index so partial selection
        # is stable and matches full-sort behavior.
        return (
            -int(item[1]),
            -float(item[2]),
            -float(item[3]),
            float(item[0]),
        )

    if max_segments is None:
        segments.sort(key=sort_key)
        return tuple(segments)

    limit = max(0, int(max_segments))
    if limit <= 0:
        return ()
    if len(segments) <= limit:
        segments.sort(key=sort_key)
        return tuple(segments)
    return tuple(heapq.nsmallest(limit, segments, key=sort_key))


def _format_focus_segment_report(search_result: _PathSearchResult) -> str:
    lines = ["Focused segment report:"]
    for start_label, end_label in (("372", "373"), ("380", "381")):
        lines.extend(_describe_focus_pair(search_result, start_label, end_label))
    return "\n".join(lines)


def _format_failure_diagnostics(
    search_result: _PathSearchResult,
    *,
    bridge_trigger_joint_delta_deg: float,
) -> str:
    lines = [
        "No strictly acceptable continuous path was found under the fixed Frame-A orientation "
        "and the allowed per-point Frame-2 Y/Z origin optimization.",
        (
            "Summary: "
            f"ik_empty_rows={search_result.ik_empty_row_count}, "
            f"config_switches={search_result.config_switches}, "
            f"bridge_like_segments={search_result.bridge_like_segments}, "
            f"worst_joint_step={search_result.worst_joint_step_deg:.3f} deg."
        ),
    ]

    if search_result.ik_empty_row_count > 0:
        empty_rows = [
            search_result.row_labels[index]
            for index, layer in enumerate(search_result.ik_layers)
            if not layer.candidates
        ]
        lines.append("IK-empty rows: " + ", ".join(empty_rows[:12]))

    problem_segments = _collect_problem_segments(
        search_result.selected_path,
        bridge_trigger_joint_delta_deg=bridge_trigger_joint_delta_deg,
        max_segments=8,
    )
    if problem_segments:
        lines.append("Failing segments:")
        for segment_index, config_changed, max_joint_delta, mean_joint_delta in problem_segments[:8]:
            left_label = search_result.row_labels[segment_index]
            right_label = search_result.row_labels[segment_index + 1]
            left_profile = search_result.frame_a_origin_yz_profile_mm[segment_index]
            right_profile = search_result.frame_a_origin_yz_profile_mm[segment_index + 1]
            left_layer = search_result.ik_layers[segment_index]
            right_layer = search_result.ik_layers[segment_index + 1]
            left_families = len({candidate.config_flags for candidate in left_layer.candidates})
            right_families = len({candidate.config_flags for candidate in right_layer.candidates})
            lines.append(
                f"  {left_label}->{right_label}: "
                f"config_changed={config_changed}, "
                f"max_joint_delta={max_joint_delta:.3f} deg, "
                f"mean_joint_delta={mean_joint_delta:.3f} deg, "
                f"profile=({left_profile[0]:.3f}, {left_profile[1]:.3f}) -> "
                f"({right_profile[0]:.3f}, {right_profile[1]:.3f}) mm, "
                f"candidate_families={left_families}->{right_families}, "
                f"candidate_counts={len(left_layer.candidates)}->{len(right_layer.candidates)}"
            )

    lines.append(_format_focus_segment_report(search_result))
    return "\n".join(lines)


def _build_local_seed_joints(
    search_result: _PathSearchResult,
    window_start: int,
    window_end: int,
) -> tuple[tuple[float, ...], ...]:
    seeds: list[tuple[float, ...]] = []
    seen: set[tuple[float, ...]] = set()

    def append_seed(candidate: _IKCandidate | None) -> None:
        if candidate is None:
            return
        seed = tuple(float(value) for value in candidate.joints)
        if seed in seen:
            return
        seen.add(seed)
        seeds.append(seed)

    for row_index in range(max(0, window_start - 1), min(len(search_result.selected_path), window_end + 2)):
        append_seed(search_result.selected_path[row_index])

    for layer in search_result.ik_layers[max(0, window_start - 1) : min(len(search_result.ik_layers), window_end + 2)]:
        for candidate in layer.candidates[:4]:
            append_seed(candidate)

    return tuple(seeds)


def _build_window_offset_states(
    current_offset: tuple[float, float],
    *,
    max_abs_offset_mm: float,
    step_mm: float,
) -> tuple[tuple[float, float], ...]:
    dy0, dz0 = (float(current_offset[0]), float(current_offset[1]))
    multipliers = (-3, -2, -1, 0, 1, 2, 3)
    states: list[tuple[float, float]] = []
    seen: set[tuple[float, float]] = set()

    def append_state(dy_mm: float, dz_mm: float) -> None:
        clipped_state = (
            round(max(-max_abs_offset_mm, min(max_abs_offset_mm, dy_mm)), 6),
            round(max(-max_abs_offset_mm, min(max_abs_offset_mm, dz_mm)), 6),
        )
        if clipped_state in seen:
            return
        seen.add(clipped_state)
        states.append(clipped_state)

    append_state(dy0, dz0)
    for dy_multiplier in multipliers:
        for dz_multiplier in multipliers:
            append_state(dy0 + dy_multiplier * step_mm, dz0 + dz_multiplier * step_mm)

    for dy_edge in (-max_abs_offset_mm, 0.0, max_abs_offset_mm):
        for dz_edge in (-max_abs_offset_mm, 0.0, max_abs_offset_mm):
            append_state(dy_edge, dz_edge)

    states.sort(key=lambda value: (abs(value[0] - dy0) + abs(value[1] - dz0), abs(value[0]) + abs(value[1]), value))
    return tuple(states)


def _window_state_node_cost(state: _WindowState) -> float:
    return (
        0.22 * (state.candidate.joint_limit_penalty + state.candidate.singularity_penalty)
        + 0.20 * math.hypot(state.dy_mm, state.dz_mm)
    )


def _offset_transition_penalty(
    previous_offset: tuple[float, float],
    current_offset: tuple[float, float],
) -> float:
    return 6.0 * math.hypot(
        float(current_offset[0]) - float(previous_offset[0]),
        float(current_offset[1]) - float(previous_offset[1]),
    )


def _passes_offset_step_constraint(
    previous_offset: tuple[float, float],
    current_offset: tuple[float, float],
    *,
    limit_mm: float,
) -> bool:
    return (
        abs(float(current_offset[0]) - float(previous_offset[0])) <= limit_mm + 1e-9
        and abs(float(current_offset[1]) - float(previous_offset[1])) <= limit_mm + 1e-9
    )


def _describe_focus_pair(
    search_result: _PathSearchResult,
    start_label: str,
    end_label: str,
) -> list[str]:
    labels = list(search_result.row_labels)
    if start_label not in labels or end_label not in labels:
        return [f"  {start_label}->{end_label}: labels not present in current path."]

    start_index = labels.index(start_label)
    end_index = labels.index(end_label)
    if end_index <= start_index:
        return [f"  {start_label}->{end_label}: label order is not monotonic in current path."]

    lines: list[str] = []
    if end_index == start_index + 1:
        lines.append(_format_single_segment_line(search_result, start_index))
        return lines

    lines.append(
        f"  {start_label}->{end_label}: {end_index - start_index - 1} inserted transition sample(s) in between."
    )
    for segment_index in range(start_index, end_index):
        lines.append(_format_single_segment_line(search_result, segment_index))
    return lines


def _format_single_segment_line(
    search_result: _PathSearchResult,
    segment_index: int,
) -> str:
    left_label = search_result.row_labels[segment_index]
    right_label = search_result.row_labels[segment_index + 1]
    if segment_index + 1 >= len(search_result.selected_path):
        return f"    {left_label}->{right_label}: no selected joint path available."

    previous_candidate = search_result.selected_path[segment_index]
    current_candidate = search_result.selected_path[segment_index + 1]
    joint_deltas = [
        abs(current - previous)
        for previous, current in zip(previous_candidate.joints, current_candidate.joints)
    ]
    max_joint_delta = max(joint_deltas, default=0.0)
    previous_profile = search_result.frame_a_origin_yz_profile_mm[segment_index]
    current_profile = search_result.frame_a_origin_yz_profile_mm[segment_index + 1]
    return (
        f"    {left_label}->{right_label}: "
        f"config={previous_candidate.config_flags}->{current_candidate.config_flags}, "
        f"max_joint_delta={max_joint_delta:.3f} deg, "
        f"profile=({previous_profile[0]:.3f}, {previous_profile[1]:.3f}) -> "
        f"({current_profile[0]:.3f}, {current_profile[1]:.3f}) mm"
    )
