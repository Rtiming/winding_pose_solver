from __future__ import annotations

from typing import Sequence

from src.core.geometry import (
    _build_pose,
    _clip_seed_to_limits,
    _extract_joint_tuple,
    _is_within_joint_limits,
    _passes_user_joint_constraints,
    _trim_joint_vector,
)
from src.core.types import _IKCandidate, _IKLayer, _PathOptimizerSettings
from src.search.path_optimizer import (
    _ROBOT_CONFIG_FLAGS_CACHE,
    _joint_limit_penalty,
    _singularity_penalty,
)

_CONFIG_FLAG_COUNT = 3
_IK_DEDUP_DECIMALS = 6
_POSE_CACHE_DECIMALS = 9
_IK_CANDIDATE_CACHE_MAX_SIZE = 100000

_IK_CANDIDATE_CACHE: dict[tuple[object, ...], tuple[_IKCandidate, ...]] = {}
_IK_CANDIDATE_CACHE_HITS = 0
_IK_CANDIDATE_CACHE_MISSES = 0


def reset_ik_candidate_collection_cache() -> None:
    _IK_CANDIDATE_CACHE.clear()
    global _IK_CANDIDATE_CACHE_HITS, _IK_CANDIDATE_CACHE_MISSES
    _IK_CANDIDATE_CACHE_HITS = 0
    _IK_CANDIDATE_CACHE_MISSES = 0


def ik_candidate_collection_cache_stats() -> dict[str, int]:
    return {
        "entries": len(_IK_CANDIDATE_CACHE),
        "hits": _IK_CANDIDATE_CACHE_HITS,
        "misses": _IK_CANDIDATE_CACHE_MISSES,
    }


def _pose_cache_key(pose) -> tuple[float, ...] | tuple[int]:
    try:
        return tuple(
            round(float(pose[row_index, column_index]), _POSE_CACHE_DECIMALS)
            for row_index in range(4)
            for column_index in range(4)
        )
    except Exception:
        pass

    rows = getattr(pose, "rows", None)
    if rows is not None:
        try:
            return tuple(
                round(float(value), _POSE_CACHE_DECIMALS)
                for row in rows
                for value in row
            )
        except Exception:
            pass

    return (id(pose),)


def _seed_cache_key(
    seed_joints: Sequence[tuple[float, ...]],
) -> tuple[tuple[float, ...], ...]:
    return tuple(
        tuple(round(float(value), _IK_DEDUP_DECIMALS) for value in seed)
        for seed in seed_joints
    )


def _candidate_collection_cache_key(
    *,
    robot,
    pose,
    lower_limits: tuple[float, ...],
    upper_limits: tuple[float, ...],
    seed_joints: Sequence[tuple[float, ...]],
    joint_count: int,
    optimizer_settings: _PathOptimizerSettings,
    a1_lower_deg: float,
    a1_upper_deg: float,
    a2_max_deg: float,
    joint_constraint_tolerance_deg: float,
) -> tuple[object, ...]:
    return (
        id(robot),
        _pose_cache_key(pose),
        lower_limits,
        upper_limits,
        int(joint_count),
        optimizer_settings,
        round(float(a1_lower_deg), _IK_DEDUP_DECIMALS),
        round(float(a1_upper_deg), _IK_DEDUP_DECIMALS),
        round(float(a2_max_deg), _IK_DEDUP_DECIMALS),
        round(float(joint_constraint_tolerance_deg), _POSE_CACHE_DECIMALS),
        None if getattr(robot, "ik_seed_invariant", False) else _seed_cache_key(seed_joints),
    )


def _build_ik_layers(
    pose_rows: Sequence[dict[str, float]],
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
    seed_joints_override: Sequence[tuple[float, ...]] | None = None,
    lower_limits_override: Sequence[float] | None = None,
    upper_limits_override: Sequence[float] | None = None,
    log_summary: bool = True,
) -> list[_IKLayer]:
    if lower_limits_override is None or upper_limits_override is None:
        lower_limits_raw, upper_limits_raw, _ = robot.JointLimits()
        lower_limits = _trim_joint_vector(lower_limits_raw.list(), joint_count)
        upper_limits = _trim_joint_vector(upper_limits_raw.list(), joint_count)
    else:
        lower_limits = tuple(float(value) for value in lower_limits_override[:joint_count])
        upper_limits = tuple(float(value) for value in upper_limits_override[:joint_count])

    if seed_joints_override is None:
        if getattr(robot, "ik_seed_invariant", False):
            seed_joints = ()
        else:
            seed_joints = _build_seed_joint_strategies(
                robot=robot,
                lower_limits=lower_limits,
                upper_limits=upper_limits,
                joint_count=joint_count,
            )
    else:
        seed_joints = tuple(seed_joints_override)

    ik_layers: list[_IKLayer] = []
    total_candidates = 0
    for row_index, row in enumerate(pose_rows):
        pose = _build_pose(row, mat_type)
        candidates = _collect_ik_candidates(
            robot,
            pose,
            tool_pose=tool_pose,
            reference_pose=reference_pose,
            lower_limits=lower_limits,
            upper_limits=upper_limits,
            seed_joints=seed_joints,
            joint_count=joint_count,
            optimizer_settings=optimizer_settings,
            a1_lower_deg=a1_lower_deg,
            a1_upper_deg=a1_upper_deg,
            a2_max_deg=a2_max_deg,
            joint_constraint_tolerance_deg=joint_constraint_tolerance_deg,
        )
        if not candidates:
            raise RuntimeError(
                f"No IK candidates remain for CSV pose row {row_index} after applying hard "
                f"constraints: A1 in [{a1_lower_deg:.1f}, {a1_upper_deg:.1f}] deg, "
                f"A2 < {a2_max_deg:.1f} deg."
            )

        ik_layers.append(_IKLayer(pose=pose, candidates=tuple(candidates)))
        total_candidates += len(candidates)

    if log_summary:
        print(
            f"Collected {total_candidates} IK candidate(s) across {len(ik_layers)} target pose(s)."
        )
    return ik_layers


def _collect_ik_candidates(
    robot,
    pose,
    *,
    tool_pose,
    reference_pose,
    lower_limits: tuple[float, ...],
    upper_limits: tuple[float, ...],
    seed_joints: Sequence[tuple[float, ...]],
    joint_count: int,
    optimizer_settings: _PathOptimizerSettings,
    a1_lower_deg: float,
    a1_upper_deg: float,
    a2_max_deg: float,
    joint_constraint_tolerance_deg: float,
) -> tuple[_IKCandidate, ...]:
    global _IK_CANDIDATE_CACHE_HITS, _IK_CANDIDATE_CACHE_MISSES

    cache_key = _candidate_collection_cache_key(
        robot=robot,
        pose=pose,
        lower_limits=lower_limits,
        upper_limits=upper_limits,
        seed_joints=seed_joints,
        joint_count=joint_count,
        optimizer_settings=optimizer_settings,
        a1_lower_deg=a1_lower_deg,
        a1_upper_deg=a1_upper_deg,
        a2_max_deg=a2_max_deg,
        joint_constraint_tolerance_deg=joint_constraint_tolerance_deg,
    )
    cached_candidates = _IK_CANDIDATE_CACHE.get(cache_key)
    if cached_candidates is not None:
        _IK_CANDIDATE_CACHE_HITS += 1
        return cached_candidates

    _IK_CANDIDATE_CACHE_MISSES += 1
    candidates: list[_IKCandidate] = []
    seen: set[tuple[float, ...]] = set()

    solve_all_filtered = getattr(robot, "SolveIK_AllFiltered", None)
    if callable(solve_all_filtered):
        all_solutions = solve_all_filtered(
            pose,
            tool_pose,
            reference_pose,
            a1_lower_deg=a1_lower_deg,
            a1_upper_deg=a1_upper_deg,
            a2_max_deg=a2_max_deg,
            tolerance_deg=joint_constraint_tolerance_deg,
        )
    else:
        all_solutions = robot.SolveIK_All(pose, tool_pose, reference_pose)
    for raw_solution in all_solutions:
        _append_candidate_if_unique(
            candidates,
            seen,
            robot=robot,
            raw_joints=raw_solution,
            lower_limits=lower_limits,
            upper_limits=upper_limits,
            joint_count=joint_count,
            optimizer_settings=optimizer_settings,
            a1_lower_deg=a1_lower_deg,
            a1_upper_deg=a1_upper_deg,
            a2_max_deg=a2_max_deg,
            joint_constraint_tolerance_deg=joint_constraint_tolerance_deg,
        )

    if not getattr(robot, "ik_seed_invariant", False):
        for seed in seed_joints:
            raw_solution = robot.SolveIK(pose, list(seed), tool_pose, reference_pose)
            _append_candidate_if_unique(
                candidates,
                seen,
                robot=robot,
                raw_joints=raw_solution,
                lower_limits=lower_limits,
                upper_limits=upper_limits,
                joint_count=joint_count,
                optimizer_settings=optimizer_settings,
                a1_lower_deg=a1_lower_deg,
                a1_upper_deg=a1_upper_deg,
                a2_max_deg=a2_max_deg,
                joint_constraint_tolerance_deg=joint_constraint_tolerance_deg,
            )

    candidates.sort(
        key=lambda candidate: (
            candidate.config_flags,
            candidate.joint_limit_penalty + candidate.singularity_penalty,
            candidate.joints,
        )
    )

    # Keep enough candidates for fixed-point fallback to explore alternate
    # branches, while still bounding the O(N*K^2) DP cost.
    max_candidates_per_config_family = max(
        1,
        int(getattr(optimizer_settings, "ik_max_candidates_per_config_family", 4)),
    )
    if len(candidates) > max_candidates_per_config_family * 2:
        from itertools import groupby
        filtered: list[_IKCandidate] = []
        for flags, group in groupby(candidates, key=lambda c: c.config_flags):
            family = sorted(
                group,
                key=lambda c: c.joint_limit_penalty + c.singularity_penalty,
            )
            filtered.extend(family[:max_candidates_per_config_family])
        candidates = filtered
        candidates.sort(
            key=lambda candidate: (
                candidate.config_flags,
                candidate.joint_limit_penalty + candidate.singularity_penalty,
                candidate.joints,
            )
        )

    cached_result = tuple(candidates)
    if len(_IK_CANDIDATE_CACHE) >= _IK_CANDIDATE_CACHE_MAX_SIZE:
        _IK_CANDIDATE_CACHE.pop(next(iter(_IK_CANDIDATE_CACHE)))
    _IK_CANDIDATE_CACHE[cache_key] = cached_result
    return cached_result


def _append_candidate_if_unique(
    candidates: list[_IKCandidate],
    seen: set[tuple[float, ...]],
    *,
    robot,
    raw_joints,
    lower_limits: tuple[float, ...],
    upper_limits: tuple[float, ...],
    joint_count: int,
    optimizer_settings: _PathOptimizerSettings,
    a1_lower_deg: float,
    a1_upper_deg: float,
    a2_max_deg: float,
    joint_constraint_tolerance_deg: float,
) -> None:
    joints = _extract_joint_tuple(raw_joints, joint_count)
    if not joints:
        return

    if not _is_within_joint_limits(joints, lower_limits, upper_limits):
        return

    if not _passes_user_joint_constraints(
        joints,
        a1_lower_deg=a1_lower_deg,
        a1_upper_deg=a1_upper_deg,
        a2_max_deg=a2_max_deg,
        tolerance_deg=joint_constraint_tolerance_deg,
    ):
        return

    dedup_key = tuple(round(value, _IK_DEDUP_DECIMALS) for value in joints)
    if dedup_key in seen:
        return
    seen.add(dedup_key)

    config_flags = _candidate_config_flags(robot, joints)
    branch_id = _candidate_branch_id(robot, joints)
    candidates.append(
        _IKCandidate(
            joints=joints,
            config_flags=config_flags,
            joint_limit_penalty=_joint_limit_penalty(
                joints,
                lower_limits,
                upper_limits,
                optimizer_settings,
            ),
            singularity_penalty=_singularity_penalty(robot, joints, optimizer_settings),
            branch_id=branch_id,
        )
    )


def _candidate_config_flags(robot, joints: tuple[float, ...]) -> tuple[int, ...]:
    cache_key = (id(robot), joints)
    cached_flags = _ROBOT_CONFIG_FLAGS_CACHE.get(cache_key)
    if cached_flags is not None:
        return cached_flags

    config_values = robot.JointsConfig(list(joints)).list()
    config_flags = tuple(int(round(value)) for value in config_values[:_CONFIG_FLAG_COUNT])
    _ROBOT_CONFIG_FLAGS_CACHE[cache_key] = config_flags
    return config_flags


def _candidate_branch_id(
    robot,
    joints: tuple[float, ...],
) -> tuple[int, ...] | None:
    joint_branch_id = getattr(robot, "JointBranchId", None)
    if not callable(joint_branch_id):
        return None

    try:
        branch_id = joint_branch_id(list(joints))
    except Exception:
        return None
    if branch_id is None:
        return None
    try:
        return tuple(int(value) for value in branch_id)
    except Exception:
        return None


def _build_seed_joint_strategies(
    *,
    robot,
    lower_limits: tuple[float, ...],
    upper_limits: tuple[float, ...],
    joint_count: int,
) -> tuple[tuple[float, ...], ...]:
    seeds: list[tuple[float, ...]] = []

    current_joints = _trim_joint_vector(robot.Joints().list(), joint_count)
    if current_joints:
        seeds.append(current_joints)

    home_joints = _trim_joint_vector(robot.JointsHome().list(), joint_count)
    if home_joints:
        seeds.append(home_joints)

    midpoint = tuple((lower + upper) * 0.5 for lower, upper in zip(lower_limits, upper_limits))
    seeds.append(midpoint)
    seeds.append(_clip_seed_to_limits((0.0,) * joint_count, lower_limits, upper_limits))

    for ratio in (0.15, 0.50, 0.85):
        seeds.append(
            tuple(
                lower + (upper - lower) * ratio
                for lower, upper in zip(lower_limits, upper_limits)
            )
        )

    if joint_count >= 6:
        for joint5 in (-90.0, 90.0):
            for joint6 in (-180.0, 0.0, 180.0):
                wrist_seed = list(midpoint)
                wrist_seed[4] = joint5
                wrist_seed[5] = joint6
                seeds.append(_clip_seed_to_limits(wrist_seed, lower_limits, upper_limits))

    unique_seeds: list[tuple[float, ...]] = []
    seen: set[tuple[float, ...]] = set()
    for seed in seeds:
        rounded = tuple(round(value, _IK_DEDUP_DECIMALS) for value in seed)
        if rounded in seen:
            continue
        seen.add(rounded)
        unique_seeds.append(seed)
    return tuple(unique_seeds)


def _append_seed_if_unique(
    seeds: list[tuple[float, ...]],
    seen: set[tuple[float, ...]],
    seed: Sequence[float],
) -> None:
    normalized_seed = tuple(float(value) for value in seed)
    dedup_key = tuple(round(value, _IK_DEDUP_DECIMALS) for value in normalized_seed)
    if dedup_key in seen:
        return
    seen.add(dedup_key)
    seeds.append(normalized_seed)
