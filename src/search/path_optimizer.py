from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache
from typing import Sequence

from src.core.types import (
    _IKCandidate,
    _IKLayer,
    _PathOptimizerSettings,
)
from src.core.geometry import (
    _normalize_step_limits,
    _pose_translation_distance_mm,
    _pose_rotation_distance_deg,
    _trim_joint_vector,
)


# ??joint tuple ??? RoboDK ?????????????????????????????????????????????
_ROBOT_CONFIG_FLAGS_CACHE: dict[tuple[int, tuple[float, ...]], tuple[int, ...]] = {}
_ROBOT_SINGULARITY_PENALTY_CACHE: dict[
    tuple[int, tuple[float, ...], "_PathOptimizerSettings"],
    float,
] = {}
_ROBOT_SINGULARITY_PENALTY_CACHE_MAX_ENTRIES = 131072

_CLOSED_WINDING_TERMINAL_PREFIX_JOINT_COUNT = 5


@dataclass(frozen=True)
class _JointPairMetrics:
    transition_cost: float
    max_joint_delta_deg: float
    joint6_delta_deg: float
    min_abs_joint5_deg: float
    passes_joint_continuity: bool
    passes_preferred_continuity: bool


@lru_cache(maxsize=131072)
def _joint_pair_metrics(
    previous_joints: tuple[float, ...],
    current_joints: tuple[float, ...],
    optimizer_settings: _PathOptimizerSettings,
) -> _JointPairMetrics:
    """Cache the shared per-pair joint math used by DP and guidance checks."""

    deltas = tuple(
        current - previous
        for previous, current in zip(previous_joints, current_joints)
    )
    abs_deltas = tuple(abs(delta) for delta in deltas)

    abs_cost = sum(
        weight * abs_delta
        for weight, abs_delta in zip(optimizer_settings.joint_delta_weights, abs_deltas)
    )
    squared_cost = sum(
        weight * delta * delta
        for weight, delta in zip(optimizer_settings.joint_delta_weights, deltas)
    )
    transition_cost = (
        optimizer_settings.abs_delta_weight * abs_cost
        + optimizer_settings.squared_delta_weight * squared_cost
    )

    max_joint_delta_deg = max(abs_deltas, default=0.0)
    if max_joint_delta_deg > optimizer_settings.large_jump_threshold_deg:
        excess = max_joint_delta_deg - optimizer_settings.large_jump_threshold_deg
        transition_cost += optimizer_settings.large_jump_penalty_weight * excess * excess

    if optimizer_settings.enable_joint_continuity_constraint:
        passes_joint_continuity = all(
            delta_deg <= limit_deg
            for delta_deg, limit_deg in zip(
                abs_deltas,
                optimizer_settings.max_joint_step_deg,
            )
        )
    else:
        passes_joint_continuity = True

    passes_preferred_continuity = all(
        delta_deg <= limit_deg
        for delta_deg, limit_deg in zip(
            abs_deltas,
            optimizer_settings.preferred_joint_step_deg,
        )
    )

    joint6_delta_deg = 0.0
    if len(previous_joints) >= 6 and len(current_joints) >= 6:
        joint6_delta_deg = abs(current_joints[5] - previous_joints[5])

    min_abs_joint5_deg = math.inf
    if len(previous_joints) >= 5 and len(current_joints) >= 5:
        min_abs_joint5_deg = min(abs(previous_joints[4]), abs(current_joints[4]))

    return _JointPairMetrics(
        transition_cost=transition_cost,
        max_joint_delta_deg=max_joint_delta_deg,
        joint6_delta_deg=joint6_delta_deg,
        min_abs_joint5_deg=min_abs_joint5_deg,
        passes_joint_continuity=passes_joint_continuity,
        passes_preferred_continuity=passes_preferred_continuity,
    )


def _candidate_lineage_key(candidate: _IKCandidate) -> tuple[int, ...]:
    if candidate.branch_id is None:
        return candidate.config_flags
    return (*candidate.config_flags, *candidate.branch_id)


def _candidate_joint_prefix_key(
    candidate: _IKCandidate,
    joint_count: int,
) -> tuple[float, ...] | None:
    if len(candidate.joints) < joint_count:
        return None
    return tuple(float(value) for value in candidate.joints[:joint_count])


def _group_candidates_by_joint_prefix(
    candidates: Sequence[_IKCandidate],
    *,
    joint_count: int,
) -> dict[tuple[float, ...], tuple[_IKCandidate, ...]]:
    grouped: dict[tuple[float, ...], list[_IKCandidate]] = {}
    for candidate in candidates:
        prefix_key = _candidate_joint_prefix_key(candidate, joint_count)
        if prefix_key is None:
            continue
        grouped.setdefault(prefix_key, []).append(candidate)
    return {prefix_key: tuple(group) for prefix_key, group in grouped.items()}


def _joint_tuple(values: Sequence[float]) -> tuple[float, ...]:
    if isinstance(values, tuple):
        return values
    return tuple(values)


def _build_optimizer_settings(
    joint_count: int,
    motion_settings,
) -> _PathOptimizerSettings:
    """??????????????DP ????????"""

    base_weights = (1.0, 1.0, 1.1, 1.4, 2.0, 2.7)
    preferred_step_defaults = (5.0, 5.0, 5.0, 25.0, 25.0, 25.0)
    if joint_count <= len(base_weights):
        weights = base_weights[:joint_count]
    else:
        weights = base_weights + (base_weights[-1],) * (joint_count - len(base_weights))

    hard_step_limits = _normalize_step_limits(motion_settings.max_joint_step_deg, joint_count)
    if joint_count <= len(preferred_step_defaults):
        preferred_step_limits = preferred_step_defaults[:joint_count]
    else:
        preferred_step_limits = preferred_step_defaults + (
            preferred_step_defaults[-1],
        ) * (joint_count - len(preferred_step_defaults))

    # "????????????????????????????????????????????????????????????????
    preferred_step_limits = tuple(
        min(hard_limit, preferred_limit)
        for hard_limit, preferred_limit in zip(hard_step_limits, preferred_step_limits)
    )

    return _PathOptimizerSettings(
        joint_delta_weights=weights,
        enable_joint_continuity_constraint=motion_settings.enable_joint_continuity_constraint,
        max_joint_step_deg=hard_step_limits,
        ik_max_candidates_per_config_family=max(
            1,
            int(getattr(motion_settings, "ik_max_candidates_per_config_family", 4)),
        ),
        use_guided_config_path=bool(
            getattr(motion_settings, "use_guided_config_path", True)
        ),
        preferred_joint_step_deg=preferred_step_limits,
        wrist_phase_lock_threshold_deg=motion_settings.wrist_phase_lock_threshold_deg,
        rear_switch_penalty=float(getattr(motion_settings, "rear_switch_penalty", 2000.0)),
        lower_switch_penalty=float(getattr(motion_settings, "lower_switch_penalty", 2000.0)),
        flip_switch_penalty=float(getattr(motion_settings, "flip_switch_penalty", 2000.0)),
        closed_path_joint6_turns=int(getattr(motion_settings, "closed_path_joint6_turns", 1)),
        closed_path_joint6_turn_tolerance_deg=float(
            getattr(motion_settings, "closed_path_joint6_turn_tolerance_deg", 1e-3)
        ),
        closed_path_single_config=bool(
            getattr(motion_settings, "closed_path_single_config", False)
        ),
        closed_path_locked_config_indices=tuple(
            int(index)
            for index in getattr(motion_settings, "closed_path_locked_config_indices", ())
        ),
        closed_path_joint6_direction_sample_count=int(
            getattr(motion_settings, "closed_path_joint6_direction_sample_count", 40)
        ),
        closed_path_joint6_direction_min_delta_deg=float(
            getattr(motion_settings, "closed_path_joint6_direction_min_delta_deg", 1.0)
        ),
    )


def _summarize_selected_path(
    selected_path: Sequence[_IKCandidate],
    *,
    bridge_trigger_joint_delta_deg: float,
    config_switch_min_joint_delta_deg: float = 5.0,
    benign_wrist_flip_abs_j5_deg: float = 12.0,
) -> tuple[int, int, float, float]:
    """??????????joint path ???????????????

    Config switches where max_joint_delta < config_switch_min_joint_delta_deg
    are treated as benign (e.g. a wrist-flip bit that crosses zero while J5 is
    near the wrist singularity).  Only meaningful config transitions ??those
    accompanied by a joint movement of at least config_switch_min_joint_delta_deg
    ??are counted against the path quality score.
    """

    if len(selected_path) <= 1:
        return 0, 0, 0.0, 0.0

    config_switches = 0
    bridge_like_segments = 0
    worst_joint_step_deg = 0.0
    mean_joint_step_sum = 0.0

    for previous_candidate, current_candidate in zip(selected_path, selected_path[1:]):
        joint_deltas = [
            abs(current - previous)
            for previous, current in zip(previous_candidate.joints, current_candidate.joints)
        ]
        max_joint_delta = max(joint_deltas, default=0.0)
        worst_joint_step_deg = max(worst_joint_step_deg, max_joint_delta)
        if joint_deltas:
            mean_joint_step_sum += sum(joint_deltas) / len(joint_deltas)

        config_changed = previous_candidate.config_flags != current_candidate.config_flags
        # Only count config switches that move joints by a meaningful amount.
        # Near-zero changes (e.g. wrist-flip at J5???) are benign and must not
        # be treated as bridge-like segments or invalidate the path.
        benign_wrist_flip = _is_benign_wrist_singularity_config_change(
            previous_candidate,
            current_candidate,
            max_joint_delta_deg=max_joint_delta,
            bridge_trigger_joint_delta_deg=bridge_trigger_joint_delta_deg,
            benign_wrist_flip_abs_j5_deg=benign_wrist_flip_abs_j5_deg,
        )
        meaningful_switch = (
            config_changed
            and not benign_wrist_flip
            and max_joint_delta >= config_switch_min_joint_delta_deg
        )
        if meaningful_switch:
            config_switches += 1
        if meaningful_switch or max_joint_delta > bridge_trigger_joint_delta_deg:
            bridge_like_segments += 1

    mean_joint_step_deg = mean_joint_step_sum / max(1, len(selected_path) - 1)
    return config_switches, bridge_like_segments, worst_joint_step_deg, mean_joint_step_deg


def _is_benign_wrist_singularity_config_change(
    previous_candidate: _IKCandidate,
    current_candidate: _IKCandidate,
    *,
    max_joint_delta_deg: float,
    bridge_trigger_joint_delta_deg: float,
    benign_wrist_flip_abs_j5_deg: float = 12.0,
) -> bool:
    """Return True for smooth wrist-flip-bit changes through J5~=0.

    This does not make A6 periodic in the middle of the path.  It only prevents
    a small, explicitly sampled wrist-singularity crossing from being counted as
    a configuration-family switch when the real per-axis joint deltas are already
    below the bridge/problem threshold.
    """

    previous_flags = previous_candidate.config_flags
    current_flags = current_candidate.config_flags
    if previous_flags == current_flags:
        return False
    if len(previous_flags) < 3 or len(current_flags) < 3:
        return False
    if previous_flags[:2] != current_flags[:2] or previous_flags[2] == current_flags[2]:
        return False
    if len(previous_candidate.joints) < 5 or len(current_candidate.joints) < 5:
        return False
    if max_joint_delta_deg > bridge_trigger_joint_delta_deg + 1e-9:
        return False

    previous_j5 = float(previous_candidate.joints[4])
    current_j5 = float(current_candidate.joints[4])
    if previous_j5 * current_j5 > 0.0:
        return False
    return max(abs(previous_j5), abs(current_j5)) <= benign_wrist_flip_abs_j5_deg + 1e-9


def _path_is_clean_enough_for_program_generation(
    selected_path: Sequence[_IKCandidate],
    *,
    bridge_trigger_joint_delta_deg: float,
    preferred_joint_step_deg: Sequence[float],
) -> bool:
    """??? exact path ??????????????????????????????????"""

    if not selected_path:
        return False

    from src.search.local_repair import _collect_problem_segments
    problem_segments = _collect_problem_segments(
        selected_path,
        bridge_trigger_joint_delta_deg=bridge_trigger_joint_delta_deg,
        max_segments=1,
    )
    if problem_segments:
        return False

    _config_switches, _bridge_like_segments, worst_joint_step_deg, _mean_joint_step_deg = (
        _summarize_selected_path(
            selected_path,
            bridge_trigger_joint_delta_deg=bridge_trigger_joint_delta_deg,
        )
    )
    preferred_limit = max(preferred_joint_step_deg, default=0.0)
    return worst_joint_step_deg <= preferred_limit + 1e-9


def _optimize_joint_path(
    ik_layers: Sequence[_IKLayer],
    *,
    robot,
    move_type: str,
    start_joints: tuple[float, ...],
    optimizer_settings: _PathOptimizerSettings,
    require_terminal_match_start: bool = False,
    selection_bridge_trigger_joint_delta_deg: float | None = None,
) -> tuple[list[_IKCandidate], float]:
    """?????????????????????????????????????????????"""

    if not ik_layers:
        return [], 0.0

    if require_terminal_match_start and len(ik_layers) > 1:
        return _optimize_closed_joint_path(
            ik_layers,
            robot=robot,
            move_type=move_type,
            start_joints=start_joints,
            optimizer_settings=optimizer_settings,
            selection_bridge_trigger_joint_delta_deg=selection_bridge_trigger_joint_delta_deg,
        )

    corridor_scores = _compute_candidate_corridor_scores(ik_layers, optimizer_settings)
    guided_config_path = None
    if bool(getattr(optimizer_settings, "use_guided_config_path", True)):
        guided_config_path = _build_guided_config_path(
            ik_layers,
            start_joints=start_joints,
            optimizer_settings=optimizer_settings,
        )
    if (
        guided_config_path is not None
        and not _guided_config_path_is_feasible(
            ik_layers,
            guided_config_path=guided_config_path,
            optimizer_settings=optimizer_settings,
        )
    ):
        guided_config_path = None

    # ??0 ?????? = ???????????????????????????? + ??????????????????
    previous_costs = []
    for candidate_index, candidate in enumerate(ik_layers[0].candidates):
        if guided_config_path is not None and candidate.config_flags != guided_config_path[0]:
            previous_costs.append(math.inf)
            continue
        previous_costs.append(
            _candidate_node_cost(
                candidate,
                corridor_score=corridor_scores[0][candidate_index],
                optimizer_settings=optimizer_settings,
            )
            + optimizer_settings.start_transition_weight
            * _joint_transition_penalty(start_joints, candidate.joints, optimizer_settings)
        )
    backpointers: list[list[int]] = []

    previous_layer = ik_layers[0]
    for layer_index in range(1, len(ik_layers)):
        current_layer = ik_layers[layer_index]
        previous_guided_flags = None
        current_guided_flags = None
        if guided_config_path is not None:
            previous_guided_flags = guided_config_path[layer_index - 1]
            current_guided_flags = guided_config_path[layer_index]

        active_previous_candidates = [
            (previous_index, previous_candidate, previous_costs[previous_index])
            for previous_index, previous_candidate in enumerate(previous_layer.candidates)
            if math.isfinite(previous_costs[previous_index])
            and (
                previous_guided_flags is None
                or previous_candidate.config_flags == previous_guided_flags
            )
        ]

        # MoveL ????????"??????????????????????????????
        # ??????????????????????????????????????????????
        move_l_cache: list[tuple[float, tuple[float, ...] | None]] | None = None
        if move_type == "MoveL":
            move_l_cache = [
                _evaluate_move_l_transition(
                    robot,
                    start_joints=candidate.joints,
                    target_pose=current_layer.pose,
                    joint_count=len(candidate.joints),
                    optimizer_settings=optimizer_settings,
                )
                for candidate in previous_layer.candidates
            ]

        current_costs = [math.inf] * len(current_layer.candidates)
        current_backpointers = [-1] * len(current_layer.candidates)

        for current_index, current_candidate in enumerate(current_layer.candidates):
            if (
                current_guided_flags is not None
                and current_candidate.config_flags != current_guided_flags
            ):
                continue
            node_cost = _candidate_node_cost(
                current_candidate,
                corridor_score=corridor_scores[layer_index][current_index],
                optimizer_settings=optimizer_settings,
            )
            best_cost = math.inf
            best_previous_index = -1

            for previous_index, previous_candidate, previous_cost in active_previous_candidates:
                pair_metrics = _joint_pair_metrics(
                    previous_candidate.joints,
                    current_candidate.joints,
                    optimizer_settings,
                )
                if not pair_metrics.passes_joint_continuity:
                    continue

                transition_cost = _candidate_transition_penalty_from_metrics(
                    previous_candidate,
                    current_candidate,
                    optimizer_settings,
                    pair_metrics,
                )

                if move_l_cache is not None:
                    linear_penalty, reached_joints = move_l_cache[previous_index]
                    if not math.isfinite(linear_penalty):
                        continue
                    transition_cost += linear_penalty

                    # ??? RoboDK ????????????????????????????????????????
                    # ???????????????????????????????????????????
                    if reached_joints is not None:
                        transition_cost += optimizer_settings.move_l_branch_mismatch_weight * (
                            _joint_transition_penalty(
                                reached_joints,
                                current_candidate.joints,
                                optimizer_settings,
                            )
                        )

                total_cost = previous_cost + transition_cost + node_cost
                if total_cost < best_cost:
                    best_cost = total_cost
                    best_previous_index = previous_index

            current_costs[current_index] = best_cost
            current_backpointers[current_index] = best_previous_index

        if not any(math.isfinite(cost) for cost in current_costs):
            message = (
                f"No globally feasible {move_type} sequence could be found for target index "
                f"{layer_index}."
            )
            if guided_config_path is not None:
                message += " The guided config-family path is too restrictive."
            if optimizer_settings.enable_joint_continuity_constraint:
                message += (
                    " The joint continuity constraint may be too strict for the current path."
                )
            raise RuntimeError(message)

        backpointers.append(current_backpointers)
        previous_layer = current_layer
        previous_costs = current_costs

    end_index = _choose_best_end_index(
        ik_layers,
        previous_costs=previous_costs,
        backpointers=backpointers,
        selection_bridge_trigger_joint_delta_deg=selection_bridge_trigger_joint_delta_deg,
    )
    total_cost = previous_costs[end_index]
    selected_path = _reconstruct_selected_path(ik_layers, backpointers, end_index)
    return selected_path, total_cost


def _choose_best_end_index(
    ik_layers: Sequence[_IKLayer],
    *,
    previous_costs: Sequence[float],
    backpointers: Sequence[Sequence[int]],
    selection_bridge_trigger_joint_delta_deg: float | None,
) -> int:
    if selection_bridge_trigger_joint_delta_deg is None:
        return min(range(len(previous_costs)), key=previous_costs.__getitem__)

    best_index = -1
    best_key: tuple[float, ...] | None = None
    for candidate_index, total_cost in enumerate(previous_costs):
        if not math.isfinite(total_cost):
            continue
        candidate_path = _reconstruct_selected_path(
            ik_layers,
            backpointers,
            candidate_index,
        )
        quality_key = _selected_path_quality_key(
            candidate_path,
            total_cost=float(total_cost),
            bridge_trigger_joint_delta_deg=float(selection_bridge_trigger_joint_delta_deg),
        )
        if best_key is None or quality_key < best_key:
            best_key = quality_key
            best_index = candidate_index
    if best_index >= 0:
        return best_index
    return min(range(len(previous_costs)), key=previous_costs.__getitem__)


def _reconstruct_selected_path(
    ik_layers: Sequence[_IKLayer],
    backpointers: Sequence[Sequence[int]],
    end_index: int,
) -> list[_IKCandidate]:
    selected_path = [ik_layers[-1].candidates[end_index]]
    for layer_index in range(len(ik_layers) - 2, -1, -1):
        end_index = backpointers[layer_index][end_index]
        selected_path.append(ik_layers[layer_index].candidates[end_index])
    selected_path.reverse()
    return selected_path


def _optimize_closed_joint_path(
    ik_layers: Sequence[_IKLayer],
    *,
    robot,
    move_type: str,
    start_joints: tuple[float, ...],
    optimizer_settings: _PathOptimizerSettings,
    selection_bridge_trigger_joint_delta_deg: float | None = None,
) -> tuple[list[_IKCandidate], float]:
    best_path: list[_IKCandidate] | None = None
    best_cost = math.inf
    best_quality_key: tuple[float, ...] | None = None
    joint6_turn_direction_cache: dict[tuple[float, ...], int] = {}
    for start_candidate in ik_layers[0].candidates:
        start_joints_key = tuple(float(value) for value in start_candidate.joints)
        joint6_turn_direction = joint6_turn_direction_cache.get(start_joints_key)
        if joint6_turn_direction is None:
            joint6_turn_direction = _infer_closed_path_joint6_turn_direction(
                ik_layers,
                start_candidate=start_candidate,
                robot=robot,
                move_type=move_type,
                start_joints=start_joints,
                optimizer_settings=optimizer_settings,
            )
            joint6_turn_direction_cache[start_joints_key] = joint6_turn_direction
        terminal_candidates = _build_closed_winding_terminal_candidates(
            start_candidate,
            ik_layers[-1].candidates,
            required_joint6_turns=optimizer_settings.closed_path_joint6_turns,
            required_turn_direction=joint6_turn_direction,
            tolerance_deg=optimizer_settings.closed_path_joint6_turn_tolerance_deg,
        )
        if not terminal_candidates:
            continue

        constrained_layers = _build_closed_winding_layers(
            ik_layers,
            start_candidate=start_candidate,
            terminal_candidates=terminal_candidates,
            single_config=optimizer_settings.closed_path_single_config,
            optimizer_settings=optimizer_settings,
        )
        if constrained_layers is None:
            continue
        try:
            candidate_path, candidate_cost = _optimize_joint_path(
                constrained_layers,
                robot=robot,
                move_type=move_type,
                start_joints=start_joints,
                optimizer_settings=optimizer_settings,
                require_terminal_match_start=False,
                selection_bridge_trigger_joint_delta_deg=selection_bridge_trigger_joint_delta_deg,
            )
        except RuntimeError:
            continue

        if selection_bridge_trigger_joint_delta_deg is None:
            candidate_quality_key = (float(candidate_cost),)
        else:
            candidate_quality_key = _selected_path_quality_key(
                candidate_path,
                total_cost=float(candidate_cost),
                bridge_trigger_joint_delta_deg=float(selection_bridge_trigger_joint_delta_deg),
            )
        if best_quality_key is None or candidate_quality_key < best_quality_key:
            best_path = candidate_path
            best_cost = candidate_cost
            best_quality_key = candidate_quality_key

    if best_path is None:
        raise RuntimeError(
            "No globally feasible closed winding joint sequence could be found with "
            "I1-I5 matching the start, I6 offset by the required full turn, and the "
            "configured closed-path config constraints."
        )
    return best_path, best_cost


def _build_closed_winding_terminal_candidates(
    start_candidate: _IKCandidate,
    terminal_candidates: Sequence[_IKCandidate],
    *,
    required_joint6_turns: int,
    required_turn_direction: int,
    tolerance_deg: float,
) -> tuple[_IKCandidate, ...]:
    """Return terminal candidates with A6 represented as the required full turn.

    IK backends commonly deduplicate periodic solutions and return the terminal
    row with A6 near the start value.  For closed winding, that is physically
    the same pose but violates the required selected-path representation.  This
    helper expands only the synthetic terminal row so I1-I5 stay fixed while I6
    is exactly one signed full turn away from the start.
    """

    if len(start_candidate.joints) < 6 or int(required_joint6_turns) == 0:
        return ()

    directions = (
        (int(required_turn_direction),)
        if int(required_turn_direction) in {-1, 1}
        else (-1, 1)
    )
    expanded: list[_IKCandidate] = []
    seen: set[tuple[float, ...]] = set()
    for candidate in terminal_candidates:
        if len(candidate.joints) != len(start_candidate.joints) or len(candidate.joints) < 6:
            continue
        if not _closed_winding_terminal_prefix_matches(
            start_candidate,
            candidate,
            tolerance_deg=tolerance_deg,
        ):
            continue
        for direction in directions:
            target_joint6 = float(start_candidate.joints[5]) + (
                360.0 * int(required_joint6_turns) * int(direction)
            )
            current_joint6 = float(candidate.joints[5])
            turn_index = round((target_joint6 - current_joint6) / 360.0)
            expanded_joint6 = current_joint6 + 360.0 * turn_index
            if abs(expanded_joint6 - target_joint6) > tolerance_deg:
                continue
            joints = (
                *tuple(float(value) for value in candidate.joints[:5]),
                float(expanded_joint6),
                *tuple(float(value) for value in candidate.joints[6:]),
            )
            expanded_candidate = _IKCandidate(
                joints=joints,
                config_flags=candidate.config_flags,
                joint_limit_penalty=candidate.joint_limit_penalty,
                singularity_penalty=candidate.singularity_penalty,
                branch_id=candidate.branch_id,
            )
            if not _candidates_satisfy_closed_winding_terminal(
                start_candidate,
                expanded_candidate,
                required_joint6_turns=required_joint6_turns,
                required_turn_direction=direction,
                tolerance_deg=tolerance_deg,
            ):
                continue
            dedup_key = tuple(round(value, 6) for value in expanded_candidate.joints)
            if dedup_key in seen:
                continue
            seen.add(dedup_key)
            expanded.append(expanded_candidate)
    return tuple(expanded)


def _infer_closed_path_joint6_turn_direction(
    ik_layers: Sequence[_IKLayer],
    *,
    start_candidate: _IKCandidate,
    robot,
    move_type: str,
    start_joints: tuple[float, ...],
    optimizer_settings: _PathOptimizerSettings,
) -> int:
    sample_count = min(
        len(ik_layers),
        max(2, int(optimizer_settings.closed_path_joint6_direction_sample_count)),
    )
    if sample_count < 2 or len(start_candidate.joints) < 6:
        return 0

    prefix_layers = _build_closed_winding_layers(
        ik_layers[:sample_count],
        start_candidate=start_candidate,
        terminal_candidates=ik_layers[sample_count - 1].candidates,
        single_config=optimizer_settings.closed_path_single_config,
        optimizer_settings=optimizer_settings,
    )
    if prefix_layers is None:
        return 0

    try:
        prefix_path, _prefix_cost = _optimize_joint_path(
            prefix_layers,
            robot=robot,
            move_type=move_type,
            start_joints=start_joints,
            optimizer_settings=optimizer_settings,
            require_terminal_match_start=False,
        )
    except RuntimeError:
        return 0
    if len(prefix_path) < 2 or len(prefix_path[-1].joints) < 6:
        return 0

    delta = float(prefix_path[-1].joints[5]) - float(prefix_path[0].joints[5])
    if abs(delta) < optimizer_settings.closed_path_joint6_direction_min_delta_deg:
        return 0
    return 1 if delta > 0.0 else -1


def _build_closed_winding_layers(
    ik_layers: Sequence[_IKLayer],
    *,
    start_candidate: _IKCandidate,
    terminal_candidates: Sequence[_IKCandidate],
    single_config: bool,
    optimizer_settings: _PathOptimizerSettings,
) -> tuple[_IKLayer, ...] | None:
    locked_indices = tuple(int(index) for index in optimizer_settings.closed_path_locked_config_indices)
    middle_layers: list[_IKLayer] = []
    for layer in ik_layers[1:-1]:
        layer_candidates = tuple(
            candidate
            for candidate in layer.candidates
            if _candidate_matches_closed_path_config(
                start_candidate,
                candidate,
                single_config=single_config,
                locked_indices=locked_indices,
            )
        )
        if not layer_candidates:
            return None
        middle_layers.append(_IKLayer(pose=layer.pose, candidates=layer_candidates))

    terminal_layer_candidates = tuple(
        candidate
        for candidate in terminal_candidates
        if _candidate_matches_closed_path_config(
            start_candidate,
            candidate,
            single_config=single_config,
            locked_indices=locked_indices,
        )
    )
    if not terminal_layer_candidates:
        return None

    return (
        _IKLayer(pose=ik_layers[0].pose, candidates=(start_candidate,)),
        *middle_layers,
        _IKLayer(pose=ik_layers[-1].pose, candidates=terminal_layer_candidates),
    )


def _candidate_matches_closed_path_config(
    start_candidate: _IKCandidate,
    candidate: _IKCandidate,
    *,
    single_config: bool,
    locked_indices: tuple[int, ...],
) -> bool:
    if single_config:
        return candidate.config_flags == start_candidate.config_flags

    for index in locked_indices:
        if index >= len(start_candidate.config_flags) or index >= len(candidate.config_flags):
            return False
        if candidate.config_flags[index] != start_candidate.config_flags[index]:
            return False
    return True


def _closed_winding_terminal_prefix_matches(
    first_candidate: _IKCandidate,
    second_candidate: _IKCandidate,
    *,
    tolerance_deg: float,
) -> bool:
    if len(first_candidate.joints) < _CLOSED_WINDING_TERMINAL_PREFIX_JOINT_COUNT:
        return False
    if len(second_candidate.joints) < _CLOSED_WINDING_TERMINAL_PREFIX_JOINT_COUNT:
        return False

    for joint_index in range(_CLOSED_WINDING_TERMINAL_PREFIX_JOINT_COUNT):
        if (
            abs(
                float(first_candidate.joints[joint_index])
                - float(second_candidate.joints[joint_index])
            )
            > tolerance_deg
        ):
            return False
    return True


def _candidates_satisfy_closed_winding_terminal(
    first_candidate: _IKCandidate,
    second_candidate: _IKCandidate,
    *,
    required_joint6_turns: int,
    required_turn_direction: int,
    tolerance_deg: float,
) -> bool:
    if len(first_candidate.joints) != len(second_candidate.joints):
        return False
    if not _closed_winding_terminal_prefix_matches(
        first_candidate,
        second_candidate,
        tolerance_deg=tolerance_deg,
    ):
        return False
    return _joint6_terminal_values_match(
        float(first_candidate.joints[5]),
        float(second_candidate.joints[5]),
        required_turns=required_joint6_turns,
        required_turn_direction=required_turn_direction,
        tolerance_deg=tolerance_deg,
    )


def _joint6_terminal_values_match(
    first_joint_deg: float,
    second_joint_deg: float,
    *,
    required_turns: int,
    required_turn_direction: int,
    tolerance_deg: float,
) -> bool:
    """Return whether A6 satisfies the closed winding turn constraint.

    This is only used for the synthetic closed-path terminal row.  Intermediate
    path continuity still uses the real signed joint values.
    """

    delta = float(second_joint_deg) - float(first_joint_deg)
    nearest_full_turn = round(delta / 360.0)
    if abs(delta - 360.0 * nearest_full_turn) > tolerance_deg:
        return False
    signed_turns = int(nearest_full_turn)
    if abs(signed_turns) != int(required_turns):
        return False
    if required_turn_direction == 0 or required_turns == 0:
        return True
    return signed_turns == int(required_turn_direction) * int(required_turns)


def _build_guided_config_path(
    ik_layers: Sequence[_IKLayer],
    *,
    start_joints: Sequence[float],
    optimizer_settings: _PathOptimizerSettings,
) -> tuple[tuple[int, ...], ...] | None:
    """??? `config_flags` ??????????????????

    ????? DP ?????????????config family ???????????????????????
    ????????????????????????????????????????????????????????

    ??????????????????????
    - ???????????????????family??
    - ??????????????????
    - ????????????????????????????????????
    """

    if not ik_layers:
        return ()

    candidate_groups_by_layer = [
        _group_candidates_by_config(layer.candidates)
        for layer in ik_layers
    ]

    previous_costs: dict[tuple[int, ...], float] = {}
    for config_flags, candidates in candidate_groups_by_layer[0].items():
        previous_costs[config_flags] = min(
            optimizer_settings.start_transition_weight
            * _joint_transition_penalty(start_joints, candidate.joints, optimizer_settings)
            for candidate in candidates
        )

    backpointers: list[dict[tuple[int, ...], tuple[int, ...]]] = []
    for layer_index in range(1, len(candidate_groups_by_layer)):
        previous_groups = candidate_groups_by_layer[layer_index - 1]
        current_groups = candidate_groups_by_layer[layer_index]
        current_costs: dict[tuple[int, ...], float] = {}
        current_backpointers: dict[tuple[int, ...], tuple[int, ...]] = {}

        for current_flags, current_candidates in current_groups.items():
            best_cost = math.inf
            best_previous_flags: tuple[int, ...] | None = None
            for previous_flags, previous_candidates in previous_groups.items():
                if previous_flags not in previous_costs:
                    continue

                transition_cost = _best_config_group_transition_cost(
                    previous_candidates,
                    current_candidates,
                    optimizer_settings,
                )
                if not math.isfinite(transition_cost):
                    continue

                total_cost = previous_costs[previous_flags] + transition_cost
                if previous_flags != current_flags:
                    total_cost += optimizer_settings.family_switch_penalty
                else:
                    total_cost -= optimizer_settings.same_config_stay_bonus

                if total_cost < best_cost:
                    best_cost = total_cost
                    best_previous_flags = previous_flags

            if best_previous_flags is not None:
                current_costs[current_flags] = best_cost
                current_backpointers[current_flags] = best_previous_flags

        if not current_costs:
            return None

        previous_costs = current_costs
        backpointers.append(current_backpointers)

    end_flags = min(previous_costs, key=previous_costs.__getitem__)
    guided_flags = [end_flags]
    for layer_index in range(len(backpointers) - 1, -1, -1):
        end_flags = backpointers[layer_index][end_flags]
        guided_flags.append(end_flags)
    guided_flags.reverse()
    return tuple(guided_flags)


def _group_candidates_by_config(
    candidates: Sequence[_IKCandidate],
) -> dict[tuple[int, ...], tuple[_IKCandidate, ...]]:
    """??branch-aware lineage key ????????????????"""

    grouped: dict[tuple[int, ...], list[_IKCandidate]] = {}
    for candidate in candidates:
        grouped.setdefault(candidate.config_flags, []).append(candidate)
    return {config_flags: tuple(group) for config_flags, group in grouped.items()}


def _group_candidate_indices_by_config(
    candidates: Sequence[_IKCandidate],
) -> dict[tuple[int, ...], tuple[int, ...]]:
    grouped: dict[tuple[int, ...], list[int]] = {}
    for index, candidate in enumerate(candidates):
        grouped.setdefault(candidate.config_flags, []).append(index)
    return {config_flags: tuple(indices) for config_flags, indices in grouped.items()}


def _best_config_group_transition_cost(
    previous_candidates: Sequence[_IKCandidate],
    current_candidates: Sequence[_IKCandidate],
    optimizer_settings: _PathOptimizerSettings,
) -> float:
    """?????? config-family ??????????????????"""

    best_cost = math.inf
    for previous_candidate in previous_candidates:
        for current_candidate in current_candidates:
            if not _passes_joint_continuity_constraint(
                previous_candidate.joints,
                current_candidate.joints,
                optimizer_settings,
            ):
                continue
            best_cost = min(
                best_cost,
                _joint_transition_penalty(
                    previous_candidate.joints,
                    current_candidate.joints,
                    optimizer_settings,
                ),
            )
    return best_cost


def _guided_config_path_is_feasible(
    ik_layers: Sequence[_IKLayer],
    *,
    guided_config_path: Sequence[tuple[int, ...]],
    optimizer_settings: _PathOptimizerSettings,
) -> bool:
    """????family guidance ?????????????????????????"""

    if len(guided_config_path) != len(ik_layers):
        return False

    reachable_indices = [
        index
        for index, candidate in enumerate(ik_layers[0].candidates)
        if candidate.config_flags == guided_config_path[0]
    ]
    if not reachable_indices:
        return False

    for layer_index in range(1, len(ik_layers)):
        current_reachable: list[int] = []
        for current_index, current_candidate in enumerate(ik_layers[layer_index].candidates):
            if current_candidate.config_flags != guided_config_path[layer_index]:
                continue
            for previous_index in reachable_indices:
                previous_candidate = ik_layers[layer_index - 1].candidates[previous_index]
                if _passes_joint_continuity_constraint(
                    previous_candidate.joints,
                    current_candidate.joints,
                    optimizer_settings,
                ):
                    current_reachable.append(current_index)
                    break
        if not current_reachable:
            return False
        reachable_indices = current_reachable

    return True


def _evaluate_move_l_transition(
    robot,
    *,
    start_joints: tuple[float, ...],
    target_pose,
    joint_count: int,
    optimizer_settings: _PathOptimizerSettings,
) -> tuple[float, tuple[float, ...] | None]:
    """????????????????????MoveL ????????"""

    status = robot.MoveL_Test(list(start_joints), target_pose)
    if status != 0:
        return optimizer_settings.move_l_unreachable_penalty, None

    reached_joints = _trim_joint_vector(robot.Joints().list(), joint_count)
    return 0.0, reached_joints


def _candidate_transition_penalty_from_metrics(
    previous_candidate: _IKCandidate,
    current_candidate: _IKCandidate,
    optimizer_settings: _PathOptimizerSettings,
    pair_metrics: _JointPairMetrics,
) -> float:
    cost = pair_metrics.transition_cost

    previous_flags = previous_candidate.config_flags
    current_flags = current_candidate.config_flags
    if len(previous_flags) >= 1 and len(current_flags) >= 1 and previous_flags[0] != current_flags[0]:
        cost += optimizer_settings.rear_switch_penalty
    if len(previous_flags) >= 2 and len(current_flags) >= 2 and previous_flags[1] != current_flags[1]:
        cost += optimizer_settings.lower_switch_penalty
    if len(previous_flags) >= 3 and len(current_flags) >= 3 and previous_flags[2] != current_flags[2]:
        cost += optimizer_settings.flip_switch_penalty

    # J5 ?????????????? wrist flip??????????
    if len(previous_candidate.joints) >= 5:
        if previous_candidate.joints[4] * current_candidate.joints[4] < 0.0:
            cost += optimizer_settings.wrist_flip_sign_penalty

    # ?????J6 ??????????????????????????????????
    if len(previous_candidate.joints) >= 6:
        if pair_metrics.joint6_delta_deg > optimizer_settings.joint6_spin_threshold_deg:
            joint6_delta = pair_metrics.joint6_delta_deg
            cost += (
                joint6_delta - optimizer_settings.joint6_spin_threshold_deg
            ) * optimizer_settings.joint6_spin_penalty_per_deg

        # ????FINA11.src??? A5 ??? 0 ??????????????? A6 ????????????
        # ??????????????????????????A4 ??????????????? A6 ????????
        if pair_metrics.min_abs_joint5_deg < optimizer_settings.wrist_phase_lock_threshold_deg:
            normalized = (
                optimizer_settings.wrist_phase_lock_threshold_deg - pair_metrics.min_abs_joint5_deg
            ) / optimizer_settings.wrist_phase_lock_threshold_deg
            cost += (
                optimizer_settings.wrist_phase_lock_penalty_per_deg
                * normalized
                * pair_metrics.joint6_delta_deg
            )

    # ??????????????????????????????????????????????????
    # ??DP ?????????????????"??exact-pose ???????
    if previous_flags == current_flags:
        cost -= optimizer_settings.same_config_stay_bonus
        if pair_metrics.passes_preferred_continuity:
            cost -= optimizer_settings.preferred_transition_bonus

    return cost


def _candidate_transition_penalty(
    previous_candidate: _IKCandidate,
    current_candidate: _IKCandidate,
    optimizer_settings: _PathOptimizerSettings,
) -> float:
    """??????????????????????"""

    pair_metrics = _joint_pair_metrics(
        previous_candidate.joints,
        current_candidate.joints,
        optimizer_settings,
    )
    return _candidate_transition_penalty_from_metrics(
        previous_candidate,
        current_candidate,
        optimizer_settings,
        pair_metrics,
    )


def _passes_joint_continuity_constraint(
    previous_joints: Sequence[float],
    current_joints: Sequence[float],
    optimizer_settings: _PathOptimizerSettings,
) -> bool:
    """??????????????????????????????"""

    return _joint_pair_metrics(
        _joint_tuple(previous_joints),
        _joint_tuple(current_joints),
        optimizer_settings,
    ).passes_joint_continuity


def _passes_preferred_continuity(
    previous_joints: Sequence[float],
    current_joints: Sequence[float],
    optimizer_settings: _PathOptimizerSettings,
) -> bool:
    """???????????????????????????

    ??????????????????????????
    ?????????????????????????????? exact-pose ?????????
    """

    return _joint_pair_metrics(
        _joint_tuple(previous_joints),
        _joint_tuple(current_joints),
        optimizer_settings,
    ).passes_preferred_continuity


def _candidate_node_cost(
    candidate: _IKCandidate,
    *,
    corridor_score: float,
    optimizer_settings: _PathOptimizerSettings,
) -> float:
    """?????????????????????

    ?????????????????
    1. ?????/ ?????????????????????????????????????????
    2. ?????????????????DP ???????????????????????
    """

    raw_penalty = candidate.joint_limit_penalty + candidate.singularity_penalty
    corridor_bonus = min(optimizer_settings.corridor_bonus_cap, corridor_score) * (
        optimizer_settings.corridor_bonus_per_step
    )
    return optimizer_settings.node_penalty_scale * raw_penalty - corridor_bonus


def _compute_candidate_corridor_scores(
    ik_layers: Sequence[_IKLayer],
    optimizer_settings: _PathOptimizerSettings,
) -> list[list[float]]:
    """??????????????"?????????"????????

    ?????
    1. ?????"??? config_flags ??????????????????
    2. ???????????????????? DP??
    3. ???????????????????????

    ?????????????????????????????????????????????
    """

    if not ik_layers:
        return []

    forward_lengths: list[list[int]] = [
        [1] * len(layer.candidates) for layer in ik_layers
    ]
    backward_lengths: list[list[int]] = [
        [1] * len(layer.candidates) for layer in ik_layers
    ]
    candidate_indices_by_config = [
        _group_candidate_indices_by_config(layer.candidates)
        for layer in ik_layers
    ]

    for layer_index in range(len(ik_layers) - 2, -1, -1):
        current_layer = ik_layers[layer_index]
        next_layer = ik_layers[layer_index + 1]
        next_indices_by_config = candidate_indices_by_config[layer_index + 1]
        next_forward_lengths = forward_lengths[layer_index + 1]
        for current_index, current_candidate in enumerate(current_layer.candidates):
            best_reach = 0
            for next_index in next_indices_by_config.get(current_candidate.config_flags, ()):
                next_candidate = next_layer.candidates[next_index]
                if not _passes_preferred_continuity(
                    current_candidate.joints,
                    next_candidate.joints,
                    optimizer_settings,
                ):
                    continue
                if next_forward_lengths[next_index] > best_reach:
                    best_reach = next_forward_lengths[next_index]
            forward_lengths[layer_index][current_index] = 1 + best_reach

    for layer_index in range(1, len(ik_layers)):
        previous_layer = ik_layers[layer_index - 1]
        current_layer = ik_layers[layer_index]
        previous_indices_by_config = candidate_indices_by_config[layer_index - 1]
        previous_backward_lengths = backward_lengths[layer_index - 1]
        for current_index, current_candidate in enumerate(current_layer.candidates):
            best_reach = 0
            for previous_index in previous_indices_by_config.get(
                current_candidate.config_flags,
                (),
            ):
                previous_candidate = previous_layer.candidates[previous_index]
                if not _passes_preferred_continuity(
                    previous_candidate.joints,
                    current_candidate.joints,
                    optimizer_settings,
                ):
                    continue
                if previous_backward_lengths[previous_index] > best_reach:
                    best_reach = previous_backward_lengths[previous_index]
            backward_lengths[layer_index][current_index] = 1 + best_reach

    corridor_scores: list[list[float]] = []
    for layer_index, layer in enumerate(ik_layers):
        layer_scores = []
        for candidate_index, _candidate in enumerate(layer.candidates):
            forward_length = forward_lengths[layer_index][candidate_index]
            backward_length = backward_lengths[layer_index][candidate_index]
            layer_scores.append(float(forward_length + backward_length - 2))
        corridor_scores.append(layer_scores)

    return corridor_scores


def _joint_transition_penalty(
    previous_joints: Sequence[float],
    current_joints: Sequence[float],
    optimizer_settings: _PathOptimizerSettings,
) -> float:
    """??????????????

    ???????????
    1. ????????????
    2. ??????????????
    3. ??????????????????
    """

    return _joint_pair_metrics(
        _joint_tuple(previous_joints),
        _joint_tuple(current_joints),
        optimizer_settings,
    ).transition_cost


def _joint_limit_penalty(
    joints: Sequence[float],
    lower_limits: Sequence[float],
    upper_limits: Sequence[float],
    optimizer_settings: _PathOptimizerSettings,
) -> float:
    """????????????????????????????"""

    penalty = 0.0
    for joint, lower, upper in zip(joints, lower_limits, upper_limits):
        span = upper - lower
        if span <= 0.0:
            continue

        margin = min(joint - lower, upper - joint)
        margin_ratio = margin / span
        if margin_ratio < optimizer_settings.joint_limit_margin_ratio:
            normalized = (
                optimizer_settings.joint_limit_margin_ratio - margin_ratio
            ) / optimizer_settings.joint_limit_margin_ratio
            penalty += optimizer_settings.joint_limit_penalty_weight * normalized * normalized
    return penalty


def _singularity_penalty(
    robot,
    joints: Sequence[float],
    optimizer_settings: _PathOptimizerSettings,
) -> float:
    """??????????????

    ????????????????????????????????????????????
    1. J5 ??? 0 ??????????????????
    2. ????????????????????????????????
    """

    joints_key = tuple(float(value) for value in joints)
    cache_key = (id(robot), joints_key, optimizer_settings)
    cached_penalty = _ROBOT_SINGULARITY_PENALTY_CACHE.get(cache_key)
    if cached_penalty is not None:
        return cached_penalty

    penalty = 0.0

    if len(joints) >= 5:
        wrist_measure = abs(math.sin(math.radians(joints[4])))
        threshold = math.sin(math.radians(optimizer_settings.wrist_singularity_threshold_deg))
        if wrist_measure < threshold:
            normalized = (threshold - wrist_measure) / threshold
            penalty += (
                optimizer_settings.wrist_singularity_penalty_weight * normalized * normalized
            )

    geometry_metrics_fn = getattr(robot, "JointGeometryMetrics", None)
    arm_measure: float | None = None
    if callable(geometry_metrics_fn):
        try:
            arm_measure = float(geometry_metrics_fn(joints).arm_singularity_measure)
        except Exception:
            arm_measure = None

    if arm_measure is None:
        from src.core.geometry import _normalized_cross_measure, _subtract_vectors, _translation_from_pose

        joint_poses = robot.JointPoses(list(joints))
        if len(joint_poses) >= 4:
            shoulder = _translation_from_pose(joint_poses[1])
            elbow = _translation_from_pose(joint_poses[2])
            wrist = _translation_from_pose(joint_poses[3])
            arm_measure = _normalized_cross_measure(
                _subtract_vectors(elbow, shoulder),
                _subtract_vectors(wrist, elbow),
            )

    if arm_measure is not None and arm_measure < optimizer_settings.arm_singularity_threshold:
        normalized = (
            optimizer_settings.arm_singularity_threshold - arm_measure
        ) / optimizer_settings.arm_singularity_threshold
        penalty += optimizer_settings.arm_singularity_penalty_weight * normalized * normalized

    if len(_ROBOT_SINGULARITY_PENALTY_CACHE) >= _ROBOT_SINGULARITY_PENALTY_CACHE_MAX_ENTRIES:
        _ROBOT_SINGULARITY_PENALTY_CACHE.clear()
    _ROBOT_SINGULARITY_PENALTY_CACHE[cache_key] = penalty
    return penalty


def _passes_step_limit(
    previous_joints: Sequence[float],
    current_joints: Sequence[float],
    step_limits: Sequence[float],
) -> bool:
    """????????????????????????????"""

    return all(
        abs(current - previous) <= limit
        for previous, current, limit in zip(previous_joints, current_joints, step_limits)
    )


def _selected_path_quality_key(
    selected_path: Sequence[_IKCandidate],
    *,
    total_cost: float,
    bridge_trigger_joint_delta_deg: float,
) -> tuple[float, ...]:
    """?????????????????????????????"""

    (
        config_switches,
        bridge_like_segments,
        worst_joint_step_deg,
        mean_joint_step_deg,
    ) = _summarize_selected_path(
        selected_path,
        bridge_trigger_joint_delta_deg=bridge_trigger_joint_delta_deg,
    )
    return (
        float(bridge_like_segments),
        float(config_switches),
        float(worst_joint_step_deg),
        float(mean_joint_step_deg),
        float(total_cost),
    )
