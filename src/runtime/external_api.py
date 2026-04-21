from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any, NoReturn

import numpy as np

from src.core.collab_models import (
    EvaluationBatchRequest,
    ProfileEvaluationRequest,
    RemoteSearchRequest,
)
from src.core.program_ir import program_runtime_capabilities
from src.robodk_runtime.eval_worker import evaluate_batch_request, evaluate_single_request
from src.runtime.delivery import (
    result_has_continuity_warnings,
    result_is_strictly_valid,
    result_quality_summary,
    result_semantic_status,
)
from src.runtime.model_identity import kinematics_hash
from src.six_axis_ik import config as ik_config
from src.six_axis_ik.interface import SixAxisIKSolver
from src.six_axis_ik.kinematics import (
    JOINT_COUNT,
    RobotModel,
    as_joint_vector,
    as_transform,
    joint_distance_deg,
    rotation_error_deg,
)


class ExternalAPIError(RuntimeError):
    def __init__(
        self,
        code: str,
        message: str,
        *,
        status_code: int = 400,
        details: Any | None = None,
    ) -> None:
        super().__init__(message)
        self.code = str(code)
        self.message = str(message)
        self.status_code = int(status_code)
        self.details = details


def _mat_from_payload(value: Any, fallback: np.ndarray | None = None) -> np.ndarray:
    if value is None:
        if fallback is None:
            raise ValueError("matrix is required")
        return fallback.copy()
    matrix = np.asarray(value, dtype=float)
    return as_transform(matrix)


def _joint_vector_from_payload(
    value: Any,
    fallback: np.ndarray | None = None,
) -> np.ndarray:
    if value is None:
        if fallback is None:
            raise ValueError("joint vector is required")
        return fallback.copy()
    return as_joint_vector(value)


def _to_rows(matrix: np.ndarray) -> list[list[float]]:
    matrix = as_transform(matrix)
    return [[float(matrix[row, col]) for col in range(4)] for row in range(4)]


def _vec6(values: np.ndarray) -> list[float]:
    return [float(v) for v in np.asarray(values, dtype=float).reshape(-1)[:JOINT_COUNT]]


def _safe_float(value: Any, default: float) -> float:
    try:
        parsed = float(value)
    except Exception:
        return float(default)
    if not np.isfinite(parsed):
        return float(default)
    return parsed


def _quality_score(
    position_error_mm: float,
    orientation_error_deg: float,
    include_orientation: bool,
) -> float:
    if include_orientation:
        return float(position_error_mm) + 0.5 * float(orientation_error_deg)
    return float(position_error_mm)


def _select_interactive_ik_candidate(
    candidates: list[Any],
    *,
    previous_q_deg: np.ndarray,
    include_orientation: bool,
    continuity_weight: float,
    quality_weight: float,
    quality_slack: float,
) -> tuple[np.ndarray, float, float, float, int]:
    if not candidates:
        raise ValueError("candidates must not be empty")

    prepared: list[tuple[np.ndarray, float, float, float]] = []
    best_quality = float("inf")
    for candidate in candidates:
        q_candidate = as_joint_vector(candidate.joints_deg)
        pos_err = float(candidate.position_error_mm)
        ori_err = float(candidate.orientation_error_deg)
        quality = _quality_score(pos_err, ori_err, include_orientation)
        prepared.append((q_candidate, pos_err, ori_err, quality))
        best_quality = min(best_quality, quality)

    shortlist = [item for item in prepared if item[3] <= best_quality + quality_slack]
    if not shortlist:
        shortlist = prepared

    best_idx = 0
    best_score = float("inf")
    best_joint_dist = 0.0
    for idx, (q_candidate, _pos_err, _ori_err, quality) in enumerate(shortlist):
        continuity = float(joint_distance_deg(q_candidate, previous_q_deg))
        score = quality_weight * quality + continuity_weight * continuity
        if score < best_score:
            best_score = score
            best_idx = idx
            best_joint_dist = continuity

    chosen_q, chosen_pos_err, chosen_ori_err, _ = shortlist[best_idx]
    return chosen_q.copy(), chosen_pos_err, chosen_ori_err, best_joint_dist, len(shortlist)


def _kinematics_hash(
    *,
    axes: np.ndarray,
    points: np.ndarray,
    senses: np.ndarray,
    home_flange: np.ndarray,
) -> str:
    return kinematics_hash(
        axes=axes,
        points=points,
        senses=senses,
        home_flange=home_flange,
    )


def _fk_partial_world_rows(
    robot_model: RobotModel,
    robot_base_pose: np.ndarray,
    q_deg: np.ndarray,
) -> list[list[list[float]]]:
    rows: list[list[list[float]]] = []
    for joint_count in range(JOINT_COUNT + 1):
        partial = robot_model.fk_partial(q_deg, n_joints=joint_count)
        world = robot_base_pose @ partial
        rows.append(_to_rows(world))
    return rows


def _kinematics_from_robot_payload(
    robot_payload: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, str]:
    kin = robot_payload.get("kinematics_inferred")
    if isinstance(kin, dict):
        try:
            axes = np.asarray(kin.get("joint_axes_base"), dtype=float).reshape((JOINT_COUNT, 3))
            points = np.asarray(kin.get("joint_points_base_mm"), dtype=float).reshape((JOINT_COUNT, 3))
            senses = np.asarray(kin.get("joint_senses"), dtype=float).reshape((JOINT_COUNT,))
            home_flange = np.asarray(kin.get("home_flange"), dtype=float).reshape((4, 4))
            return axes, points, senses, home_flange, "metadata.kinematics_inferred"
        except Exception as exc:
            raise ValueError(
                "payload.robot.kinematics_inferred is present but malformed; "
                "expected joint_axes_base(6x3), joint_points_base_mm(6x3), "
                "joint_senses(6), and home_flange(4x4)."
            ) from exc
    if kin is not None:
        raise ValueError("payload.robot.kinematics_inferred must be an object when provided.")

    axes = np.asarray(ik_config.LOCAL_JOINT_AXIS_DIRECTIONS_BASE, dtype=float).reshape((JOINT_COUNT, 3))
    points = np.asarray(ik_config.LOCAL_JOINT_AXIS_POINTS_BASE_MM, dtype=float).reshape((JOINT_COUNT, 3))
    senses = np.asarray(ik_config.LOCAL_JOINT_SENSES, dtype=float).reshape((JOINT_COUNT,))
    home_flange = np.asarray(ik_config.LOCAL_HOME_FLANGE_MATRIX, dtype=float).reshape((4, 4))
    return axes, points, senses, home_flange, "project.config.fallback"


def _limits_from_robot_payload(robot_payload: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    lower_raw = robot_payload.get("joint_limits_lower_deg")
    upper_raw = robot_payload.get("joint_limits_upper_deg")
    try:
        lower = np.asarray(lower_raw, dtype=float).reshape((JOINT_COUNT,))
        upper = np.asarray(upper_raw, dtype=float).reshape((JOINT_COUNT,))
        return lower, upper
    except Exception:
        return (
            np.asarray(ik_config.get_configured_lower_limits_deg(), dtype=float).reshape((JOINT_COUNT,)),
            np.asarray(ik_config.get_configured_upper_limits_deg(), dtype=float).reshape((JOINT_COUNT,)),
        )


def _result_sort_key(result: Any) -> tuple[float, ...]:
    return (
        float(getattr(result, "invalid_row_count", 0)),
        float(getattr(result, "ik_empty_row_count", 0)),
        float(getattr(result, "bridge_like_segments", 0)),
        float(getattr(result, "big_circle_step_count", 0)),
        float(getattr(result, "worst_joint_step_deg", 0.0)),
        float(getattr(result, "mean_joint_step_deg", 0.0)),
        float(getattr(result, "config_switches", 0)),
        float(getattr(result, "total_cost", 0.0)),
        float(getattr(result, "timing_seconds", 0.0)),
    )


def _select_best_result(results: tuple[Any, ...]) -> Any | None:
    if not results:
        return None
    return sorted(results, key=_result_sort_key)[0]


@dataclass
class SolverSession:
    robot_model: RobotModel | None = None
    robot_base_pose: np.ndarray = field(default_factory=lambda: np.eye(4, dtype=float))
    q_init_deg: np.ndarray = field(default_factory=lambda: np.zeros((JOINT_COUNT,), dtype=float))
    q_last_solution_deg: np.ndarray = field(default_factory=lambda: np.zeros((JOINT_COUNT,), dtype=float))
    solver: SixAxisIKSolver | None = None
    kinematics_source: str = "unconfigured"
    kinematics_hash: str | None = None
    model_id: str | None = None
    lock: threading.Lock = field(default_factory=threading.Lock)

    def configured(self) -> bool:
        return self.robot_model is not None and self.solver is not None

    def configure(self, payload: dict[str, Any]) -> dict[str, Any]:
        robot_payload = payload.get("robot")
        if not isinstance(robot_payload, dict):
            raise ValueError("payload.robot is required")

        strict_kinematics = bool(
            payload.get("strict_kinematics", False)
            or robot_payload.get("strict_kinematics", False)
        )
        if strict_kinematics and robot_payload.get("kinematics_inferred") is None:
            raise ValueError(
                "payload.robot.kinematics_inferred is required when strict_kinematics=true."
            )

        axes, points, senses, home_flange, kinematics_source = _kinematics_from_robot_payload(robot_payload)
        model_id_raw = robot_payload.get("model_id", robot_payload.get("name"))
        model_id = None if model_id_raw in (None, "") else str(model_id_raw)
        kinematics_hash = _kinematics_hash(
            axes=axes,
            points=points,
            senses=senses,
            home_flange=home_flange,
        )
        lower, upper = _limits_from_robot_payload(robot_payload)

        tool_pose = _mat_from_payload(robot_payload.get("pose_tool"), ik_config.get_configured_tool_pose())
        frame_pose = _mat_from_payload(robot_payload.get("pose_frame"), ik_config.get_configured_frame_pose())
        base_pose = _mat_from_payload(robot_payload.get("pose_abs"), np.eye(4, dtype=float))
        q_init = _joint_vector_from_payload(
            robot_payload.get("joints_current_deg"),
            np.zeros((JOINT_COUNT,), dtype=float),
        )

        model = RobotModel(
            joint_axis_directions_base=axes,
            joint_axis_points_base_mm=points,
            joint_senses=senses,
            home_flange_T=home_flange,
            joint_min_deg=lower,
            joint_max_deg=upper,
            tool_T=tool_pose,
            frame_T=frame_pose,
        )

        solver = SixAxisIKSolver(
            model,
            ik_backend="numeric",
        )

        with self.lock:
            self.robot_model = model
            self.robot_base_pose = base_pose
            self.q_init_deg = q_init
            self.q_last_solution_deg = q_init.copy()
            self.solver = solver
            self.kinematics_source = kinematics_source
            self.kinematics_hash = kinematics_hash
            self.model_id = model_id

        return {
            "ok": True,
            "kinematics_source": self.kinematics_source,
            "kinematics_hash": kinematics_hash,
            "model_id": model_id,
            "limits_lower_deg": _vec6(lower),
            "limits_upper_deg": _vec6(upper),
            "q_init_deg": _vec6(q_init),
        }

    def fk(self, payload: dict[str, Any]) -> dict[str, Any]:
        with self.lock:
            if not self.configured():
                raise RuntimeError("solver is not configured")
            assert self.robot_model is not None
            robot_model = self.robot_model
            base_pose = self.robot_base_pose.copy()

        q_input = _joint_vector_from_payload(payload.get("q_deg"))
        q_clipped = robot_model.clip_joints(q_input)

        tcp_robot = robot_model.fk_tcp_in_robot_base(q_clipped)
        tcp_world = base_pose @ tcp_robot
        tcp_frame = robot_model.fk_tcp_in_frame(q_clipped)
        joint_frames_world = _fk_partial_world_rows(robot_model, base_pose, q_clipped)

        return {
            "ok": True,
            "q_deg": _vec6(q_clipped),
            "tcp_robot_pose": _to_rows(tcp_robot),
            "tcp_world_pose": _to_rows(tcp_world),
            "tcp_frame_pose": _to_rows(tcp_frame),
            "joint_frames_world": joint_frames_world,
        }

    def ik(self, payload: dict[str, Any]) -> dict[str, Any]:
        with self.lock:
            if not self.configured():
                raise RuntimeError("solver is not configured")
            assert self.robot_model is not None
            assert self.solver is not None
            robot_model = self.robot_model
            solver = self.solver
            q_last_solution = self.q_last_solution_deg.copy()

        target_frame_pose = _mat_from_payload(payload.get("target_frame_pose"))
        include_orientation = bool(payload.get("include_orientation", True))
        seed = _joint_vector_from_payload(payload.get("seed_q_deg"), self.q_init_deg)
        seed = robot_model.clip_joints(seed)
        previous_q = _joint_vector_from_payload(payload.get("previous_q_deg"), q_last_solution)
        previous_q = robot_model.clip_joints(previous_q)

        continuity_weight = _safe_float(payload.get("continuity_weight"), 1.0)
        quality_weight = _safe_float(payload.get("quality_weight"), 8.0)
        quality_slack = _safe_float(payload.get("quality_slack"), 0.35)
        max_joint_jump_warn_deg = _safe_float(payload.get("max_joint_jump_warn_deg"), 30.0)

        selected_by = "seed_fallback"
        candidate_count = 0
        shortlisted_count = 0
        joint_distance_to_previous = float(joint_distance_deg(seed, previous_q))

        if include_orientation:
            solve_all = None
            solve_all_error = ""
            try:
                solve_all = solver.solve_ik_all(
                    target_frame_pose,
                    target_space="frame",
                    seed_joints_deg=seed,
                    rotation_weight_mm=ik_config.NUMERIC_ROTATION_WEIGHT_MM,
                    max_nfev=ik_config.NUMERIC_MAX_NFEV,
                )
            except Exception as exc:
                solve_all_error = f"ik_all unavailable: {type(exc).__name__}: {exc}"

            if solve_all is not None:
                candidate_pool = solve_all.filtered_solutions or solve_all.all_solutions
                candidate_count = len(candidate_pool)
                if candidate_pool:
                    (
                        q_solution,
                        pos_err_est,
                        ori_err_est,
                        joint_distance_to_previous,
                        shortlisted_count,
                    ) = _select_interactive_ik_candidate(
                        candidate_pool,
                        previous_q_deg=previous_q,
                        include_orientation=include_orientation,
                        continuity_weight=continuity_weight,
                        quality_weight=quality_weight,
                        quality_slack=quality_slack,
                    )
                    q_solution = robot_model.clip_joints(q_solution)
                    selected_by = "ik_all_continuity"
                else:
                    q_solution = seed.copy()
                    pos_err_est = float("inf")
                    ori_err_est = float("inf")
                backend_ok = bool(solve_all.success and candidate_count > 0)
                backend_message = solve_all.failure_reason or ""
            else:
                solve_one = solver.solve_ik(
                    target_frame_pose,
                    target_space="frame",
                    seed_joints_deg=seed,
                    rotation_weight_mm=ik_config.NUMERIC_ROTATION_WEIGHT_MM,
                    max_nfev=ik_config.NUMERIC_MAX_NFEV,
                )
                one_solution = solve_one.preferred_solution
                if one_solution is None and solve_one.all_solutions:
                    one_solution = solve_one.all_solutions[0]
                if one_solution is not None:
                    q_solution = robot_model.clip_joints(one_solution.joints_deg)
                    pos_err_est = float(one_solution.position_error_mm)
                    ori_err_est = float(one_solution.orientation_error_deg)
                    candidate_count = int(len(solve_one.all_solutions))
                    shortlisted_count = int(len(solve_one.filtered_solutions)) if solve_one.filtered_solutions else 1
                    selected_by = "ik_single_fallback"
                else:
                    q_solution = seed.copy()
                    pos_err_est = float("inf")
                    ori_err_est = float("inf")
                joint_distance_to_previous = float(joint_distance_deg(q_solution, previous_q))
                backend_ok = bool(solve_one.success and one_solution is not None)
                backend_message = solve_one.failure_reason or solve_all_error

            need_numeric_refine = (not backend_ok) or (pos_err_est > 1.0) or (ori_err_est > 1.2)
            if need_numeric_refine:
                numeric = robot_model.ik_numeric(
                    target_frame_pose,
                    q0_deg=q_solution,
                    rotation_weight_mm=ik_config.NUMERIC_ROTATION_WEIGHT_MM,
                    max_nfev=ik_config.NUMERIC_MAX_NFEV,
                    raise_on_fail=False,
                )
                q_numeric = robot_model.clip_joints(numeric.q_deg)
                pose_numeric = robot_model.fk_tcp_in_frame(q_numeric)
                pos_numeric = float(np.linalg.norm(pose_numeric[:3, 3] - target_frame_pose[:3, 3]))
                ori_numeric = float(rotation_error_deg(pose_numeric, target_frame_pose))
                metric_numeric = _quality_score(pos_numeric, ori_numeric, include_orientation)

                pose_current = robot_model.fk_tcp_in_frame(q_solution)
                pos_current = float(np.linalg.norm(pose_current[:3, 3] - target_frame_pose[:3, 3]))
                ori_current = float(rotation_error_deg(pose_current, target_frame_pose))
                metric_current = _quality_score(pos_current, ori_current, include_orientation)

                if metric_numeric + 1e-9 < metric_current:
                    q_solution = q_numeric
                    joint_distance_to_previous = float(joint_distance_deg(q_solution, previous_q))
                    selected_by = "numeric_refine"
                if numeric.success:
                    backend_ok = True
                backend_message = backend_message if backend_message else str(numeric.message)
        else:
            numeric = robot_model.ik_numeric(
                target_frame_pose,
                q0_deg=seed,
                rotation_weight_mm=0.0,
                max_nfev=ik_config.NUMERIC_MAX_NFEV,
                raise_on_fail=False,
            )
            q_solution = robot_model.clip_joints(numeric.q_deg)
            backend_ok = bool(numeric.success)
            backend_message = str(numeric.message)
            joint_distance_to_previous = float(joint_distance_deg(q_solution, previous_q))
            selected_by = "numeric_position_only"

        solved_frame = robot_model.fk_tcp_in_frame(q_solution)
        pos_err_mm = float(np.linalg.norm(solved_frame[:3, 3] - target_frame_pose[:3, 3]))
        ori_err_deg = float(rotation_error_deg(solved_frame, target_frame_pose))
        discontinuous_warning = bool(joint_distance_to_previous > max_joint_jump_warn_deg)

        with self.lock:
            self.q_last_solution_deg = q_solution.copy()

        return {
            "ok": True,
            "backend_success": backend_ok,
            "backend_message": backend_message,
            "selected_by": selected_by,
            "candidate_count": int(candidate_count),
            "shortlisted_count": int(shortlisted_count),
            "joint_distance_to_previous_deg": float(joint_distance_to_previous),
            "discontinuous_warning": discontinuous_warning,
            "q_deg": _vec6(q_solution),
            "position_error_mm": pos_err_mm,
            "orientation_error_deg": ori_err_deg,
            "solved_frame_pose": _to_rows(solved_frame),
        }


DEFAULT_SOLVER_SESSION = SolverSession()


def get_default_session() -> SolverSession:
    return DEFAULT_SOLVER_SESSION


def get_runtime_capabilities() -> dict[str, Any]:
    return {
        "ok": True,
        "solver_kernel": {
            "ik_fk_path_continuity": "implemented",
            "owner": "winding_pose_solver",
        },
        "program_runtime": program_runtime_capabilities(),
    }


def _raise_reserved_endpoint(
    *,
    capability: str,
    payload: dict[str, Any],
) -> NoReturn:
    raise ExternalAPIError(
        f"{capability}_not_implemented",
        (
            f"{capability} is a reserved interface. "
            "The contract exists, but execution/export logic has not been implemented yet."
        ),
        status_code=501,
        details={
            "payload_keys": sorted(payload.keys()) if isinstance(payload, dict) else None,
            "capabilities": program_runtime_capabilities(),
        },
    )


def control_simulation(payload: dict[str, Any]) -> dict[str, Any]:
    _raise_reserved_endpoint(capability="simulation_control", payload=payload)


def plan_program(payload: dict[str, Any]) -> dict[str, Any]:
    _raise_reserved_endpoint(capability="program_plan", payload=payload)


def export_program(payload: dict[str, Any]) -> dict[str, Any]:
    _raise_reserved_endpoint(capability="program_export", payload=payload)


def configure_robot(
    payload: dict[str, Any],
    *,
    session: SolverSession | None = None,
) -> dict[str, Any]:
    active_session = session or DEFAULT_SOLVER_SESSION
    try:
        return active_session.configure(payload)
    except Exception as exc:
        raise ExternalAPIError(
            "configure_failed",
            f"{type(exc).__name__}: {exc}",
            status_code=400,
            details={"payload_keys": sorted(payload.keys()) if isinstance(payload, dict) else None},
        ) from exc


def solve_fk(
    payload: dict[str, Any],
    *,
    session: SolverSession | None = None,
) -> dict[str, Any]:
    active_session = session or DEFAULT_SOLVER_SESSION
    try:
        return active_session.fk(payload)
    except Exception as exc:
        raise ExternalAPIError(
            "fk_failed",
            f"{type(exc).__name__}: {exc}",
            status_code=400,
        ) from exc


def solve_ik(
    payload: dict[str, Any],
    *,
    session: SolverSession | None = None,
) -> dict[str, Any]:
    active_session = session or DEFAULT_SOLVER_SESSION
    try:
        return active_session.ik(payload)
    except Exception as exc:
        raise ExternalAPIError(
            "ik_failed",
            f"{type(exc).__name__}: {exc}",
            status_code=400,
        ) from exc


def _as_profile_request(payload: dict[str, Any]) -> ProfileEvaluationRequest:
    source = payload
    if isinstance(payload.get("request"), dict):
        source = payload["request"]
    return ProfileEvaluationRequest.from_dict(source)


def _as_batch_request(payload: dict[str, Any]) -> EvaluationBatchRequest:
    source = payload
    if isinstance(payload.get("batch"), dict):
        source = payload["batch"]
    if "evaluations" in source:
        return EvaluationBatchRequest.from_dict(source)
    if "base_request" in source:
        remote_request = RemoteSearchRequest.from_dict(source)
        return EvaluationBatchRequest(evaluations=(remote_request.base_request,))
    if isinstance(source.get("request"), dict):
        return EvaluationBatchRequest(
            evaluations=(ProfileEvaluationRequest.from_dict(source["request"]),)
        )
    return EvaluationBatchRequest(evaluations=(ProfileEvaluationRequest.from_dict(source),))


def solve_path_request(payload: dict[str, Any]) -> dict[str, Any]:
    try:
        request = _as_profile_request(payload)
    except Exception as exc:
        raise ExternalAPIError(
            "path_request_invalid",
            f"{type(exc).__name__}: {exc}",
            status_code=400,
        ) from exc

    try:
        result, _search_result = evaluate_single_request(request)
    except Exception as exc:
        raise ExternalAPIError(
            "path_solve_failed",
            f"{type(exc).__name__}: {exc}",
            status_code=400,
            details={"request_id": request.request_id},
        ) from exc

    quality = result_quality_summary(result)
    return {
        "ok": True,
        "request_id": str(result.request_id),
        "result": result.to_dict(),
        "quality": quality,
        "strictly_valid": bool(result_is_strictly_valid(result)),
        "continuity_warnings": bool(result_has_continuity_warnings(result)),
        "semantic_status": result_semantic_status(result),
    }


def solve_path_batch(payload: dict[str, Any]) -> dict[str, Any]:
    try:
        batch_request = _as_batch_request(payload)
    except Exception as exc:
        raise ExternalAPIError(
            "path_batch_request_invalid",
            f"{type(exc).__name__}: {exc}",
            status_code=400,
        ) from exc

    try:
        batch_result = evaluate_batch_request(batch_request)
    except Exception as exc:
        raise ExternalAPIError(
            "path_batch_solve_failed",
            f"{type(exc).__name__}: {exc}",
            status_code=400,
            details={"request_count": len(batch_request.evaluations)},
        ) from exc

    quality_summaries = [result_quality_summary(item) for item in batch_result.results]
    best_result = _select_best_result(batch_result.results)
    best_quality = None if best_result is None else result_quality_summary(best_result)

    return {
        "ok": True,
        "result": batch_result.to_dict(),
        "quality_summaries": quality_summaries,
        "best_request_id": None if best_result is None else str(best_result.request_id),
        "best_quality": best_quality,
        "strictly_valid": all(result_is_strictly_valid(item) for item in batch_result.results),
        "continuity_warnings": any(
            result_has_continuity_warnings(item) for item in batch_result.results
        ),
    }
