from __future__ import annotations

import argparse
import json
import threading
import sys
from dataclasses import dataclass, field
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import numpy as np

_THIS_FILE = Path(__file__).resolve()
_REPO_ROOT = _THIS_FILE.parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

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


def _mat_from_payload(value: Any, fallback: np.ndarray | None = None) -> np.ndarray:
    if value is None:
        if fallback is None:
            raise ValueError("matrix is required")
        return fallback.copy()
    matrix = np.asarray(value, dtype=float)
    return as_transform(matrix)


def _joint_vector_from_payload(value: Any, fallback: np.ndarray | None = None) -> np.ndarray:
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
        number = float(value)
    except Exception:
        return float(default)
    if not np.isfinite(number):
        return float(default)
    return number


def _quality_score(position_error_mm: float, orientation_error_deg: float, include_orientation: bool) -> float:
    if include_orientation:
        return float(position_error_mm) + 0.5 * float(orientation_error_deg)
    return float(position_error_mm)


def _select_continuous_candidate(
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

    shortlist = [entry for entry in prepared if entry[3] <= best_quality + quality_slack]
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


def _fk_partial_world_rows(robot_model: RobotModel, robot_base_pose: np.ndarray, q_deg: np.ndarray) -> list[list[list[float]]]:
    rows: list[list[list[float]]] = []
    for joint_count in range(JOINT_COUNT + 1):
        partial = robot_model.fk_partial(q_deg, n_joints=joint_count)
        world = robot_base_pose @ partial
        rows.append(_to_rows(world))
    return rows


def _kinematics_from_robot_payload(robot_payload: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    kin = robot_payload.get("kinematics_inferred")
    if isinstance(kin, dict):
        try:
            axes = np.asarray(kin.get("joint_axes_base"), dtype=float).reshape((JOINT_COUNT, 3))
            points = np.asarray(kin.get("joint_points_base_mm"), dtype=float).reshape((JOINT_COUNT, 3))
            senses = np.asarray(kin.get("joint_senses"), dtype=float).reshape((JOINT_COUNT,))
            home_flange = np.asarray(kin.get("home_flange"), dtype=float).reshape((4, 4))
            return axes, points, senses, home_flange
        except Exception:
            pass

    axes = np.asarray(ik_config.LOCAL_JOINT_AXIS_DIRECTIONS_BASE, dtype=float).reshape((JOINT_COUNT, 3))
    points = np.asarray(ik_config.LOCAL_JOINT_AXIS_POINTS_BASE_MM, dtype=float).reshape((JOINT_COUNT, 3))
    senses = np.asarray(ik_config.LOCAL_JOINT_SENSES, dtype=float).reshape((JOINT_COUNT,))
    home_flange = np.asarray(ik_config.LOCAL_HOME_FLANGE_MATRIX, dtype=float).reshape((4, 4))
    return axes, points, senses, home_flange


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


@dataclass
class SolverSession:
    robot_model: RobotModel | None = None
    robot_base_pose: np.ndarray = field(default_factory=lambda: np.eye(4, dtype=float))
    q_init_deg: np.ndarray = field(default_factory=lambda: np.zeros((JOINT_COUNT,), dtype=float))
    q_last_solution_deg: np.ndarray = field(default_factory=lambda: np.zeros((JOINT_COUNT,), dtype=float))
    solver: SixAxisIKSolver | None = None
    kinematics_source: str = "unconfigured"
    lock: threading.Lock = field(default_factory=threading.Lock)

    def configured(self) -> bool:
        return self.robot_model is not None and self.solver is not None

    def configure(self, payload: dict[str, Any]) -> dict[str, Any]:
        robot_payload = payload.get("robot")
        if not isinstance(robot_payload, dict):
            raise ValueError("payload.robot is required")

        axes, points, senses, home_flange = _kinematics_from_robot_payload(robot_payload)
        lower, upper = _limits_from_robot_payload(robot_payload)

        tool_pose = _mat_from_payload(robot_payload.get("pose_tool"), ik_config.get_configured_tool_pose())
        frame_pose = _mat_from_payload(robot_payload.get("pose_frame"), ik_config.get_configured_frame_pose())
        base_pose = _mat_from_payload(robot_payload.get("pose_abs"), np.eye(4, dtype=float))
        q_init = _joint_vector_from_payload(robot_payload.get("joints_current_deg"), np.zeros((JOINT_COUNT,), dtype=float))

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
            # The exported/inferred model may not satisfy pure-analytic backend assumptions.
            # Use numeric backend for robust interactive IK across arbitrary imported stations.
            ik_backend="numeric",
        )

        with self.lock:
            self.robot_model = model
            self.robot_base_pose = base_pose
            self.q_init_deg = q_init
            self.q_last_solution_deg = q_init.copy()
            self.solver = solver
            self.kinematics_source = (
                "metadata.kinematics_inferred"
                if isinstance(robot_payload.get("kinematics_inferred"), dict)
                else "project.config.fallback"
            )

        return {
            "ok": True,
            "kinematics_source": self.kinematics_source,
            "limits_lower_deg": _vec6(lower),
            "limits_upper_deg": _vec6(upper),
            "q_init_deg": _vec6(q_init),
        }

    def fk(self, payload: dict[str, Any]) -> dict[str, Any]:
        with self.lock:
            if not self.configured():
                raise RuntimeError("solver is not configured")
            assert self.robot_model is not None
            assert self.solver is not None
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
                    ) = _select_continuous_candidate(
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
                backend_message = (
                    backend_message
                    if backend_message
                    else str(numeric.message)
                )
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


SESSION = SolverSession()


class RequestHandler(BaseHTTPRequestHandler):
    server_version = "ModelDemoSolverAPI/1.0"

    def _write_json(self, payload: dict[str, Any], status: int = HTTPStatus.OK) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self) -> None:  # noqa: N802
        self._write_json({"ok": True})

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/api/health":
            self._write_json(
                {
                    "ok": True,
                    "configured": SESSION.configured(),
                    "kinematics_source": SESSION.kinematics_source,
                }
            )
            return
        self._write_json({"ok": False, "error": "not found"}, status=HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:  # noqa: N802
        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload_raw = self.rfile.read(length) if length > 0 else b"{}"
            payload = json.loads(payload_raw.decode("utf-8"))
            if not isinstance(payload, dict):
                raise ValueError("json body must be an object")

            if self.path == "/api/configure":
                self._write_json(SESSION.configure(payload))
                return
            if self.path == "/api/fk":
                self._write_json(SESSION.fk(payload))
                return
            if self.path == "/api/ik":
                self._write_json(SESSION.ik(payload))
                return

            self._write_json({"ok": False, "error": "not found"}, status=HTTPStatus.NOT_FOUND)
        except Exception as exc:
            self._write_json({"ok": False, "error": f"{type(exc).__name__}: {exc}"}, status=HTTPStatus.BAD_REQUEST)


def main() -> int:
    parser = argparse.ArgumentParser(description="Use src.six_axis_ik as backend solver for model_demo.")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8898)
    args = parser.parse_args()

    server = ThreadingHTTPServer((args.host, args.port), RequestHandler)
    print(f"[model_demo_solver_api] listening on http://{args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
