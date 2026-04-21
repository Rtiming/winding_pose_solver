from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

_THIS_FILE = Path(__file__).resolve()
_REPO_ROOT = _THIS_FILE.parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.six_axis_ik import config as ik_config
from src.six_axis_ik.kinematics import JOINT_COUNT, RobotModel, rotation_error_deg


def _as_matrix4(value: Any, fallback: np.ndarray | None = None) -> np.ndarray:
    if value is None:
        if fallback is None:
            raise ValueError("matrix value is required")
        return np.asarray(fallback, dtype=float).reshape((4, 4))
    arr = np.asarray(value, dtype=float)
    if arr.shape != (4, 4):
        raise ValueError(f"matrix shape must be (4,4), got {arr.shape}")
    return arr


def _as_joint_vec(value: Any) -> np.ndarray:
    arr = np.asarray(value, dtype=float).reshape((-1,))
    if arr.shape != (JOINT_COUNT,):
        raise ValueError(f"joint vector must have {JOINT_COUNT} elements, got {arr.shape}")
    return arr


def _pose_from_request_row(row: dict[str, Any]) -> np.ndarray:
    pose = np.eye(4, dtype=float)
    pose[0, 0] = float(row["r11"])
    pose[0, 1] = float(row["r12"])
    pose[0, 2] = float(row["r13"])
    pose[1, 0] = float(row["r21"])
    pose[1, 1] = float(row["r22"])
    pose[1, 2] = float(row["r23"])
    pose[2, 0] = float(row["r31"])
    pose[2, 1] = float(row["r32"])
    pose[2, 2] = float(row["r33"])
    pose[0, 3] = float(row["x_mm"])
    pose[1, 3] = float(row["y_mm"])
    pose[2, 3] = float(row["z_mm"])
    return pose


def _kinematics_from_metadata_robot(robot_payload: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    kin = robot_payload.get("kinematics_inferred")
    if isinstance(kin, dict):
        axes = np.asarray(kin.get("joint_axes_base"), dtype=float).reshape((JOINT_COUNT, 3))
        points = np.asarray(kin.get("joint_points_base_mm"), dtype=float).reshape((JOINT_COUNT, 3))
        senses = np.asarray(kin.get("joint_senses"), dtype=float).reshape((JOINT_COUNT,))
        home_flange = np.asarray(kin.get("home_flange"), dtype=float).reshape((4, 4))
        return axes, points, senses, home_flange

    axes = np.asarray(ik_config.LOCAL_JOINT_AXIS_DIRECTIONS_BASE, dtype=float).reshape((JOINT_COUNT, 3))
    points = np.asarray(ik_config.LOCAL_JOINT_AXIS_POINTS_BASE_MM, dtype=float).reshape((JOINT_COUNT, 3))
    senses = np.asarray(ik_config.LOCAL_JOINT_SENSES, dtype=float).reshape((JOINT_COUNT,))
    home_flange = np.asarray(ik_config.LOCAL_HOME_FLANGE_MATRIX, dtype=float).reshape((4, 4))
    return axes, points, senses, home_flange


def _build_robot_model(metadata: dict[str, Any]) -> tuple[RobotModel, str]:
    robots = metadata.get("robots") or []
    if not robots:
        raise RuntimeError("metadata has no robots[]")
    robot = robots[0]
    axes, points, senses, home_flange = _kinematics_from_metadata_robot(robot)
    lower = np.asarray(robot.get("joint_limits_lower_deg"), dtype=float).reshape((JOINT_COUNT,))
    upper = np.asarray(robot.get("joint_limits_upper_deg"), dtype=float).reshape((JOINT_COUNT,))
    tool = _as_matrix4(robot.get("pose_tool"), ik_config.get_configured_tool_pose())
    frame = _as_matrix4(robot.get("pose_frame"), ik_config.get_configured_frame_pose())
    model = RobotModel(
        joint_axis_directions_base=axes,
        joint_axis_points_base_mm=points,
        joint_senses=senses,
        home_flange_T=home_flange,
        joint_min_deg=lower,
        joint_max_deg=upper,
        tool_T=tool,
        frame_T=frame,
    )
    source = "metadata.kinematics_inferred" if isinstance(robot.get("kinematics_inferred"), dict) else "project.config.fallback"
    return model, source


def _load_trajectory(csv_path: Path) -> np.ndarray:
    rows: list[list[float]] = []
    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        required = [f"j{i}_deg" for i in range(1, JOINT_COUNT + 1)]
        for key in required:
            if key not in (reader.fieldnames or []):
                raise RuntimeError(f"CSV missing required column: {key}")
        for row in reader:
            rows.append([float(row[f"j{i}_deg"]) for i in range(1, JOINT_COUNT + 1)])
    if not rows:
        raise RuntimeError("trajectory CSV has no data rows")
    return np.asarray(rows, dtype=float)


def _load_request_reference(request_path: Path) -> list[np.ndarray]:
    payload = json.loads(request_path.read_text(encoding="utf-8"))
    rows = payload.get("reference_pose_rows") or []
    return [_pose_from_request_row(r) for r in rows]


def _read_eval_metrics(eval_result_path: Path) -> dict[str, Any]:
    payload = json.loads(eval_result_path.read_text(encoding="utf-8"))
    results = payload.get("results") or []
    if not results:
        return {}
    first = results[0]
    return {
        "status": first.get("status"),
        "invalid_row_count": first.get("invalid_row_count"),
        "ik_empty_row_count": first.get("ik_empty_row_count"),
        "config_switches": first.get("config_switches"),
        "bridge_like_segments": first.get("bridge_like_segments"),
        "big_circle_step_count": first.get("big_circle_step_count"),
        "worst_joint_step_deg": first.get("worst_joint_step_deg"),
        "mean_joint_step_deg": first.get("mean_joint_step_deg"),
        "gate_tier": first.get("gate_tier"),
        "block_reasons": first.get("block_reasons"),
    }


def _percentiles(arr: np.ndarray, pcts: tuple[int, ...] = (50, 90, 95, 99)) -> dict[str, float]:
    if arr.size == 0:
        return {f"p{p}": float("nan") for p in pcts}
    return {f"p{p}": float(np.percentile(arr, p)) for p in pcts}


def _analyze_mesh_assets(metadata_path: Path, metadata: dict[str, Any]) -> dict[str, Any]:
    scene_objects = metadata.get("scene_objects") or []
    meshes_dir = metadata_path.parent / "meshes"

    required_robot_links = {"base", "j1", "j2", "j3", "j4", "j5", "j6"}
    robot_links_found: set[str] = set()
    mesh_file_missing = 0
    mesh_file_too_small = 0
    mesh_file_ok = 0

    for obj in scene_objects:
        name = str(obj.get("name") or "").lower()
        mesh_file = obj.get("mesh_file")
        if name in required_robot_links:
            robot_links_found.add(name)
        if not mesh_file:
            continue
        mesh_path = meshes_dir / str(mesh_file)
        if not mesh_path.exists():
            mesh_file_missing += 1
            continue
        size = mesh_path.stat().st_size
        if size < 84:
            mesh_file_too_small += 1
        else:
            mesh_file_ok += 1

    failed_exports = [obj for obj in scene_objects if obj.get("mesh_export_ok") is False]
    placeholder_failed = []
    critical_failed = []
    for obj in failed_exports:
        err = str(obj.get("mesh_export_error") or "").lower()
        if "placeholder" in err or "suspiciously small" in err:
            placeholder_failed.append(obj)
        else:
            critical_failed.append(obj)
    missing_robot_links = sorted(required_robot_links - robot_links_found)
    return {
        "scene_objects_count": len(scene_objects),
        "mesh_export_failed_count": len(failed_exports),
        "mesh_export_failed_names": [str(obj.get("name") or "") for obj in failed_exports],
        "mesh_export_placeholder_failed_count": len(placeholder_failed),
        "mesh_export_critical_failed_count": len(critical_failed),
        "mesh_export_critical_failed_names": [str(obj.get("name") or "") for obj in critical_failed],
        "mesh_files_ok_count": mesh_file_ok,
        "mesh_files_missing_count": mesh_file_missing,
        "mesh_files_too_small_count": mesh_file_too_small,
        "robot_links_found": sorted(robot_links_found),
        "robot_links_missing": missing_robot_links,
        "required_robot_links_ok": len(missing_robot_links) == 0,
    }


def _analyze_trajectory_metrics(traj_deg: np.ndarray) -> dict[str, Any]:
    n = int(traj_deg.shape[0])
    if n <= 1:
        return {
            "point_count": n,
            "step_count": 0,
            "max_abs_joint_step_deg_per_axis": [0.0] * JOINT_COUNT,
            "worst_joint_step_deg": 0.0,
            "mean_joint_step_deg": 0.0,
            "closure_delta_deg": [0.0] * JOINT_COUNT,
            "closure_i1_i5_max_abs_deg": 0.0,
            "closure_i6_delta_deg": 0.0,
            "closure_i6_full_turn_error_deg": 360.0,
        }

    step = np.diff(traj_deg, axis=0)
    abs_step = np.abs(step)
    max_abs_per_axis = np.max(abs_step, axis=0)
    worst_joint_step_deg = float(np.max(max_abs_per_axis))
    mean_joint_step_deg = float(np.mean(np.max(abs_step, axis=1)))

    closure_delta = traj_deg[-1, :] - traj_deg[0, :]
    i1_i5_max = float(np.max(np.abs(closure_delta[:5])))
    i6 = float(closure_delta[5])
    i6_full_turn_error = min(abs(i6 - 360.0), abs(i6 + 360.0))

    return {
        "point_count": n,
        "step_count": n - 1,
        "max_abs_joint_step_deg_per_axis": [float(v) for v in max_abs_per_axis],
        "worst_joint_step_deg": worst_joint_step_deg,
        "mean_joint_step_deg": mean_joint_step_deg,
        "closure_delta_deg": [float(v) for v in closure_delta],
        "closure_i1_i5_max_abs_deg": i1_i5_max,
        "closure_i6_delta_deg": i6,
        "closure_i6_full_turn_error_deg": float(i6_full_turn_error),
        "step_norm_percentiles_deg": _percentiles(np.max(abs_step, axis=1)),
    }


def _analyze_fk_vs_reference(model: RobotModel, traj_deg: np.ndarray, ref_poses: list[np.ndarray]) -> dict[str, Any]:
    n = traj_deg.shape[0]
    m = len(ref_poses)
    compared = min(n, m)
    if compared <= 0:
        return {
            "compared_count": 0,
            "trajectory_count": n,
            "reference_count": m,
            "position_error_mm": None,
            "orientation_error_deg": None,
        }

    pos_errors = np.zeros((compared,), dtype=float)
    ori_errors = np.zeros((compared,), dtype=float)
    for i in range(compared):
        pose = model.fk_tcp_in_frame(traj_deg[i, :])
        ref = ref_poses[i]
        pos_errors[i] = float(np.linalg.norm(pose[:3, 3] - ref[:3, 3]))
        ori_errors[i] = float(rotation_error_deg(pose, ref))

    return {
        "compared_count": compared,
        "trajectory_count": n,
        "reference_count": m,
        "position_error_mm": {
            "max": float(np.max(pos_errors)),
            "mean": float(np.mean(pos_errors)),
            **_percentiles(pos_errors),
        },
        "orientation_error_deg": {
            "max": float(np.max(ori_errors)),
            "mean": float(np.mean(ori_errors)),
            **_percentiles(ori_errors),
        },
    }


def _derive_pass_flags(mesh_summary: dict[str, Any], traj_summary: dict[str, Any], fk_summary: dict[str, Any]) -> dict[str, bool]:
    mesh_ok = (
        mesh_summary.get("mesh_export_critical_failed_count", 1) == 0
        and mesh_summary.get("mesh_files_missing_count", 1) == 0
        and mesh_summary.get("mesh_files_too_small_count", 1) == 0
        and bool(mesh_summary.get("required_robot_links_ok"))
    )
    traj_ok = (
        float(traj_summary.get("worst_joint_step_deg", 999.0)) <= 60.0
        and float(traj_summary.get("closure_i1_i5_max_abs_deg", 999.0)) <= 1e-3
        and float(traj_summary.get("closure_i6_full_turn_error_deg", 999.0)) <= 1e-3
    )

    fk_pos = ((fk_summary.get("position_error_mm") or {}).get("max"))
    fk_ori = ((fk_summary.get("orientation_error_deg") or {}).get("max"))
    if fk_pos is None or fk_ori is None:
        fk_ok = False
    else:
        fk_ok = float(fk_pos) <= 0.05 and float(fk_ori) <= 0.05

    return {
        "mesh_ok": mesh_ok,
        "trajectory_ok": traj_ok,
        "fk_match_ok": fk_ok,
        "overall_ok": bool(mesh_ok and traj_ok and fk_ok),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Check model asset quality + continuous trajectory quality for model_demo.")
    parser.add_argument("--metadata", type=Path, required=True, help="Path to model_demo metadata.json")
    parser.add_argument("--trajectory-csv", type=Path, required=True, help="Path to selected_joint_path.csv")
    parser.add_argument("--request", type=Path, required=True, help="Path to request.json (with reference_pose_rows)")
    parser.add_argument("--eval-result", type=Path, default=None, help="Optional eval_result.json")
    parser.add_argument("--output", type=Path, default=None, help="Optional output JSON path")
    args = parser.parse_args()

    metadata_path = args.metadata.expanduser().resolve()
    csv_path = args.trajectory_csv.expanduser().resolve()
    request_path = args.request.expanduser().resolve()
    eval_result_path = args.eval_result.expanduser().resolve() if args.eval_result else None

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    model, kinematics_source = _build_robot_model(metadata)
    trajectory_deg = _load_trajectory(csv_path)
    reference_poses = _load_request_reference(request_path)

    mesh_summary = _analyze_mesh_assets(metadata_path, metadata)
    traj_summary = _analyze_trajectory_metrics(trajectory_deg)
    fk_summary = _analyze_fk_vs_reference(model, trajectory_deg, reference_poses)

    report: dict[str, Any] = {
        "metadata": str(metadata_path),
        "trajectory_csv": str(csv_path),
        "request": str(request_path),
        "kinematics_source": kinematics_source,
        "mesh_summary": mesh_summary,
        "trajectory_summary": traj_summary,
        "fk_vs_reference_summary": fk_summary,
    }

    if eval_result_path:
        report["eval_result"] = str(eval_result_path)
        report["eval_metrics"] = _read_eval_metrics(eval_result_path)

    report["pass_flags"] = _derive_pass_flags(mesh_summary, traj_summary, fk_summary)

    if args.output:
        out = args.output.expanduser().resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["pass_flags"]["overall_ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
