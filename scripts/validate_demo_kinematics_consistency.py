from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np

_THIS_FILE = Path(__file__).resolve()
_REPO_ROOT = _THIS_FILE.parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.six_axis_ik import config as ik_config
from src.six_axis_ik.kinematics import RobotModel, rotation_error_deg


def _as_matrix4(value: Any, fallback: np.ndarray | None = None) -> np.ndarray:
    if value is None:
        if fallback is None:
            raise ValueError("matrix value is required")
        return fallback.copy()
    arr = np.asarray(value, dtype=float)
    if arr.shape != (4, 4):
        raise ValueError(f"matrix shape must be (4,4), got {arr.shape}")
    return arr


def _kinematics_from_robot_payload(robot: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    kin = robot.get("kinematics_inferred")
    if not isinstance(kin, dict):
        raise ValueError("metadata does not contain robots[0].kinematics_inferred")

    axes = np.asarray(kin.get("joint_axes_base"), dtype=float).reshape((6, 3))
    points = np.asarray(kin.get("joint_points_base_mm"), dtype=float).reshape((6, 3))
    senses = np.asarray(kin.get("joint_senses"), dtype=float).reshape((6,))
    home_flange = np.asarray(kin.get("home_flange"), dtype=float).reshape((4, 4))
    return axes, points, senses, home_flange


def _build_model_from_metadata(robot: dict[str, Any]) -> RobotModel:
    axes, points, senses, home_flange = _kinematics_from_robot_payload(robot)
    lower = np.asarray(robot.get("joint_limits_lower_deg"), dtype=float).reshape((6,))
    upper = np.asarray(robot.get("joint_limits_upper_deg"), dtype=float).reshape((6,))
    tool = _as_matrix4(robot.get("pose_tool"), ik_config.get_configured_tool_pose())
    frame = _as_matrix4(robot.get("pose_frame"), ik_config.get_configured_frame_pose())
    return RobotModel(
        joint_axis_directions_base=axes,
        joint_axis_points_base_mm=points,
        joint_senses=senses,
        home_flange_T=home_flange,
        joint_min_deg=lower,
        joint_max_deg=upper,
        tool_T=tool,
        frame_T=frame,
    )


def _build_model_from_project_config(robot: dict[str, Any]) -> RobotModel:
    lower = np.asarray(robot.get("joint_limits_lower_deg"), dtype=float).reshape((6,))
    upper = np.asarray(robot.get("joint_limits_upper_deg"), dtype=float).reshape((6,))
    tool = _as_matrix4(robot.get("pose_tool"), ik_config.get_configured_tool_pose())
    frame = _as_matrix4(robot.get("pose_frame"), ik_config.get_configured_frame_pose())
    return RobotModel(
        joint_axis_directions_base=np.asarray(ik_config.LOCAL_JOINT_AXIS_DIRECTIONS_BASE, dtype=float),
        joint_axis_points_base_mm=np.asarray(ik_config.LOCAL_JOINT_AXIS_POINTS_BASE_MM, dtype=float),
        joint_senses=np.asarray(ik_config.LOCAL_JOINT_SENSES, dtype=float),
        home_flange_T=np.asarray(ik_config.LOCAL_HOME_FLANGE_MATRIX, dtype=float),
        joint_min_deg=lower,
        joint_max_deg=upper,
        tool_T=tool,
        frame_T=frame,
    )


def _rand_joint(lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    return np.array([random.uniform(float(lower[i]), float(upper[i])) for i in range(6)], dtype=float)


def _mat_to_robodk_mat(rows: np.ndarray):
    from robodk.robomath import Mat

    return Mat(rows.tolist())


def _robodk_mat_to_np(mat: Any) -> np.ndarray:
    out = np.zeros((4, 4), dtype=float)
    for row in range(4):
        for col in range(4):
            out[row, col] = float(mat[row, col])
    return out


def _compare_models(model_a: RobotModel, model_b: RobotModel, samples: int) -> dict[str, Any]:
    max_pos = 0.0
    max_ori = 0.0
    mean_pos = 0.0
    mean_ori = 0.0
    lower = model_a.joint_min_deg
    upper = model_a.joint_max_deg
    for _ in range(samples):
        q = _rand_joint(lower, upper)
        Ta = model_a.fk_tcp_in_frame(q)
        Tb = model_b.fk_tcp_in_frame(q)
        pos = float(np.linalg.norm(Ta[:3, 3] - Tb[:3, 3]))
        ori = float(rotation_error_deg(Ta, Tb))
        max_pos = max(max_pos, pos)
        max_ori = max(max_ori, ori)
        mean_pos += pos
        mean_ori += ori
    return {
        "samples": int(samples),
        "max_position_error_mm": max_pos,
        "max_orientation_error_deg": max_ori,
        "mean_position_error_mm": mean_pos / max(1, samples),
        "mean_orientation_error_deg": mean_ori / max(1, samples),
    }


def _compare_with_robodk(
    model: RobotModel,
    robot_name: str,
    station_path_hint: str | None,
    station_file: str | None,
    samples: int,
) -> dict[str, Any]:
    from robodk import robolink

    rdk = robolink.Robolink()
    if station_file:
        station_path = Path(station_file)
        if station_path.exists():
            loaded = rdk.AddFile(str(station_path))
            if loaded.Valid() and loaded.Type() == robolink.ITEM_TYPE_STATION:
                rdk.setActiveStation(loaded)
    elif station_path_hint:
        station_path = Path(station_path_hint)
        if station_path.exists():
            loaded = rdk.AddFile(str(station_path))
            if loaded.Valid() and loaded.Type() == robolink.ITEM_TYPE_STATION:
                rdk.setActiveStation(loaded)

    robot = rdk.Item(robot_name, robolink.ITEM_TYPE_ROBOT)
    if not robot.Valid():
        raise RuntimeError(f"RoboDK robot '{robot_name}' not found.")

    tool_mat = _mat_to_robodk_mat(model.tool_T)
    frame_mat = _mat_to_robodk_mat(model.frame_T)

    max_pos = 0.0
    max_ori = 0.0
    mean_pos = 0.0
    mean_ori = 0.0
    for _ in range(samples):
        q = _rand_joint(model.joint_min_deg, model.joint_max_deg)
        pose_rdk = robot.SolveFK([float(v) for v in q], tool=tool_mat, reference=frame_mat)
        pose_rdk_np = _robodk_mat_to_np(pose_rdk)
        pose_model = model.fk_tcp_in_frame(q)
        pos = float(np.linalg.norm(pose_model[:3, 3] - pose_rdk_np[:3, 3]))
        ori = float(rotation_error_deg(pose_model, pose_rdk_np))
        max_pos = max(max_pos, pos)
        max_ori = max(max_ori, ori)
        mean_pos += pos
        mean_ori += ori

    return {
        "samples": int(samples),
        "max_position_error_mm": max_pos,
        "max_orientation_error_deg": max_ori,
        "mean_position_error_mm": mean_pos / max(1, samples),
        "mean_orientation_error_deg": mean_ori / max(1, samples),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate metadata/demo kinematics consistency.")
    parser.add_argument("--metadata", type=Path, required=True, help="Path to metadata.json exported from RoboDK.")
    parser.add_argument("--samples", type=int, default=80)
    parser.add_argument("--robodk-samples", type=int, default=30)
    parser.add_argument("--station", type=Path, default=None, help="Optional .rdk station file for RoboDK check.")
    parser.add_argument("--output", type=Path, default=None, help="Optional output JSON path.")
    parser.add_argument("--skip-robodk", action="store_true")
    args = parser.parse_args()

    metadata_path = args.metadata.expanduser().resolve()
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    robots = payload.get("robots") or []
    if not robots:
        raise RuntimeError("metadata has no robots[]")
    robot = robots[0]

    model_meta = _build_model_from_metadata(robot)
    model_cfg = _build_model_from_project_config(robot)

    report: dict[str, Any] = {
        "metadata": str(metadata_path),
        "robot_name": str(robot.get("name") or "unknown"),
        "station_path_hint": payload.get("station_path_hint"),
        "meta_vs_project_config": _compare_models(model_meta, model_cfg, samples=max(1, int(args.samples))),
    }

    if args.skip_robodk:
        report["meta_vs_robodk"] = {"skipped": True}
    else:
        try:
            report["meta_vs_robodk"] = _compare_with_robodk(
                model_meta,
                robot_name=str(robot.get("name") or "KUKA"),
                station_path_hint=payload.get("station_path_hint"),
                station_file=str(args.station.resolve()) if args.station else None,
                samples=max(1, int(args.robodk_samples)),
            )
        except Exception as exc:
            report["meta_vs_robodk"] = {
                "error": f"{type(exc).__name__}: {exc}",
            }

    cfg = report["meta_vs_project_config"]
    cfg_ok = (
        float(cfg["max_position_error_mm"]) <= 1e-6
        and float(cfg["max_orientation_error_deg"]) <= 1e-6
    )
    rdk = report["meta_vs_robodk"]
    if isinstance(rdk, dict) and "max_position_error_mm" in rdk:
        rdk_ok = (
            float(rdk["max_position_error_mm"]) <= 0.05
            and float(rdk["max_orientation_error_deg"]) <= 0.05
        )
    elif isinstance(rdk, dict) and rdk.get("skipped"):
        rdk_ok = True
    else:
        rdk_ok = False

    report["pass"] = bool(cfg_ok and rdk_ok)
    if args.output:
        output_path = args.output.expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
