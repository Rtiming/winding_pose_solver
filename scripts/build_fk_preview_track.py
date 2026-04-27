from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

_THIS_FILE = Path(__file__).resolve()
_REPO_ROOT = _THIS_FILE.parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.runtime.external_api import SolverSession


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _vec6(value: Any) -> list[float]:
    values = np.asarray(value, dtype=float).reshape((6,))
    return [float(item) for item in values]


def _interpolate(left: list[float], right: list[float], alpha: float) -> list[float]:
    return [float(a + (b - a) * alpha) for a, b in zip(left, right, strict=True)]


def _trajectory_from_csv(path: Path) -> list[list[float]]:
    rows: list[list[float]] = []
    with path.expanduser().resolve().open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames:
            joint_columns = [f"j{i}_deg" for i in range(1, 7)]
            if all(column in reader.fieldnames for column in joint_columns):
                for row in reader:
                    rows.append([float(row[column]) for column in joint_columns])
                return rows

        handle.seek(0)
        raw_reader = csv.reader(handle)
        for row in raw_reader:
            numeric: list[float] = []
            for value in row:
                try:
                    numeric.append(float(value))
                except ValueError:
                    continue
            if len(numeric) >= 6:
                rows.append(numeric[-6:])
    if not rows:
        raise ValueError(f"trajectory CSV contains no 6-axis joint rows: {path}")
    return rows


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a browser preview FK track using winding_pose_solver FK."
    )
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=120)
    parser.add_argument("--duration-ms", type=int, default=9000)
    parser.add_argument(
        "--mode",
        choices=("export-hold", "home-export-home"),
        default="export-hold",
        help=(
            "export-hold shows the actual RoboDK export/current pose without "
            "inventing motion; home-export-home keeps the old diagnostic sweep."
        ),
    )
    parser.add_argument(
        "--trajectory-csv",
        type=Path,
        default=None,
        help=(
            "Optional selected_joint_path.csv. When provided, q_deg samples are "
            "loaded from the CSV and FK is evaluated for each row."
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    metadata = _load_json(args.metadata.expanduser().resolve())
    robots = metadata.get("robots")
    if not isinstance(robots, list) or not robots:
        raise ValueError("metadata.robots[0] is required")
    robot = robots[0]

    q_home = _vec6(robot.get("joints_home_deg", [0, 0, 0, 0, 0, 0]))
    q_export = _vec6(robot.get("joints_at_export_deg", robot.get("joints_current_deg", q_home)))
    trajectory_rows = _trajectory_from_csv(args.trajectory_csv) if args.trajectory_csv is not None else None
    sample_count = len(trajectory_rows) if trajectory_rows is not None else max(1, int(args.samples))
    session = SolverSession()
    configure_result = session.configure({"robot": robot, "strict_kinematics": True})
    home_fk = session.fk({"q_deg": q_home})
    export_fk = session.fk({"q_deg": q_export})

    samples: list[dict[str, Any]] = []
    for index in range(sample_count):
        phase = index / float(max(1, sample_count - 1))
        if trajectory_rows is not None:
            q = trajectory_rows[index]
        elif args.mode == "export-hold":
            q = q_export
        else:
            if phase <= 0.5:
                alpha = phase * 2.0
                q = _interpolate(q_home, q_export, alpha)
            else:
                alpha = (phase - 0.5) * 2.0
                q = _interpolate(q_export, q_home, alpha)
        fk = session.fk({"q_deg": q})
        samples.append(
            {
                "time_ms": round(args.duration_ms * phase, 6),
                "q_deg": fk["q_deg"],
                "joint_frames_robot": fk["joint_frames_robot"],
            }
        )

    payload = {
        "schema_version": 1,
        "source": "winding_pose_solver.fk",
        "model_id": configure_result.get("model_id"),
        "kinematics_hash": configure_result.get("kinematics_hash"),
        "frame_space": "robot",
        "profile": "selected-joint-path" if trajectory_rows is not None else args.mode,
        "source_csv": str(args.trajectory_csv.expanduser().resolve()) if args.trajectory_csv is not None else None,
        "home_q_deg": home_fk["q_deg"],
        "home_joint_frames_robot": home_fk["joint_frames_robot"],
        "q_export_deg": export_fk["q_deg"],
        "export_joint_frames_robot": export_fk["joint_frames_robot"],
        "preferred_progress": 0,
        "autoplay": trajectory_rows is not None or args.mode != "export-hold",
        "duration_ms": int(args.duration_ms),
        "sample_count": len(samples),
        "samples": samples,
    }
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "ok": True,
                "output": str(output),
                "sample_count": len(samples),
                "model_id": payload["model_id"],
                "kinematics_hash": payload["kinematics_hash"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
