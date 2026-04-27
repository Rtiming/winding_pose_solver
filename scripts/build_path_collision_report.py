from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

_THIS_FILE = Path(__file__).resolve()
_REPO_ROOT = _THIS_FILE.parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.runtime.external_api import SolverSession, check_collision, configure_robot


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a solver-owned path collision report for a preview FK track."
    )
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--asset-manifest", type=Path, required=True)
    parser.add_argument("--track", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sample-stride", type=int, default=1)
    parser.add_argument("--padding-mm", type=float, default=0.0)
    parser.add_argument("--stop-on-first-collision", action="store_true")
    parser.add_argument("--without-static-collision", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    metadata_path = args.metadata.expanduser().resolve()
    manifest_path = args.asset_manifest.expanduser().resolve()
    track_path = args.track.expanduser().resolve()

    metadata = _load_json(metadata_path)
    robots = metadata.get("robots")
    if not isinstance(robots, list) or not robots:
        raise ValueError("metadata.robots[0] is required")
    robot = robots[0]

    track = _load_json(track_path)
    samples = track.get("samples")
    if not isinstance(samples, list) or not samples:
        raise ValueError("track.samples is required")
    q_path = [sample["q_deg"] for sample in samples if isinstance(sample, dict) and "q_deg" in sample]
    if len(q_path) != len(samples):
        raise ValueError("every track sample must contain q_deg")

    session = SolverSession()
    configure_result = configure_robot(
        {"robot": robot, "strict_kinematics": True},
        session=session,
    )
    result = check_collision(
        {
            "asset_manifest_path": str(manifest_path),
            "q_path_deg": q_path,
            "padding_mm": float(args.padding_mm),
            "ignore_adjacent_links": True,
            "include_static_collision": not bool(args.without_static_collision),
            "sample_stride": int(args.sample_stride),
            "stop_on_first_collision": bool(args.stop_on_first_collision),
            "include_sample_details": True,
        },
        session=session,
    )

    payload = {
        "schema_version": 1,
        "source": "winding_pose_solver.collision_check",
        "metadata_file": str(metadata_path),
        "asset_manifest_file": str(manifest_path),
        "track_file": str(track_path),
        "model_id": configure_result.get("model_id"),
        "kinematics_hash": configure_result.get("kinematics_hash"),
        "collision": result,
    }
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(
        json.dumps(
            {
                "ok": True,
                "output": str(output),
                "mode": result.get("mode"),
                "method": result.get("method"),
                "path_collision_free": result.get("path_collision_free"),
                "sample_count": result.get("sample_count"),
                "evaluated_sample_count": result.get("evaluated_sample_count"),
                "collision_count": result.get("collision_count"),
                "first_collision_sample_index": result.get("first_collision_sample_index"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
