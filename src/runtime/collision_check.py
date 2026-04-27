from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from src.six_axis_ik.kinematics import JOINT_COUNT, as_joint_vector, as_transform


@dataclass(frozen=True)
class LinkCollisionAsset:
    link_id: int
    name: str
    mesh_path: Path
    bounds_path: Path | None
    bind_pose_robot: np.ndarray
    visual_pose_correction_robot: np.ndarray


@dataclass(frozen=True)
class StaticCollisionAsset:
    static_id: int
    name: str
    mesh_path: Path
    bounds_path: Path | None
    pose_robot: np.ndarray


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _manifest_from_payload(payload: dict[str, Any]) -> tuple[dict[str, Any], Path]:
    manifest_payload = payload.get("asset_manifest")
    if isinstance(manifest_payload, dict):
        return manifest_payload, Path.cwd()

    manifest_path_raw = payload.get("asset_manifest_path")
    if not manifest_path_raw:
        raise ValueError("asset_manifest_path or asset_manifest is required")
    manifest_path = Path(str(manifest_path_raw)).expanduser().resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError(f"asset manifest not found: {manifest_path}")
    return _read_json(manifest_path), manifest_path.parent


def _asset_path(asset_root: Path, ref: dict[str, Any]) -> Path:
    file_value = ref.get("file")
    if not file_value:
        raise ValueError("collision asset is missing file")
    path = Path(str(file_value))
    if not path.is_absolute():
        path = asset_root / path
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"collision asset not found: {path}")
    return path


def _optional_asset_path(asset_root: Path, ref: dict[str, Any], key: str) -> Path | None:
    file_value = ref.get(key)
    if not file_value:
        return None
    path = Path(str(file_value))
    if not path.is_absolute():
        path = asset_root / path
    path = path.resolve()
    if not path.is_file():
        return None
    return path


def _as_bind_pose(value: Any) -> np.ndarray:
    if value is None:
        return np.eye(4, dtype=float)
    return as_transform(np.asarray(value, dtype=float))


def _load_link_assets(manifest: dict[str, Any], asset_root: Path) -> list[LinkCollisionAsset]:
    entries = manifest.get("robot_link_visuals")
    if not isinstance(entries, list) or not entries:
        raise ValueError("asset manifest is missing robot_link_visuals")

    assets: list[LinkCollisionAsset] = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        collision = entry.get("collision_asset")
        if not isinstance(collision, dict):
            raise ValueError(f"robot_link_visuals[{entry.get('link_id')}] is missing collision_asset")
        if bool(collision.get("shared_with_visual")) or collision.get("source") == "visual_mesh_fallback":
            raise ValueError(
                f"robot_link_visuals[{entry.get('link_id')}] collision_asset is visual fallback; "
                "strict collision checking requires a dedicated collision mesh"
            )
        link_id = int(entry.get("link_id"))
        if link_id < 0 or link_id > JOINT_COUNT:
            raise ValueError(f"invalid link_id: {link_id}")
        assets.append(
            LinkCollisionAsset(
                link_id=link_id,
                name=str(entry.get("name") or f"link_{link_id}"),
                mesh_path=_asset_path(asset_root, collision),
                bounds_path=_optional_asset_path(asset_root, collision, "bounds_file"),
                bind_pose_robot=_as_bind_pose(entry.get("bind_pose_robot")),
                visual_pose_correction_robot=_as_bind_pose(entry.get("visual_pose_correction_robot")),
            )
        )
    return sorted(assets, key=lambda item: item.link_id)


def _load_static_assets(
    manifest: dict[str, Any],
    asset_root: Path,
    robot_base_pose: np.ndarray,
    *,
    allow_visual_fallback: bool,
) -> list[StaticCollisionAsset]:
    entries = manifest.get("static_visuals")
    if not isinstance(entries, list):
        return []

    robot_from_world = np.linalg.inv(robot_base_pose)
    assets: list[StaticCollisionAsset] = []
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            continue
        collision = entry.get("collision_asset")
        if not isinstance(collision, dict):
            continue
        if (
            not allow_visual_fallback
            and (bool(collision.get("shared_with_visual")) or collision.get("source") == "visual_mesh_fallback")
        ):
            raise ValueError(
                f"static_visuals[{index}] collision_asset is visual fallback; "
                "static collision checking requires a dedicated collision mesh"
            )
        pose_world = _as_bind_pose(entry.get("pose_world"))
        assets.append(
            StaticCollisionAsset(
                static_id=index,
                name=str(entry.get("name") or f"static_{index}"),
                mesh_path=_asset_path(asset_root, collision),
                bounds_path=_optional_asset_path(asset_root, collision, "bounds_file"),
                pose_robot=robot_from_world @ pose_world,
            )
        )
    return assets


@lru_cache(maxsize=128)
def _mesh_vertices(mesh_path_text: str) -> np.ndarray:
    try:
        import trimesh  # type: ignore
    except Exception as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("trimesh is required for collision checking") from exc

    mesh = trimesh.load_mesh(mesh_path_text, process=False)
    vertices = np.asarray(mesh.vertices, dtype=float)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or len(vertices) == 0:
        raise ValueError(f"collision mesh has no vertices: {mesh_path_text}")
    return vertices


@lru_cache(maxsize=64)
def _mesh_component_bounds(mesh_path_text: str) -> tuple[tuple[tuple[float, float, float], tuple[float, float, float]], ...]:
    try:
        import trimesh  # type: ignore
    except Exception as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("trimesh is required for collision checking") from exc

    mesh = trimesh.load_mesh(mesh_path_text, process=False)
    try:
        components = mesh.split(only_watertight=False)
    except Exception:
        components = []
    if not components:
        vertices = _mesh_vertices(mesh_path_text)
        return ((tuple(vertices.min(axis=0)), tuple(vertices.max(axis=0))),)

    bounds: list[tuple[tuple[float, float, float], tuple[float, float, float]]] = []
    for component in components:
        vertices = np.asarray(component.vertices, dtype=float)
        if vertices.ndim == 2 and vertices.shape[1] == 3 and len(vertices) > 0:
            bounds.append((tuple(vertices.min(axis=0)), tuple(vertices.max(axis=0))))
    if not bounds:
        vertices = _mesh_vertices(mesh_path_text)
        return ((tuple(vertices.min(axis=0)), tuple(vertices.max(axis=0))),)
    return tuple(bounds)


def _transform_vertices(vertices: np.ndarray, transform: np.ndarray) -> np.ndarray:
    ones = np.ones((vertices.shape[0], 1), dtype=float)
    homogeneous = np.concatenate([vertices, ones], axis=1)
    transformed = (transform @ homogeneous.T).T
    return transformed[:, :3]


def _transform_bound(
    min_xyz: tuple[float, float, float],
    max_xyz: tuple[float, float, float],
    transform: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    lower = np.asarray(min_xyz, dtype=float)
    upper = np.asarray(max_xyz, dtype=float)
    corners = np.asarray(
        [
            [lower[0], lower[1], lower[2]],
            [lower[0], lower[1], upper[2]],
            [lower[0], upper[1], lower[2]],
            [lower[0], upper[1], upper[2]],
            [upper[0], lower[1], lower[2]],
            [upper[0], lower[1], upper[2]],
            [upper[0], upper[1], lower[2]],
            [upper[0], upper[1], upper[2]],
        ],
        dtype=float,
    )
    transformed = _transform_vertices(corners, transform)
    return transformed.min(axis=0), transformed.max(axis=0)


def _bounds_for_link_components(
    asset: LinkCollisionAsset,
    current_frame_robot: np.ndarray,
) -> list[tuple[np.ndarray, np.ndarray]]:
    transform = current_frame_robot @ asset.visual_pose_correction_robot @ np.linalg.inv(asset.bind_pose_robot)
    if asset.bounds_path is not None:
        payload = _read_json(asset.bounds_path)
        entries = payload.get("bounds")
        if isinstance(entries, list) and entries:
            bounds: list[tuple[np.ndarray, np.ndarray]] = []
            for entry in entries:
                if not isinstance(entry, dict):
                    continue
                min_xyz = entry.get("min")
                max_xyz = entry.get("max")
                if min_xyz is None or max_xyz is None:
                    continue
                bounds.append(_transform_bound(tuple(min_xyz), tuple(max_xyz), transform))
            if bounds:
                return bounds
    return [
        _transform_bound(min_xyz, max_xyz, transform)
        for min_xyz, max_xyz in _mesh_component_bounds(str(asset.mesh_path))
    ]


def _bounds_for_static_components(asset: StaticCollisionAsset) -> list[tuple[np.ndarray, np.ndarray]]:
    if asset.bounds_path is not None:
        payload = _read_json(asset.bounds_path)
        entries = payload.get("bounds")
        if isinstance(entries, list) and entries:
            bounds: list[tuple[np.ndarray, np.ndarray]] = []
            for entry in entries:
                if not isinstance(entry, dict):
                    continue
                min_xyz = entry.get("min")
                max_xyz = entry.get("max")
                if min_xyz is None or max_xyz is None:
                    continue
                bounds.append(_transform_bound(tuple(min_xyz), tuple(max_xyz), asset.pose_robot))
            if bounds:
                return bounds
    return [
        _transform_bound(min_xyz, max_xyz, asset.pose_robot)
        for min_xyz, max_xyz in _mesh_component_bounds(str(asset.mesh_path))
    ]


def _aabb_overlap(
    left_min: np.ndarray,
    left_max: np.ndarray,
    right_min: np.ndarray,
    right_max: np.ndarray,
    padding_mm: float,
) -> tuple[bool, float]:
    overlap = np.minimum(left_max, right_max) - np.maximum(left_min, right_min)
    overlap = overlap + float(padding_mm)
    if np.any(overlap <= 0.0):
        return False, 0.0
    return True, float(np.min(overlap))


def _normalize_link_pair(left_id: int, right_id: int) -> tuple[int, int]:
    return (min(int(left_id), int(right_id)), max(int(left_id), int(right_id)))


def _allowed_link_pairs_from_payload(payload: dict[str, Any], manifest: dict[str, Any]) -> set[tuple[int, int]]:
    pairs: set[tuple[int, int]] = set()
    for source in (manifest.get("allowed_collision_pairs"), payload.get("allowed_link_collision_pairs")):
        if not isinstance(source, list):
            continue
        for entry in source:
            if isinstance(entry, dict):
                left = entry.get("left_link_id", entry.get("left"))
                right = entry.get("right_link_id", entry.get("right"))
            elif isinstance(entry, (list, tuple)) and len(entry) >= 2:
                left, right = entry[0], entry[1]
            else:
                continue
            try:
                pairs.add(_normalize_link_pair(int(left), int(right)))
            except Exception:
                continue
    return pairs


def _should_ignore_pair(
    left_id: int,
    right_id: int,
    ignore_adjacent: bool,
    allowed_link_pairs: set[tuple[int, int]],
) -> bool:
    if left_id == right_id:
        return True
    if ignore_adjacent and abs(left_id - right_id) <= 1:
        return True
    if _normalize_link_pair(left_id, right_id) in allowed_link_pairs:
        return True
    return False


def _joint_path_from_payload(payload: dict[str, Any]) -> list[tuple[int, np.ndarray]] | None:
    raw = payload.get("q_path_deg")
    if raw is None:
        raw = payload.get("path_q_deg")
    if raw is None:
        return None
    rows = np.asarray(raw, dtype=float)
    if rows.ndim != 2 or rows.shape[1] != JOINT_COUNT:
        raise ValueError("q_path_deg must be an Nx6 joint array")
    if rows.shape[0] == 0:
        raise ValueError("q_path_deg must not be empty")
    stride = int(payload.get("sample_stride", 1) or 1)
    if stride < 1:
        raise ValueError("sample_stride must be >= 1")
    return [
        (int(index), as_joint_vector(rows[index]))
        for index in range(0, rows.shape[0], stride)
    ]


def _collision_method(static_assets: list[StaticCollisionAsset]) -> str:
    return "link_static_aabb_broadphase" if static_assets else "link_aabb_broadphase"


def _check_pose_collision(
    *,
    q_input: np.ndarray,
    session: Any,
    link_assets: list[LinkCollisionAsset],
    static_assets: list[StaticCollisionAsset],
    static_bounds: dict[int, list[tuple[np.ndarray, np.ndarray]]],
    padding_mm: float,
    ignore_adjacent: bool,
    allowed_link_pairs: set[tuple[int, int]],
    check_static_base_link: bool,
) -> dict[str, Any]:
    fk = session.fk({"q_deg": q_input})
    joint_frames_robot = [
        as_transform(np.asarray(matrix, dtype=float))
        for matrix in fk["joint_frames_robot"]
    ]

    bounds: dict[int, list[tuple[np.ndarray, np.ndarray]]] = {}
    for asset in link_assets:
        if asset.link_id >= len(joint_frames_robot):
            raise ValueError(f"FK response missing frame for link_id={asset.link_id}")
        bounds[asset.link_id] = _bounds_for_link_components(asset, joint_frames_robot[asset.link_id])

    checked_pairs: list[dict[str, Any]] = []
    ignored_pairs: list[dict[str, Any]] = []
    collisions: list[dict[str, Any]] = []
    for left_index, left in enumerate(link_assets):
        for right in link_assets[left_index + 1 :]:
            pair = {"left_link_id": left.link_id, "right_link_id": right.link_id}
            if _should_ignore_pair(left.link_id, right.link_id, ignore_adjacent, allowed_link_pairs):
                ignored_pairs.append(pair)
                continue
            checked_pairs.append(pair)
            for left_component_index, (left_min, left_max) in enumerate(bounds[left.link_id]):
                for right_component_index, (right_min, right_max) in enumerate(bounds[right.link_id]):
                    overlaps, overlap_min = _aabb_overlap(left_min, left_max, right_min, right_max, padding_mm)
                    if overlaps:
                        collisions.append(
                            {
                                **pair,
                                "left_kind": "robot_link",
                                "right_kind": "robot_link",
                                "left_name": left.name,
                                "right_name": right.name,
                                "left_component_index": left_component_index,
                                "right_component_index": right_component_index,
                                "minimum_overlap_mm": overlap_min,
                            }
                        )

    for link in link_assets:
        if link.link_id == 0 and not check_static_base_link:
            continue
        for static in static_assets:
            pair = {
                "left_kind": "robot_link",
                "right_kind": "static_object",
                "left_link_id": link.link_id,
                "right_static_id": static.static_id,
            }
            checked_pairs.append(pair)
            for left_component_index, (left_min, left_max) in enumerate(bounds[link.link_id]):
                for right_component_index, (right_min, right_max) in enumerate(static_bounds[static.static_id]):
                    overlaps, overlap_min = _aabb_overlap(left_min, left_max, right_min, right_max, padding_mm)
                    if not overlaps:
                        continue
                    collisions.append(
                        {
                            **pair,
                            "left_component_index": left_component_index,
                            "right_component_index": right_component_index,
                            "left_name": link.name,
                            "right_name": static.name,
                            "minimum_overlap_mm": overlap_min,
                        }
                    )

    return {
        "q_deg": [float(value) for value in fk["q_deg"]],
        "collision_free": len(collisions) == 0,
        "checked_pair_count": len(checked_pairs),
        "ignored_pair_count": len(ignored_pairs),
        "collision_count": len(collisions),
        "collisions": collisions,
        "checked_pairs": checked_pairs,
        "ignored_pairs": ignored_pairs,
    }


def check_collision_with_session(payload: dict[str, Any], session: Any) -> dict[str, Any]:
    if not session.configured():
        raise RuntimeError("solver is not configured")
    manifest, asset_root = _manifest_from_payload(payload)
    robot_base_pose = as_transform(np.asarray(session.robot_base_pose, dtype=float))

    manifest_hash = manifest.get("kinematics_hash")
    if (
        manifest_hash
        and session.kinematics_hash
        and str(manifest_hash) != str(session.kinematics_hash)
        and not bool(payload.get("allow_kinematics_mismatch", False))
    ):
        raise ValueError(
            "asset manifest kinematics_hash does not match the active solver model"
        )

    link_assets = _load_link_assets(manifest, asset_root)
    static_assets = (
        _load_static_assets(
            manifest,
            asset_root,
            robot_base_pose,
            allow_visual_fallback=bool(payload.get("allow_static_visual_fallback", False)),
        )
        if bool(payload.get("include_static_collision", True))
        else []
    )
    padding_mm = float(payload.get("padding_mm", 0.0) or 0.0)
    ignore_adjacent = bool(payload.get("ignore_adjacent_links", True))
    allowed_link_pairs = _allowed_link_pairs_from_payload(payload, manifest)
    check_static_base_link = bool(payload.get("check_static_base_link", False))
    stop_on_first_collision = bool(payload.get("stop_on_first_collision", False))
    include_sample_details = bool(payload.get("include_sample_details", True))
    method = _collision_method(static_assets)

    static_bounds: dict[int, list[tuple[np.ndarray, np.ndarray]]] = {
        asset.static_id: _bounds_for_static_components(asset)
        for asset in static_assets
    }

    common = {
        "ok": True,
        "method": method,
        "model_id": session.model_id,
        "kinematics_hash": session.kinematics_hash,
        "link_count": len(link_assets),
        "static_object_count": len(static_assets),
        "padding_mm": padding_mm,
        "include_static_collision": bool(static_assets),
    }

    path = _joint_path_from_payload(payload)
    if path is not None:
        sample_reports: list[dict[str, Any]] = []
        all_collisions: list[dict[str, Any]] = []
        checked_pair_count = 0
        ignored_pair_count = 0
        first_collision_sample_index: int | None = None
        evaluated_sample_count = 0

        for sample_index, q_input in path:
            evaluated_sample_count += 1
            pose_result = _check_pose_collision(
                q_input=q_input,
                session=session,
                link_assets=link_assets,
                static_assets=static_assets,
                static_bounds=static_bounds,
                padding_mm=padding_mm,
                ignore_adjacent=ignore_adjacent,
                allowed_link_pairs=allowed_link_pairs,
                check_static_base_link=check_static_base_link,
            )
            checked_pair_count += int(pose_result["checked_pair_count"])
            ignored_pair_count += int(pose_result["ignored_pair_count"])
            collisions = [
                {**collision, "sample_index": sample_index}
                for collision in pose_result["collisions"]
            ]
            if collisions and first_collision_sample_index is None:
                first_collision_sample_index = sample_index
            all_collisions.extend(collisions)
            if include_sample_details:
                sample_reports.append(
                    {
                        "sample_index": sample_index,
                        "q_deg": pose_result["q_deg"],
                        "collision_free": pose_result["collision_free"],
                        "collision_count": pose_result["collision_count"],
                        "collisions": collisions,
                    }
                )
            if collisions and stop_on_first_collision:
                break

        return {
            **common,
            "mode": "path",
            "collision_free": len(all_collisions) == 0,
            "path_collision_free": len(all_collisions) == 0,
            "sample_count": len(path),
            "evaluated_sample_count": evaluated_sample_count,
            "first_collision_sample_index": first_collision_sample_index,
            "colliding_sample_count": len(
                {int(collision["sample_index"]) for collision in all_collisions}
            ),
            "checked_pair_count": checked_pair_count,
            "ignored_pair_count": ignored_pair_count,
            "collision_count": len(all_collisions),
            "collisions": all_collisions,
            "samples": sample_reports,
            "stop_on_first_collision": stop_on_first_collision,
        }

    q_input = as_joint_vector(payload.get("q_deg"))
    pose_result = _check_pose_collision(
        q_input=q_input,
        session=session,
        link_assets=link_assets,
        static_assets=static_assets,
        static_bounds=static_bounds,
        padding_mm=padding_mm,
        ignore_adjacent=ignore_adjacent,
        allowed_link_pairs=allowed_link_pairs,
        check_static_base_link=check_static_base_link,
    )
    return {
        **common,
        "mode": "pose",
        **pose_result,
    }
