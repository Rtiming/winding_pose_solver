from __future__ import annotations

import heapq
import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from src.six_axis_ik.kinematics import JOINT_COUNT, as_joint_vector, as_transform

PAIR_MATRIX_LIMIT = 250_000
BVH_LEAF_TRIANGLE_COUNT = 96
TRIANGLE_DISTANCE_BATCH_SIZE = 5_000
TRIANGLE_INTERSECTION_EPS = 1e-8
BROADPHASE_COVERAGE_TOLERANCE_MM = 1e-6
BoundsArrays = tuple[np.ndarray, np.ndarray]


@dataclass(frozen=True)
class LinkCollisionAsset:
    link_id: int
    name: str
    mesh_path: Path
    visual_mesh_path: Path
    bounds_path: Path | None
    bind_pose_robot: np.ndarray
    visual_pose_correction_robot: np.ndarray


@dataclass(frozen=True)
class StaticCollisionAsset:
    static_id: int
    name: str
    mesh_path: Path
    visual_mesh_path: Path
    bounds_path: Path | None
    pose_robot: np.ndarray


@dataclass(frozen=True)
class ToolCollisionAsset:
    tool_id: int
    name: str
    mesh_path: Path
    visual_mesh_path: Path
    bounds_path: Path | None
    fk_frame_index: int
    pose_frame_local: np.ndarray


@dataclass(frozen=True)
class TriangleBVH:
    node_min: np.ndarray
    node_max: np.ndarray
    left_child: np.ndarray
    right_child: np.ndarray
    start: np.ndarray
    count: np.ndarray
    triangle_indices: np.ndarray


@dataclass(frozen=True)
class MeshPose:
    triangles: np.ndarray
    bvh: TriangleBVH | None = None
    node_min: np.ndarray | None = None
    node_max: np.ndarray | None = None
    leaf_nodes: np.ndarray | None = None
    leaf_bounds: BoundsArrays | None = None


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _file_signature(path_text: str) -> dict[str, Any]:
    path = Path(path_text)
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


@lru_cache(maxsize=256)
def _read_bounds_entries(
    bounds_path_text: str,
) -> tuple[tuple[tuple[float, float, float], tuple[float, float, float]], ...]:
    payload = _read_json(Path(bounds_path_text))
    entries = payload.get("bounds")
    if not isinstance(entries, list):
        return ()

    bounds: list[tuple[tuple[float, float, float], tuple[float, float, float]]] = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        min_xyz = entry.get("min")
        max_xyz = entry.get("max")
        if min_xyz is None or max_xyz is None:
            continue
        bounds.append(
            (
                tuple(np.asarray(min_xyz, dtype=float).tolist()),
                tuple(np.asarray(max_xyz, dtype=float).tolist()),
            )
        )
    return tuple(bounds)


@lru_cache(maxsize=256)
def _read_bounds_arrays(bounds_path_text: str) -> BoundsArrays:
    return _raw_bounds_arrays(_read_bounds_entries(bounds_path_text))


def _empty_bounds_arrays() -> BoundsArrays:
    empty = np.zeros((0, 3), dtype=float)
    return empty, empty


def _raw_bounds_arrays(
    raw_bounds: tuple[tuple[tuple[float, float, float], tuple[float, float, float]], ...],
) -> BoundsArrays:
    if not raw_bounds:
        return _empty_bounds_arrays()
    return (
        np.asarray([item[0] for item in raw_bounds], dtype=float),
        np.asarray([item[1] for item in raw_bounds], dtype=float),
    )


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


def _mesh_asset_path(asset_root: Path, ref: Any, fallback: Path) -> Path:
    if isinstance(ref, dict) and ref.get("file"):
        return _asset_path(asset_root, ref)
    return fallback


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
                visual_mesh_path=_mesh_asset_path(asset_root, entry.get("visual_asset"), _asset_path(asset_root, collision)),
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
                visual_mesh_path=_mesh_asset_path(asset_root, entry.get("visual_asset"), _asset_path(asset_root, collision)),
                bounds_path=_optional_asset_path(asset_root, collision, "bounds_file"),
                pose_robot=robot_from_world @ pose_world,
            )
        )
    return assets


def _load_tool_assets(manifest: dict[str, Any], asset_root: Path) -> list[ToolCollisionAsset]:
    entries = manifest.get("tool_visuals")
    if not isinstance(entries, list):
        return []

    assets: list[ToolCollisionAsset] = []
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            continue
        collision = entry.get("collision_asset")
        if not isinstance(collision, dict):
            continue
        if bool(collision.get("shared_with_visual")) or collision.get("source") == "visual_mesh_fallback":
            raise ValueError(
                f"tool_visuals[{index}] collision_asset is visual fallback; "
                "strict collision checking requires a dedicated collision mesh"
            )
        fk_frame_index = int(entry.get("fk_frame_index", JOINT_COUNT))
        if fk_frame_index < 0:
            raise ValueError(f"tool_visuals[{index}] has invalid fk_frame_index={fk_frame_index}")
        assets.append(
            ToolCollisionAsset(
                tool_id=index,
                name=str(entry.get("name") or f"tool_{index}"),
                mesh_path=_asset_path(asset_root, collision),
                visual_mesh_path=_mesh_asset_path(asset_root, entry.get("visual_asset"), _asset_path(asset_root, collision)),
                bounds_path=_optional_asset_path(asset_root, collision, "bounds_file"),
                fk_frame_index=fk_frame_index,
                pose_frame_local=_as_bind_pose(entry.get("pose_frame_local")),
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


@lru_cache(maxsize=32)
def _mesh_triangles(mesh_path_text: str) -> np.ndarray:
    try:
        import trimesh  # type: ignore
    except Exception as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("trimesh is required for collision checking") from exc

    mesh = trimesh.load_mesh(mesh_path_text, process=False)
    if not hasattr(mesh, "faces"):
        meshes = [item for item in mesh.dump() if hasattr(item, "faces")]
        if not meshes:
            raise ValueError(f"collision visual mesh has no triangles: {mesh_path_text}")
        mesh = trimesh.util.concatenate(meshes)
    triangles = np.asarray(mesh.triangles, dtype=float)
    if triangles.ndim != 3 or triangles.shape[1:] != (3, 3) or len(triangles) == 0:
        raise ValueError(f"collision visual mesh has no triangles: {mesh_path_text}")
    return triangles


def _transform_triangles(triangles: np.ndarray, transform: np.ndarray) -> np.ndarray:
    return triangles @ transform[:3, :3].T + transform[:3, 3]


@lru_cache(maxsize=32)
def _mesh_bvh(mesh_path_text: str) -> TriangleBVH:
    triangles = _mesh_triangles(mesh_path_text)
    tri_min, tri_max = _triangle_bounds(triangles)
    centers = (tri_min + tri_max) * 0.5

    node_min: list[np.ndarray] = []
    node_max: list[np.ndarray] = []
    left_child: list[int] = []
    right_child: list[int] = []
    start: list[int] = []
    count: list[int] = []
    ordered_triangles: list[int] = []

    def build(indices: np.ndarray) -> int:
        node_index = len(node_min)
        node_min.append(tri_min[indices].min(axis=0))
        node_max.append(tri_max[indices].max(axis=0))
        left_child.append(-1)
        right_child.append(-1)
        start.append(-1)
        count.append(0)

        if indices.size <= BVH_LEAF_TRIANGLE_COUNT:
            start[node_index] = len(ordered_triangles)
            count[node_index] = int(indices.size)
            ordered_triangles.extend(int(value) for value in indices)
            return node_index

        span = np.ptp(centers[indices], axis=0)
        axis = int(np.argmax(span))
        if not np.isfinite(span[axis]) or span[axis] <= TRIANGLE_INTERSECTION_EPS:
            sorted_indices = np.sort(indices, kind="mergesort")
        else:
            sorted_indices = indices[np.argsort(centers[indices, axis], kind="mergesort")]
        mid = max(1, sorted_indices.size // 2)
        left_child[node_index] = build(sorted_indices[:mid])
        right_child[node_index] = build(sorted_indices[mid:])
        return node_index

    build(np.arange(triangles.shape[0], dtype=int))
    return TriangleBVH(
        node_min=np.asarray(node_min, dtype=float),
        node_max=np.asarray(node_max, dtype=float),
        left_child=np.asarray(left_child, dtype=int),
        right_child=np.asarray(right_child, dtype=int),
        start=np.asarray(start, dtype=int),
        count=np.asarray(count, dtype=int),
        triangle_indices=np.asarray(ordered_triangles, dtype=int),
    )


def _mesh_pose_from_transform(mesh_path_text: str, transform: np.ndarray) -> MeshPose:
    return MeshPose(
        triangles=_transform_triangles(_mesh_triangles(mesh_path_text), transform),
    )


def _mesh_bvh_pose_from_transform(mesh_path_text: str, transform: np.ndarray) -> MeshPose:
    bvh = _mesh_bvh(mesh_path_text)
    node_min, node_max = _transform_bounds_arrays(bvh.node_min, bvh.node_max, transform)
    leaf_nodes = np.flatnonzero(bvh.count > 0)
    return MeshPose(
        triangles=_transform_triangles(_mesh_triangles(mesh_path_text), transform),
        bvh=bvh,
        node_min=node_min,
        node_max=node_max,
        leaf_nodes=leaf_nodes,
        leaf_bounds=(node_min[leaf_nodes], node_max[leaf_nodes]),
    )


@lru_cache(maxsize=32)
def _cached_mesh_bvh_pose(
    mesh_path_text: str,
    transform_key: tuple[float, ...],
) -> MeshPose:
    transform = np.asarray(transform_key, dtype=float).reshape(4, 4)
    return _mesh_bvh_pose_from_transform(mesh_path_text, transform)


@lru_cache(maxsize=32)
def _cached_transformed_triangles(
    mesh_path_text: str,
    transform_key: tuple[float, ...],
) -> np.ndarray:
    transform = np.asarray(transform_key, dtype=float).reshape(4, 4)
    return _transform_triangles(_mesh_triangles(mesh_path_text), transform)


@lru_cache(maxsize=32)
def _cached_mesh_pose(
    mesh_path_text: str,
    transform_key: tuple[float, ...],
) -> MeshPose:
    transform = np.asarray(transform_key, dtype=float).reshape(4, 4)
    return _mesh_pose_from_transform(mesh_path_text, transform)


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


def _transform_bounds(
    raw_bounds: tuple[tuple[tuple[float, float, float], tuple[float, float, float]], ...],
    transform: np.ndarray,
) -> BoundsArrays:
    lower, upper = _raw_bounds_arrays(raw_bounds)
    return _transform_bounds_arrays(lower, upper, transform)


def _transform_bounds_arrays(
    lower: np.ndarray,
    upper: np.ndarray,
    transform: np.ndarray,
) -> BoundsArrays:
    if lower.size == 0 or upper.size == 0:
        return _empty_bounds_arrays()
    corners = np.stack(
        (
            np.column_stack((lower[:, 0], lower[:, 1], lower[:, 2])),
            np.column_stack((lower[:, 0], lower[:, 1], upper[:, 2])),
            np.column_stack((lower[:, 0], upper[:, 1], lower[:, 2])),
            np.column_stack((lower[:, 0], upper[:, 1], upper[:, 2])),
            np.column_stack((upper[:, 0], lower[:, 1], lower[:, 2])),
            np.column_stack((upper[:, 0], lower[:, 1], upper[:, 2])),
            np.column_stack((upper[:, 0], upper[:, 1], lower[:, 2])),
            np.column_stack((upper[:, 0], upper[:, 1], upper[:, 2])),
        ),
        axis=1,
    )
    transformed = corners @ transform[:3, :3].T + transform[:3, 3]
    min_xyz = transformed.min(axis=1)
    max_xyz = transformed.max(axis=1)
    return min_xyz, max_xyz


def _matrix_cache_key(transform: np.ndarray) -> tuple[float, ...]:
    return tuple(float(value) for value in np.asarray(transform, dtype=float).reshape(-1))


@lru_cache(maxsize=128)
def _cached_transformed_bounds(
    bounds_path_text: str,
    transform_key: tuple[float, ...],
) -> BoundsArrays:
    transform = np.asarray(transform_key, dtype=float).reshape(4, 4)
    lower, upper = _read_bounds_arrays(bounds_path_text)
    return _transform_bounds_arrays(lower, upper, transform)


@lru_cache(maxsize=64)
def _coverage_patched_bounds_arrays(
    mesh_path_text: str,
    bounds_path_text: str,
) -> BoundsArrays:
    cache_path = Path(f"{bounds_path_text}.coverage-patched.json")
    mesh_signature = _file_signature(mesh_path_text)
    bounds_signature = _file_signature(bounds_path_text)
    try:
        cached_payload = _read_json(cache_path)
        if (
            cached_payload.get("schema") == "winding_pose_solver.coverage_patched_bounds.v1"
            and cached_payload.get("source_mesh") == mesh_signature
            and cached_payload.get("source_bounds") == bounds_signature
            and isinstance(cached_payload.get("bounds"), list)
        ):
            return _raw_bounds_arrays(
                tuple(
                    (
                        tuple(np.asarray(entry.get("min"), dtype=float).tolist()),
                        tuple(np.asarray(entry.get("max"), dtype=float).tolist()),
                    )
                    for entry in cached_payload["bounds"]
                    if isinstance(entry, dict) and entry.get("min") is not None and entry.get("max") is not None
                )
            )
    except Exception:
        pass

    bounds_min, bounds_max = _read_bounds_arrays(bounds_path_text)
    triangles = _mesh_triangles(mesh_path_text)
    tri_min, tri_max = _triangle_bounds(triangles)
    if bounds_min.size == 0:
        return tri_min, tri_max

    tri_pair, bounds_pair = _axis_sweep_candidate_pairs(
        tri_min,
        tri_max,
        bounds_min,
        bounds_max,
        BROADPHASE_COVERAGE_TOLERANCE_MM,
    )
    covered = np.zeros(tri_min.shape[0], dtype=bool)
    if tri_pair.size:
        contact_rows, _distance, _overlap = _aabb_contact_rows(
            tri_min[tri_pair],
            tri_max[tri_pair],
            bounds_min[bounds_pair],
            bounds_max[bounds_pair],
            BROADPHASE_COVERAGE_TOLERANCE_MM,
        )
        if contact_rows.size:
            covered[np.unique(tri_pair[contact_rows])] = True

    if np.all(covered):
        patched_min, patched_max = bounds_min, bounds_max
    else:
        patched_min, patched_max = (
            np.vstack((bounds_min, tri_min[~covered])),
            np.vstack((bounds_max, tri_max[~covered])),
        )
    try:
        cache_payload = {
            "schema": "winding_pose_solver.coverage_patched_bounds.v1",
            "source_mesh": mesh_signature,
            "source_bounds": bounds_signature,
            "triangle_count": int(tri_min.shape[0]),
            "original_bound_count": int(bounds_min.shape[0]),
            "patched_bound_count": int(patched_min.shape[0]),
            "patched_triangle_bound_count": int(max(0, patched_min.shape[0] - bounds_min.shape[0])),
            "bounds": [
                {"min": patched_min[index].tolist(), "max": patched_max[index].tolist()}
                for index in range(patched_min.shape[0])
            ],
        }
        cache_path.write_text(json.dumps(cache_payload, ensure_ascii=False), encoding="utf-8")
    except Exception:
        pass
    return patched_min, patched_max


@lru_cache(maxsize=128)
def _cached_coverage_patched_transformed_bounds(
    mesh_path_text: str,
    bounds_path_text: str,
    transform_key: tuple[float, ...],
) -> BoundsArrays:
    transform = np.asarray(transform_key, dtype=float).reshape(4, 4)
    lower, upper = _coverage_patched_bounds_arrays(mesh_path_text, bounds_path_text)
    return _transform_bounds_arrays(lower, upper, transform)


@lru_cache(maxsize=64)
def _bounds_coverage_stats(mesh_path_text: str, bounds_path_text: str) -> dict[str, int]:
    bounds_min, _bounds_max = _read_bounds_arrays(bounds_path_text)
    patched_min, _patched_max = _coverage_patched_bounds_arrays(mesh_path_text, bounds_path_text)
    triangle_count = int(_mesh_triangles(mesh_path_text).shape[0])
    original_bound_count = int(bounds_min.shape[0])
    patched_bound_count = int(patched_min.shape[0])
    return {
        "triangle_count": triangle_count,
        "original_bound_count": original_bound_count,
        "patched_bound_count": patched_bound_count,
        "patched_triangle_bound_count": max(0, patched_bound_count - original_bound_count),
    }


def _broadphase_coverage_summary(
    link_assets: list[LinkCollisionAsset],
    tool_assets: list[ToolCollisionAsset],
    static_assets: list[StaticCollisionAsset],
) -> dict[str, Any]:
    assets: list[dict[str, Any]] = []
    for kind, asset_id, name, mesh_path, bounds_path in (
        [
            ("robot_link", asset.link_id, asset.name, asset.visual_mesh_path, asset.bounds_path)
            for asset in link_assets
        ]
        + [
            ("tool", asset.tool_id, asset.name, asset.visual_mesh_path, asset.bounds_path)
            for asset in tool_assets
        ]
        + [
            ("static_object", asset.static_id, asset.name, asset.visual_mesh_path, asset.bounds_path)
            for asset in static_assets
        ]
    ):
        if bounds_path is None:
            continue
        stats = _bounds_coverage_stats(str(mesh_path), str(bounds_path))
        assets.append(
            {
                "kind": kind,
                "id": int(asset_id),
                "name": name,
                **stats,
            }
        )
    return {
        "mode": "visual_mesh_triangle_coverage_patched_surface_aabb",
        "asset_count": len(assets),
        "triangle_count": sum(int(item["triangle_count"]) for item in assets),
        "original_bound_count": sum(int(item["original_bound_count"]) for item in assets),
        "patched_bound_count": sum(int(item["patched_bound_count"]) for item in assets),
        "patched_triangle_bound_count": sum(int(item["patched_triangle_bound_count"]) for item in assets),
        "assets": assets,
    }


def _preload_collision_mesh_data(
    link_assets: list[LinkCollisionAsset],
    tool_assets: list[ToolCollisionAsset],
    static_assets: list[StaticCollisionAsset],
) -> None:
    seen_meshes: set[str] = set()
    for mesh_path, bounds_path in (
        [
            (asset.visual_mesh_path, asset.bounds_path)
            for asset in link_assets
        ]
        + [
            (asset.visual_mesh_path, asset.bounds_path)
            for asset in tool_assets
        ]
        + [
            (asset.visual_mesh_path, asset.bounds_path)
            for asset in static_assets
        ]
    ):
        mesh_path_text = str(mesh_path)
        if mesh_path_text not in seen_meshes:
            seen_meshes.add(mesh_path_text)
            _mesh_triangles(mesh_path_text)
            _mesh_bvh(mesh_path_text)
        if bounds_path is not None:
            _coverage_patched_bounds_arrays(mesh_path_text, str(bounds_path))


def _link_transform(asset: LinkCollisionAsset, current_frame_robot: np.ndarray) -> np.ndarray:
    return current_frame_robot @ asset.visual_pose_correction_robot @ np.linalg.inv(asset.bind_pose_robot)


def _tool_transform(asset: ToolCollisionAsset, current_frame_robot: np.ndarray) -> np.ndarray:
    return current_frame_robot @ asset.pose_frame_local


def _bounds_for_link_components(
    asset: LinkCollisionAsset,
    current_frame_robot: np.ndarray,
) -> BoundsArrays:
    transform = _link_transform(asset, current_frame_robot)
    if asset.bounds_path is not None:
        lower, upper = _coverage_patched_bounds_arrays(str(asset.visual_mesh_path), str(asset.bounds_path))
        bounds = _transform_bounds_arrays(lower, upper, transform)
        if bounds[0].size:
            return bounds
    return _transform_bounds(_mesh_component_bounds(str(asset.mesh_path)), transform)


def _bounds_for_static_components(asset: StaticCollisionAsset) -> BoundsArrays:
    if asset.bounds_path is not None:
        bounds = _cached_coverage_patched_transformed_bounds(
            str(asset.visual_mesh_path),
            str(asset.bounds_path),
            _matrix_cache_key(asset.pose_robot),
        )
        if bounds[0].size:
            return bounds
    return _transform_bounds(_mesh_component_bounds(str(asset.mesh_path)), asset.pose_robot)


def _bounds_for_tool_components(
    asset: ToolCollisionAsset,
    current_frame_robot: np.ndarray,
) -> BoundsArrays:
    transform = _tool_transform(asset, current_frame_robot)
    if asset.bounds_path is not None:
        lower, upper = _coverage_patched_bounds_arrays(str(asset.visual_mesh_path), str(asset.bounds_path))
        bounds = _transform_bounds_arrays(lower, upper, transform)
        if bounds[0].size:
            return bounds
    return _transform_bounds(_mesh_component_bounds(str(asset.mesh_path)), transform)


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


def _aabb_contact_rows(
    left_min: np.ndarray,
    left_max: np.ndarray,
    right_min: np.ndarray,
    right_max: np.ndarray,
    clearance: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    separated_gap = np.maximum(
        np.maximum(right_min - left_max, left_min - right_max),
        0.0,
    )
    distance = np.linalg.norm(separated_gap, axis=1)
    mask = distance <= clearance
    if not np.any(mask):
        empty_int = np.zeros(0, dtype=int)
        empty_float = np.zeros(0, dtype=float)
        return empty_int, empty_float, empty_float

    overlap = np.minimum(left_max[mask], right_max[mask]) - np.maximum(left_min[mask], right_min[mask])
    overlap_min = np.maximum(0.0, overlap.min(axis=1))
    return np.flatnonzero(mask), distance[mask], overlap_min


def _axis_sweep_candidate_pairs(
    left_min: np.ndarray,
    left_max: np.ndarray,
    right_min: np.ndarray,
    right_max: np.ndarray,
    clearance: float,
) -> tuple[np.ndarray, np.ndarray]:
    span_min = np.minimum(left_min.min(axis=0), right_min.min(axis=0))
    span_max = np.maximum(left_max.max(axis=0), right_max.max(axis=0))
    axis = int(np.argmax(span_max - span_min))

    order = np.argsort(right_min[:, axis], kind="mergesort")
    sorted_right_min = right_min[order, axis]
    sorted_right_max = right_max[order, axis]
    left_pairs: list[np.ndarray] = []
    right_pairs: list[np.ndarray] = []
    for left_index in range(left_min.shape[0]):
        lo = left_min[left_index, axis] - clearance
        hi = left_max[left_index, axis] + clearance
        end = int(np.searchsorted(sorted_right_min, hi, side="right"))
        if end <= 0:
            continue
        candidates = order[:end][sorted_right_max[:end] >= lo]
        if candidates.size == 0:
            continue
        left_pairs.append(np.full(candidates.shape, left_index, dtype=int))
        right_pairs.append(candidates.astype(int, copy=False))

    if not left_pairs:
        empty = np.zeros(0, dtype=int)
        return empty, empty
    return np.concatenate(left_pairs), np.concatenate(right_pairs)


def _component_contacts(
    left_bounds: BoundsArrays,
    right_bounds: BoundsArrays,
    clearance_mm: float,
) -> list[tuple[int, int, float, float]]:
    left_min, left_max = left_bounds
    right_min, right_max = right_bounds
    if left_min.size == 0 or right_min.size == 0:
        return []
    left_indices = np.arange(left_min.shape[0])
    right_indices = np.arange(right_min.shape[0])

    clearance = float(clearance_mm)
    left_global_min = left_min.min(axis=0) - clearance
    left_global_max = left_max.max(axis=0) + clearance
    right_mask = np.all((right_max >= left_global_min) & (right_min <= left_global_max), axis=1)
    if not np.any(right_mask):
        return []
    right_min = right_min[right_mask]
    right_max = right_max[right_mask]
    right_indices = right_indices[right_mask]

    right_global_min = right_min.min(axis=0) - clearance
    right_global_max = right_max.max(axis=0) + clearance
    left_mask = np.all((left_max >= right_global_min) & (left_min <= right_global_max), axis=1)
    if not np.any(left_mask):
        return []
    left_min = left_min[left_mask]
    left_max = left_max[left_mask]
    left_indices = left_indices[left_mask]

    pair_count = left_min.shape[0] * right_min.shape[0]
    if pair_count <= PAIR_MATRIX_LIMIT:
        separated_gap = np.maximum(
            np.maximum(
                right_min[None, :, :] - left_max[:, None, :],
                left_min[:, None, :] - right_max[None, :, :],
            ),
            0.0,
        )
        distance = np.linalg.norm(separated_gap, axis=2)
        mask = distance <= clearance
        if not np.any(mask):
            return []
        overlap = np.minimum(left_max[:, None, :], right_max[None, :, :]) - np.maximum(
            left_min[:, None, :],
            right_min[None, :, :],
        )
        pairs = np.argwhere(mask)
        return [
            (
                int(left_indices[left_index]),
                int(right_indices[right_index]),
                float(distance[left_index, right_index]),
                float(max(0.0, np.min(overlap[left_index, right_index]))),
            )
            for left_index, right_index in pairs
        ]

    left_pair, right_pair = _axis_sweep_candidate_pairs(left_min, left_max, right_min, right_max, clearance)
    if left_pair.size == 0:
        return []
    contact_rows, distance, overlap_min = _aabb_contact_rows(
        left_min[left_pair],
        left_max[left_pair],
        right_min[right_pair],
        right_max[right_pair],
        clearance,
    )
    if contact_rows.size == 0:
        return []
    return [
        (
            int(left_indices[left_pair[row_index]]),
            int(right_indices[right_pair[row_index]]),
            float(distance[contact_index]),
            float(overlap_min[contact_index]),
        )
        for contact_index, row_index in enumerate(contact_rows)
    ]


def _triangle_bounds(triangles: np.ndarray) -> BoundsArrays:
    return triangles.min(axis=1), triangles.max(axis=1)


def _triangle_candidate_pairs(
    left_triangles: np.ndarray,
    right_triangles: np.ndarray,
    clearance_mm: float,
) -> tuple[np.ndarray, np.ndarray]:
    left_min, left_max = _triangle_bounds(left_triangles)
    right_min, right_max = _triangle_bounds(right_triangles)
    if left_min.size == 0 or right_min.size == 0:
        empty = np.zeros(0, dtype=int)
        return empty, empty

    left_pair, right_pair = _axis_sweep_candidate_pairs(
        left_min,
        left_max,
        right_min,
        right_max,
        float(clearance_mm),
    )
    if left_pair.size == 0:
        return left_pair, right_pair
    contact_rows, _distance, _overlap = _aabb_contact_rows(
        left_min[left_pair],
        left_max[left_pair],
        right_min[right_pair],
        right_max[right_pair],
        float(clearance_mm),
    )
    if contact_rows.size == 0:
        empty = np.zeros(0, dtype=int)
        return empty, empty
    return left_pair[contact_rows], right_pair[contact_rows]


def _point_triangle_distance_squared(points: np.ndarray, triangles: np.ndarray) -> np.ndarray:
    try:
        import trimesh  # type: ignore
    except Exception as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("trimesh is required for mesh narrowphase collision checking") from exc
    closest = trimesh.triangles.closest_point(triangles, points)
    delta = points - closest
    return np.einsum("ij,ij->i", delta, delta)


def _segment_segment_distance_squared(
    p1: np.ndarray,
    q1: np.ndarray,
    p2: np.ndarray,
    q2: np.ndarray,
) -> np.ndarray:
    d1 = q1 - p1
    d2 = q2 - p2
    r = p1 - p2
    a = np.einsum("ij,ij->i", d1, d1)
    e = np.einsum("ij,ij->i", d2, d2)
    f = np.einsum("ij,ij->i", d2, r)
    c = np.einsum("ij,ij->i", d1, r)
    b = np.einsum("ij,ij->i", d1, d2)
    eps = TRIANGLE_INTERSECTION_EPS
    s = np.zeros_like(a)
    t = np.zeros_like(a)

    both = (a > eps) & (e > eps)
    denom = (a * e) - (b * b)
    denom_mask = both & (np.abs(denom) > eps)
    s[denom_mask] = np.clip(((b[denom_mask] * f[denom_mask]) - (c[denom_mask] * e[denom_mask])) / denom[denom_mask], 0.0, 1.0)
    parallel = both & ~denom_mask
    s[parallel] = 0.0
    t[both] = ((b[both] * s[both]) + f[both]) / e[both]

    low_t = both & (t < 0.0)
    t[low_t] = 0.0
    s[low_t] = np.clip(-c[low_t] / a[low_t], 0.0, 1.0)

    high_t = both & (t > 1.0)
    t[high_t] = 1.0
    s[high_t] = np.clip((b[high_t] - c[high_t]) / a[high_t], 0.0, 1.0)

    p_degenerate = (a <= eps) & (e > eps)
    t[p_degenerate] = np.clip(f[p_degenerate] / e[p_degenerate], 0.0, 1.0)

    q_degenerate = (a > eps) & (e <= eps)
    s[q_degenerate] = np.clip(-c[q_degenerate] / a[q_degenerate], 0.0, 1.0)

    closest_left = p1 + (d1 * s[:, None])
    closest_right = p2 + (d2 * t[:, None])
    delta = closest_left - closest_right
    return np.einsum("ij,ij->i", delta, delta)


def _segments_intersect_triangles(
    start: np.ndarray,
    end: np.ndarray,
    triangles: np.ndarray,
) -> np.ndarray:
    direction = end - start
    edge1 = triangles[:, 1] - triangles[:, 0]
    edge2 = triangles[:, 2] - triangles[:, 0]
    h = np.cross(direction, edge2)
    a = np.einsum("ij,ij->i", edge1, h)
    valid = np.abs(a) > TRIANGLE_INTERSECTION_EPS
    inv_a = np.zeros_like(a)
    inv_a[valid] = 1.0 / a[valid]
    s = start - triangles[:, 0]
    u = inv_a * np.einsum("ij,ij->i", s, h)
    q = np.cross(s, edge1)
    v = inv_a * np.einsum("ij,ij->i", direction, q)
    t = inv_a * np.einsum("ij,ij->i", edge2, q)
    eps = TRIANGLE_INTERSECTION_EPS
    return valid & (u >= -eps) & (v >= -eps) & ((u + v) <= 1.0 + eps) & (t >= -eps) & (t <= 1.0 + eps)


def _triangle_pair_distance_squared(left_triangles: np.ndarray, right_triangles: np.ndarray) -> np.ndarray:
    count = left_triangles.shape[0]
    distances = np.full(count, np.inf, dtype=float)

    for vertex_index in range(3):
        distances = np.minimum(
            distances,
            _point_triangle_distance_squared(left_triangles[:, vertex_index], right_triangles),
        )
        distances = np.minimum(
            distances,
            _point_triangle_distance_squared(right_triangles[:, vertex_index], left_triangles),
        )

    edge_indices = ((0, 1), (1, 2), (2, 0))
    intersects = np.zeros(count, dtype=bool)
    for edge_start, edge_end in edge_indices:
        intersects |= _segments_intersect_triangles(
            left_triangles[:, edge_start],
            left_triangles[:, edge_end],
            right_triangles,
        )
        intersects |= _segments_intersect_triangles(
            right_triangles[:, edge_start],
            right_triangles[:, edge_end],
            left_triangles,
        )
        for right_start, right_end in edge_indices:
            distances = np.minimum(
                distances,
                _segment_segment_distance_squared(
                    left_triangles[:, edge_start],
                    left_triangles[:, edge_end],
                    right_triangles[:, right_start],
                    right_triangles[:, right_end],
                ),
            )

    distances[intersects] = 0.0
    return distances


def _mesh_contacts(
    left_triangles: np.ndarray,
    right_triangles: np.ndarray,
    clearance_mm: float,
    *,
    stop_on_first_collision: bool,
) -> list[tuple[int, int, float, float]]:
    left_pair, right_pair = _triangle_candidate_pairs(left_triangles, right_triangles, clearance_mm)
    if left_pair.size == 0:
        return []

    clearance_squared = float(clearance_mm) ** 2
    contacts: list[tuple[int, int, float, float]] = []
    for start in range(0, left_pair.size, TRIANGLE_DISTANCE_BATCH_SIZE):
        end = min(start + TRIANGLE_DISTANCE_BATCH_SIZE, left_pair.size)
        left_batch = left_pair[start:end]
        right_batch = right_pair[start:end]
        distance_squared = _triangle_pair_distance_squared(
            left_triangles[left_batch],
            right_triangles[right_batch],
        )
        contact_rows = np.flatnonzero(distance_squared <= clearance_squared)
        for row in contact_rows:
            contacts.append(
                (
                    int(left_batch[row]),
                    int(right_batch[row]),
                    float(np.sqrt(max(0.0, distance_squared[row]))),
                    0.0,
                )
            )
            if stop_on_first_collision:
                return contacts
    return contacts


def _aabb_contact_value(
    left_min: np.ndarray,
    left_max: np.ndarray,
    right_min: np.ndarray,
    right_max: np.ndarray,
    clearance_mm: float,
) -> tuple[bool, float, float]:
    separated_gap = np.maximum(np.maximum(right_min - left_max, left_min - right_max), 0.0)
    distance = float(np.linalg.norm(separated_gap))
    if distance > float(clearance_mm):
        return False, distance, 0.0
    overlap = np.minimum(left_max, right_max) - np.maximum(left_min, right_min)
    return True, distance, float(max(0.0, np.min(overlap)))


def _node_surface_area(node_min: np.ndarray, node_max: np.ndarray) -> float:
    extent = np.maximum(0.0, node_max - node_min)
    return float(2.0 * ((extent[0] * extent[1]) + (extent[1] * extent[2]) + (extent[0] * extent[2])))


def _mesh_contacts_bvh(
    left_pose: MeshPose,
    right_pose: MeshPose,
    clearance_mm: float,
    *,
    stop_on_first_collision: bool,
) -> list[tuple[int, int, float, float]]:
    contacts: list[tuple[int, int, float, float]] = []
    seen_triangle_pairs: set[tuple[int, int]] = set()
    if stop_on_first_collision:
        return _mesh_contacts_bvh_first_hit(
            left_pose,
            right_pose,
            clearance_mm,
            seen_triangle_pairs=seen_triangle_pairs,
        )

    stack: list[tuple[int, int]] = [(0, 0)]
    while stack:
        left_node, right_node = stack.pop()
        in_range, _distance, _overlap = _aabb_contact_value(
            left_pose.node_min[left_node],
            left_pose.node_max[left_node],
            right_pose.node_min[right_node],
            right_pose.node_max[right_node],
            clearance_mm,
        )
        if not in_range:
            continue

        left_is_leaf = left_pose.bvh.left_child[left_node] < 0
        right_is_leaf = right_pose.bvh.left_child[right_node] < 0
        if left_is_leaf and right_is_leaf:
            left_start = int(left_pose.bvh.start[left_node])
            left_end = left_start + int(left_pose.bvh.count[left_node])
            right_start = int(right_pose.bvh.start[right_node])
            right_end = right_start + int(right_pose.bvh.count[right_node])
            left_indices = left_pose.bvh.triangle_indices[left_start:left_end]
            right_indices = right_pose.bvh.triangle_indices[right_start:right_end]
            for left_index, right_index, distance_mm, overlap_min in _mesh_contacts(
                left_pose.triangles[left_indices],
                right_pose.triangles[right_indices],
                clearance_mm,
                stop_on_first_collision=stop_on_first_collision,
            ):
                left_triangle_index = int(left_indices[left_index])
                right_triangle_index = int(right_indices[right_index])
                key = (left_triangle_index, right_triangle_index)
                if key in seen_triangle_pairs:
                    continue
                seen_triangle_pairs.add(key)
                contacts.append((left_triangle_index, right_triangle_index, distance_mm, overlap_min))
                if stop_on_first_collision:
                    return contacts
            continue

        if right_is_leaf or (
            not left_is_leaf
            and _node_surface_area(left_pose.node_min[left_node], left_pose.node_max[left_node])
            >= _node_surface_area(right_pose.node_min[right_node], right_pose.node_max[right_node])
        ):
            stack.append((int(left_pose.bvh.left_child[left_node]), right_node))
            stack.append((int(left_pose.bvh.right_child[left_node]), right_node))
        else:
            stack.append((left_node, int(right_pose.bvh.left_child[right_node])))
            stack.append((left_node, int(right_pose.bvh.right_child[right_node])))
    return contacts


def _mesh_contacts_bvh_first_hit(
    left_pose: MeshPose,
    right_pose: MeshPose,
    clearance_mm: float,
    *,
    seen_triangle_pairs: set[tuple[int, int]],
) -> list[tuple[int, int, float, float]]:
    queue: list[tuple[float, float, float, float, int, int, int]] = []
    sequence = 0

    def push_node_pair(left_node: int, right_node: int) -> None:
        nonlocal sequence
        in_range, distance, overlap = _aabb_contact_value(
            left_pose.node_min[left_node],
            left_pose.node_max[left_node],
            right_pose.node_min[right_node],
            right_pose.node_max[right_node],
            clearance_mm,
        )
        if not in_range:
            return
        left_area = _node_surface_area(left_pose.node_min[left_node], left_pose.node_max[left_node])
        right_area = _node_surface_area(right_pose.node_min[right_node], right_pose.node_max[right_node])
        # For first-hit live checking, examine actual AABB overlap before near-miss
        # clearance, then prefer deeper overlap and smaller combined node area.
        priority_distance = 0.0 if distance <= TRIANGLE_INTERSECTION_EPS else float(distance)
        heapq.heappush(
            queue,
            (
                priority_distance,
                -float(overlap),
                float(left_area + right_area),
                float(max(left_area, right_area)),
                sequence,
                int(left_node),
                int(right_node),
            ),
        )
        sequence += 1

    push_node_pair(0, 0)
    while queue:
        _distance, _neg_overlap, _area, _max_area, _sequence, left_node, right_node = heapq.heappop(queue)
        left_is_leaf = left_pose.bvh.left_child[left_node] < 0
        right_is_leaf = right_pose.bvh.left_child[right_node] < 0
        if left_is_leaf and right_is_leaf:
            left_start = int(left_pose.bvh.start[left_node])
            left_end = left_start + int(left_pose.bvh.count[left_node])
            right_start = int(right_pose.bvh.start[right_node])
            right_end = right_start + int(right_pose.bvh.count[right_node])
            left_indices = left_pose.bvh.triangle_indices[left_start:left_end]
            right_indices = right_pose.bvh.triangle_indices[right_start:right_end]
            for left_index, right_index, distance_mm, overlap_min in _mesh_contacts(
                left_pose.triangles[left_indices],
                right_pose.triangles[right_indices],
                clearance_mm,
                stop_on_first_collision=True,
            ):
                left_triangle_index = int(left_indices[left_index])
                right_triangle_index = int(right_indices[right_index])
                key = (left_triangle_index, right_triangle_index)
                if key in seen_triangle_pairs:
                    continue
                return [(left_triangle_index, right_triangle_index, distance_mm, overlap_min)]
            continue

        if right_is_leaf or (
            not left_is_leaf
            and _node_surface_area(left_pose.node_min[left_node], left_pose.node_max[left_node])
            >= _node_surface_area(right_pose.node_min[right_node], right_pose.node_max[right_node])
        ):
            push_node_pair(int(left_pose.bvh.left_child[left_node]), right_node)
            push_node_pair(int(left_pose.bvh.right_child[left_node]), right_node)
        else:
            push_node_pair(left_node, int(right_pose.bvh.left_child[right_node]))
            push_node_pair(left_node, int(right_pose.bvh.right_child[right_node]))
    return []


def _mesh_contacts_leaf_broadphase(
    left_pose: MeshPose,
    right_pose: MeshPose,
    clearance_mm: float,
    *,
    stop_on_first_collision: bool,
) -> list[tuple[int, int, float, float]]:
    broadphase_contacts = _component_contacts(left_pose.leaf_bounds, right_pose.leaf_bounds, clearance_mm)
    if not broadphase_contacts:
        return []

    contacts: list[tuple[int, int, float, float]] = []
    seen_triangle_pairs: set[tuple[int, int]] = set()
    ordered_contacts = sorted(broadphase_contacts, key=lambda item: item[2])
    for left_leaf_index, right_leaf_index, _distance_mm, _overlap_min in ordered_contacts:
        left_node = int(left_pose.leaf_nodes[int(left_leaf_index)])
        right_node = int(right_pose.leaf_nodes[int(right_leaf_index)])
        left_start = int(left_pose.bvh.start[left_node])
        left_end = left_start + int(left_pose.bvh.count[left_node])
        right_start = int(right_pose.bvh.start[right_node])
        right_end = right_start + int(right_pose.bvh.count[right_node])
        left_indices = left_pose.bvh.triangle_indices[left_start:left_end]
        right_indices = right_pose.bvh.triangle_indices[right_start:right_end]
        for left_index, right_index, distance_mm, overlap_min in _mesh_contacts(
            left_pose.triangles[left_indices],
            right_pose.triangles[right_indices],
            clearance_mm,
            stop_on_first_collision=stop_on_first_collision,
        ):
            left_triangle_index = int(left_indices[left_index])
            right_triangle_index = int(right_indices[right_index])
            key = (left_triangle_index, right_triangle_index)
            if key in seen_triangle_pairs:
                continue
            seen_triangle_pairs.add(key)
            contacts.append((left_triangle_index, right_triangle_index, distance_mm, overlap_min))
            if stop_on_first_collision:
                return contacts
    return contacts


def _triangle_indices_near_bounds(
    triangles: np.ndarray,
    query_bounds: BoundsArrays,
    clearance_mm: float,
) -> np.ndarray:
    return _triangle_indices_near_bounds_arrays(
        _triangle_bounds(triangles),
        query_bounds,
        clearance_mm,
    )


def _triangle_indices_near_bounds_arrays(
    triangle_bounds: BoundsArrays,
    query_bounds: BoundsArrays,
    clearance_mm: float,
) -> np.ndarray:
    query_min, query_max = query_bounds
    tri_min, tri_max = triangle_bounds
    if tri_min.size == 0 or query_min.size == 0:
        return np.zeros(0, dtype=int)
    if query_min.shape[0] <= 16:
        clearance = float(clearance_mm)
        mask = np.zeros(tri_min.shape[0], dtype=bool)
        for row in range(query_min.shape[0]):
            separated_gap = np.maximum(
                np.maximum(query_min[row] - tri_max, tri_min - query_max[row]),
                0.0,
            )
            mask |= np.linalg.norm(separated_gap, axis=1) <= clearance
        return np.flatnonzero(mask)
    tri_pair, query_pair = _axis_sweep_candidate_pairs(
        tri_min,
        tri_max,
        query_min,
        query_max,
        float(clearance_mm),
    )
    if tri_pair.size == 0:
        return tri_pair
    contact_rows, _distance, _overlap = _aabb_contact_rows(
        tri_min[tri_pair],
        tri_max[tri_pair],
        query_min[query_pair],
        query_max[query_pair],
        float(clearance_mm),
    )
    if contact_rows.size == 0:
        return np.zeros(0, dtype=int)
    return np.unique(tri_pair[contact_rows])


def _bounds_subset(bounds: BoundsArrays, indices: np.ndarray) -> BoundsArrays:
    min_xyz, max_xyz = bounds
    if indices.size == 0:
        return _empty_bounds_arrays()
    return min_xyz[indices], max_xyz[indices]


def _mesh_contacts_for_broadphase(
    *,
    left_triangles: np.ndarray,
    right_triangles: np.ndarray,
    left_bounds: BoundsArrays,
    right_bounds: BoundsArrays,
    broadphase_contacts: list[tuple[int, int, float, float]],
    clearance_mm: float,
    stop_on_first_collision: bool,
) -> list[tuple[int, int, float, float]]:
    if not broadphase_contacts:
        return []
    left_triangle_bounds = _triangle_bounds(left_triangles)
    right_triangle_bounds = _triangle_bounds(right_triangles)
    contacts: list[tuple[int, int, float, float]] = []
    seen_triangle_pairs: set[tuple[int, int]] = set()
    ordered_broadphase_contacts = sorted(broadphase_contacts, key=lambda item: item[2])
    for left_box_index, right_box_index, _distance_mm, _overlap_min in ordered_broadphase_contacts:
        left_box = np.asarray([int(left_box_index)], dtype=int)
        right_box = np.asarray([int(right_box_index)], dtype=int)
        left_triangle_indices = _triangle_indices_near_bounds_arrays(
            left_triangle_bounds,
            _bounds_subset(right_bounds, right_box),
            clearance_mm,
        )
        right_triangle_indices = _triangle_indices_near_bounds_arrays(
            right_triangle_bounds,
            _bounds_subset(left_bounds, left_box),
            clearance_mm,
        )
        if left_triangle_indices.size == 0 or right_triangle_indices.size == 0:
            continue
        for left_index, right_index, distance_mm, overlap_min in _mesh_contacts(
            left_triangles[left_triangle_indices],
            right_triangles[right_triangle_indices],
            clearance_mm,
            stop_on_first_collision=stop_on_first_collision,
        ):
            left_triangle_index = int(left_triangle_indices[left_index])
            right_triangle_index = int(right_triangle_indices[right_index])
            key = (left_triangle_index, right_triangle_index)
            if key in seen_triangle_pairs:
                continue
            seen_triangle_pairs.add(key)
            contacts.append(
                (
                    left_triangle_index,
                    right_triangle_index,
                    distance_mm,
                    overlap_min,
                )
            )
            if stop_on_first_collision:
                return contacts
    return contacts


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


def _narrowphase_mode_from_payload(payload: dict[str, Any]) -> str:
    raw = str(payload.get("narrowphase_mode", "mesh") or "mesh").strip().lower()
    if raw in {"mesh", "visual_mesh", "triangle_mesh", "narrowphase"}:
        return "mesh"
    if raw in {"aabb", "proxy", "surface_aabb", "boxes", "box"}:
        return "aabb"
    raise ValueError(f"unsupported narrowphase_mode: {raw}")


def _collision_method(static_assets: list[StaticCollisionAsset], tool_assets: list[ToolCollisionAsset], narrowphase_mode: str) -> str:
    suffix = "mesh_narrowphase" if narrowphase_mode == "mesh" else "aabb_distance"
    if static_assets and tool_assets:
        return f"link_tool_static_{suffix}"
    if static_assets:
        return f"link_static_{suffix}"
    if tool_assets:
        return f"link_tool_{suffix}"
    return f"link_{suffix}"


def _pose_collision_result(
    fk: dict[str, Any],
    checked_pairs: list[dict[str, Any]],
    ignored_pairs: list[dict[str, Any]],
    collisions: list[dict[str, Any]],
    *,
    include_fk_frames: bool,
) -> dict[str, Any]:
    result = {
        "q_deg": [float(value) for value in fk["q_deg"]],
        "collision_free": len(collisions) == 0,
        "checked_pair_count": len(checked_pairs),
        "ignored_pair_count": len(ignored_pairs),
        "collision_count": len(collisions),
        "collisions": collisions,
        "checked_pairs": checked_pairs,
        "ignored_pairs": ignored_pairs,
    }
    if include_fk_frames:
        result["joint_frames_robot"] = fk.get("joint_frames_robot")
    return result


def _check_pose_collision(
    *,
    q_input: np.ndarray,
    session: Any,
    link_assets: list[LinkCollisionAsset],
    tool_assets: list[ToolCollisionAsset],
    static_assets: list[StaticCollisionAsset],
    static_bounds: dict[int, BoundsArrays],
    clearance_mm: float,
    ignore_adjacent: bool,
    allowed_link_pairs: set[tuple[int, int]],
    check_static_base_link: bool,
    tool_parent_ignore_depth: int,
    stop_on_first_collision: bool,
    include_fk_frames: bool,
    narrowphase_mode: str,
) -> dict[str, Any]:
    fk = session.fk({"q_deg": q_input})
    joint_frames_robot = [
        as_transform(np.asarray(matrix, dtype=float))
        for matrix in fk["joint_frames_robot"]
    ]

    bounds: dict[int, BoundsArrays] = {}
    for asset in link_assets:
        if asset.link_id >= len(joint_frames_robot):
            raise ValueError(f"FK response missing frame for link_id={asset.link_id}")
        bounds[asset.link_id] = _bounds_for_link_components(asset, joint_frames_robot[asset.link_id])

    tool_bounds: dict[int, BoundsArrays] = {}
    for asset in tool_assets:
        if asset.fk_frame_index >= len(joint_frames_robot):
            raise ValueError(f"FK response missing frame for tool fk_frame_index={asset.fk_frame_index}")
        tool_bounds[asset.tool_id] = _bounds_for_tool_components(asset, joint_frames_robot[asset.fk_frame_index])

    mesh_cache: dict[tuple[str, int], MeshPose] = {}

    def link_mesh_pose(asset: LinkCollisionAsset) -> MeshPose:
        key = ("link", asset.link_id)
        cached = mesh_cache.get(key)
        if cached is None:
            cached = _mesh_bvh_pose_from_transform(
                str(asset.visual_mesh_path),
                _link_transform(asset, joint_frames_robot[asset.link_id]),
            )
            mesh_cache[key] = cached
        return cached

    def tool_mesh_pose(asset: ToolCollisionAsset) -> MeshPose:
        key = ("tool", asset.tool_id)
        cached = mesh_cache.get(key)
        if cached is None:
            cached = _mesh_bvh_pose_from_transform(
                str(asset.visual_mesh_path),
                _tool_transform(asset, joint_frames_robot[asset.fk_frame_index]),
            )
            mesh_cache[key] = cached
        return cached

    def static_mesh_pose(asset: StaticCollisionAsset) -> MeshPose:
        key = ("static", asset.static_id)
        cached = mesh_cache.get(key)
        if cached is None:
            cached = _cached_mesh_bvh_pose(
                str(asset.visual_mesh_path),
                _matrix_cache_key(asset.pose_robot),
            )
            mesh_cache[key] = cached
        return cached

    def final_contacts(
        broadphase_contacts: list[tuple[int, int, float, float]],
        left_bounds: BoundsArrays,
        right_bounds: BoundsArrays,
        left_mesh_getter: Any,
        right_mesh_getter: Any,
    ) -> list[tuple[int, int, float, float]]:
        if narrowphase_mode != "mesh" or not broadphase_contacts:
            return broadphase_contacts
        return _mesh_contacts_bvh(
            left_mesh_getter(),
            right_mesh_getter(),
            clearance_mm,
            stop_on_first_collision=stop_on_first_collision,
        )

    checked_pairs: list[dict[str, Any]] = []
    ignored_pairs: list[dict[str, Any]] = []
    collisions: list[dict[str, Any]] = []
    pending_checks: list[dict[str, Any]] = []
    check_sequence = 0

    def enqueue_check(
        *,
        pair: dict[str, Any],
        collision_base: dict[str, Any],
        left_bounds: BoundsArrays,
        right_bounds: BoundsArrays,
        left_mesh_getter: Any,
        right_mesh_getter: Any,
    ) -> None:
        nonlocal check_sequence
        checked_pairs.append(pair)
        broadphase_contacts = _component_contacts(
            left_bounds,
            right_bounds,
            clearance_mm,
        )
        if broadphase_contacts:
            distances = [float(item[2]) for item in broadphase_contacts]
            overlaps = [float(item[3]) for item in broadphase_contacts]
            pending_checks.append(
                {
                    "sequence": check_sequence,
                    "min_distance": min(distances),
                    "max_overlap": max(overlaps),
                    "overlap_sum": sum(overlaps),
                    "contact_count": len(broadphase_contacts),
                    "broadphase_contacts": broadphase_contacts,
                    "left_bounds": left_bounds,
                    "right_bounds": right_bounds,
                    "left_mesh_getter": left_mesh_getter,
                    "right_mesh_getter": right_mesh_getter,
                    "collision_base": collision_base,
                }
            )
        check_sequence += 1

    for left_index, left in enumerate(link_assets):
        for right in link_assets[left_index + 1 :]:
            pair = {"left_link_id": left.link_id, "right_link_id": right.link_id}
            if _should_ignore_pair(left.link_id, right.link_id, ignore_adjacent, allowed_link_pairs):
                ignored_pairs.append(pair)
                continue
            enqueue_check(
                pair=pair,
                collision_base={
                    **pair,
                    "left_kind": "robot_link",
                    "right_kind": "robot_link",
                    "left_name": left.name,
                    "right_name": right.name,
                },
                left_bounds=bounds[left.link_id],
                right_bounds=bounds[right.link_id],
                left_mesh_getter=lambda left=left: link_mesh_pose(left),
                right_mesh_getter=lambda right=right: link_mesh_pose(right),
            )

    for tool in tool_assets:
        for link in link_assets:
            pair = {
                "left_kind": "tool",
                "right_kind": "robot_link",
                "left_tool_id": tool.tool_id,
                "right_link_id": link.link_id,
            }
            if link.link_id == tool.fk_frame_index or (
                ignore_adjacent
                and 0 <= tool.fk_frame_index - link.link_id <= tool_parent_ignore_depth
            ):
                ignored_pairs.append(pair)
                continue
            enqueue_check(
                pair=pair,
                collision_base={
                    **pair,
                    "left_name": tool.name,
                    "right_name": link.name,
                },
                left_bounds=tool_bounds[tool.tool_id],
                right_bounds=bounds[link.link_id],
                left_mesh_getter=lambda tool=tool: tool_mesh_pose(tool),
                right_mesh_getter=lambda link=link: link_mesh_pose(link),
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
            enqueue_check(
                pair=pair,
                collision_base={
                    **pair,
                    "left_name": link.name,
                    "right_name": static.name,
                },
                left_bounds=bounds[link.link_id],
                right_bounds=static_bounds[static.static_id],
                left_mesh_getter=lambda link=link: link_mesh_pose(link),
                right_mesh_getter=lambda static=static: static_mesh_pose(static),
            )

    for tool in tool_assets:
        for static in static_assets:
            pair = {
                "left_kind": "tool",
                "right_kind": "static_object",
                "left_tool_id": tool.tool_id,
                "right_static_id": static.static_id,
            }
            enqueue_check(
                pair=pair,
                collision_base={
                    **pair,
                    "left_name": tool.name,
                    "right_name": static.name,
                },
                left_bounds=tool_bounds[tool.tool_id],
                right_bounds=static_bounds[static.static_id],
                left_mesh_getter=lambda tool=tool: tool_mesh_pose(tool),
                right_mesh_getter=lambda static=static: static_mesh_pose(static),
            )

    if stop_on_first_collision:
        pending_checks.sort(
            key=lambda item: (
                -float(item["max_overlap"]),
                -float(item["overlap_sum"]),
                float(item["min_distance"]),
                -int(item["contact_count"]),
                int(item["sequence"]),
            )
        )

    for pending in pending_checks:
        for left_component_index, right_component_index, distance_mm, overlap_min in final_contacts(
            pending["broadphase_contacts"],
            pending["left_bounds"],
            pending["right_bounds"],
            pending["left_mesh_getter"],
            pending["right_mesh_getter"],
        ):
            collisions.append(
                {
                    **pending["collision_base"],
                    "left_component_index": left_component_index,
                    "right_component_index": right_component_index,
                    "minimum_distance_mm": distance_mm,
                    "minimum_overlap_mm": overlap_min,
                    "clearance_mm": clearance_mm,
                }
            )
            if stop_on_first_collision:
                return _pose_collision_result(
                    fk,
                    checked_pairs,
                    ignored_pairs,
                    collisions,
                    include_fk_frames=include_fk_frames,
                )

    return _pose_collision_result(
        fk,
        checked_pairs,
        ignored_pairs,
        collisions,
        include_fk_frames=include_fk_frames,
    )


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
    tool_assets = _load_tool_assets(manifest, asset_root)
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
    clearance_mm = max(0.0, float(payload.get("clearance_mm", padding_mm) or 0.0))
    ignore_adjacent = bool(payload.get("ignore_adjacent_links", True))
    allowed_link_pairs = _allowed_link_pairs_from_payload(payload, manifest)
    check_static_base_link = bool(payload.get("check_static_base_link", False))
    tool_parent_ignore_depth = max(0, int(payload.get("tool_parent_ignore_depth", 2) or 0))
    stop_on_first_collision = bool(payload.get("stop_on_first_collision", False))
    include_sample_details = bool(payload.get("include_sample_details", True))
    include_fk_frames = bool(payload.get("include_fk_frames", False))
    narrowphase_mode = _narrowphase_mode_from_payload(payload)
    method = _collision_method(static_assets, tool_assets, narrowphase_mode)
    include_broadphase_coverage = bool(
        payload.get(
            "include_broadphase_coverage",
            payload.get("q_path_deg") is not None or payload.get("path_q_deg") is not None,
        )
    )
    if narrowphase_mode == "mesh" and bool(payload.get("preload_collision_assets", False)):
        _preload_collision_mesh_data(link_assets, tool_assets, static_assets)

    common = {
        "ok": True,
        "method": method,
        "model_id": session.model_id,
        "kinematics_hash": session.kinematics_hash,
        "link_count": len(link_assets),
        "tool_count": len(tool_assets),
        "static_object_count": len(static_assets),
        "padding_mm": padding_mm,
        "clearance_mm": clearance_mm,
        "include_static_collision": bool(static_assets),
        "narrowphase_mode": narrowphase_mode,
    }
    if include_broadphase_coverage:
        common["broadphase_coverage"] = _broadphase_coverage_summary(link_assets, tool_assets, static_assets)
    if bool(payload.get("preload_collision_assets_only", False)):
        return {
            **common,
            "mode": "preload",
            "preloaded_collision_assets": narrowphase_mode == "mesh",
        }

    static_bounds: dict[int, BoundsArrays] = {
        asset.static_id: _bounds_for_static_components(asset)
        for asset in static_assets
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
                tool_assets=tool_assets,
                static_assets=static_assets,
                static_bounds=static_bounds,
                clearance_mm=clearance_mm,
                ignore_adjacent=ignore_adjacent,
                allowed_link_pairs=allowed_link_pairs,
                check_static_base_link=check_static_base_link,
                tool_parent_ignore_depth=tool_parent_ignore_depth,
                stop_on_first_collision=stop_on_first_collision,
                include_fk_frames=False,
                narrowphase_mode=narrowphase_mode,
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
        tool_assets=tool_assets,
        static_assets=static_assets,
        static_bounds=static_bounds,
        clearance_mm=clearance_mm,
        ignore_adjacent=ignore_adjacent,
        allowed_link_pairs=allowed_link_pairs,
        check_static_base_link=check_static_base_link,
        tool_parent_ignore_depth=tool_parent_ignore_depth,
        stop_on_first_collision=stop_on_first_collision,
        include_fk_frames=include_fk_frames,
        narrowphase_mode=narrowphase_mode,
    )
    return {
        **common,
        "mode": "pose",
        **pose_result,
    }
