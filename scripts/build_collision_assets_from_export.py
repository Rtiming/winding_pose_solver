from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


LINK_NAME_TO_ID = {
    "base": 0,
    "j1": 1,
    "j2": 2,
    "j3": 3,
    "j4": 4,
    "j5": 5,
    "j6": 6,
}


def _resolve_export_dir(path: Path) -> Path:
    candidate = path.expanduser().resolve()
    if (candidate / "metadata.json").is_file() and (candidate / "meshes").is_dir():
        return candidate
    children = [
        child
        for child in candidate.iterdir()
        if child.is_dir() and (child / "metadata.json").is_file() and (child / "meshes").is_dir()
    ]
    if not children:
        raise FileNotFoundError(f"No export directory with metadata.json and meshes found under {candidate}")
    return max(children, key=lambda item: item.stat().st_mtime)


def _safe_name(value: Any) -> str:
    text = str(value or "item").strip()
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in text).strip("._-") or "item"


def _load_metadata(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _bounds_from_vertices(vertices: Any) -> tuple[list[float], list[float]]:
    array = np.asarray(vertices, dtype=float)
    return array.min(axis=0).tolist(), array.max(axis=0).tolist()


def _box_from_bounds(min_xyz: list[float], max_xyz: list[float]) -> Any:
    import trimesh  # type: ignore

    lower = np.asarray(min_xyz, dtype=float)
    upper = np.asarray(max_xyz, dtype=float)
    center = (lower + upper) * 0.5
    extents = np.maximum(upper - lower, 1e-6)
    transform = np.eye(4, dtype=float)
    transform[:3, 3] = center
    return trimesh.creation.box(extents=extents, transform=transform)


def _grid_bounds(mesh: Any, cell_size_mm: float) -> list[tuple[list[float], list[float]]]:
    vertices = mesh.vertices
    array = np.asarray(vertices, dtype=float)
    if array.ndim != 2 or array.shape[1] != 3 or len(array) == 0:
        return []
    lower = array.min(axis=0)
    upper = array.max(axis=0)
    cell = max(float(cell_size_mm), 1.0)
    span = np.maximum(upper - lower, 1e-9)
    grid_shape = np.maximum(np.ceil(span / cell).astype(int), 1)
    max_index = grid_shape - 1
    faces = np.asarray(getattr(mesh, "faces", []), dtype=int)
    seen: set[tuple[int, int, int]] = set()
    if faces.ndim == 2 and faces.shape[1] >= 3 and len(faces) > 0:
        for face in faces[:, :3]:
            try:
                tri = array[face]
            except Exception:
                continue
            tri_min = tri.min(axis=0)
            tri_max = tri.max(axis=0)
            start = np.clip(np.floor((tri_min - lower) / cell).astype(int), 0, max_index)
            end = np.clip(np.floor((tri_max - lower) / cell).astype(int), 0, max_index)
            for ix in range(int(start[0]), int(end[0]) + 1):
                for iy in range(int(start[1]), int(end[1]) + 1):
                    for iz in range(int(start[2]), int(end[2]) + 1):
                        seen.add((ix, iy, iz))
    else:
        indices = np.clip(np.floor((array - lower) / cell).astype(int), 0, max_index)
        seen = {tuple(row.tolist()) for row in indices}
    bounds: list[tuple[list[float], list[float]]] = []
    for index in sorted(seen):
        start = lower + np.asarray(index, dtype=float) * cell
        end = np.minimum(start + cell, upper)
        bounds.append((start.tolist(), end.tolist()))
    return bounds


def _cell_index(point: np.ndarray, lower: np.ndarray, cell: float, max_index: np.ndarray) -> tuple[int, int, int]:
    index = np.clip(np.floor((point - lower) / cell).astype(int), 0, max_index)
    return int(index[0]), int(index[1]), int(index[2])


def _update_cell_bound(
    cells: dict[tuple[int, int, int], tuple[np.ndarray, np.ndarray]],
    point: np.ndarray,
    lower: np.ndarray,
    cell: float,
    max_index: np.ndarray,
) -> None:
    key = _cell_index(point, lower, cell, max_index)
    if key not in cells:
        cells[key] = (point.copy(), point.copy())
        return
    min_xyz, max_xyz = cells[key]
    cells[key] = (np.minimum(min_xyz, point), np.maximum(max_xyz, point))


def _triangle_sample_points(tri: np.ndarray, step_mm: float) -> np.ndarray:
    edge_lengths = np.asarray(
        [
            np.linalg.norm(tri[1] - tri[0]),
            np.linalg.norm(tri[2] - tri[1]),
            np.linalg.norm(tri[0] - tri[2]),
        ],
        dtype=float,
    )
    subdivisions = int(np.clip(np.ceil(float(edge_lengths.max(initial=0.0)) / max(float(step_mm), 1.0)), 1, 8))
    if subdivisions <= 1:
        return np.vstack((tri, tri.mean(axis=0)))

    points: list[np.ndarray] = []
    for i in range(subdivisions + 1):
        for j in range(subdivisions + 1 - i):
            a = i / subdivisions
            b = j / subdivisions
            c = 1.0 - a - b
            points.append((a * tri[0]) + (b * tri[1]) + (c * tri[2]))
    return np.asarray(points, dtype=float)


def _tighten_patch_bounds(
    min_xyz: np.ndarray,
    max_xyz: np.ndarray,
    mesh_lower: np.ndarray,
    mesh_upper: np.ndarray,
    min_extent_mm: float,
) -> tuple[list[float], list[float]]:
    extent = max_xyz - min_xyz
    target = np.maximum(extent, float(min_extent_mm))
    extra = (target - extent) * 0.5
    lower = np.maximum(mesh_lower, min_xyz - extra)
    upper = np.minimum(mesh_upper, max_xyz + extra)
    return lower.tolist(), upper.tolist()


def _surface_patch_bounds(mesh: Any, cell_size_mm: float) -> list[tuple[list[float], list[float]]]:
    """Build local AABBs from actual mesh surface samples.

    Full grid voxels are conservative but visually and computationally too large
    for large STEP faces. These bounds only wrap sampled surface points in each
    spatial cell, so they stay close to the real mesh while still using boxes for
    broadphase and clearance diagnostics.
    """
    vertices = np.asarray(mesh.vertices, dtype=float)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or len(vertices) == 0:
        return []
    lower = vertices.min(axis=0)
    upper = vertices.max(axis=0)
    cell = max(float(cell_size_mm), 1.0)
    span = np.maximum(upper - lower, 1e-9)
    grid_shape = np.maximum(np.ceil(span / cell).astype(int), 1)
    max_index = grid_shape - 1
    cells: dict[tuple[int, int, int], tuple[np.ndarray, np.ndarray]] = {}

    faces = np.asarray(getattr(mesh, "faces", []), dtype=int)
    if faces.ndim == 2 and faces.shape[1] >= 3 and len(faces) > 0:
        for face in faces[:, :3]:
            try:
                tri = vertices[face]
            except Exception:
                continue
            for point in _triangle_sample_points(tri, cell):
                _update_cell_bound(cells, point, lower, cell, max_index)
    else:
        for point in vertices:
            _update_cell_bound(cells, point, lower, cell, max_index)

    min_extent = min(max(cell * 0.02, 0.5), 2.0)
    return [
        _tighten_patch_bounds(min_xyz, max_xyz, lower, upper, min_extent)
        for _, (min_xyz, max_xyz) in sorted(cells.items())
    ]


def _make_collision_mesh(mesh: Any, strategy: str, grid_cell_mm: float) -> tuple[Any, list[dict[str, list[float]]]]:
    def box_mesh(box: Any) -> Any:
        if hasattr(box, "as_trimesh"):
            return box.as_trimesh()
        return box.to_mesh()

    def package_single(collision_mesh: Any) -> tuple[Any, list[dict[str, list[float]]]]:
        vertices = np.asarray(collision_mesh.vertices, dtype=float)
        min_xyz, max_xyz = _bounds_from_vertices(vertices)
        return collision_mesh, [{"min": min_xyz, "max": max_xyz}]

    if strategy == "grid-aabb":
        import trimesh  # type: ignore

        bounds = _grid_bounds(mesh, grid_cell_mm)
        if bounds:
            boxes = [_box_from_bounds(min_xyz, max_xyz) for min_xyz, max_xyz in bounds]
            return (
                trimesh.util.concatenate(boxes),
                [{"min": min_xyz, "max": max_xyz} for min_xyz, max_xyz in bounds],
            )
        return package_single(box_mesh(mesh.bounding_box))
    if strategy == "surface-aabb":
        import trimesh  # type: ignore

        bounds = _surface_patch_bounds(mesh, grid_cell_mm)
        if bounds:
            boxes = [_box_from_bounds(min_xyz, max_xyz) for min_xyz, max_xyz in bounds]
            return (
                trimesh.util.concatenate(boxes),
                [{"min": min_xyz, "max": max_xyz} for min_xyz, max_xyz in bounds],
            )
        return package_single(box_mesh(mesh.bounding_box))
    if strategy == "component-aabb":
        try:
            import trimesh  # type: ignore

            components = mesh.split(only_watertight=False)
            boxes = []
            bounds = []
            for component in components:
                if len(component.vertices) <= 0:
                    continue
                min_xyz, max_xyz = _bounds_from_vertices(component.vertices)
                boxes.append(_box_from_bounds(min_xyz, max_xyz))
                bounds.append({"min": min_xyz, "max": max_xyz})
            if boxes:
                return trimesh.util.concatenate(boxes), bounds
        except Exception:
            pass
        return package_single(box_mesh(mesh.bounding_box))
    if strategy == "convex-hull":
        try:
            return package_single(mesh.convex_hull)
        except Exception:
            return package_single(box_mesh(mesh.bounding_box))
    if strategy == "oriented-box":
        try:
            return package_single(box_mesh(mesh.bounding_box_oriented))
        except Exception:
            return package_single(box_mesh(mesh.bounding_box))
    return package_single(box_mesh(mesh.bounding_box))


def _collision_asset(relative_file: str, bounds_file: str, strategy: str) -> dict[str, Any]:
    return {
        "available": True,
        "usage": "collision",
        "format": "stl",
        "units": "mm",
        "source": f"trimesh_{strategy}",
        "quality": f"simplified_{strategy}",
        "shared_with_visual": False,
        "file": relative_file,
        "bounds_file": bounds_file,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build simplified collision STL assets from a RoboDK model export."
    )
    parser.add_argument("--export-dir", type=Path, required=True)
    parser.add_argument(
        "--strategy",
        choices=("aabb", "grid-aabb", "surface-aabb", "component-aabb", "oriented-box", "convex-hull"),
        default="aabb",
        help=(
            "Collision simplification strategy. aabb is fastest; surface-aabb "
            "keeps local boxes close to sampled mesh surfaces."
        ),
    )
    parser.add_argument(
        "--grid-cell-mm",
        type=float,
        default=250.0,
        help="Cell size for --strategy grid-aabb.",
    )
    parser.add_argument(
        "--tool-mesh",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help="Additional tool mesh to simplify and attach to metadata.tools by name.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _parse_tool_meshes(values: list[str]) -> list[tuple[str, Path]]:
    meshes: list[tuple[str, Path]] = []
    for value in values:
        if "=" not in value:
            raise ValueError(f"--tool-mesh must be NAME=PATH, got: {value}")
        name, path_text = value.split("=", 1)
        name = name.strip()
        if not name:
            raise ValueError(f"--tool-mesh has empty NAME: {value}")
        path = Path(path_text).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"tool mesh not found for {name}: {path}")
        meshes.append((name, path))
    return meshes


def _build_one_asset(
    *,
    trimesh_module: Any,
    source_path: Path,
    source_label: str,
    out_path: Path,
    relative_file: str,
    strategy: str,
    grid_cell_mm: float,
) -> tuple[dict[str, Any], list[dict[str, list[float]]]]:
    mesh = trimesh_module.load_mesh(source_path, process=False)
    collision_mesh, bounds = _make_collision_mesh(mesh, strategy, grid_cell_mm)
    collision_mesh.export(out_path)
    bounds_name = f"{out_path.name}.bounds.json"
    bounds_path = out_path.parent / bounds_name
    bounds_payload = {
        "schema": "winding_pose_solver.collision_bounds.v1",
        "source_mesh": source_label,
        "strategy": strategy,
        "grid_cell_mm": grid_cell_mm if strategy in {"grid-aabb", "surface-aabb"} else None,
        "bounds": bounds,
    }
    bounds_path.write_text(json.dumps(bounds_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return _collision_asset(relative_file, f"collision_meshes/{bounds_name}", strategy), bounds


def main() -> int:
    args = _parse_args()
    export_dir = _resolve_export_dir(args.export_dir)
    metadata_path = export_dir / "metadata.json"
    metadata = _load_metadata(metadata_path)
    meshes_dir = export_dir / "meshes"
    collision_dir = export_dir / "collision_meshes"
    collision_dir.mkdir(parents=True, exist_ok=True)

    try:
        import trimesh  # type: ignore
    except Exception as exc:
        print(f"trimesh is required: {type(exc).__name__}: {exc}")
        return 2

    entries: list[dict[str, Any]] = []
    link_entries: list[dict[str, Any]] = []
    tool_entries: list[dict[str, Any]] = []
    static_entries: list[dict[str, Any]] = []
    for scene_object in metadata.get("scene_objects", []):
        name = str(scene_object.get("name") or "").lower()
        mesh_file = scene_object.get("mesh_file")
        if not scene_object.get("mesh_export_ok") or not mesh_file:
            continue
        source_path = meshes_dir / str(mesh_file)
        if not source_path.is_file():
            continue

        if name in LINK_NAME_TO_ID:
            link_id = LINK_NAME_TO_ID[name]
            out_name = f"link_{link_id:02d}_{_safe_name(name)}.collision.stl"
            entry_kind = "robot_link"
        else:
            try:
                object_index = int(scene_object.get("object_index", len(static_entries) + 1))
            except Exception:
                object_index = len(static_entries) + 1
            link_id = None
            out_name = f"static_{object_index:03d}_{_safe_name(name)}.collision.stl"
            entry_kind = "static_object"

        out_path = collision_dir / out_name
        if out_path.exists() and not args.overwrite:
            raise FileExistsError(f"Collision asset exists: {out_path} (use --overwrite)")

        relative_file = f"collision_meshes/{out_name}"
        collision_asset, _bounds = _build_one_asset(
            trimesh_module=trimesh,
            source_path=source_path,
            source_label=str(mesh_file),
            out_path=out_path,
            relative_file=relative_file,
            strategy=args.strategy,
            grid_cell_mm=args.grid_cell_mm,
        )
        scene_object["collision_asset"] = collision_asset
        entry = {
            "kind": entry_kind,
            "name": name,
            "source_visual": str(mesh_file),
            "collision_asset": collision_asset,
        }
        if link_id is not None:
            entry["link_id"] = link_id
            link_entries.append(entry)
        else:
            entry["object_index"] = scene_object.get("object_index")
            static_entries.append(entry)
        entries.append(entry)

    for tool_index, (tool_name, tool_mesh_path) in enumerate(_parse_tool_meshes(args.tool_mesh)):
        out_name = f"tool_{tool_index:03d}_{_safe_name(tool_name)}.collision.stl"
        out_path = collision_dir / out_name
        if out_path.exists() and not args.overwrite:
            raise FileExistsError(f"Collision asset exists: {out_path} (use --overwrite)")
        relative_file = f"collision_meshes/{out_name}"
        collision_asset, _bounds = _build_one_asset(
            trimesh_module=trimesh,
            source_path=tool_mesh_path,
            source_label=str(tool_mesh_path),
            out_path=out_path,
            relative_file=relative_file,
            strategy=args.strategy,
            grid_cell_mm=args.grid_cell_mm,
        )
        matched_tool: dict[str, Any] | None = None
        for tool in metadata.get("tools", []):
            if isinstance(tool, dict) and str(tool.get("name") or "") == tool_name:
                matched_tool = tool
                break
        if matched_tool is not None:
            matched_tool["collision_asset"] = collision_asset
        entry = {
            "kind": "tool",
            "name": tool_name,
            "source_visual": str(tool_mesh_path),
            "tool_index": tool_index,
            "collision_asset": collision_asset,
        }
        tool_entries.append(entry)
        entries.append(entry)

    metadata["collision_build"] = {
        "strategy": args.strategy,
        "collision_asset_count": len(entries),
        "robot_link_collision_asset_count": len(link_entries),
        "tool_collision_asset_count": len(tool_entries),
        "static_collision_asset_count": len(static_entries),
        "entries": entries,
    }
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "ok": True,
                "export_dir": str(export_dir),
                "strategy": args.strategy,
                "collision_asset_count": len(entries),
                "metadata": str(metadata_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
