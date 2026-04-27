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


def _grid_bounds(vertices: Any, cell_size_mm: float) -> list[tuple[list[float], list[float]]]:
    array = np.asarray(vertices, dtype=float)
    if array.ndim != 2 or array.shape[1] != 3 or len(array) == 0:
        return []
    lower = array.min(axis=0)
    upper = array.max(axis=0)
    cell = max(float(cell_size_mm), 1.0)
    indices = np.floor((array - lower) / cell).astype(int)
    seen = {tuple(row.tolist()) for row in indices}
    bounds: list[tuple[list[float], list[float]]] = []
    for index in sorted(seen):
        start = lower + np.asarray(index, dtype=float) * cell
        end = np.minimum(start + cell, upper)
        bounds.append((start.tolist(), end.tolist()))
    return bounds


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

        bounds = _grid_bounds(mesh.vertices, grid_cell_mm)
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
        choices=("aabb", "grid-aabb", "component-aabb", "oriented-box", "convex-hull"),
        default="aabb",
        help=(
            "Collision simplification strategy. aabb is fastest; grid-aabb avoids "
            "turning large fixture assemblies into one huge box."
        ),
    )
    parser.add_argument(
        "--grid-cell-mm",
        type=float,
        default=250.0,
        help="Cell size for --strategy grid-aabb.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


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

        mesh = trimesh.load_mesh(source_path, process=False)
        collision_mesh, bounds = _make_collision_mesh(mesh, args.strategy, args.grid_cell_mm)
        collision_mesh.export(out_path)
        relative_file = f"collision_meshes/{out_name}"
        bounds_name = f"{out_name}.bounds.json"
        bounds_path = collision_dir / bounds_name
        bounds_payload = {
            "schema": "winding_pose_solver.collision_bounds.v1",
            "source_mesh": str(mesh_file),
            "strategy": args.strategy,
            "grid_cell_mm": args.grid_cell_mm if args.strategy == "grid-aabb" else None,
            "bounds": bounds,
        }
        bounds_path.write_text(json.dumps(bounds_payload, ensure_ascii=False, indent=2), encoding="utf-8")
        collision_asset = _collision_asset(relative_file, f"collision_meshes/{bounds_name}", args.strategy)
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

    metadata["collision_build"] = {
        "strategy": args.strategy,
        "collision_asset_count": len(entries),
        "robot_link_collision_asset_count": len(link_entries),
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
