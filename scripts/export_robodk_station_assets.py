from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

_THIS_FILE = Path(__file__).resolve()
_REPO_ROOT = _THIS_FILE.parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.runtime.model_identity import kinematics_hash


def _sanitize_token(text: str) -> str:
    token = re.sub(r"[^0-9A-Za-z._-]+", "_", str(text).strip())
    token = token.strip("._-")
    return token or "item"


def _mat_to_rows(mat: Any) -> list[list[float]] | None:
    if mat is None:
        return None
    try:
        return [
            [float(mat[row, col]) for col in range(4)]
            for row in range(4)
        ]
    except Exception:
        pass

    try:
        rows = mat.tolist()
    except Exception:
        return None

    if not isinstance(rows, list):
        return None
    if len(rows) == 4 and all(isinstance(row, list) and len(row) == 4 for row in rows):
        try:
            return [[float(value) for value in row] for row in rows]
        except Exception:
            return None
    return None


def _mat_to_vector(mat: Any) -> list[float] | None:
    if mat is None:
        return None
    try:
        values = mat.list()
        return [float(value) for value in values]
    except Exception:
        pass
    try:
        return [float(value) for value in list(mat)]
    except Exception:
        return None


def _safe_item_valid(item: Any) -> bool:
    try:
        return bool(item.Valid())
    except Exception:
        return False


def _safe_item_type(item: Any) -> int | None:
    try:
        return int(item.Type())
    except Exception:
        return None


def _safe_item_name(item: Any) -> str:
    try:
        return str(item.Name())
    except Exception:
        return "Unnamed"


def _safe_parent(item: Any) -> Any | None:
    try:
        parent = item.Parent()
    except Exception:
        return None
    return parent if _safe_item_valid(parent) else None


def _item_metadata(item: Any) -> dict[str, Any]:
    parent = _safe_parent(item)
    return {
        "name": _safe_item_name(item),
        "type": _safe_item_type(item),
        "pose_abs": _mat_to_rows(_safe_call(item, "PoseAbs")),
        "pose_local": _mat_to_rows(_safe_call(item, "Pose")),
        "parent_name": _safe_item_name(parent) if parent is not None else None,
        "parent_type": _safe_item_type(parent) if parent is not None else None,
    }


def _safe_call(item: Any, method_name: str) -> Any:
    try:
        fn = getattr(item, method_name)
    except Exception:
        return None
    if not callable(fn):
        return None
    try:
        return fn()
    except Exception:
        return None


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _rgba_from_value(value: Any) -> list[float] | None:
    if value is None:
        return None
    try:
        values = list(value)
    except Exception:
        return None
    if len(values) < 3:
        return None

    try:
        rgba = [float(values[i]) for i in range(min(4, len(values)))]
    except Exception:
        return None
    if len(rgba) == 3:
        rgba.append(1.0)

    if max(rgba[:3]) > 1.0 or rgba[3] > 1.0:
        rgba = [component / 255.0 for component in rgba]
    return [_clamp01(component) for component in rgba[:4]]


def _rgba_to_hex(rgba: list[float]) -> str:
    r, g, b = [int(round(_clamp01(component) * 255.0)) for component in rgba[:3]]
    return f"#{r:02x}{g:02x}{b:02x}"


def _safe_item_color(item: Any) -> list[float] | None:
    return _rgba_from_value(_safe_call(item, "Color"))


def _classify_visual_role(
    name: str,
    *,
    is_robot_link: bool = False,
    link_id: int | None = None,
    item_type: int | None = None,
) -> str:
    lowered = str(name or "").lower()
    if is_robot_link:
        if link_id == 0 or "base" in lowered:
            return "robot_base"
        return "robot_link"
    if lowered == "base" or lowered.endswith("_base"):
        return "robot_base"
    if lowered in {"j1", "j2", "j3", "j4", "j5", "j6"}:
        return "robot_link"
    if "tool" in lowered or "gripper" in lowered or "tcp" in lowered:
        return "tool"
    if "target" in lowered:
        return "target"
    if "frame" in lowered or "base" in lowered:
        return "frame"
    return "fixture"


def _default_material_for_role(role: str) -> dict[str, Any]:
    presets: dict[str, dict[str, Any]] = {
        "robot_link": {
            "base_color_rgba": [1.0, 0.27, 0.0, 1.0],
            "metalness": 0.15,
            "roughness": 0.42,
            "double_sided": False,
        },
        "robot_base": {
            "base_color_rgba": [0.015, 0.018, 0.018, 1.0],
            "metalness": 0.25,
            "roughness": 0.55,
            "double_sided": False,
        },
        "fixture": {
            "base_color_rgba": [0.62, 0.68, 0.86, 0.72],
            "metalness": 0.05,
            "roughness": 0.62,
            "double_sided": True,
        },
        "tool": {
            "base_color_rgba": [0.82, 0.84, 0.90, 0.92],
            "metalness": 0.35,
            "roughness": 0.38,
            "double_sided": False,
        },
        "frame": {
            "base_color_rgba": [0.82, 0.88, 1.0, 0.85],
            "metalness": 0.0,
            "roughness": 0.5,
            "double_sided": True,
        },
        "target": {
            "base_color_rgba": [0.15, 0.80, 0.20, 1.0],
            "metalness": 0.0,
            "roughness": 0.35,
            "double_sided": True,
        },
    }
    payload = dict(presets.get(role, presets["fixture"]))
    rgba = list(payload["base_color_rgba"])
    payload.update(
        {
            "role": role,
            "source": "preset.robodk_like",
            "base_color_hex": _rgba_to_hex(rgba),
            "opacity": float(rgba[3]),
            "transparent": bool(rgba[3] < 0.999),
        }
    )
    return payload


def _material_payload(item: Any, role: str) -> dict[str, Any]:
    payload = _default_material_for_role(role)
    item_rgba = _safe_item_color(item)
    if item_rgba is None:
        return payload
    payload["base_color_rgba"] = item_rgba
    payload["base_color_hex"] = _rgba_to_hex(item_rgba)
    payload["opacity"] = float(item_rgba[3])
    payload["transparent"] = bool(item_rgba[3] < 0.999)
    payload["source"] = "robodk.item_color"
    return payload


def _mesh_asset_payload(
    export_result: ExportResult,
    *,
    mesh_ext: str,
    usage: str,
    source: str,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "available": bool(export_result.ok and export_result.file_name),
        "usage": usage,
        "format": mesh_ext.lstrip(".").lower(),
        "units": "mm",
        "source": source,
    }
    if export_result.file_name:
        payload["file"] = export_result.file_name
    if export_result.message:
        payload["error"] = export_result.message
    return payload


def _collision_asset_from_visual(export_result: ExportResult, *, mesh_ext: str) -> dict[str, Any]:
    payload = _mesh_asset_payload(
        export_result,
        mesh_ext=mesh_ext,
        usage="collision",
        source="visual_mesh_fallback",
    )
    payload["quality"] = "preview_only"
    payload["shared_with_visual"] = bool(export_result.ok and export_result.file_name)
    return payload


def _viewer_hints_payload(preset: str) -> dict[str, Any]:
    if preset == "neutral":
        return {
            "preset": "neutral_inspection",
            "units": "mm",
            "background": {"type": "solid", "color": "#101217"},
            "rendering": {
                "antialias": True,
                "shadows": True,
                "ambient_occlusion": True,
                "tone_mapping": "aces",
                "output_color_space": "srgb",
                "pixel_ratio_max": 2.0,
            },
            "camera": {"projection": "perspective", "fit": "scene_bounds", "padding_ratio": 0.18},
        }

    return {
        "preset": "robodk_like_inspection",
        "units": "mm",
        "background": {
            "type": "vertical_gradient",
            "top": "#07073f",
            "bottom": "#515b97",
        },
        "rendering": {
            "antialias": True,
            "shadows": True,
            "ambient_occlusion": True,
            "tone_mapping": "aces",
            "output_color_space": "srgb",
            "pixel_ratio_max": 2.0,
            "sort_transparent_objects": True,
        },
        "lighting": [
            {"type": "hemisphere", "sky_color": "#eef3ff", "ground_color": "#222640", "intensity": 1.8},
            {"type": "directional", "color": "#ffffff", "intensity": 3.2, "position_mm": [2500, -3500, 5000], "cast_shadow": True},
            {"type": "directional", "color": "#d6e2ff", "intensity": 0.8, "position_mm": [-4000, 2500, 2400], "cast_shadow": False},
        ],
        "camera": {
            "projection": "perspective",
            "fit": "scene_bounds",
            "padding_ratio": 0.18,
            "near_mm": 1.0,
            "far_mm": 50000.0,
        },
        "overlays": {
            "show_world_axes": True,
            "show_target_axes": True,
            "show_tcp_axes": True,
            "axis_length_mm": 180.0,
            "axis_radius_mm": 5.0,
            "label_font_family": "monospace",
            "label_color": "#ffffff",
        },
        "interaction": {
            "orbit_controls": True,
            "zoom_to_cursor": True,
            "double_click_focus": True,
            "preserve_solver_camera_independence": True,
        },
        "quality_notes": [
            "Use GLB when available for colored previews; retain original mesh files for exact inspection.",
            "Use solver FK/path results for link transforms; do not compute FK in the viewer adapter.",
            "Collision assets may initially share visual meshes and should be replaced by simplified meshes later.",
        ],
    }


@dataclass(frozen=True)
class ExportResult:
    file_name: str | None
    ok: bool
    message: str | None = None


def _export_item_mesh(item: Any, output_path: Path) -> ExportResult:
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        item.Save(str(output_path))
    except Exception as exc:
        return ExportResult(file_name=None, ok=False, message=f"{type(exc).__name__}: {exc}")

    if not output_path.exists():
        return ExportResult(file_name=None, ok=False, message="export returned but file was not created")
    file_size = output_path.stat().st_size
    if file_size <= 128:
        try:
            output_path.unlink()
        except Exception:
            pass
        return ExportResult(
            file_name=None,
            ok=False,
            message=f"exported mesh is suspiciously small ({file_size} bytes), skipped as placeholder",
        )
    return ExportResult(file_name=output_path.name, ok=True, message=None)


def _build_station_path_hint(rdk: Any, robolink_mod: Any) -> str | None:
    try:
        value = rdk.getParam(robolink_mod.PATH_OPENSTATION)
    except Exception:
        return None
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _export_robot_links(
    robot: Any,
    *,
    mesh_dir: Path,
    mesh_ext: str,
    max_links: int,
) -> tuple[list[dict[str, Any]], list[Any]]:
    robot_name = _safe_item_name(robot)
    robot_token = _sanitize_token(robot_name)
    links: list[dict[str, Any]] = []
    exported_link_items: list[Any] = []

    missing_in_a_row = 0
    for link_id in range(max_links):
        link_item = _safe_call_with_arg(robot, "ObjectLink", link_id)
        if not _safe_item_valid(link_item):
            missing_in_a_row += 1
            if link_id > 0 and missing_in_a_row >= 2:
                break
            continue

        missing_in_a_row = 0
        exported_link_items.append(link_item)
        link_name = _safe_item_name(link_item)
        visual_role = _classify_visual_role(
            link_name,
            is_robot_link=True,
            link_id=link_id,
            item_type=_safe_item_type(link_item),
        )
        file_name = f"{robot_token}__link_{link_id:02d}__{_sanitize_token(link_name)}.{mesh_ext}"
        out_path = mesh_dir / file_name
        export_result = _export_item_mesh(link_item, out_path)
        link_payload = {
            "link_id": int(link_id),
            "kinematic_link_index": int(link_id),
            **_item_metadata(link_item),
            "visual_role": visual_role,
            "material": _material_payload(link_item, visual_role),
            "visual_asset": _mesh_asset_payload(
                export_result,
                mesh_ext=mesh_ext,
                usage="visual",
                source="robodk_object_link_export",
            ),
            "collision_asset": _collision_asset_from_visual(export_result, mesh_ext=mesh_ext),
            "mesh_file": export_result.file_name,
            "mesh_export_ok": bool(export_result.ok),
            "mesh_export_error": export_result.message,
        }
        links.append(link_payload)

    return links, exported_link_items


def _safe_call_with_arg(item: Any, method_name: str, arg0: Any) -> Any:
    try:
        fn = getattr(item, method_name)
    except Exception:
        return None
    if not callable(fn):
        return None
    try:
        return fn(arg0)
    except Exception:
        return None


def _items_equal(left: Any, right: Any) -> bool:
    if left is None or right is None:
        return False
    try:
        return bool(left.equals(right))
    except Exception:
        return False


def _is_robot_link_item(item: Any, robot_link_items: list[Any]) -> bool:
    for link_item in robot_link_items:
        if _items_equal(item, link_item):
            return True
    return False


def _export_scene_objects(
    rdk: Any,
    *,
    mesh_dir: Path,
    mesh_ext: str,
    robolink_mod: Any,
    skip_items: list[Any],
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for index, obj in enumerate(rdk.ItemList(filter=robolink_mod.ITEM_TYPE_OBJECT), start=1):
        if not _safe_item_valid(obj):
            continue
        if _is_robot_link_item(obj, skip_items):
            continue
        obj_name = _safe_item_name(obj)
        visual_role = _classify_visual_role(
            obj_name,
            is_robot_link=False,
            item_type=_safe_item_type(obj),
        )
        file_name = f"scene_object_{index:03d}__{_sanitize_token(obj_name)}.{mesh_ext}"
        out_path = mesh_dir / file_name
        export_result = _export_item_mesh(obj, out_path)
        payload = {
            "object_index": int(index),
            **_item_metadata(obj),
            "visual_role": visual_role,
            "material": _material_payload(obj, visual_role),
            "visual_asset": _mesh_asset_payload(
                export_result,
                mesh_ext=mesh_ext,
                usage="visual",
                source="robodk_scene_object_export",
            ),
            "collision_asset": _collision_asset_from_visual(export_result, mesh_ext=mesh_ext),
            "mesh_file": export_result.file_name,
            "mesh_export_ok": bool(export_result.ok),
            "mesh_export_error": export_result.message,
        }
        results.append(payload)
    return results


def _collect_items_metadata(rdk: Any, *, item_type: int) -> list[dict[str, Any]]:
    payloads: list[dict[str, Any]] = []
    for item in rdk.ItemList(filter=item_type):
        if not _safe_item_valid(item):
            continue
        name = _safe_item_name(item)
        visual_role = _classify_visual_role(name, item_type=item_type)
        payload = {
            **_item_metadata(item),
            "visual_role": visual_role,
            "material": _material_payload(item, visual_role),
        }
        payloads.append(payload)
    return payloads


def _ensure_output_dir(path: Path, overwrite: bool) -> None:
    if path.exists() and any(path.iterdir()) and not overwrite:
        raise FileExistsError(
            f"Output directory is not empty: {path} (use --overwrite to allow reuse)."
        )
    path.mkdir(parents=True, exist_ok=True)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export meshes and hierarchy metadata from the current RoboDK station "
            "for reuse in custom viewers (for example glTF + JSON pipelines)."
        )
    )
    parser.add_argument(
        "--station",
        type=Path,
        default=None,
        help="Optional .rdk station path to open before exporting. Defaults to active station.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts") / "model_rebuild",
        help="Output folder. A timestamp subfolder is created automatically.",
    )
    parser.add_argument(
        "--robot-name",
        type=str,
        default=None,
        help="Optional robot item name. By default, export all robots in the station.",
    )
    parser.add_argument(
        "--mesh-ext",
        type=str,
        default="stl",
        help="Mesh extension for exports, usually stl/step/iges.",
    )
    parser.add_argument(
        "--max-robot-links",
        type=int,
        default=32,
        help="Maximum robot link index to probe through ObjectLink(link_id).",
    )
    parser.add_argument(
        "--export-robot-links",
        action="store_true",
        help=(
            "Also try exporting robot links through ObjectLink(link_id). "
            "This is optional because some stations expose non-mesh link handles."
        ),
    )
    parser.add_argument(
        "--skip-scene-objects",
        action="store_true",
        help="Skip exporting non-robot scene objects.",
    )
    parser.add_argument(
        "--skip-frames",
        action="store_true",
        help="Skip exporting frame metadata.",
    )
    parser.add_argument(
        "--skip-tools",
        action="store_true",
        help="Skip exporting tool metadata.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow writing to a non-empty output directory.",
    )
    parser.add_argument(
        "--build-glb",
        action="store_true",
        help=(
            "Try packing exported meshes into a single scene.glb "
            "(requires trimesh + numpy)."
        ),
    )
    parser.add_argument(
        "--visual-preset",
        choices=("robodk", "neutral"),
        default="robodk",
        help="Viewer hint preset to include in metadata.json.",
    )
    parser.add_argument(
        "--glb-no-pose-abs",
        action="store_true",
        help="When building GLB, do not apply pose_abs transforms from metadata.",
    )
    return parser.parse_args()


def _iter_mesh_entries(payload: dict[str, Any]) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for robot in payload.get("robots", []):
        for link in robot.get("links", []):
            if bool(link.get("mesh_export_ok")) and link.get("mesh_file"):
                entries.append(link)
    for item in payload.get("scene_objects", []):
        if bool(item.get("mesh_export_ok")) and item.get("mesh_file"):
            entries.append(item)
    return entries


def _build_glb(
    *,
    payload: dict[str, Any],
    output_root: Path,
    mesh_dir: Path,
    apply_pose_abs: bool,
) -> tuple[bool, str]:
    try:
        import numpy as np  # type: ignore
        import trimesh  # type: ignore
    except Exception as exc:
        return False, f"missing dependency for glb build: {type(exc).__name__}: {exc}"

    scene = trimesh.Scene()
    mesh_entries = _iter_mesh_entries(payload)
    if not mesh_entries:
        return False, "no mesh entries available for glb build"

    for index, entry in enumerate(mesh_entries, start=1):
        visual_asset = entry.get("visual_asset") if isinstance(entry.get("visual_asset"), dict) else {}
        mesh_file = visual_asset.get("file") or entry.get("mesh_file")
        if not mesh_file:
            continue
        mesh_path = mesh_dir / str(mesh_file)
        if not mesh_path.exists():
            continue
        try:
            geom = trimesh.load(mesh_path, force="mesh")
        except Exception:
            continue

        material = entry.get("material") if isinstance(entry.get("material"), dict) else {}
        rgba = _rgba_from_value(material.get("base_color_rgba"))
        if rgba is not None:
            color = np.array([int(round(_clamp01(component) * 255.0)) for component in rgba], dtype=np.uint8)
            try:
                face_count = int(len(getattr(geom, "faces", [])))
                if face_count > 0:
                    geom.visual.face_colors = np.tile(color, (face_count, 1))
                else:
                    vertex_count = int(len(getattr(geom, "vertices", [])))
                    if vertex_count > 0:
                        geom.visual.vertex_colors = np.tile(color, (vertex_count, 1))
            except Exception:
                pass

        transform = np.eye(4)
        if apply_pose_abs:
            pose_abs = entry.get("pose_abs")
            if isinstance(pose_abs, list) and len(pose_abs) == 4:
                try:
                    transform = np.array(pose_abs, dtype=float)
                except Exception:
                    transform = np.eye(4)

        node_name = f"{index:04d}_{_sanitize_token(str(entry.get('name', 'item')))}"
        scene.add_geometry(geom, node_name=node_name, transform=transform)

    glb_path = output_root / "scene.glb"
    try:
        scene.export(glb_path)
    except Exception as exc:
        return False, f"failed to export scene.glb: {type(exc).__name__}: {exc}"

    if not glb_path.exists():
        return False, "scene.glb export returned but file was not created"
    return True, str(glb_path)


def main() -> int:
    args = _parse_args()

    try:
        from robodk import robolink  # type: ignore
    except Exception as exc:
        print(
            "RoboDK Python API is unavailable. Install/select a Python environment "
            "with package 'robodk'."
        )
        print(f"Import error: {type(exc).__name__}: {exc}")
        return 2

    rdk = robolink.Robolink()

    if args.station is not None:
        station_path = args.station.expanduser().resolve()
        if not station_path.exists():
            print(f"Station file not found: {station_path}")
            return 2
        loaded = rdk.AddFile(str(station_path))
        if not _safe_item_valid(loaded):
            print(f"Failed to open station: {station_path}")
            return 2
        if _safe_item_type(loaded) == robolink.ITEM_TYPE_STATION:
            rdk.setActiveStation(loaded)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_root = args.output_dir.expanduser().resolve() / timestamp
    mesh_dir = output_root / "meshes"
    _ensure_output_dir(output_root, overwrite=args.overwrite)
    mesh_dir.mkdir(parents=True, exist_ok=True)

    if args.robot_name:
        robot_items = [rdk.Item(args.robot_name, robolink.ITEM_TYPE_ROBOT)]
        robot_items = [item for item in robot_items if _safe_item_valid(item)]
        if not robot_items:
            print(f"Robot not found: {args.robot_name}")
            return 2
    else:
        robot_items = [
            item
            for item in rdk.ItemList(filter=robolink.ITEM_TYPE_ROBOT)
            if _safe_item_valid(item)
        ]

    if not robot_items:
        print("No valid robot items found in station.")
        return 2

    payload: dict[str, Any] = {
        "schema_version": 1,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "station_path_hint": _build_station_path_hint(rdk, robolink),
        "mesh_extension": args.mesh_ext.lstrip("."),
        "viewer_hints": _viewer_hints_payload(args.visual_preset),
        "robots": [],
        "scene_objects": [],
        "frames": [],
        "tools": [],
    }

    all_robot_link_items: list[Any] = []
    for robot in robot_items:
        robot_name = _safe_item_name(robot)
        links: list[dict[str, Any]] = []
        exported_link_items: list[Any] = []
        if args.export_robot_links:
            links, exported_link_items = _export_robot_links(
                robot,
                mesh_dir=mesh_dir,
                mesh_ext=args.mesh_ext.lstrip("."),
                max_links=max(1, int(args.max_robot_links)),
            )
        all_robot_link_items.extend(exported_link_items)
        lower, upper, joint_type = _safe_joint_limits(robot)
        kinematics_inferred = _infer_robot_kinematics(robot)
        inferred_hash = _kinematics_hash_from_inferred(kinematics_inferred)
        joints_current = _mat_to_vector(_safe_call(robot, "Joints"))
        robot_payload = {
            "name": robot_name,
            "model_id": _sanitize_token(robot_name),
            "kinematics_hash": inferred_hash,
            "type": _safe_item_type(robot),
            "pose_abs": _mat_to_rows(_safe_call(robot, "PoseAbs")),
            "pose_frame": _mat_to_rows(_safe_call(robot, "PoseFrame")),
            "pose_tool": _mat_to_rows(_safe_call(robot, "PoseTool")),
            "joints_current_deg": joints_current,
            "joints_at_export_deg": joints_current,
            "joints_home_deg": _mat_to_vector(_safe_call(robot, "JointsHome")),
            "joint_limits_lower_deg": lower,
            "joint_limits_upper_deg": upper,
            "joint_type": joint_type,
            "kinematics_inferred": kinematics_inferred,
            "visual_style": {
                "default_link_role": "robot_link",
                "base_link_role": "robot_base",
                "viewer_transform_source": "solver_fk_or_path_result",
            },
            "links": links,
        }
        payload["robots"].append(robot_payload)

    if not args.skip_scene_objects:
        payload["scene_objects"] = _export_scene_objects(
            rdk,
            mesh_dir=mesh_dir,
            mesh_ext=args.mesh_ext.lstrip("."),
            robolink_mod=robolink,
            skip_items=all_robot_link_items,
        )

    if not args.skip_frames:
        payload["frames"] = _collect_items_metadata(rdk, item_type=robolink.ITEM_TYPE_FRAME)
    if not args.skip_tools:
        payload["tools"] = _collect_items_metadata(rdk, item_type=robolink.ITEM_TYPE_TOOL)

    metadata_path = output_root / "metadata.json"
    metadata_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    glb_result: dict[str, Any] | None = None
    if args.build_glb:
        ok, detail = _build_glb(
            payload=payload,
            output_root=output_root,
            mesh_dir=mesh_dir,
            apply_pose_abs=not bool(args.glb_no_pose_abs),
        )
        glb_result = {
            "ok": bool(ok),
            "detail": detail,
            "apply_pose_abs": bool(not args.glb_no_pose_abs),
        }
        payload["glb_result"] = glb_result
        metadata_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    summary = {
        "output_root": str(output_root),
        "metadata": str(metadata_path),
        "mesh_count": int(len(list(mesh_dir.glob(f"*.{args.mesh_ext.lstrip('.')}")))),
        "robot_count": int(len(payload["robots"])),
        "scene_object_count": int(len(payload["scene_objects"])),
    }
    if glb_result is not None:
        summary["glb"] = glb_result
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def _safe_joint_limits(robot: Any) -> tuple[list[float] | None, list[float] | None, float | None]:
    try:
        lower, upper, joint_type = robot.JointLimits()
    except Exception:
        return None, None, None
    return _mat_to_vector(lower), _mat_to_vector(upper), float(joint_type)


def _mat3_transpose(mat3: list[list[float]]) -> list[list[float]]:
    return [
        [mat3[0][0], mat3[1][0], mat3[2][0]],
        [mat3[0][1], mat3[1][1], mat3[2][1]],
        [mat3[0][2], mat3[1][2], mat3[2][2]],
    ]


def _mat3_mul(left: list[list[float]], right: list[list[float]]) -> list[list[float]]:
    return [
        [
            left[row][0] * right[0][col]
            + left[row][1] * right[1][col]
            + left[row][2] * right[2][col]
            for col in range(3)
        ]
        for row in range(3)
    ]


def _axis_angle_from_rotation(rot: list[list[float]]) -> tuple[list[float], float]:
    import math

    trace = float(rot[0][0] + rot[1][1] + rot[2][2])
    cos_theta = max(-1.0, min(1.0, (trace - 1.0) / 2.0))
    theta = float(math.acos(cos_theta))
    if theta < 1e-10:
        return [0.0, 0.0, 1.0], 0.0

    denom = 2.0 * math.sin(theta)
    if abs(denom) < 1e-12:
        return [0.0, 0.0, 1.0], theta

    ax = (rot[2][1] - rot[1][2]) / denom
    ay = (rot[0][2] - rot[2][0]) / denom
    az = (rot[1][0] - rot[0][1]) / denom
    norm = math.sqrt(ax * ax + ay * ay + az * az)
    if norm < 1e-12:
        return [0.0, 0.0, 1.0], theta
    return [ax / norm, ay / norm, az / norm], theta


def _safe_joint_poses_rows(robot: Any, joints_deg: list[float]) -> list[list[list[float]]] | None:
    poses = _safe_call_with_arg(robot, "JointPoses", joints_deg)
    if not isinstance(poses, list):
        return None
    rows_list: list[list[list[float]]] = []
    for pose in poses:
        rows = _mat_to_rows(pose)
        if rows is None:
            return None
        rows_list.append(rows)
    return rows_list


def _infer_robot_kinematics(robot: Any) -> dict[str, Any] | None:
    joints_home = _mat_to_vector(_safe_call(robot, "JointsHome"))
    if joints_home is None or len(joints_home) < 6:
        return None
    q_home = [float(joints_home[i]) for i in range(6)]

    poses_home = _safe_joint_poses_rows(robot, q_home)
    if poses_home is None or len(poses_home) < 7:
        return None

    axes: list[list[float]] = []
    points: list[list[float]] = []
    senses = [1.0] * 6
    step_deg = 1.0
    for joint_index in range(6):
        q_step = q_home.copy()
        q_step[joint_index] = q_step[joint_index] + step_deg
        poses_step = _safe_joint_poses_rows(robot, q_step)
        if poses_step is None or len(poses_step) < 7:
            return None

        pose0 = poses_home[joint_index + 1]
        pose1 = poses_step[joint_index + 1]
        rot0 = [[float(pose0[r][c]) for c in range(3)] for r in range(3)]
        rot1 = [[float(pose1[r][c]) for c in range(3)] for r in range(3)]
        drot = _mat3_mul(rot1, _mat3_transpose(rot0))
        axis, _theta = _axis_angle_from_rotation(drot)

        point = [
            float(pose0[0][3]),
            float(pose0[1][3]),
            float(pose0[2][3]),
        ]
        axes.append(axis)
        points.append(point)

    return {
        "joint_axes_base": axes,
        "joint_points_base_mm": points,
        "joint_senses": senses,
        "home_flange": poses_home[6],
        "joints_home_deg": q_home,
        "source": "robot.JointPoses(home,+1deg)",
    }


def _kinematics_hash_from_inferred(kinematics_inferred: dict[str, Any] | None) -> str | None:
    if not isinstance(kinematics_inferred, dict):
        return None
    try:
        return kinematics_hash(
            axes=kinematics_inferred.get("joint_axes_base"),
            points=kinematics_inferred.get("joint_points_base_mm"),
            senses=kinematics_inferred.get("joint_senses"),
            home_flange=kinematics_inferred.get("home_flange"),
        )
    except Exception:
        return None


if __name__ == "__main__":
    raise SystemExit(main())
