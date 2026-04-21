# Model Assets And Collision Plan

This document defines the low-risk path for robot model display and future
collision-model selection.

## Boundary

`winding_pose_solver` owns robot identity, kinematics validation, FK, IK, path
search, repair, and future collision policy.

`winding-motion-module` may load visual assets and apply transforms from the
solver API, but it must not implement FK, IK, path search, repair, or collision
decision logic.

## Current Sources

- `src/runtime/external_api.py`: formal Python API for `configure`, FK, IK, and
  path solving.
- `src/runtime/http_service.py`: formal HTTP facade.
- `scripts/export_robodk_station_assets.py`: RoboDK-side mesh and metadata export.
- `scripts/check_model_demo_quality.py`: mesh, trajectory, and FK consistency
  validation.

## Immediate Display Rules

- Treat `POST /api/configure` as the source of truth for the active robot model.
- Use `kinematics_source`, `kinematics_hash`, and `model_id` from `configure` or
  `/health` to verify the UI is showing assets for the same robot model.
- Use `/api/fk` or path-solve results for transforms. Do not compute robot FK in
  TypeScript.
- If a model asset export was produced at a non-home joint pose, consumers must
  account for `joints_at_export_deg` before applying live FK transforms.
- Load `viewer_hints`, `visual_role`, and `material` from exported metadata
  instead of hard-coding viewer colors in the adapter.
- Prefer `visual_asset` for rendering and `collision_asset` for collision
  previews. In the first stage, `collision_asset` may explicitly point to the
  visual mesh as `visual_mesh_fallback`.

## RoboDK-Class Display Recipe

The viewer can match or exceed the RoboDK screenshot only if model assets carry
enough display metadata. The export script now emits:

- `viewer_hints`: background, lighting, camera fit, antialiasing, shadows,
  ambient occlusion, tone mapping, axes, and label defaults.
- `visual_role`: semantic role for each mesh, such as `robot_link`,
  `robot_base`, `fixture`, `tool`, or `target`.
- `material`: RGBA color, opacity, roughness, metalness, transparency, and
  double-sided rendering hints.
- `visual_asset`: exact mesh for rendering.
- `collision_asset`: collision mesh reference. It can initially share the
  visual mesh but must declare that fallback explicitly.

Recommended frontend rendering choices:

- Use glTF/GLB when `scene.glb` exists, because it preserves a single scene and
  exported colors better than many standalone STL files.
- Use physically based materials, ACES tone mapping, sRGB output, antialiasing,
  shadows, and screen-space ambient occlusion.
- Keep the dark blue inspection background and high-contrast axis labels from
  `viewer_hints`.
- Render fixtures with light blue translucent materials, robot links in orange,
  and the base in black unless RoboDK exported a specific item color.
- Fit the camera to the scene bounds and expose orbit controls, but keep robot
  transforms driven by solver FK/path responses.
- Validate mesh-to-FK alignment using `joints_at_export_deg`,
  `model_id`, and `kinematics_hash`; do not silently display assets for a
  different robot model.

## Manifest Direction

Keep the first manifest custom and small. Avoid introducing URDF/SDF until
external interoperability becomes a hard requirement.

Recommended future manifest shape:

```json
{
  "schema_version": 1,
  "model_id": "KUKA",
  "kinematics_hash": "sha256...",
  "joints_at_export_deg": [0, 0, 0, 0, 0, 0],
  "visual": {
    "links": [
      {
        "link_id": 0,
        "mesh_file": "KUKA__link_00__base.stl",
        "visual_asset": {
          "file": "KUKA__link_00__base.stl",
          "format": "stl",
          "units": "mm",
          "source": "robodk_object_link_export"
        },
        "material": {
          "role": "robot_base",
          "base_color_rgba": [0.015, 0.018, 0.018, 1.0],
          "metalness": 0.25,
          "roughness": 0.55
        },
        "mesh_format": "stl",
        "pose_abs_at_export": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
      }
    ]
  },
  "collision": {
    "links": [
      {
        "link_id": 0,
        "mesh_file": "KUKA__link_00__base.collision.stl",
        "mesh_format": "stl",
        "source": "simplified_convex"
      }
    ]
  }
}
```

In the first stage, collision entries may point to the same mesh files as visual
entries. Simplified collision meshes can be added later without changing the
solver API shape.

## API Direction

Current low-risk fields:

- `POST /api/configure` returns `model_id`, `kinematics_source`, and
  `kinematics_hash`.
- `GET /health` returns the same active model identity fields for reconnect
  checks.
- `strict_kinematics=true` may be sent to `configure` to reject fallback to the
  default configured model when `kinematics_inferred` is missing.

Future additions should stay solver-owned:

- `GET /api/models`: list available model descriptors/manifests.
- `POST /api/configure` with `model_id`: select a known model descriptor instead
  of sending inline kinematics.
- `GET /api/models/{model_id}/assets`: return visual/collision asset references.

## Failure Modes

Hard failures:

- `kinematics_inferred` is present but malformed.
- `strict_kinematics=true` and no `kinematics_inferred` was provided.
- A requested `model_id` does not exist in the future model catalog.
- A selected collision mode requires collision assets and they are missing.

Warnings:

- Visual assets are missing but the request is solve-only.
- `model_id` is absent but the caller explicitly accepts fallback configuration.
- Collision assets fall back to visual meshes in a preview-only mode.

## Validation

Use the smallest validation that proves each layer:

```powershell
python -m compileall src scripts/model_demo_solver_api.py
```

```powershell
python scripts/export_robodk_station_assets.py --export-robot-links --build-glb
```

```powershell
python scripts/check_model_demo_quality.py --metadata <metadata.json> --trajectory-csv <selected_joint_path.csv> --request <request.json>
```

The export and quality commands require a suitable RoboDK station/artifacts.
For server-side compute validation, use the existing offline `six_axis_ik`
path-solve smoke tests instead of requiring RoboDK.
