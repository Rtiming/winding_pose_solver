# AGENTS.md

This file applies to `src/core/`.

## Own This Layer For

- geometry and frame math
- CSV readers and writers
- request and result schemas
- pose solving from centerline data
- backend-agnostic robot interface helpers
- visualization that does not need a live RoboDK station

## Good Edit Targets

- `frame_math.py`, `geometry.py`, `simple_mat.py`: reusable math and transforms
- `pose_solver.py`, `pose_csv.py`: centerline-to-pose generation and CSV plumbing
- `collab_models.py`, `types.py`: shared request/result payload shapes and typing
- `robot_interface.py`: backend-agnostic abstraction boundary
- `visualization.py`: shared visualization helpers that do not require station access

## Keep Out Of This Layer

- live RoboDK station access
- SSH or remote orchestration
- high-level path-search policy
- batch-sweep coordination or runtime logging

If the code needs an open RoboDK station, it probably belongs in `src/robodk_runtime/`.

## Editing Guidance

- Favor pure functions and data-shape clarity here because this layer is reused by local runs, online flows, and diagnostics scripts.
- If you change schemas in `collab_models.py`, inspect all producers and consumers in `src/runtime/`, `online_roundtrip.py`, and worker-side code.
- Keep backend-agnostic abstractions honest. Do not quietly leak RoboDK-only assumptions into shared interfaces.

## Validation

- Prefer import checks and small focused payload or CSV roundtrips.
- If a schema changes, validate at least one caller and one consumer.
