# `src/` Layout

This repository is organized into four main layers:

- `core/`
  - Shared math, schemas, CSV loading, pose solving, and backend-agnostic helpers.
- `search/`
  - IK candidate collection, DP path search, and local repair.
- `runtime/`
  - High-level local and online orchestration.
- `robodk_runtime/`
  - Live RoboDK evaluation and final program generation.

There is also:

- `six_axis_ik/`
  - The embedded local IK/FK implementation.

The flat files in `src/` are mostly compatibility wrappers kept for older imports.
When adding new logic, prefer placing it inside the package directories above instead of creating new flat modules here.

