# `src/` Layout

This repository is organized into four main layers:

- `core/`
  - Shared math, schemas, CSV loading, pose solving, and backend-agnostic helpers.
  - `pose_solver.py` exposes both full CSV loading and reusable dataset-based
    pose generation for sweeps.
- `search/`
  - IK candidate collection, DP path search, continuity diagnostics, and local
    repair.
- `runtime/`
  - High-level local and online orchestration.
  - `main_entrypoint.py` is the canonical implementation behind root `main.py`.
  - `runtime/online/` contains coordinator/server/receiver online entrypoints.
  - `delivery.py` owns the official/debug/diagnostic gate decision.
  - `origin_sweep.py` owns grid/adaptive/smart-square Frame-A origin search.
- `robodk_runtime/`
  - Live RoboDK evaluation and final program generation.

There is also:

- `six_axis_ik/`
  - The embedded local IK/FK implementation.

The flat files in `src/` are mostly compatibility wrappers kept for older imports.
When adding new logic, prefer placing it inside the package directories above instead of creating new flat modules here.

Keep these boundaries stable:

- Do not add live RoboDK dependencies to `core/` or server compute paths.
- Do not put DP/rewrite heuristics in `runtime/`; use `search/`.
- Do not put SSH or Slurm orchestration in `search/`; use `runtime/online/`.
- Keep root `main.py` and root `online_*.py` wrappers thin.

