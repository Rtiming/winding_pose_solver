# AGENTS.md

This file applies to `src/runtime/`.

## Own This Layer For

- local app orchestration
- request construction
- requester-side remote search flow
- runtime profiling
- run logging and archive emission
- origin sweep coordination

## Good Edit Targets

- `app.py`: high-level local flow and user-facing runtime coordination
- `request_builder.py`: request payload assembly
- `remote_search.py`: requester-side remote search logic
- `run_logging.py`, `profiler.py`: logging and profiling behavior
- `origin_sweep.py`: coordinated Y/Z sweep and adaptive iteration helpers
- `local_retry.py`: local retry and profile-refinement orchestration

## Keep Out Of This Layer

- pure geometry and schemas that belong in `src/core/`
- live RoboDK station behavior that belongs in `src/robodk_runtime/`
- low-level kinematics internals that belong in `src/six_axis_ik/`
- detailed DP or repair heuristics that belong in `src/search/`

## Editing Guidance

- Keep top-level flows understandable: `main.py` and the online entrypoints should stay thin and wire into this layer rather than accumulating logic themselves.
- Preserve artifact layout under `artifacts/local_runs/`, `artifacts/online_runs/`, and `artifacts/run_logs/` unless the task explicitly changes conventions.
- If you change request or result wiring, verify both the builder side and the consumer side.
- `origin_sweep.py` and related profiling helpers can become CPU-heavy; treat large experiments as Slurm work.

## Validation

- Import checks are enough for small refactors.
- For request-building changes, validate the produced payload shape.
- For orchestration changes, prefer the smallest local solve or online smoke flow that hits the changed path.
