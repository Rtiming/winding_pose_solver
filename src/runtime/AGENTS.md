# AGENTS.md

This file applies to `src/runtime/`.

## Own This Layer For

- local app orchestration
- local machine profile resolution for Mac/Windows/Linux coordinator behavior
- request construction
- requester-side remote search flow
- runtime profiling
- run logging and archive emission
- origin sweep coordination

## Good Edit Targets

- `app.py`: high-level local flow and user-facing runtime coordination
- `local_profile.py`: `auto|mac|windows|linux` profile normalization,
  detection, and local Conda Python candidate paths
- `request_builder.py`: request payload assembly
- `remote_search.py`: server-side retry candidate proposal and profile
  finalization
- `run_logging.py`, `profiler.py`: logging and profiling behavior
- `origin_sweep.py`: coordinated Y/Z sweep and adaptive iteration helpers
- `local_retry.py`: local retry and profile-refinement orchestration
- `online/roundtrip.py`: online sync, Slurm compute, server polish, receiver
  orchestration

## Keep Out Of This Layer

- pure geometry and schemas that belong in `src/core/`
- live RoboDK station behavior that belongs in `src/robodk_runtime/`
- low-level kinematics internals that belong in `src/six_axis_ik/`
- detailed DP or repair heuristics that belong in `src/search/`

## Editing Guidance

- Keep top-level flows understandable: `main.py` and the online entrypoints should stay thin and wire into this layer rather than accumulating logic themselves.
- Keep platform-specific local command/path decisions in `local_profile.py` or
  the orchestration boundary. Search, scoring, and IK behavior should remain
  platform-independent.
- Preserve artifact layout under `artifacts/local_runs/`, `artifacts/online_runs/`, and `artifacts/run_logs/` unless the task explicitly changes conventions.
- If you change request or result wiring, verify both the builder side and the consumer side.
- `origin_sweep.py` and related profiling helpers can become CPU-heavy; treat large experiments as Slurm work.
- Keep online retry/repair server-side. The receiver should consume the
  selected handoff and must not silently run a new search before RoboDK
  materialization.
- Preserve the current polish controls unless the task explicitly changes
  policy: `ONLINE_PROFILE_RETRY_*`, `WPS_ONLINE_RETRY_*`,
  `WPS_ONLINE_QUALITY_POLISH`, `WPS_ONLINE_QUALITY_POLISH_TARGET_DEG`, and
  `WPS_LOCAL_RETRY_FOCUS_SEGMENT_LIMIT`.
- Avoid nested process pools in server candidate evaluation. If cloning a
  request for parallel remote-search candidates, force inner local parallelism
  to a single worker unless benchmarking proves otherwise.

## Validation

- Import checks are enough for small refactors.
- For request-building changes, validate the produced payload shape.
- For orchestration changes, prefer the smallest local solve or online smoke flow that hits the changed path.
- For retry/polish changes, compare strict delivery fields and
  `worst_joint_step_deg`; do not accept a lower score if it introduces bridge,
  big-circle, or closed-terminal violations.
