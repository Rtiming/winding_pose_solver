# winding_pose_solver

Pose generation, IK validation, and RoboDK program materialization for the winding task.

This repository supports two interchangeable IK backends:

- `robodk`: uses RoboDK as the live truth source
- `six_axis_ik`: uses the embedded local analytic solver in `src/six_axis_ik/`

The codebase is now organized by responsibility so the runtime entrypoints, live RoboDK integration, and path-search logic are easier to maintain independently.

## Current Default Setup

The main run config lives at the top of `main.py`.

Current important defaults:

```python
TARGET_FRAME_A_ORIGIN_IN_FRAME2_MM = (1126.0, -650.0, 1130.0)
TARGET_FRAME_A_ROTATION_IN_FRAME2_XYZ_DEG = (0, 0, -180.0)
IK_BACKEND = "six_axis_ik"
```

With this orientation and the current strict constraints (`A2_MAX_DEG = 115`), the local pipeline currently produces a valid continuous path (`config_switches=0`, `bridge_like_segments=0`).

## Repository Layout

```text
main.py
app_settings.py
online_requester.py
online_worker.py
online_roundtrip.py
scripts/
src/
  core/
  runtime/
  robodk_runtime/
  search/
  six_axis_ik/
  README.md
```

### Package responsibilities

- `src/core/`
  - Shared math, CSV loading, schema models, pose solving, visualization, and backend-agnostic helpers.
- `src/runtime/`
  - High-level local and online orchestration.
  - Includes app flow, request building, requester-side search runner, runtime profiling, and run-log formatting helpers (`src/runtime/run_logging.py`).
  - Includes origin sweep utilities (`src/runtime/origin_sweep.py`) used for parallel Y/Z search and adaptive iteration (`run_grid_sweep`, `run_adaptive_sweep`).
- `src/robodk_runtime/`
  - RoboDK-station-bound logic.
  - Includes live evaluation against the open station and final RoboDK program creation.
- `src/search/`
  - Exact path search and repair pipeline.
  - Includes IK candidate collection, DP path selection, window refinement, and inserted-transition repair.
- `src/six_axis_ik/`
  - Embedded local six-axis IK/FK model and analytic backend.
  - Can run without a live RoboDK station for evaluation-only workflows.

Each package directory now also contains a short local `README.md` describing its role, which helps both humans and coding agents quickly find the right edit surface.

### Compatibility wrappers

Older flat imports are still supported through thin wrapper files such as:

- `src/app_runner.py`
- `src/request_builder.py`
- `src/remote_search_runner.py`
- `src/runtime_profiler.py`
- `src/robodk_eval_worker.py`
- `src/robodk_program.py`
- `src/bridge_builder.py`
- `src/collab_models.py`
- `src/frame_math.py`
- `src/geometry.py`
- `src/motion_settings.py`
- `src/pose_csv.py`
- `src/pose_solver.py`
- `src/robot_interface.py`
- `src/types.py`
- `src/visualization.py`
- `src/global_search.py`
- `src/ik_collection.py`
- `src/local_repair.py`
- `src/path_optimizer.py`

These wrappers re-export the new package contents so existing scripts do not break while the codebase transitions to the new structure.

## Main Workflow

For the single-machine flow, the pipeline is:

1. Read `data/validation_centerline.csv`.
2. Build each process-local frame from the centerline geometry.
3. Solve the required tool pose in `Frame 2` for the fixed target `Frame A`.
4. Write `data/tool_poses_frame2.csv`.
5. Collect IK candidates with the selected backend.
6. Run exact path search and optional local repair.
7. Validate continuity and, if valid, create a RoboDK program.

The key idea is:

```text
T_frame2_tool(i) = T_frame2_A * inverse(T_tool_proc(i))
```

So each row in the centerline defines a local process frame on the tool, and the code solves the tool pose needed to place that local frame onto the fixed target `A` in `Frame 2`.

## Continuous Y/Z Optimization Workflow

The path solver keeps orientation fixed and optimizes a continuous Frame-A origin Y/Z profile along the whole trajectory.

- Global stage:
  - Build IK candidates and run exact DP.
  - Apply full-path continuous Y/Z refinement (LSQ-based) and re-evaluate with exact search.
- Local stage:
  - Run window repair and handover corridor refinement near problematic transitions.
  - Keep strict hard constraints (`A1`, `A2`, joint continuity) and only accept exact improvements.
- Optional outer search:
  - Sweep `TARGET_FRAME_A_ORIGIN_IN_FRAME2_MM` over Y/Z with parallel workers to find a better basin, then run the exact pipeline.

This means the system is not only selecting discrete IK points; it is also optimizing a trajectory-wide continuous Y/Z profile under the same strict constraints.

## Entry Points

### Single-machine

```powershell
python main.py
```

Default `single` action is now `solve` (compute path only, no RoboDK program import).
This mode can run with `ik_backend="six_axis_ik"` without requiring RoboDK.

Explicitly choose single-machine action:

```powershell
python main.py --mode single --single-action solve
python main.py --mode single --single-action program
```

Visualization only (legacy shortcut still supported):

```powershell
python main.py --visualize
```

Single-run artifacts are written under `artifacts/local_runs/<run_id>/`:

- `request.json`: exact evaluation request used for this run.
- `eval_result.json`: full evaluation result payload.
- `selected_joint_path.csv`: solved execution path (`j1..j6` joint angles + config flags).
- `run_archive.json`: compact run archive with basic settings, status, and key metrics.

`run_archive.json` is written for both success and runtime failure paths.

### Online requester

```powershell
python online_requester.py propose --request request.json --candidates candidates.json
python online_requester.py summarize --results results.json --summary summary.json
```

### Online worker

```powershell
python online_worker.py eval --request request.json --result result.json
python online_worker.py eval-batch --request candidates.json --result results.json
```

### Online roundtrip controller

```powershell
python online_roundtrip.py build-request --request artifacts/online_runs/request.json --candidate-limit 4
python online_roundtrip.py run-round --host master --request artifacts/online_runs/request.json --run-id smoke_round1
```

If the request uses `ik_backend="six_axis_ik"` and does not ask for final program generation during candidate evaluation, you can let the server run the offline IK stage itself:

```powershell
python online_roundtrip.py run-round --host master --request artifacts/online_runs/request.json --run-id smoke_round1 --server-eval-when-possible
```

The final RoboDK program-generation step still stays local.

## IK Backends

Switch the backend in `main.py`:

```python
IK_BACKEND = "six_axis_ik"  # or "robodk"
```

### `robodk`

- Uses RoboDK `SolveIK_All` and related live station behavior.
- Requires RoboDK and the target station to be open.

### `six_axis_ik`

- Uses the embedded local solver in `src/six_axis_ik/`.
- Supports offline evaluation when program creation is disabled.
- Shares the same high-level search pipeline through `src/core/robot_interface.py`.
- In single mode, RoboDK is only required when you choose `--single-action program`
  (or when `ik_backend="robodk"`).

## Diagnostics Scripts

### Parallel target-origin sweep

Search for feasible Y/Z target-origin basins in parallel:

```powershell
python scripts/sweep_target_origin_yz.py --mode grid --x 1126 --y-values=-700,-650,-600 --z-values=1130,1180 --workers 4 --strategy exact_profile --skip-inserted-repair
```

Adaptive iteration around a seed point:

```powershell
python scripts/sweep_target_origin_yz.py --mode adaptive --x 1126 --start-y -650 --start-z 1130 --step-y 20 --step-z 20 --max-iters 6 --workers 4 --strategy full_search
```

The script writes ranked JSON results under `artifacts/tmp/` and is built on `src/runtime/origin_sweep.py` so the algorithm can be reused in tests and future optimization modules.

Recent refactor notes for this sweep path:

- `scripts/sweep_target_origin_yz.py` is now a thin CLI wrapper.
- Core search logic lives in `src/runtime/origin_sweep.py`, which is easier to import directly in tests.
- Parallel case execution now runs in worker batches and reuses per-process offline IK context to reduce process overhead without changing exact scoring/constraints.

### Backend parity

Compare solvability row by row:

```powershell
python scripts/compare_ik_backends.py
```

Current result with the default orientation above:

- `SixAxisIK`: `496/496`
- `RoboDK`: `496/496`
- backend disagreement rows: `0`

### Focused window diagnosis

Inspect a local row window:

```powershell
python scripts/diagnose_ik_window.py --start 395 --end 406 --padding 4
```

This writes a CSV report under:

- `artifacts/diagnostics/`

### Import a diagnostic window into RoboDK

```powershell
python scripts/import_ik_window_to_robodk.py --start 395 --end 406 --padding 4
```

This creates diagnostic targets in the live RoboDK station so you can inspect the neighborhood visually.

## Dependencies

The dependency files are split by runtime role:

- `requirements.shared.txt`
- `requirements.server.txt`
- `requirements.local-worker.txt`
- `environment.server.yml`
- `environment.local-worker.yml`

Use the local-worker environment when running RoboDK-related flows.
The shared dependency set now includes `scipy`, which is required by the embedded `six_axis_ik` backend.

## RoboDK Assumptions

The live station is expected to contain at least:

- robot: `KUKA`
- frame: `Frame 2`

When `six_axis_ik` is used together with a live RoboDK station, the worker now checks whether the live Tool and Frame calibration match the embedded `six_axis_ik` configuration and prints a warning if they diverge.

## Recommended Edit Zones

When you want to change behavior, the most useful files are:

- Project run config: `main.py`
- Default settings and tuning: `app_settings.py`
- Shared math / schemas / pose generation: `src/core/`
- Live evaluation and final validation: `src/robodk_runtime/eval_worker.py`
- Program creation in RoboDK: `src/robodk_runtime/program.py`
- Exact path search and repair: `src/search/`
- Local analytic IK model: `src/six_axis_ik/`

## Smoke Checks After Refactors

These commands are a good minimum regression set:

```powershell
python main.py --help
python online_requester.py --help
python online_worker.py --help
python scripts/sweep_target_origin_yz.py --mode grid --x 1126 --y-values=-650 --z-values=1130 --workers 1 --strategy exact_profile --skip-inserted-repair
python scripts/compare_ik_backends.py
python scripts/diagnose_ik_window.py --start 395 --end 406 --padding 4
```

If you change the target orientation, tool calibration, or frame calibration, rerun the backend comparison and the focused diagnostic window before trusting the new path results.
