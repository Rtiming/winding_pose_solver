# winding_pose_solver

Winding path pose solving, RoboDK truth evaluation, and multi-mode requester/worker orchestration in one repository.

## What This Repository Supports

This project now has two top-level runtime modes:

- `单机模式`
  - Runs on local Windows with RoboDK.
  - Covers `pose solve -> IK collection -> path search / repair -> validation -> final RoboDK program generation`.
- `联机模式`
  - Still uses one repository, but splits runtime roles:
  - `发送端 / requester`
    - Runs on `master`.
    - Generates candidate `Frame-A` Y/Z profiles, batches experiments, and summarizes failures.
  - `接收端 / local worker`
    - Runs on local Windows with RoboDK.
    - Performs IK, continuity checks, exact evaluation, and optional final program generation.

The online flow is intentionally local-initiated:

1. Local builds a JSON request.
2. Local sends it to `master` with `ssh/scp`.
3. `master` proposes candidate batches.
4. Local worker evaluates those candidates against the live RoboDK station.
5. Results go back to `master` for sorting and summary.
6. The best candidate is re-run locally for final RoboDK program generation when online final-generate is enabled.

The final generation step is not a separate lightweight exporter. It reuses the same local RoboDK worker validation and program-materialization path that single-machine mode uses, so the final online output is checked against the same continuity and path-quality rules before a RoboDK program is kept.

No extra service, no open port, no split repo.

## Mode Matrix

| Mode | Where it runs | Uses RoboDK | Main entry |
| --- | --- | --- | --- |
| Single-machine | Local Windows | Yes | `python main.py` |
| Online requester | `master` | No | `python online_requester.py ...` |
| Online worker | Local Windows | Yes | `python online_worker.py ...` |
| Online controller | Local Windows | Only when it invokes worker | `python online_roundtrip.py ...` |

## Module Boundaries

### Shared / server-safe modules

These modules do not require RoboDK to import or run:

- `src/frame_math.py`
- `src/pose_solver.py`
- `src/geometry.py`
- `src/bridge_builder.py`
- `src/motion_settings.py`
- `src/pose_csv.py`
- `src/collab_models.py`
- `src/runtime_profiler.py`
- `src/request_builder.py`
- `src/remote_search_runner.py`

### Local worker / RoboDK-bound runtime modules

These modules depend on the live RoboDK station at runtime:

- `src/ik_collection.py`
- `src/path_optimizer.py`
- `src/global_search.py`
- `src/local_repair.py`
- `src/robodk_eval_worker.py`
- `src/robodk_program.py`

## Entry Points

### Preferred local usage

The recommended day-to-day workflow is now:

1. Edit the small top-level run config block in `main.py`
2. Set `RUN_MODE`, `ONLINE_ACTION`, request / run-id / host / env fields as needed
3. Run:

```powershell
python main.py
```

`main.py` is now the unified IDE-friendly entrypoint. It decides whether to:

- run single-machine program generation
- run single-machine visualization
- run online requester/worker roundtrip
- run local worker-only evaluation
- build an online request only
- run online best-candidate final program generation as part of roundtrip
- set up the server requester environment

Detailed subprocess / SSH / conda logs can be written to a log file while the local terminal stays focused on key stage feedback.

### 1. Single-machine mode

Local full solve and final program generation:

```powershell
conda activate winding_pose_solver
python main.py
```

Visualization only:

```powershell
python main.py --visualize
```

### 2. Online worker

Evaluate one request:

```powershell
python online_worker.py eval --request request.json --result result.json
```

Evaluate a batch:

```powershell
python online_worker.py eval-batch --request candidates.json --result results.json
```

### 3. Online requester

Generate candidate batches on `master`:

```bash
python online_requester.py propose --request request.json --candidates candidates.json
```

Summarize evaluated results on `master`:

```bash
python online_requester.py summarize --results results.json --summary summary.json
```

### 4. Online controller

Sync requester code and create the server conda environment:

```powershell
python online_roundtrip.py setup-server --host master --server-dir ~/apps/winding_pose_solver --env winding_pose_solver
```

Build a round-1 request from the current local project settings:

```powershell
python online_roundtrip.py build-request --request artifacts/online_runs/request.json --candidate-limit 4
```

Run one full requester/worker round:

```powershell
python online_roundtrip.py run-round --host master --request artifacts/online_runs/request.json --run-id smoke_round1
```

By default, `run-round` now also re-runs the best candidate locally and attempts final RoboDK program generation.
Use `--skip-final-generate` only when you want search and evaluation without final program output.
If final generation fails, the local terminal will still show the stage name and the paths to `request.json`, `candidates.json`, `results.json`, `summary.json`, and `final_generate_result.json` so you can inspect exactly where it stopped.

If the current interpreter does not contain RoboDK, pass the worker interpreter explicitly:

```powershell
python online_roundtrip.py run-round --host master --request artifacts/online_runs/request.json --run-id smoke_round1 --local-python C:\Users\22290\anaconda3\envs\winding_pose_solver\python.exe
```

All online artifacts are stored under:

- Local: `artifacts/online_runs/<run_id>/`
- Server: `~/apps/winding_pose_solver/artifacts/online_runs/<run_id>/`

## Dependencies

Dependency files are now separated by mode:

- `requirements.shared.txt`
  - `numpy`
  - `pandas`
- `requirements.server.txt`
  - requester-only dependencies
  - intentionally does not include `robodk`
  - intentionally does not include `matplotlib`
- `requirements.local-worker.txt`
  - shared dependencies
  - `matplotlib`
  - `robodk`

Conda environment files:

- `environment.server.yml`
  - `python=3.12`
  - installs `requirements.server.txt`
- `environment.local-worker.yml`
  - `python=3.10`
  - installs `requirements.local-worker.txt`

## RoboDK Station Requirements

The live local station must contain at least:

- robot: `KUKA`
- frame: `Frame 2`

The worker and single-machine modes assume the station is already open in RoboDK and the selected Python interpreter can import `robodk`.

The requester mode on `master` does not require RoboDK and should stay that way.

## Current Constraint Model

The live RoboDK station remains the runtime truth source.

The process frame `A` in `Frame 2` is defined by:

- nominal origin: `TARGET_FRAME_A_ORIGIN_IN_FRAME2_MM = (1126.0, -400.0, 1200.0)`
- fixed calibrated orientation: `TARGET_FRAME_A_ROTATION_IN_FRAME2_XYZ_DEG = (-180.0, -14.0, -180.0)`

The optimizer is allowed to search only the smooth per-point origin offsets of frame `A` in `Frame 2`:

- `x_i = nominal_x`
- `y_i = nominal_y + dy_i`
- `z_i = nominal_z + dz_i`

Important constraints:

- `X` stays fixed.
- Process-frame orientation stays fixed.
- `dy_i / dz_i` are global path variables.
- Local repair may refine the `Frame 2` Y/Z profile but does not replace it with joint interpolation.

## Current Observed Runtime Behavior

As of local regression runs on `2026-04-12`:

- `main.py` still fails explicitly instead of exporting a misleading program.
- The current full-search baseline reports:
  - `ik_empty_rows=57`
  - `config_switches=496`
  - `bridge_like_segments=496`
- The current hard failure is dominated by IK-empty rows:
  - `387, 388, 389, 390, 391, 392, 393, 394, 395, 396, 397, 398`
- `online_worker.py eval` with the same full-search request matches the single-machine baseline.
- A real requester/worker smoke round against `master` also ran through successfully:
  - requester proposed candidates on `master`
  - worker evaluated them locally with RoboDK
  - requester summarized results on `master`
  - the best smoke-round candidate reached `ik_empty_rows=59`

So the architecture is now runnable and mode-separated, but the current live station still does not yield a final fully acceptable path.

## What Changed In This Refactor

- Added explicit shared motion-setting types in `src/motion_settings.py`
- Added independent pose-CSV loading in `src/pose_csv.py`
- Added JSON request/result schemas in `src/collab_models.py`
- Added runtime section profiling in `src/runtime_profiler.py`
- Added request construction in `src/request_builder.py`
- Added local RoboDK evaluator worker in `src/robodk_eval_worker.py`
- Added server requester in `src/remote_search_runner.py`
- Added online entry points:
  - `online_worker.py`
  - `online_requester.py`
  - `online_roundtrip.py`
- Refactored `src/robodk_program.py` so single-machine mode reuses the worker evaluation flow instead of maintaining a separate search pipeline

## Validation Checklist

When checking this repository after changes, verify:

- `python main.py` still runs the local full solve and fails explicitly if continuity cannot be proven
- `python online_requester.py --help` works on `master` without RoboDK installed
- `python online_worker.py --help` works locally
- `python online_roundtrip.py setup-server ...` creates the `winding_pose_solver` environment on `master`
- `python online_roundtrip.py run-round ...` produces:
  - `request.json`
  - `candidates.json`
  - `results.json`
  - `summary.json`
  - `final_generate_request.json`
  - `final_generate_result.json`
- `online_worker.py eval` full-search metrics match `main.py` for the same request
