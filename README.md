# winding_pose_solver

`winding_pose_solver` generates winding-tool poses from centerline data,
evaluates IK reachability, searches for a usable robot joint path, and can
materialize the selected path as a RoboDK program.

The project is coordinate-driven: the winding tool/work frame must visit the
target coordinate points in order. Posture quality metrics such as config
switches and worst joint step are diagnostics and optimization targets, not the
root goal unless explicitly promoted to hard constraints for a run.

Input centerline order is a hard part of the task. Row 0 in
`data/validation_centerline.csv` is the user-selected start point; closed-path
handling appends a copy of that first row as the terminal row and must not move
the start to another seam.

## Current Defaults

The top-level `main.py` is now a thin control panel. Implementation is in
`src/runtime/main_entrypoint.py`.

```python
TARGET_FRAME_A_ORIGIN_IN_FRAME2_MM = (1126.0, -247.5, 977.5)
TARGET_FRAME_A_ROTATION_IN_FRAME2_XYZ_DEG = (0.0, 0.0, -180.0)
RUN_MODE = "online"
SINGLE_ACTION = "program"
ONLINE_ROLE = "coordinator"
ONLINE_SERVER_DIR = "/home/tzwang/program/winding_pose_solver"
REMOTE_SYNC_MODE = "push"  # off | guard | push
ENABLE_TARGET_ORIGIN_YZ_SEARCH = True
TARGET_ORIGIN_YZ_SEARCH_SQUARE_SIZE_MM = 400.0
TARGET_ORIGIN_YZ_SEARCH_INITIAL_STEP_MM = 100.0
TARGET_ORIGIN_YZ_SEARCH_MIN_STEP_MM = 5.0
```

`main.py` is intentionally a commented control panel. Edit those top-level
parameters first; deeper defaults and advanced tuning stay in
`src/runtime/main_entrypoint.py` and `app_settings.py`.

When `ENABLE_TARGET_ORIGIN_YZ_SEARCH=True`, a plain `python main.py` starts
with origin search before any post-dispatch flow.

Closed winding rule currently enforced by the search/import path:

- Terminal I1-I5 must equal start I1-I5.
- Terminal I6 must equal start I6 plus or minus one full turn.
- A fake `0` degree A6 terminal closure is rejected.

Official RoboDK delivery also uses the current strict motion-quality gate:

- `invalid_row_count == 0`
- `ik_empty_row_count == 0`
- complete `selected_path`
- closed-winding terminal rule passes when the path is closed
- `bridge_like_segments == 0`
- `big_circle_step_count == 0`
- `worst_joint_step_deg <= 60`

`config_switches` is reported as a diagnostic signal. It is not a standalone
blocker because a config label can change benignly near a singularity when the
joint and TCP motion remain continuous.

## Architecture

```text
main.py                         thin user control panel (compat exports)
app_settings.py                 shared runtime tuning
online_roundtrip.py             compatibility wrapper
scripts/                        thin diagnostics and operation CLIs
src/
  core/                         math, schemas, CSV, pose solving
  search/                       IK candidates, DP path search, repair
  runtime/                      orchestration, logging, origin search
    main_entrypoint.py          canonical main-mode implementation
    online/                     coordinator/server/receiver implementations
  robodk_runtime/               live RoboDK import/program generation
  six_axis_ik/                  embedded local IK/FK backend
artifacts/                      generated runs, logs, temporary outputs
data/                           input CSVs and small generated pose CSVs
```

Keep reusable logic out of `main.py`. When adding behavior, put it in the
owning package and keep `main.py` as a small switchboard.

## Main Workflows

### Online Coordinator

Windows builds the request, sends compute to `master`, downloads the handoff,
and optionally generates the RoboDK program locally.

If `ENABLE_TARGET_ORIGIN_YZ_SEARCH=True` in `main.py`, turn it off before a
direct online smoke. Otherwise the entrypoint intentionally starts with origin
search.

```powershell
python main.py --mode online --online-role coordinator --run-id smoke --skip-final-generate
```

Useful defaults in `main.py`:

```python
ONLINE_HOST = "master"
ONLINE_SERVER_DIR = "/home/tzwang/program/winding_pose_solver"
ONLINE_ENV_NAME = "winding_pose_solver"
ONLINE_FINAL_GENERATE_PROGRAM = True
REMOTE_SYNC_MODE = "push"  # off | guard | push
ENFORCE_REMOTE_SYNC_GUARD = True
```

Retry budgets can also be overridden by env vars:
`WPS_ONLINE_RETRY_CANDIDATE_LIMIT`, `WPS_ONLINE_RETRY_REPAIR_LIMIT`,
`WPS_ONLINE_RETRY_MAX_ROUNDS`.

Direct coordinator CLI also supports:

```powershell
python online_roundtrip.py run-online --remote-sync-mode push --request artifacts/online_runs/main_request.json --run-id smoke_sync --skip-final-generate
```

### Server Role Only

Run pure server compute from an existing request:

```powershell
python online_roundtrip.py run-server --request artifacts/online_runs/main_request.json --run-id server_smoke --allow-invalid-outputs
```

On `master`, heavy server compute should run through Slurm. The online
coordinator and origin-search runner already wrap remote compute with `srun`
when Slurm is available.

### Receiver Role Only

Use a server handoff to import the selected path into the local RoboDK station:

```powershell
python online_roundtrip.py run-receiver --handoff artifacts/online_runs/<run_id>/handoff_package.json --run-id <run_id>_receiver
```

The receiver should import the selected joint path directly; it should not run
a fresh full search just to generate the RoboDK program.

### Single-Machine Solve

Use this for local debugging or when RoboDK is not needed:

```powershell
python main.py --mode single --single-action solve --run-id local_smoke
```

Use program mode only when the correct RoboDK station is open:

```powershell
python main.py --mode single --single-action program --run-id local_program
```

## Smart Frame-A Origin Search

The project can search a Y/Z square around
`TARGET_FRAME_A_ORIGIN_IN_FRAME2_MM` to find better global Frame-A origin
basins.

From `main.py`:

```python
ENABLE_TARGET_ORIGIN_YZ_SEARCH = True
TARGET_ORIGIN_YZ_SEARCH_USE_SERVER = True
TARGET_ORIGIN_YZ_SEARCH_SQUARE_SIZE_MM = 600.0
TARGET_ORIGIN_YZ_SEARCH_INITIAL_STEP_MM = 150.0
TARGET_ORIGIN_YZ_SEARCH_MIN_STEP_MM = 10.0
TARGET_ORIGIN_YZ_SEARCH_MAX_ITERS = 5
TARGET_ORIGIN_YZ_SEARCH_BEAM_WIDTH = 4
```

The top-level switch above controls whether the normal `python main.py` entry
starts with origin search. More specialized dispatch controls, usable-count
limits, diagonal policy, and validation-grid settings live in
`src/runtime/main_entrypoint.py` and can also be overridden by CLI flags.

Or from the CLI:

```powershell
python main.py --mode origin_search --run-id origin_search_300
```

Dispatch the top usable candidates directly after origin search:

```powershell
python main.py --mode origin_search --origin-search-dispatch online_role --origin-search-post-top-n 2 --run-id origin_search_300_online
```

The CLI wrapper is:

```powershell
python scripts/sweep_target_origin_yz.py --mode smart-square --seed-origin 1126,-400,1130 --square-size-mm 300 --smart-initial-step-mm 75 --smart-min-step-mm 25 --smart-max-iters 4 --beam-width 4 --workers 8 --skip-window-repair --skip-inserted-repair
```

The `smart-square` algorithm is not an exhaustive final grid search:

1. Evaluate center, edge midpoints, and corners of the square.
2. Keep a diverse beam of promising points.
3. Evaluate 3x3 neighborhoods around those points.
4. Shrink step size and repeat.
5. Optionally run a coarse validation grid with
   `--validation-grid-step-mm`.

Reusable implementation:

- `src/runtime/origin_sweep.py`: search algorithms and scoring.
- `src/runtime/origin_search_runner.py`: `main.py` local/remote launcher.
- `scripts/sweep_target_origin_yz.py`: thin CLI.

Performance note: origin sweeps reuse the parsed centerline dataset inside each
worker process, and `smart-square` case results are cached across repeated runs
under `artifacts/tmp/origin_sweep/origin_case_eval_cache.json`. The case-result
cache key includes the relevant source files, centerline CSV content hash,
Frame-A rotation, backend, terminal-append mode, parallel settings, and repair
strategy. Set `WPS_ORIGIN_SWEEP_CACHE=0` to force fresh evaluation.

Typical output:

- `artifacts/online_runs/<run_id>/origin_yz_search_results.json`
- `artifacts/online_runs/<run_id>/origin_search_selection.json`
- ranked candidate origins, baseline metrics, and trace metadata

## IK Backends

The thin `main.py` currently exposes only the parameters you most often adjust.
Backend defaults are kept in `src/runtime/main_entrypoint.py` and
`app_settings.py`:

```python
IK_BACKEND = "six_axis_ik"  # or "robodk"
```

`six_axis_ik`:

- Embedded local analytic/numeric backend.
- Works on the Linux server without RoboDK.
- Preferred for online compute and sweeps.

`robodk`:

- Uses RoboDK as the live truth source.
- Requires an open RoboDK station on Windows.
- Keep RoboDK-specific logic in `src/robodk_runtime/`.

## Important Modules

`src/core/`

- `pose_solver.py`: centerline to tool pose generation.
- `frame_math.py`, `geometry.py`: reusable math.
- `collab_models.py`: request/result schemas.
- `motion_settings.py`: shared motion settings dataclass.

`src/search/`

- `ik_collection.py`: IK candidate collection.
- `path_optimizer.py`: DP path selection, continuity costs, closed terminal
  full-turn rule.
- `local_repair.py`: local Y/Z and inserted-transition repair.
- `global_search.py`: high-level exact profile evaluation.

`src/runtime/`

- `app.py`: high-level local app flow.
- `request_builder.py`: request construction.
- `remote_search.py`: candidate proposal for online runs.
- `local_retry.py`: profile retry/repair orchestration.
- `origin_sweep.py`: grid, adaptive, candidate, and smart-square origin search.
- `origin_search_runner.py`: `main.py` origin-search launcher.
- `run_logging.py`: run logs and console tee.
- `delivery.py`: target-reachability delivery gate helpers.

`src/robodk_runtime/`

- `eval_worker.py`: evaluation worker context.
- `result_import.py`: direct handoff import into RoboDK.
- `program.py`: RoboDK target/program materialization.

## Artifacts

Common run outputs:

```text
artifacts/local_runs/<run_id>/request.json
artifacts/local_runs/<run_id>/eval_result.json
artifacts/local_runs/<run_id>/selected_joint_path.csv
artifacts/local_runs/<run_id>/run_archive.json
artifacts/online_runs/<run_id>/request.json
artifacts/online_runs/<run_id>/results.json
artifacts/online_runs/<run_id>/handoff_package.json
artifacts/online_runs/<run_id>/origin_yz_search_results.json
artifacts/run_logs/
artifacts/tmp/
```

Generated artifacts are diagnostic data. Do not commit large run outputs unless
there is a specific reason.

## Remote Sync

Canonical server checkout:

```text
master:/home/tzwang/program/winding_pose_solver
```

The old path below is retired:

```text
/home/tzwang/apps/winding_pose_solver
```

Preferred compare/sync wrapper:

```powershell
& "$env:USERPROFILE\.codex\skills\winding-master-sync\scripts\Sync-WindingMaster.ps1" -Mode Compare
& "$env:USERPROFILE\.codex\skills\winding-master-sync\scripts\Sync-WindingMaster.ps1" -Mode Push
```

Built-in online coordinator preflight modes:

- `off`: skip preflight sync checks.
- `guard`: hash-check key files and fail-fast if local/remote differ.
- `push`: upload local source tree first, then run the hash guard.

`push` cadence details:

- Upload source is now a single runtime bundle (`src/`, `scripts/`, and root
  entry files) to reduce SSH/SCP round trips.
- Coordinator writes/reads a remote bundle hash marker at
  `artifacts/tmp/runtime_bundle.sha256`.
- If local and remote bundle hashes match, `push` skips upload automatically.
- Force full upload with `WPS_FORCE_BUNDLE_SYNC=1`.

Network retry pacing env vars:

- `WPS_LOCAL_CMD_RETRY_ATTEMPTS` (default `3`)
- `WPS_LOCAL_CMD_RETRY_DELAY_SECONDS` (default `1.0`)
- `WPS_REMOTE_CMD_RETRY_ATTEMPTS` (default `3`)
- `WPS_REMOTE_CMD_RETRY_DELAY_SECONDS` (default `1.0`)

Online repair pacing guardrail:

- `WPS_SERVER_PROFILE_MIN_BATCH_SIZE` is clamped by default to
  `WPS_SERVER_PROFILE_MIN_BATCH_SIZE_CAP` (default `4`) to avoid severe
  performance regressions from oversized batch thresholds.
- Allow higher values only for benchmarking:
  `WPS_SERVER_PROFILE_ALLOW_HIGH_MIN_BATCH=1`.

When syncing manually, back up remote files before overwriting them and avoid
deleting local-only or remote-only work unless explicitly requested.

## Validation

Lightweight checks:

```powershell
python -m py_compile main.py online_roundtrip.py src/runtime/origin_sweep.py src/runtime/origin_search_runner.py scripts/sweep_target_origin_yz.py
python main.py --help
python scripts/sweep_target_origin_yz.py --help
```

Online smoke without RoboDK finalization:

```powershell
python main.py --mode online --online-role coordinator --run-id smoke --skip-final-generate
```

Origin-search smoke:

```powershell
python scripts/sweep_target_origin_yz.py --mode smart-square --seed-origin 1126,-400,1130 --square-size-mm 300 --smart-max-iters 1 --workers 1 --skip-window-repair --skip-inserted-repair
```

Remote compile:

```powershell
ssh master "cd /home/tzwang/program/winding_pose_solver && conda run -n winding_pose_solver python -m py_compile main.py src/runtime/origin_sweep.py src/runtime/origin_search_runner.py scripts/sweep_target_origin_yz.py"
```

For heavy server sweeps or repair experiments, use Slurm on `master`.

## Current Optimization Notes

Recent 300x300 smart-square search around `(1126, -400, 1130)` found better
global seeds, but did not eliminate all bridge/config warnings by origin alone.
The next practical optimization layer is focused local repair around the
remaining large wrist/config transitions, using the smart-square candidates as
seed origins.
