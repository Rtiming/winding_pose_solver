# AGENTS.md

Repo-level guidance for coding agents working in `winding_pose_solver`.

If a deeper directory has its own `AGENTS.md`, follow that file for code in
that subtree and use this root file as the fallback policy.

## Agent Operating Contract

This project is usually edited through short Chinese instructions from the
user. Agents should infer low-risk details and act, but must keep the following
project-specific contract:

- Start by reading this file and running `git status --short --branch` before
  broad edits, sync, or online runs.
- Treat the Windows checkout as the active editing workspace unless the user
  explicitly says the Linux server is source of truth.
- Never overwrite, revert, delete, or clean local/remote changes that the agent
  did not create. Work with dirty files and keep edits scoped.
- Do not use the retired server path `/home/tzwang/apps/winding_pose_solver`.
  The canonical server path is `/home/tzwang/program/winding_pose_solver`.
- Do not turn `main.py` back into a large implementation file. It is a thin
  user control panel; reusable logic belongs in `src/`.
- Do not put substantial implementation in root `online_*.py` files. They are
  compatibility wrappers; online implementation belongs in
  `src/runtime/online/`.
- Prefer concrete implementation plus validation over long proposals. Ask the
  user only when a wrong assumption could cause data loss, expensive compute,
  or architectural drift.
- When consulting external assistants or tools, treat them as advisory only and
  do not claim Kimi, Claude, web research, or RoboDK behavior was checked unless
  it was actually checked in this run.
- Keep user-facing responses concise and Chinese-first unless the user asks
  otherwise. Report commands run, outputs/artifacts, and any validation that
  could not be performed.

## Project Intent

This project generates winding-tool poses from centerline data, evaluates IK
reachability, searches for a usable joint path, and optionally materializes the
selected path as a RoboDK program.

The root objective is coordinate-driven:

- The tool/work frame must visit the requested target coordinate points in
  order.
- The imported centerline CSV row order is part of the task definition. Row 0
  is the user-selected start point; do not rotate, reverse, sort, or choose a
  different closed-curve seam unless the user explicitly asks for that.
- `TARGET_FRAME_A_ORIGIN_IN_FRAME2_MM` is the nominal fixed Frame-A origin.
- Per-row Frame-A Y/Z offsets and origin sweeps are optimization freedoms used
  to reduce physical detours and joint discontinuities.
- `config_switches`, `bridge_like_segments`, and `worst_joint_step_deg` are
  diagnostics and optimization signals unless the user explicitly promotes one
  to a hard constraint.

Current user-promoted hard rule for closed winding paths:

- Closed-path terminal rows are produced by copying the imported row 0 to the
  end; they must not replace or move the actual start row.
- When the start point is appended as the terminal point, terminal I1-I5 must
  equal the start I1-I5.
- Terminal I6 must differ from the start I6 by exactly one full turn
  (`+360` or `-360` degrees), with the sign chosen from the observed early A6
  trend when that trend is clear.
- Do not satisfy this terminal rule by accepting a fake `0` degree A6 closure.

Known motion-quality failure mode:

- A path can be target-reachable but still practically bad if a segment such as
  `59->60`, `372->373`, or another wrist/config transition causes a fast
  configuration jump or a physical detour.
- Prefer local Y/Z repair, candidate scoring, A6 periodic handling, or
  insertion strategy to reduce those transitions. Do not hide the problem by
  deleting diagnostics.

## Machines And Paths

- Windows checkout: current local workspace.
- Canonical Linux server checkout:
  `/home/tzwang/program/winding_pose_solver`
- The retired server path `/home/tzwang/apps/winding_pose_solver` must not be
  used.
- Windows coordinates online orchestration and RoboDK finalization.
- The server runs pure compute with the `six_axis_ik` backend; it must not need
  RoboDK.
- Live RoboDK behavior stays on Windows and should attach to the already-open
  station.

## Repository Map

- `main.py`: user-facing run configuration and mode selection. Keep it thin.
- `src/runtime/main_entrypoint.py`: canonical main-mode implementation.
- `app_settings.py`: shared solver/runtime tuning defaults.
- `src/runtime/online/roundtrip.py`: SSH transport, server setup, server role,
  receiver role, and coordinator flow.
- `online_roundtrip.py`, `online_requester.py`, `online_worker.py`:
  root-level compatibility wrappers only.
- `scripts/`: thin diagnostics and operational wrappers.
- `src/core/`: math, geometry, schemas, CSV handling, pose solving, and
  backend-neutral helpers.
- `src/search/`: IK candidate collection, DP path selection, continuity costs,
  local repair, and inserted-transition logic.
- `src/runtime/`: orchestration, request building, run logging, origin search,
  local retry, and remote-search helpers.
- `src/runtime/online/`: modular online coordinator/server/receiver entrypoints
  and sync preflight logic.
- `src/robodk_runtime/`: live RoboDK station import/program generation and
  RoboDK-backed evaluation.
- `src/six_axis_ik/`: embedded IK/FK model and backend-specific kinematics.

Recent modular split:

- Smart Frame-A origin search lives in `src/runtime/origin_sweep.py`.
- The `main.py --mode origin_search` launcher lives in
  `src/runtime/origin_search_runner.py`.
- `scripts/sweep_target_origin_yz.py` is a CLI wrapper around the runtime
  origin-search code.

## Edit Surface

Before editing, decide which layer owns the change:

- Pure geometry, CSV, schemas, or backend-neutral pose logic: `src/core/`
- DP costs, config switching, A6 terminal constraints, IK candidates, repair:
  `src/search/`
- Request/result orchestration, origin sweep/search, logging, retry:
  `src/runtime/`
- RoboDK import, target/program creation, live station behavior:
  `src/robodk_runtime/`
- Analytic/numeric IK model details: `src/six_axis_ik/`
- CLI-only diagnostics: `scripts/`

Keep `main.py` as a small control panel. Move reusable logic into the
appropriate package.

## Documentation Editing Rules

Documentation edits must constrain future agents instead of drifting into
general advice:

- Keep docs project-specific. Mention only paths, commands, modes, and
  constraints that are valid for this repository.
- Preserve the current architecture: thin `main.py`, compatibility root
  wrappers, implementation under `src/core`, `src/search`, `src/runtime`,
  `src/runtime/online`, `src/robodk_runtime`, and `src/six_axis_ik`.
- Do not rewrite large documents from scratch unless the user explicitly asks.
  Prefer focused sections that update the stale rule or command.
- Do not invent validation results, benchmark numbers, RoboDK output, Slurm
  availability, or server sync status. State "not run" when validation was not
  run.
- Keep command examples copy-pasteable for PowerShell on Windows unless the
  section is explicitly about the Linux server.
- Any server command in docs must use
  `/home/tzwang/program/winding_pose_solver`; never document the retired path.
- When documenting online behavior, include the sync preflight mode
  (`REMOTE_SYNC_MODE` / `--remote-sync-mode`) and whether RoboDK finalization is
  skipped or expected.
- When documenting motion quality, preserve both sides of the rule:
  coordinate order and closed-winding terminal constraints are hard; config
  switches and continuity metrics are diagnostics unless explicitly promoted.
- Add comments in code only around non-obvious algorithms, hard constraints,
  sync boundaries, or RoboDK/server separation. Do not add noisy line-by-line
  comments.
- Keep generated artifacts, logs, cache directories, and one-off run outputs out
  of documentation examples unless they are part of the stable artifact
  contract below.

## Operational Defaults

- Default `TARGET_FRAME_A_ORIGIN_IN_FRAME2_MM` is configured in `main.py`.
- Default backend is expected to remain `six_axis_ik` for server-safe compute.
- Default online final program generation remains local on Windows, but
  online SixAxisIK evaluation and retry/repair run in the server role.
- Online coordinator preflight sync supports `off|guard|push` mode.
- Default online retry/repair should stay conservative and server-side;
  tune `ONLINE_PROFILE_RETRY_*` or `WPS_ONLINE_RETRY_*` rather than moving
  continuity repair back into the Windows receiver.
- Smart origin search is available through `main.py --mode origin_search` or
  `python scripts/sweep_target_origin_yz.py --mode smart-square ...`.
- Origin sweep case-result caching is enabled by default and stored under
  `artifacts/tmp/origin_sweep/origin_case_eval_cache.json`. Disable it with
  `WPS_ORIGIN_SWEEP_CACHE=0` when validating fresh recomputation behavior.

Online cadence and sync rhythm defaults:

- `REMOTE_SYNC_MODE=push` now uploads a single runtime bundle and records a
  bundle hash marker at `artifacts/tmp/runtime_bundle.sha256` on the server.
- If the local runtime bundle hash equals the remote marker hash, `push` skips
  upload and proceeds to sync guard directly.
- Force upload even when hashes match only when explicitly needed:
  `WPS_FORCE_BUNDLE_SYNC=1`.
- Network retry pacing can be tuned (without editing code):
  `WPS_LOCAL_CMD_RETRY_ATTEMPTS`, `WPS_LOCAL_CMD_RETRY_DELAY_SECONDS`,
  `WPS_REMOTE_CMD_RETRY_ATTEMPTS`, `WPS_REMOTE_CMD_RETRY_DELAY_SECONDS`.
- Keep `WPS_SERVER_PROFILE_MIN_BATCH_SIZE` conservative for online retry/repair.
  Large values can cause major slowdowns; raise it only for controlled
  experiments and only with `WPS_SERVER_PROFILE_ALLOW_HIGH_MIN_BATCH=1`.

## Delivery Semantics

Official delivery currently requires target reachability plus the user-promoted
motion-quality gate. Debug/diagnostic artifacts may still be produced for
analysis, but they must not be treated as ready-to-run production RoboDK
programs.

Official delivery requires:

- `invalid_row_count == 0`
- `ik_empty_row_count == 0`
- a complete non-empty `selected_path`
- for closed winding payloads, terminal I1-I5/I6 hard constraints must pass
- `bridge_like_segments == 0`
- `big_circle_step_count == 0`
- `worst_joint_step_deg <= 60`

Continuity warnings should be reported clearly:

- `config_switches`
- `bridge_like_segments`
- `big_circle_step_count`
- `worst_joint_step_deg`
- focused problem segments

`config_switches` remains a diagnostic signal rather than a standalone official
delivery blocker. A config label change can be benign near singularity if the
joint path and TCP motion stay continuous.

## Remote And Slurm Rules

When actively running work on `master` under `/home/tzwang`, follow the
`slurm-run` skill:

- Lightweight inspection is fine on the login node.
- Heavy sweeps, repeated evaluations, benchmarking, large repair runs, or
  broad tests must use Slurm.
- Inspect live Slurm resources before launching heavy work.
- Use `srun` for interactive compute and `sbatch` for long detached work.

For source synchronization with master, use the `winding-master-sync` skill or
careful targeted `scp` after backing up remote files. Do not delete local-only
or remote-only files unless explicitly asked.

## Dirty Worktree Policy

Assume the worktree can contain user changes.

- Always check `git status --short --branch` before broad edits or sync.
- Never revert changes you did not make unless the user explicitly asks.
- If touching a dirty file, read the relevant section and work with existing
  edits.
- Keep edits scoped to the requested task.

## Validation

Use the smallest validation that proves the change:

```powershell
python -m py_compile main.py online_roundtrip.py src/runtime/origin_sweep.py scripts/sweep_target_origin_yz.py
python main.py --help
python scripts/sweep_target_origin_yz.py --help
```

For online compute smoke:

```powershell
python main.py --mode online --online-role coordinator --run-id smoke --skip-final-generate
```

For origin search smoke:

```powershell
python scripts/sweep_target_origin_yz.py --mode smart-square --seed-origin 1126,-400,1130 --square-size-mm 300 --smart-max-iters 1 --workers 1 --skip-window-repair --skip-inserted-repair
```

For receiver/RoboDK finalization, run only when the correct station is open:

```powershell
python online_roundtrip.py run-receiver --handoff artifacts/online_runs/<run_id>/handoff_package.json --run-id <run_id>_receiver
```

## Artifacts

Preserve these conventions unless explicitly redesigning them:

- `artifacts/local_runs/<run_id>/request.json`
- `artifacts/local_runs/<run_id>/eval_result.json`
- `artifacts/local_runs/<run_id>/selected_joint_path.csv`
- `artifacts/local_runs/<run_id>/run_archive.json`
- `artifacts/online_runs/<run_id>/request.json`
- `artifacts/online_runs/<run_id>/results.json`
- `artifacts/online_runs/<run_id>/handoff_package.json`
- `artifacts/online_runs/<run_id>/origin_yz_search_results.json`
- `artifacts/run_logs/`
- `artifacts/tmp/`

## Communication And Iteration

The user often gives compressed Chinese-first instructions. Infer low-risk
details, keep visible restatement short, and proceed. Prefer autonomous
multi-pass refinement for non-trivial work:

1. Understand the relevant layer.
2. Implement a focused first pass.
3. Self-review the diff.
4. Run the smallest useful validation.
5. Do one cheap cleanup/refinement pass if it materially improves correctness
   or readability.

Ask only when requirements are materially ambiguous or a wrong assumption would
cause high-cost validation, data loss, or architectural drift.
