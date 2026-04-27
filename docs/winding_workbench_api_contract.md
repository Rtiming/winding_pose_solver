# Winding Workbench Orchestration API (MVP)

This document defines the stable orchestration-facing API that `winding_workbench`
can call without depending on solver internal modules.

The existing low-level solver endpoints stay unchanged:

- `POST /api/configure`
- `POST /api/fk`
- `POST /api/ik`
- `POST /api/path/solve`
- `POST /api/path/solve-batch`
- `POST /api/collision/check`

## Unified Envelope (New Orchestration Endpoints)

All `/api/winding/*` endpoints use:

```json
{
  "ok": true,
  "code": "winding_run_created",
  "message": "Winding run accepted.",
  "data": {},
  "request_id": "string",
  "ts": "2026-04-22T00:00:00Z"
}
```

Error envelope:

```json
{
  "ok": false,
  "code": "run_not_found",
  "message": "Run id was not found.",
  "data": {
    "error_code": "run_not_found",
    "detail": {"run_id": "abc"},
    "suggestion": "Check run_id or create a new run first."
  },
  "request_id": "string",
  "ts": "2026-04-22T00:00:00Z"
}
```

## Endpoints

### 1) `GET /api/winding/capabilities`

Returns orchestration capability metadata:

- run modes
- online roles
- artifact conventions
- lifecycle states

### 2) `POST /api/winding/runs`

Creates an asynchronous run.

Request body (MVP):

```json
{
  "run_id": "optional",
  "run_mode": "online",
  "online_role": "coordinator",
  "single_action": "program",
  "selection": {
    "base_face_id": 10,
    "start_point": {"x": 0, "y": 0, "z": 0},
    "end_point": {"x": 10, "y": 0, "z": 0},
    "direction_hint": "forward"
  },
  "centerline_source": {
    "file_path": "C:/path/to/centerline.csv"
  },
  "process_params": {
    "target_frame_origin_mm": [1126.0, -647.5, 1477.5],
    "target_frame_rotation_xyz_deg": [0.0, 0.0, -180.0],
    "program_name": "Path_From_CSV"
  },
  "request_source": {
    "request_path": "C:/path/to/request.json",
    "handoff_path": "C:/path/to/handoff_package.json"
  },
  "options": {
    "dry_run": false,
    "timeout_sec": 3600,
    "allow_invalid_outputs": false,
    "skip_final_generate": false,
    "remote_sync_mode": "guard",
    "host": "master",
    "server_dir": "/home/tzwang/program/winding_pose_solver",
    "env_name": "winding_pose_solver",
    "retry_candidate_limit": 4,
    "retry_repair_limit": 2,
    "retry_max_rounds": 1
  }
}
```

Notes:

- `run_mode=online` is the primary integration mode.
- `centerline_source` supports:
  - `file_path`
  - `inline_points` (rows with `x/y/z`; optional `tx/ty/tz` and `nx/ny/nz`)
- `selection` is currently passed as request metadata for downstream traceability.

### 3) `GET /api/winding/runs`

List recent in-memory runs (`limit` query parameter, default 20).

### 4) `GET /api/winding/runs/{run_id}`

Returns run status:

- `queued | running | succeeded | failed | canceled`
- progress/stage
- pid/exit_code
- artifact paths

Historical run directories under `artifacts/online_runs/<run_id>` are also
recognized even if they were not created by the API.

### 5) `POST /api/winding/runs/{run_id}/cancel`

Requests cancellation for an active run.

### 6) `GET /api/winding/runs/{run_id}/summary`

Loads `summary.json`.

### 7) `GET /api/winding/runs/{run_id}/results`

Loads `results.json` (or mode-equivalent results path).

### 8) `GET /api/winding/runs/{run_id}/handoff`

Loads `handoff_package.json`.

### 9) `GET /api/winding/latest`

Returns latest available summary from `artifacts/online_runs/*/summary.json`.

## Artifact Conventions

Default online run artifact root:

- `artifacts/online_runs/<run_id>/`

Key files:

- `request.json`
- `results.json`
- `summary.json`
- `handoff_package.json`

API runtime state/log files:

- `.runtime/winding_runs/<run_id>.json`
- `.runtime/winding_runs/<run_id>.out.log`
- `.runtime/winding_runs/<run_id>.err.log`

## Backward Compatibility

This orchestration layer does not remove or change existing solver endpoints.
Low-level contracts remain unchanged for existing callers.
