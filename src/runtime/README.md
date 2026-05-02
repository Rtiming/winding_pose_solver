# `src/runtime/`

This package contains high-level orchestration code:

- local app entry flows
- request construction
- local `auto|mac|windows|linux` machine profile resolution
- requester-side remote search flow
- runtime profiling helpers
- official/debug/diagnostic delivery gate helpers
- Frame-A origin Y/Z search and dispatch
- multi-round local retry and quality polish
- server-side profile candidate proposal for branch policy, active sets,
  corridors, and basis perturbations
- canonical main entry implementation (`main_entrypoint.py`)
- modular online role orchestration (`online/roundtrip.py`)
- external FastAPI service entry (`http_service.py`)
- external reusable Python facade (`external_api.py`)
- async winding orchestration run manager (`winding_runs.py`)

If a change is about:

- how `main.py` builds or runs work
- how online requests are assembled
- how Mac/Windows/Linux local profile selection affects local command paths
- how profiling is reported
- how strict delivery status is summarized
- how origin-search candidates are generated, ranked, and dispatched
- how online retry budgets, active-set focus segments, or post-valid quality
  polish are coordinated

this is usually the right package to edit.

Important boundaries:

- `main_entrypoint.py` should remain the canonical implementation behind the
  thin root `main.py` control panel.
- `local_profile.py` owns local profile detection and Conda Python candidate
  paths. Keep platform differences there; do not branch solver/search behavior
  by OS.
- `online/roundtrip.py` owns SSH sync, Slurm-wrapped server compute, server
  handoff, and receiver orchestration. Root `online_roundtrip.py` remains the
  compatibility wrapper for the roundtrip CLI; requester and worker CLIs are
  module entrypoints in `src/runtime/online/`.
- `local_retry.py` owns multi-round profile retry/repair. It can continue
  polishing a strict-valid baseline when the caller asks for quality polish.
- `remote_search.py` owns server candidate generation for branch-policy,
  active-set window/ramp, corridor, and basis profile updates. Keep profile
  length checks, max-offset clamping, endpoint locking, smoothness filtering,
  and deduplication centralized there.
- `http_service.py` and `external_api.py` are the formal external call surface.
  Keep HTTP in protocol/validation/response shaping only; do not duplicate IK,
  search, continuity, or repair logic there.
- `origin_sweep.py` reuses parsed centerline datasets inside worker processes
  during Y/Z sweeps. Keep the cache key tied to CSV path, mtime, size, frame
  build options, and terminal-append mode so edited input data cannot reuse stale
  frames.

Current online quality defaults:

- `LOCAL_MACHINE_PROFILE = "auto"` in `main.py`; override with
  `WPS_LOCAL_MACHINE_PROFILE` when needed.
- `ONLINE_PROFILE_RETRY_CANDIDATE_LIMIT = 24`
- `ONLINE_PROFILE_RETRY_REPAIR_LIMIT = 2`
- `ONLINE_PROFILE_RETRY_MAX_ROUNDS = 3`
- `WPS_ONLINE_QUALITY_POLISH` defaults to enabled in the server role.
- `WPS_ONLINE_QUALITY_POLISH_TARGET_DEG` defaults to `20.0`.
- `WPS_LOCAL_RETRY_FOCUS_SEGMENT_LIMIT` defaults to `6` when unset or invalid.

