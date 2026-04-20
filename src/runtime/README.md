# `src/runtime/`

This package contains high-level orchestration code:

- local app entry flows
- request construction
- requester-side remote search flow
- runtime profiling helpers
- official/debug/diagnostic delivery gate helpers
- Frame-A origin Y/Z search and dispatch
- canonical main entry implementation (`main_entrypoint.py`)
- modular online role orchestration (`online/roundtrip.py`)

If a change is about:

- how `main.py` builds or runs work
- how online requests are assembled
- how profiling is reported
- how strict delivery status is summarized
- how origin-search candidates are generated, ranked, and dispatched

this is usually the right package to edit.

Important boundaries:

- `main_entrypoint.py` should remain the canonical implementation behind the
  thin root `main.py` control panel.
- `online/roundtrip.py` owns SSH sync, Slurm-wrapped server compute, server
  handoff, and receiver orchestration. Root `online_roundtrip.py`,
  `online_requester.py`, and `online_worker.py` should stay compatibility
  wrappers.
- `origin_sweep.py` reuses parsed centerline datasets inside worker processes
  during Y/Z sweeps. Keep the cache key tied to CSV path, mtime, size, frame
  build options, and terminal-append mode so edited input data cannot reuse stale
  frames.

