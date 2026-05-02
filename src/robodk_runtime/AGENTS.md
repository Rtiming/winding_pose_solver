# AGENTS.md

This file applies to `src/robodk_runtime/`.

## Own This Layer For

- code that requires an open RoboDK station
- live robot, frame, and tool lookup
- worker-side live evaluation against RoboDK
- final program generation and path materialization inside RoboDK
- importing selected server handoff paths into the active station

## Good Edit Targets

- `eval_worker.py`: worker-side live evaluation entry and RoboDK-backed execution path
- `program.py`: final program creation and station-bound materialization

## Keep Out Of This Layer

- shared math and reusable schemas
- backend-agnostic abstractions
- top-level requester orchestration
- generic path-search policy

Move those concerns into `src/core/`, `src/runtime/`, or `src/search/` instead.

## Editing Guidance

- Keep RoboDK imports and station assumptions contained here when practical.
- Do not make offline `six_axis_ik` evaluation depend on this package.
- The final program-generation step should remain local by default unless the task explicitly changes the architecture.
- When changing worker-side payload handling, check alignment with the shared schema layer and requester-side expectations.
- Receiver/finalization code should trust the selected path from the handoff.
  Do not add hidden full-search, retry, or active-set polish here; that belongs
  in `src/runtime/` and `src/search/`.
- RoboDK validation must be reported as not run when the station is unavailable
  or activation is broken.

## Validation

- Prefer the smallest live-station validation that covers the change.
- If RoboDK is unavailable, say so clearly instead of guessing.
