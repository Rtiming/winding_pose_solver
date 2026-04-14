# `scripts/`

Small diagnostics and one-off utilities live here.

Current examples:

- compare RoboDK and `six_axis_ik` solvability
- inspect a focused window of rows
- import a diagnostic window into RoboDK

Prefer keeping scripts thin: they should usually call into `src/core/`, `src/search/`, or `src/robodk_runtime/` rather than duplicating project logic.
