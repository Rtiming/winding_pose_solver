# `scripts/`

Small diagnostics and one-off utilities live here.

Current examples:

- compare RoboDK and `six_axis_ik` solvability
- inspect a focused window of rows
- run smart-square Frame-A Y/Z origin search (`sweep_target_origin_yz.py`)
- import a diagnostic window into RoboDK
- import an already selected profile result into RoboDK
- launch an unattended `overnight_codex.sh` run that keeps working after you disconnect

Prefer keeping scripts thin: they should usually call into `src/core/`, `src/search/`, or `src/robodk_runtime/` rather than duplicating project logic.
