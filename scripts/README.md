# `scripts/`

Small diagnostics and one-off utilities live here.

Current examples:

- compare RoboDK and `six_axis_ik` solvability
- run the compatibility HTTP entry `model_demo_solver_api.py` (thin wrapper over `src/runtime/http_service.py`)
- inspect a focused window of rows
- run smart-square Frame-A Y/Z origin search (`sweep_target_origin_yz.py`)
- import a diagnostic window into RoboDK
- import an already selected profile result into RoboDK
- export RoboDK station assets for external viewers with material, visual,
  collision, and viewer metadata (`export_robodk_station_assets.py`)
- validate exported assets, trajectory continuity, and FK-vs-reference quality
  (`check_model_demo_quality.py`)
- launch an unattended `overnight_codex.sh` run that keeps working after you disconnect

Prefer keeping scripts thin: they should usually call into `src/core/`, `src/search/`, or `src/robodk_runtime/` rather than duplicating project logic.
