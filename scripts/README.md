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

Prefer keeping scripts thin: they should usually call into `src/core/`,
`src/search/`, `src/runtime/`, or `src/robodk_runtime/` rather than duplicating
project logic.

Stable smoke examples:

```powershell
python scripts/sweep_target_origin_yz.py --help
python scripts/sweep_target_origin_yz.py --mode smart-square --seed-origin 1126,-400,1130 --square-size-mm 300 --smart-max-iters 1 --workers 1 --skip-window-repair --skip-inserted-repair
```

Do not put online retry, active-set profile generation, or RoboDK finalization
logic directly into scripts. Add or update the owning `src/` module first, then
keep the script as a CLI wrapper.
