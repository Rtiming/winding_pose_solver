# winding_pose_solver

Pose solving, path selection, and RoboDK program generation for the winding path in the live RoboDK station.

## Current constraint model

The live RoboDK station is the runtime source of truth.

The process frame `A` in `Frame 2` is defined by:

- Nominal origin: `TARGET_FRAME_A_ORIGIN_IN_FRAME2_MM = (1126.0, -400.0, 1200.0)`
- Fixed calibrated orientation: `TARGET_FRAME_A_ROTATION_IN_FRAME2_XYZ_DEG = (-180.0, -14.0, -180.0)`

The optimizer is allowed to change only the origin of process frame `A` in `Frame 2` as a smooth per-point profile:

- `x_i = nominal_x`
- `y_i = nominal_y + Δy_i`
- `z_i = nominal_z + Δz_i`

Important:

- `X` is fixed.
- The process-frame orientation definition is fixed and is not searched at runtime.
- `Δy_i` and `Δz_i` are global path variables. They may differ at every point, but they must remain smooth.
- Tool-local Y/Z compensation is not used.
- Local repair is allowed only as a refinement of that global `Frame 2` Y/Z profile.

## What the solver does now

### 1. Build nominal poses

`src/pose_solver.py` generates the nominal tool poses from the centerline data and the fixed calibrated frame-`A` rotation.

### 2. Search valid Frame-2 Y/Z profiles

`src/global_search.py` and `src/local_repair.py` search only over the allowed per-point `Frame 2` Y/Z origin offsets of frame `A`.

The current flow is:

1. Evaluate the fixed-orientation zero-offset baseline.
2. Run a global coarse search over uniform `Frame 2` Y/Z offsets.
3. Run local window refinement on the per-point `Δy_i / Δz_i` profile around bad segments.
4. Re-run full-path IK collection and dynamic-programming path selection after each accepted profile update.

Path comparison is ordered by:

1. fewer invalid / IK-empty rows
2. fewer config switches
3. fewer bridge-like segments
4. smaller worst joint step
5. smaller mean joint step
6. smoother offset profile
7. smaller offset magnitude
8. lower scalar DP cost

### 3. Insert transition samples only when they are mathematically valid

The old active fallback of joint interpolation plus `MoveJ` bridge points has been removed from the program-generation path.

If refinement still leaves a bad segment, the code may insert transition samples only by:

1. interpolating the neighboring process-frame/tool-pose reference rows
2. interpolating the solved `Δy / Δz` profile across that interval
3. rebuilding IK layers and re-running full-path selection

This means waypoint insertion is now:

- valid inserted process-frame samples plus valid interpolated `Δy / Δz`
- never a denser sampling of an already wrong joint-space jump

### 4. Fail explicitly if continuity still cannot be proven

If the final path still contains a config switch or an over-threshold joint jump, the code raises an explicit error and does not export a misleading RoboDK program.

The failure report includes:

- failing segment labels
- `Δy / Δz` values around the failure
- candidate-family counts
- worst joint deltas
- focused diagnostics for `372->373` and `380->381`

## Current live-station behavior

With the current live station:

- The fixed calibrated orientation `(-180.0, -14.0, -180.0)` gives IK candidates at row `0` with zero offsets.
- Runtime orientation search is gone.
- Local orientation redistribution is gone.
- Joint-interpolation bridge export is gone.
- The solver now fails explicitly instead of exporting a path that only appears to run.

Current known limitation:

- On the present station and with the present constraints/search envelope, the code still finds one unresolved configuration switch in the same bad zone, currently reported at `381->382`.
- `372->373` is explicitly reported and remains smooth in the current runs.
- Because the remaining segment is not strictly acceptable, no final RoboDK program is exported.

This explicit fail behavior is intentional and matches the project requirement to reject paths that violate the mathematical logic.

## Key files

- [main.py](main.py): project entry point and main business parameters
- [app_settings.py](app_settings.py): advanced tuning settings
- [src/pose_solver.py](src/pose_solver.py): nominal pose generation from the centerline
- [src/global_search.py](src/global_search.py): fixed-orientation Frame-2 Y/Z profile evaluation and coarse search
- [src/local_repair.py](src/local_repair.py): local profile refinement, inserted transition samples, diagnostics
- [src/bridge_builder.py](src/bridge_builder.py): process-frame transition sample insertion helpers
- [src/robodk_program.py](src/robodk_program.py): RoboDK-facing path validation and program generation

## How to run

Use the project conda environment:

```powershell
conda activate winding_pose_solver
python main.py
```

Equivalent direct interpreter:

```powershell
C:\Users\22290\anaconda3\envs\winding_pose_solver\python.exe main.py
```

Visualization only:

```powershell
python main.py --visualize
```

## RoboDK station requirements

The live station must contain at least:

- robot: `KUKA`
- frame: `Frame 2`

The current implementation assumes the live station is already opened in RoboDK and that the RoboDK Python API is available in the selected environment.

## Outputs

- `data/tool_poses_frame2.csv`
  - nominal fixed-orientation tool poses from the centerline
- `data/tool_poses_frame2_optimized.csv`
  - written only when a fully validated final path exists
  - includes:
    - `row_label`
    - `inserted_transition_point`
    - `frame_a_origin_dy_mm`
    - `frame_a_origin_dz_mm`
    - final pose matrix columns

## Validation checklist

When `python main.py` runs, verify that the log shows:

- zero-offset baseline row `0` has IK candidates
- no orientation-search acceptance messages
- no orientation-redistribution messages
- a focused report for `372->373` and `380->381`
- either:
  - a fully validated path with no remaining bad segments, or
  - an explicit failure report and no exported misleading program

For tighter negative-case validation, reduce the Y/Z envelope temporarily and confirm the solver fails explicitly instead of exporting a program.

## What changed in this revision

- Added a fixed calibrated process-frame rotation constant in `Frame 2`
- Removed runtime rigid orientation search from the active solver path
- Removed local orientation redistribution from the active solver path
- Replaced the old adjustment variable with a per-point `Frame 2` Y/Z origin profile for frame `A`
- Removed the active joint-space interpolation bridge fallback from program generation
- Added explicit whole-path failure reporting
- Updated validation output to report the real problematic zone instead of hiding it with joint interpolation
