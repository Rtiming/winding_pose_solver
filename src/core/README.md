# `src/core/`

Use this package for code that should stay reusable across:

- single-machine runs
- online requester/worker flows
- offline `six_axis_ik` evaluation
- diagnostics scripts

Good fits for `core/`:

- geometry and frame math
- CSV readers/writers
- request/result schemas
- pose solving from centerline data
- backend-agnostic robot interface helpers
- visualization that does not depend on a live RoboDK station

Avoid putting live-station behavior here. If code needs an open RoboDK station, it should usually live in `src/robodk_runtime/`.

Pose-solving structure:

- `frame_math.py` loads the centerline CSV and builds validated local frame
  records. It owns the terminal-row append used by closed winding paths.
- `pose_solver.py` converts those frame records into Frame-2 tool poses. Use
  `solve_tool_poses()` for one-shot CSV input and
  `solve_tool_poses_from_dataset()` when a caller needs to evaluate many
  Frame-A origins against the same centerline.

Do not add IK, DP scoring, or RoboDK station behavior here; those belong in
`src/search/` and `src/robodk_runtime/`.

