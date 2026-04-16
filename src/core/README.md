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

