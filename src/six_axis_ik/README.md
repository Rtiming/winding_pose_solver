# `src/six_axis_ik/`

This package contains the embedded local six-axis IK/FK implementation.

Typical edit reasons:

- robot calibration/model constants
- analytic or numeric IK logic
- FK helpers
- RoboDK bridge helpers for parity checks

Try to keep high-level path-search policy outside this package. This layer
should focus on kinematics and backend behavior, not search strategy.

Online server compute uses this backend so it can run without RoboDK. Config
flags returned by this layer are consumed by `src/search/`; lower/elbow
preference, periodic A4/A6 continuity, and closed terminal full-turn policy are
search/runtime concerns, not solver internals.

