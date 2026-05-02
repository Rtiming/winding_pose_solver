# `src/robodk_runtime/`

This package is the live RoboDK layer.

Use it for code that requires:

- an open RoboDK station
- live robot/frame/tool lookup
- program generation inside RoboDK
- final path materialization against the station
- importing a server handoff or selected joint path into the active station

Keep pure math and shared data structures out of this package when possible;
those belong in `src/core/`. Keep IK search, active-set polish, and retry
orchestration out of this package; those belong in `src/search/` and
`src/runtime/`.

The online receiver should materialize the selected handoff path directly. It
should not rerun full search or silently change the chosen path while generating
the final RoboDK program.

