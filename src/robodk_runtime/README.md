# `src/robodk_runtime/`

This package is the live RoboDK layer.

Use it for code that requires:

- an open RoboDK station
- live robot/frame/tool lookup
- program generation inside RoboDK
- final path materialization against the station

Keep pure math and shared data structures out of this package when possible; those belong in `src/core/`.

