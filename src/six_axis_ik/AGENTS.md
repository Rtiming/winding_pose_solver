# AGENTS.md

This file applies to `src/six_axis_ik/`.

## Own This Layer For

- robot calibration and model constants
- analytic and numeric IK solving
- FK helpers
- backend-specific workflow glue for the embedded solver
- RoboDK bridge helpers used for parity checks

## Good Edit Targets

- `config.py`: model constants and solver configuration
- `analytic_solver.py`, `numeric_solver.py`: solving strategies
- `kinematics.py`, `interface.py`, `workflow.py`: kinematics plumbing and solver-facing workflow
- `backends.py`, `robodk_bridge.py`: backend integration and parity helpers

## Keep Out Of This Layer

- high-level path-search policy
- run orchestration and artifact management
- live RoboDK station program generation

This package should focus on kinematics and backend behavior, not search strategy.

## Editing Guidance

- Be explicit when a change is intended to improve RoboDK parity versus when it intentionally changes local-solver behavior.
- Keep interfaces stable for callers in `src/core/robot_interface.py` and the search layer unless the task explicitly changes that contract.
- If you change numeric tolerances or model constants, document the intent in code comments when the reason is not obvious.

## Validation

- Prefer targeted solver-level checks or the smallest backend comparison that exercises the change.
- Large parity sweeps or broad batch comparisons must use Slurm.
