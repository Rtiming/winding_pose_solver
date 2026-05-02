# AGENTS.md

This file applies to `scripts/`.

## Purpose

`scripts/` is for thin diagnostics, inspection utilities, and one-off operational helpers.

Current scripts include:

- backend comparison
- focused IK-window diagnosis
- candidate-landscape inspection
- RoboDK import of a diagnostic window
- Y/Z sweep utilities

## Rules

- Keep scripts thin. Reusable logic belongs in `src/core/`, `src/search/`, `src/runtime/`, or `src/robodk_runtime/`.
- When a script starts accumulating shared logic, move that logic into `src/` and let the script remain a CLI wrapper.
- Preserve user-facing CLI behavior unless the task explicitly asks to redesign it.
- If a script writes outputs, prefer existing artifact conventions under `artifacts/`.
- If a script triggers a heavy sweep, repeated evaluation, or broad comparison, run it through Slurm rather than on the login node.
- Do not implement online retry, active-set profile generation, path scoring,
  or RoboDK finalization policy directly in `scripts/`. Put reusable behavior
  in the owning `src/` package and keep the script as a launcher.
- Server examples must use `/home/tzwang/program/winding_pose_solver`.

## Validation

- For CLI-only changes, use the lightest possible invocation such as `--help` or a minimal dry path.
- For behavior changes, validate the owning logic in `src/` rather than relying only on the script wrapper.
