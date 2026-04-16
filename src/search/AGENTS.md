# AGENTS.md

This file applies to `src/search/`.

## Own This Layer For

- IK candidate collection per row
- path scoring and dynamic-programming selection
- local repair and handover refinement
- inserted-transition logic
- continuity-focused quality metrics

## Good Edit Targets

- `ik_collection.py`: candidate generation and per-row solvability data
- `path_optimizer.py`, `global_search.py`: path selection and exact-search behavior
- `local_repair.py`, `bridge_builder.py`: repair and transition refinement
- `parallel_profile_eval.py`: profile evaluation helpers that may become CPU-heavy

## Keep Out Of This Layer

- live RoboDK station orchestration
- request transport and SSH plumbing
- top-level CLI defaults
- low-level kinematics implementation details better owned by `src/six_axis_ik/`

## Editing Guidance

- Changes here often affect both the single-machine and online flows because both consume the same search outputs.
- Preserve the meaning of summary metrics such as `ik_empty_row_count`, `config_switches`, `bridge_like_segments`, and `worst_joint_step_deg` unless the task explicitly asks to redesign them.
- Keep backend-specific calls behind the shared robot interface when possible; do not hard-wire search policy to one backend without an explicit reason.
- Prefer small, measurable heuristic changes over broad intertwined rewrites.

## Validation

- Start with the smallest diagnostic script or single solve that exercises the changed heuristic.
- Large sweeps, batch evaluations, or repeated profile experiments must use Slurm.
