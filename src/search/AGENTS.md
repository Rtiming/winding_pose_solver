# AGENTS.md

This file applies to `src/search/`.

## Own This Layer For

- IK candidate collection per row
- path scoring and dynamic-programming selection
- local repair and handover refinement
- inserted-transition logic
- continuity-focused quality metrics
- A4/A6 periodic transition scoring and closed-terminal A6 full-turn handling
- lower/elbow configuration preference or lock policy as consumed by DP

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
- Preserve the hard closed-winding rule: terminal I1-I5 match start and I6 is
  exactly one full turn from start. Periodic transition unwrapping in the
  middle of the path must not become a fake `0` degree terminal closure.
- Treat `config_switches` as diagnostic unless the user promotes it. Improve
  physical continuity through costs, repair, and candidate policy rather than
  deleting or weakening diagnostics.
- If you add profile repair mechanics, keep orchestration in `src/runtime/`;
  this package should expose reusable scoring/repair behavior.

## Validation

- Start with the smallest diagnostic script or single solve that exercises the changed heuristic.
- Large sweeps, batch evaluations, or repeated profile experiments must use Slurm.
- For periodic or lower-config policy changes, check both selected joint
  continuity and the closed terminal rule.
