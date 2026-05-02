# `src/search/`

This package owns the path-quality logic after poses are known.

Main responsibilities:

- collect IK candidates per row
- score and select a continuous joint path with DP
- enforce the closed-winding terminal family/full-turn search constraints
- keep A4/A6 periodic transition continuity out of artificial large-jump costs
- bias or lock lower/elbow configuration when requested by motion settings
- repair bad windows with Frame-A Y/Z profile adjustments
- insert transition samples when needed

If a change is about continuity, configuration switching, repair heuristics, or DP cost design, this is the package to edit first.

Important files:

- `ik_collection.py`: row-wise IK candidates and per-row solvability.
- `path_optimizer.py`: DP selection, transition cost, config-family penalties,
  periodic continuity, and closed terminal full-turn handling.
- `local_repair.py`: local Y/Z search, handover corridor/window refinement,
  same-family repair, and inserted transitions.
- `global_search.py`: exact profile evaluation entry used by runtime flows.
- `parallel_profile_eval.py`: parallel profile evaluation helpers.

Current quality rule:

- closed path terminal I1-I5/I6 behavior is a hard search/import constraint
- `big_circle_step_count`, `bridge_like_segments`, and worst joint step feed the
  strict delivery gate
- `config_switches` is still diagnostic by itself, because a config label can
  change without a physical jump near singularity

Runtime retry can generate active-set, corridor, and basis Y/Z profiles, but
the actual path scoring and repair acceptance still rely on this package's
search result metrics. Keep those metric meanings stable unless the task
explicitly redesigns the delivery contract.

