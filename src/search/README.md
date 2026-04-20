# `src/search/`

This package owns the path-quality logic after poses are known.

Main responsibilities:

- collect IK candidates per row
- score and select a continuous joint path
- enforce the closed-winding terminal family/full-turn search constraints
- repair bad windows with Frame-A Y/Z profile adjustments
- insert transition samples when needed

If a change is about continuity, configuration switching, repair heuristics, or DP cost design, this is the package to edit first.

Current quality rule:

- closed path terminal I1-I5/I6 behavior is a hard search/import constraint
- `big_circle_step_count`, `bridge_like_segments`, and worst joint step feed the
  strict delivery gate
- `config_switches` is still diagnostic by itself, because a config label can
  change without a physical jump near singularity

