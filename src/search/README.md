# `src/search/`

This package owns the path-quality logic after poses are known.

Main responsibilities:

- collect IK candidates per row
- score and select a continuous joint path
- repair bad windows with Frame-A Y/Z profile adjustments
- insert transition samples when needed

If a change is about continuity, configuration switching, repair heuristics, or DP cost design, this is the package to edit first.

