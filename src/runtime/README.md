# `src/runtime/`

This package contains high-level orchestration code:

- local app entry flows
- request construction
- requester-side remote search flow
- runtime profiling helpers

If a change is about:

- how `main.py` builds or runs work
- how online requests are assembled
- how profiling is reported

this is usually the right package to edit.

