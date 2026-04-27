# AGENTS.md

This file applies to `src/` as a whole.

If a deeper package directory contains its own `AGENTS.md`, follow that deeper file for code in that subtree.

## Purpose

`src/` contains the real implementation layers of the project. The package directories under `src/` are the preferred home for new logic.

The flat `src/*.py` files are mostly compatibility wrappers kept for older imports.

## Rules

- Put new implementation in `src/core/`, `src/search/`, `src/runtime/`, `src/robodk_runtime/`, or `src/six_axis_ik/` instead of adding more flat modules unless a compatibility reason is explicit.
- Keep flat wrapper files thin. Re-export symbols or forward calls; do not duplicate substantial business logic there.
- If you need to expose new functionality through an old import path, implement it in the package first and then add the smallest necessary wrapper change.
- When changing import surfaces, search for both `src.<package>` imports and legacy flat imports before finishing.
- Read the nearest package `README.md` and `AGENTS.md` before making nontrivial edits.

## Validation

- Import checks are usually enough for wrapper-only edits.
- If the change affects behavior rather than import wiring, validate in the owning package rather than in the wrapper layer.
