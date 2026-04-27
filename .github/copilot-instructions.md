# Repository Instructions

- Read the nearest `AGENTS.md` before making non-trivial changes. If you are working in a subdirectory with its own `AGENTS.md`, that more specific file takes precedence.
- Canonical server path is `/home/tzwang/program/winding_pose_solver`; do not use the retired `/home/tzwang/apps/winding_pose_solver` copy.
- Keep top-level entrypoints thin. New implementation usually belongs in `src/core/`, `src/search/`, `src/runtime/`, `src/robodk_runtime/`, or `src/six_axis_ik/` rather than in flat wrapper modules.
- The user prefers autonomous multi-pass refinement for non-trivial tasks. Default to a short plan, first implementation, self-review, focused validation, and one refinement pass before handing back results.
- Do not stop at a merely plausible first draft if another cheap pass would likely improve correctness, maintainability, or consistency with the existing codebase.
- The user often writes informal or Chinese-first prompts. Normalize them internally into a compact task brief, preserve the intended scope, and keep visible restatement minimal.
- Ask the user for clarification only when requirements are genuinely ambiguous or a decision has meaningful product or architecture consequences.
- This repository is developed on a shared Slurm-managed server. Heavy compute such as large sweeps, batch evaluation, benchmarking, compilation, or substantial data processing must go through Slurm rather than the login node.
- Preserve artifact conventions under `artifacts/local_runs/`, `artifacts/online_runs/`, and `artifacts/run_logs/` unless the task explicitly changes them.
- If you change request or result schemas, artifact layouts, or other cross-cutting behavior, update the relevant docs and affected producers and consumers together.
