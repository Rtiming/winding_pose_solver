# AGENTS.md

This repo-level file defines the default rules for coding agents working in `winding_pose_solver`.

If a deeper directory contains its own `AGENTS.md`, follow the deepest applicable file for code in that subtree and treat this root file as the fallback policy.

## Project Purpose

This repository generates winding-tool poses from centerline data, validates IK feasibility, searches for a continuous joint path, and optionally materializes the final RoboDK program.

There are two interchangeable IK backends:

- `six_axis_ik`: embedded local analytic backend
- `robodk`: live RoboDK-backed truth source

Preserve backend interchangeability unless the task explicitly targets one backend only.

## Fast Decision Checklist

Before editing, quickly answer these questions:

1. Which layer owns the change: `core`, `search`, `runtime`, `robodk_runtime`, `six_axis_ik`, or `scripts`?
2. Does the change affect both the single-machine flow and the online requester/worker flow?
3. Is the change backend-agnostic, RoboDK-only, or `six_axis_ik`-only?
4. Will it change request/result schemas, artifact filenames, or run-directory conventions?
5. What is the smallest validation that proves the change, and does that validation need Slurm?

## Top-Level Files

- `main.py`: convenience entrypoint plus frequently tuned local defaults. Keep reusable logic out of this file; move real implementation into `src/runtime/`, `src/core/`, or the relevant package.
- `app_settings.py`: shared runtime settings and tuning surface. Edit here when changing config wiring or default solver/runtime parameters used across flows.
- `online_requester.py`: requester-side CLI entrypoint. Keep it thin.
- `online_worker.py`: worker-side CLI entrypoint. Keep it thin and delegate to `src/robodk_runtime/` or `src/runtime/`.
- `online_roundtrip.py`: remote orchestration, SSH transport, and end-to-end roundtrip control. Do not bury pure math or search heuristics here.

## Repository Map

Main package layout:

- `src/core/`: reusable math, geometry, schemas, CSV handling, pose solving, backend-agnostic helpers
- `src/search/`: IK candidate collection, DP path selection, repair logic, continuity-focused search
- `src/runtime/`: orchestration, request building, profiling, run logging, origin sweep utilities
- `src/robodk_runtime/`: logic that requires a live RoboDK station
- `src/six_axis_ik/`: embedded IK/FK implementation and backend-specific kinematics behavior

Support directories:

- `scripts/`: thin diagnostics and one-off utilities; reusable logic belongs in `src/`
- `data/`: input CSVs and small checked-in examples
- `artifacts/`: local runs, online runs, diagnostics, logs, temporary outputs

## Edit Surface Guidance

Use these heuristics before touching code:

- If the change is pure math, geometry, CSV/schema, or backend-agnostic pose logic: start in `src/core/`
- If the change is about continuity, configuration switching, repair heuristics, DP cost design, or inserted transitions: start in `src/search/`
- If the change is about orchestration, request/result handling, profiling, run logging, or sweep coordination: start in `src/runtime/`
- If the change requires a live RoboDK station or final program generation: start in `src/robodk_runtime/`
- If the change is about IK model constants, analytic/numeric solving, FK, or RoboDK parity: start in `src/six_axis_ik/`
- If the change is only a CLI or thin diagnostic wrapper: start in `scripts/`

Avoid adding new business logic to the flat compatibility wrappers in `src/`.

## Cross-Cutting Rules

- Prefer minimal, local changes over broad refactors.
- Read the relevant package `README.md` and nearest `AGENTS.md` before editing.
- Keep the single-machine and online flows aligned. Do not silently fix one path while breaking the other.
- Keep RoboDK-dependent behavior isolated from backend-agnostic logic where practical.
- Keep high-level path-search policy out of `src/six_axis_ik/`.
- Keep live-station behavior out of `src/core/`.
- Do not do git mutations unless explicitly requested.

## Iteration And Handoff Style

- Assume the user prefers autonomous multi-pass refinement for non-trivial work rather than having to ask for repeated follow-up iterations.
- Default workflow: short plan, first implementation, self-review of the touched files or diff, smallest relevant validation, then one more refinement pass if there are obvious rough edges or fixable risks.
- Do not stop after a merely plausible first draft if another cheap pass would likely improve correctness, maintainability, or fit with the existing codebase.
- Ask the user for guidance only when requirements are genuinely unclear, a tradeoff is high-impact, or validation is blocked or disproportionately expensive.
- In final reporting, separate what changed, what was validated, and what still remains uncertain.

## Intent Alignment Style

- The user often writes short, informal, or Chinese-first prompts. Internally normalize those requests into a compact task brief before acting.
- Preserve the user's intent rather than "upgrading" the request into a broader task. Infer missing low-risk details when reasonable, but do not silently add major scope.
- Keep visible restatement minimal. If the request is clear enough, do not paraphrase it back. If a small alignment check helps, use one short sentence or a few compact points, then proceed.
- Ask clarifying questions only when ambiguity is genuinely material, or when a wrong assumption could lead to high-cost validation, large edits, or architectural drift.
- For code search, logs, and external docs, it is fine to expand the user's wording into bilingual or English-heavy internal search terms while continuing to communicate with the user in Chinese.

## Compatibility And Public Surface

Older flat imports in `src/*.py` are still supported through thin wrappers. When adding or moving reusable logic:

- put the implementation in the package directory first
- only touch the flat wrapper if an import-compatibility surface must remain exposed
- avoid duplicating logic between wrapper files and package files

If you rename exported symbols or move modules, search for both package imports and legacy flat imports before finishing.

## Environment

There are two main environment targets:

- `environment.server.yml`: Python 3.12
- `environment.local-worker.yml`: Python 3.10

The shared dependency split is:

- `requirements.shared.txt`
- `requirements.server.txt`
- `requirements.local-worker.txt`

When making dependency-sensitive changes, check which runtime target they affect.

## Slurm And Heavy Work

This repo is developed on a shared Slurm-managed server.

- Heavy compute must not run directly on the login node.
- Any training, benchmark, large sweep, large batch evaluation, compilation, or substantial data processing must go through Slurm.
- Lightweight inspection, searching, reading, and small edits may run directly.
- Follow the Slurm policy defined in `/home/tzwang/.codex/skills/slurm-run/SKILL.md`.

If you need to run an expensive sweep, batch evaluation, or profile search experiment, prefer `srun` or `sbatch` sized from current free capacity.

## Operational Defaults

- Default single-machine work should remain solve-first unless the task explicitly needs RoboDK program generation.
- `six_axis_ik` should stay usable for offline evaluation workflows.
- The final RoboDK program-generation step should remain local unless the task explicitly changes that architecture.
- If a schema or artifact change crosses local and online flows, update the relevant docs and all affected producers and consumers together.

## Diagnostics And Artifacts

Important run outputs often include:

- `artifacts/local_runs/<run_id>/request.json`
- `artifacts/local_runs/<run_id>/eval_result.json`
- `artifacts/local_runs/<run_id>/selected_joint_path.csv`
- `artifacts/local_runs/<run_id>/run_archive.json`
- `artifacts/local_runs/<run_id>/profile_retry_summary.json`
- `artifacts/online_runs/<run_id>/request.json`
- `artifacts/online_runs/<run_id>/results.json`
- `artifacts/diagnostics/`
- `artifacts/run_logs/`
- `artifacts/tmp/`

When changing run behavior or diagnostics, preserve these conventions unless the task explicitly asks to redesign them.

If you modify output schemas, filenames, or run directory structure, update the user-facing README and the relevant `AGENTS.md` files.

## Validation Expectations

There is no obvious centralized root test suite, so validation is task-specific.

Prefer the smallest relevant validation:

- import-level or module-level checks for pure refactors
- focused diagnostic scripts for search or IK behavior
- targeted single-machine solve runs for pipeline changes
- online requester/worker smoke flows for remote orchestration changes

If validation is expensive, use Slurm.

When reporting results, separate:

- what changed
- what was validated
- what was not validated

## Kimi Consultation

Codex may consult the local Kimi CLI for advisory-only second opinions.

- Use it only when it materially improves confidence.
- Good fits: design review, public-web cross-checking, risk review, Chinese-language synthesis.
- Do not treat Kimi as the primary executor for this repo.
- Do not use Kimi to make direct local file changes or Slurm decisions.

The wrapper is:

- `/home/tzwang/.codex/skills/kimi-consult/scripts/ask_kimi.sh`

## When To Update This File

Update this file when you change:

- the recommended edit surface
- primary entrypoints or package responsibilities
- artifact conventions
- environment/runtime expectations
- Slurm execution policy for this repo
- backend architecture in a way future agents should know
