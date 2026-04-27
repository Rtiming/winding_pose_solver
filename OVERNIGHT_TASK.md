# Overnight Task

## Goal

Describe the concrete outcome you want by morning in one or two sentences.

## Scope

- Which files, directories, or subsystems are in scope?
- What should stay out of scope?

## Constraints

- Preserve existing behavior unless needed for the task.
- Follow project `AGENTS.md` and shared Slurm rules.
- If a decision is high-impact, prefer the most conservative correct option and explain it in the handoff.
- Keep the visible final reply short; put the useful detail into the handoff file.

## Deliverables

- Code or config changes, if needed
- Focused validation
- A concise overnight handoff in the generated run directory

## By Morning Checklist

- [ ] The main task is implemented or a real blocker is documented
- [ ] The smallest useful validation has been run
- [ ] The handoff clearly states `done`, `partial`, or `blocked`
- [ ] Important artifact paths are listed

## Optional Notes Before Sleep

- Extra follow-up ideas you want included if there is time
- Things to avoid changing tonight
- Any preference about code style, safety, or risk tolerance

## Suggested Internal Workflow

1. Understand the relevant code and choose the smallest correct edit surface.
2. Implement the task.
3. Self-review the diff.
4. Run the smallest useful validation.
5. If obvious issues remain and another pass is cheap, refine once more.

## Task Details

Replace this section with the actual overnight request. A good pattern is:

- What is broken or missing
- What success looks like
- What files or modules are probably involved
- What evidence or validation would make you trust the result
