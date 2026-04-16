@AGENTS.md

## Cross-Agent Workflow Preferences

- The user generally prefers autonomous internal iteration for non-trivial coding tasks.
- Default loop: understand and scope, implement a first pass, self-review the diff, run the smallest relevant validation, fix issues found, and do one more cleanup or refinement pass if it is likely to improve the result.
- Do not stop after a merely plausible first draft if another cheap pass would materially improve correctness, maintainability, or clarity.
- Ask the user for input only when there is a real requirement gap, an important tradeoff, or an external blocker.
- Keep final summaries concise and explicit about what was validated versus what was not.
- The user often writes informal or Chinese-first prompts. Normalize them internally into a compact task brief, preserve the intended scope, and keep any visible restatement extremely short.
- Use clarifying questions only when ambiguity is materially risky. For search and diagnostics, internally expand into bilingual or English-heavy keywords when useful, but keep the user-facing conversation in Chinese unless asked otherwise.
