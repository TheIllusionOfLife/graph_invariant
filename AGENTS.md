# Repository Guidelines

## Project Structure & Module Organization
This repository is currently research-document driven. Keep core documents at the root:
- `SPEC.md`: authoritative implementation specification.
- `REVIEW.md`: consolidated technical review and resolved issues.
- `Research_Plan_Graph_Invariant_Discovery.md`: initial proposal and context.

When adding executable code, use a clean split:
- `src/` for implementation modules.
- `tests/` for automated tests.
- `artifacts/` for experiment outputs (logs, plots, checkpoints), with large/generated files excluded from Git.

## Build, Test, and Development Commands
There is no build pipeline yet; contributors mainly edit and validate documents/specs.
- `rg --files` lists tracked project files quickly.
- `git log --oneline -n 10` checks recent change patterns before committing.
- `uv run pytest` runs tests once Python modules are introduced.
- `uv run ruff check . && uv run ruff format .` lints and formats Python code when code exists.

If you add runnable scripts, document the exact command in `SPEC.md` and this file in the same PR.

## Coding Style & Naming Conventions
- Python: 4-space indentation, type hints for public functions, `snake_case` for functions/variables, `PascalCase` for classes.
- Keep modules focused and small; separate generation, evaluation, and logging concerns.
- Markdown: concise sections, explicit headings, and consistent terminology with `SPEC.md` (e.g., “Phase 1”, “Island Model”, “novelty_bonus”).

## Testing Guidelines
- Follow TDD: write failing tests first, then implement minimal code to pass.
- Use `pytest` with files named `tests/test_<module>.py` and test names `test_<behavior>()`.
- For numerical metrics, include deterministic seeds and tolerance-based assertions.
- Add regression tests for every bug fix, especially sandboxing and scoring logic.

## Commit & Pull Request Guidelines
Recent history favors concise, imperative messages such as:
- `Address SPEC.md review feedback (6 issues)`
- `Add implementation spec and cross-reference all documents`

PRs should include:
- Purpose and scope (what changed and why).
- Linked issue/review comment (if applicable).
- Validation performed (commands run and results).
- Any spec/document sync required across `SPEC.md`, `REVIEW.md`, and this guide.
