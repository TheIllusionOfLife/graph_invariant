# AGENTS.md

Repository-specific instructions for coding agents and contributors.

## Scope and Priority

- This file defines repository conventions and operational details.
- If this file conflicts with direct user instructions in a task, follow the user instruction for that task.

## Non-Obvious Commands

- Install dev dependencies: `uv sync --group dev`
- Run CI-equivalent local checks:
  - `uv run --group dev ruff check .`
  - `uv run --group dev ruff format --check .`
  - `uv run --group dev pytest -q`
- Run Phase 1: `uv run python -m graph_invariant.cli phase1 --config <config.json>`
- Resume Phase 1 from checkpoint:
  - `uv run python -m graph_invariant.cli phase1 --config <config.json> --resume <checkpoint.json>`
- Generate report: `uv run python -m graph_invariant.cli report --artifacts <artifacts_dir>`
- Run benchmark: `uv run python -m graph_invariant.cli benchmark --config <config.json>`

## Code Style and Architecture Rules

- Python conventions:
  - 4-space indentation
  - type hints for public APIs
  - `snake_case` for variables/functions
  - `PascalCase` for classes
- Keep high cohesion:
  - orchestration in `cli.py`
  - generation in `data.py`
  - evaluation sandbox in `sandbox.py`
  - metrics/scoring in `scoring.py`
  - baseline methods under `baselines/`
- Do not bypass `Phase1Config` for new runtime flags; add validated config fields there first.
- Preserve compatibility for artifact schema readers when changing summary payloads.

## Testing Instructions

- Preferred runner: `pytest` through `uv`.
- Default command: `uv run --group dev pytest -q`
- Bugfix workflow:
  - add or update a failing test first
  - implement minimal fix
  - run full test suite
- For stochastic or numeric behavior:
  - use deterministic seeds
  - use tolerance-based assertions
- For CLI behavior, prefer tests in `tests/test_cli.py` rather than ad-hoc shell-only checks.

## Branching, Commits, and PR Etiquette

- Never work directly on `main`; create a branch first.
- Branch naming:
  - `feat/<short-description>`
  - `fix/<short-description>`
  - `chore/<short-description>`
  - `docs/<short-description>`
- Commit messages should be imperative and specific.
- Push with explicit branch names:
  - `git push origin <branch-name>`
- PRs should include:
  - purpose/scope
  - validation commands and outcomes
  - spec/doc updates when behavior or interfaces changed

## Environment and Tooling Quirks

- Use `uv` for Python dependency and command execution.
- Ollama is local-first:
  - default endpoint: `http://localhost:11434/api/generate`
  - remote endpoints are blocked unless `allow_remote_ollama=true`
- Optional baselines:
  - random forest baseline requires `scikit-learn`
  - PySR baseline requires `pysr` and Julia runtime

## Common Gotchas

- Sandbox security is best-effort research isolation, not a hard production boundary.
- `persist_prompt_and_response_logs=true` stores raw model I/O in artifacts logs; avoid on sensitive prompts/data.
- Artifact directories can become large; keep them untracked and under ignored paths.
- Changing scoring weights (`alpha`, `beta`, `gamma`) triggers normalization if they do not sum to 1.0.

## Document Sync Policy

- Keep cross-document consistency when changing behavior:
  - `README.md` (usage)
  - `SPEC.md` (implementation spec)
  - `TECH.md` / `STRUCTURE.md` (architecture and stack)
