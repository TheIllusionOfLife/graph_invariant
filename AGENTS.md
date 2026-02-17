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
- Run OOD validation: `uv run python -m graph_invariant.cli ood-validate --summary <summary.json> --output <output_dir>`
- Run full experiment suite: `bash run_all_experiments.sh`
  - Quick profile (default): `bash run_all_experiments.sh`
  - Full profile: `PROFILE=full bash run_all_experiments.sh`
  - Override model: `MODEL=gemma3:4b bash run_all_experiments.sh`
- Compile paper: `cd paper && tectonic main.tex`
- Run analysis pipeline:
  - `uv run python analysis/analyze_experiments.py --artifacts-root artifacts/ --output analysis/results/`
  - `uv run python analysis/generate_figures.py --data analysis/results/ --output paper/figures/`

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
  - target value computation in `targets.py`
  - MAP-Elites diversity archive in `map_elites.py`
  - OOD validation in `ood_validation.py`
  - baseline methods under `baselines/`
  - baseline feature extraction in `baselines/features.py`
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
  - `docs/SPEC.md` (implementation spec)
  - `TECH.md` / `STRUCTURE.md` (architecture and stack)

## Lessons from Past Reviews

Patterns and pitfalls distilled from PR reviews across the project history.

### Security

- **numpy sandbox escape** (PR #11): `np.fromfile`/`np.tofile` allow filesystem access from within the sandbox. Whitelist only safe numpy functions in `safe_globals`.
- **sympy.simplify() uses eval()** (PR #8): SymPy's simplification internally calls `eval()`. Never pass untrusted or user-supplied strings to `sympy.simplify()`.
- **Indirect prompt injection** (PR #14): Raw LLM output concatenated into subsequent prompts can carry injection payloads. Sanitize or constrain content when building prompts from prior LLM output.

### Feature Leakage

- **Baseline features must exclude the target** (PR #16): When building feature matrices for baselines, the target invariant must be excluded from the feature set to prevent trivial leakage.

### Edge Cases

- **`deserialize_archive` must handle malformed data** (PR #15): MAP-Elites archive deserialization should wrap cell parsing in `try/except` to handle corrupted checkpoint data gracefully.
- **Grid graphs have tuple node labels** (PR #15): `nx.grid_2d_graph()` produces nodes labeled as `(row, col)` tuples. Relabel to integers before computing invariants.
- **`degree_assortativity_coefficient` can raise on uniform-degree graphs** (PR #15): Guard calls with `try/except` for graphs where all nodes have equal degree.
- **Empty feature exclusion can produce 1D array** (PR #16): When all features are excluded, `np.asarray(list(zip(...)))` may yield a 1D array instead of 2D. Guard the shape.

### Error Handling

- **Catch `ImportError` not bare `Exception`** (PR #5, #6): Optional dependency imports (e.g., `scikit-learn`, `pysr`) should catch `ImportError` specifically, not bare `Exception`.

### Code Quality

- **Extract duplicated rejection-handling logic** (PR #10): Repetitive rejection-handling code should be factored into helper functions (see `_handle_rejection` in `cli.py`).

### Config Discipline

- **New runtime flags must go through `Phase1Config`** with validation in `__post_init__`. Never bypass config with ad-hoc global variables or environment reads.
