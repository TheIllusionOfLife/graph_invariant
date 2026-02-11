# TECH.md

## Technology Stack

## Language and Runtime

- Python 3.11+

## Dependency and Environment Management

- `uv` for dependency resolution, virtualenv management, and command execution
- Project metadata and dependencies defined in `pyproject.toml`

## Core Libraries

- `networkx`: graph generation and graph statistics
- `numpy`: vectorized numerical operations
- `scipy`: statistical metrics (Spearman/Pearson)
- `sympy`: symbolic simplification for simplicity scoring
- `requests`: HTTP integration with Ollama API

## Optional Libraries

- `scikit-learn`: random-forest baseline
- `pysr` + Julia runtime: symbolic regression baseline

## Linting and Testing

- `ruff` for linting and formatting checks
- `pytest` for unit and integration-style module tests

## CI

- GitHub Actions workflows in `.github/workflows/`
- Python CI gates:
  - lint (`ruff check`)
  - format check (`ruff format --check`)
  - tests (`pytest`)

## Technical Constraints

- Sandbox evaluator is best-effort and not a full security boundary.
- Ollama endpoint must be localhost unless `allow_remote_ollama=true`.
- Artifact outputs are expected to be generated locally and kept out of Git.

## Preferred Local Validation Command Set

```bash
uv sync --group dev
uv run --group dev ruff check .
uv run --group dev ruff format --check .
uv run --group dev pytest -q
```
