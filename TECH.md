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
- `scipy.sparse` / `scipy.sparse.linalg`: sparse eigensolver for algebraic connectivity on large graphs
- `sympy`: symbolic simplification for simplicity scoring
- `requests`: HTTP integration with Ollama API
- `argparse`: CLI framework (stdlib)
- `multiprocessing`: sandbox execution via pre-spawned worker pool (stdlib)

## Optional Libraries

- `scikit-learn`: random-forest baseline
- `pysr` + Julia runtime: symbolic regression baseline (requires Julia 1.10+ installed via `juliaup` or official installer)

## Linting and Testing

- `ruff` for linting and formatting checks
- `pytest` for unit and integration-style module tests

## Experiment Orchestration

- `run_all_experiments.sh`: sequential experiment runner with OOD validation and reporting
- `configs/`: pre-built experiment configurations (quick and full profiles)

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
