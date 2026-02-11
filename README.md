# graph_invariant

LLM-driven graph invariant discovery for research workflows.  
This repository contains a Phase 1 implementation (data generation, candidate search/evaluation, novelty scoring, baselines, and reporting) plus the project specification documents.

## What This Project Does

- Generates synthetic graph datasets and target values.
- Uses an island-style evolutionary loop to search candidate invariants from an LLM.
- Evaluates candidates with a constrained Python sandbox.
- Scores by accuracy, simplicity, and novelty against known invariants.
- Produces reproducible JSON/JSONL artifacts, checkpoints, and markdown reports.

## Requirements

- Python 3.11+
- `uv` for environment/dependency management
- Optional: local Ollama model for candidate generation (default model: `gpt-oss:20b`)

## Quickstart

1. Install dependencies:

```bash
uv sync --group dev
```

2. Run local quality checks (same gates as CI):

```bash
uv run --group dev ruff check .
uv run --group dev ruff format --check .
uv run --group dev pytest -q
```

3. Run Phase 1 with a small config:

```bash
cat > /tmp/phase1_config.json <<'JSON'
{
  "num_train_graphs": 2,
  "num_val_graphs": 3,
  "num_test_graphs": 2,
  "timeout_sec": 0.5,
  "artifacts_dir": "artifacts_smoke"
}
JSON
uv run python -m graph_invariant.cli phase1 --config /tmp/phase1_config.json
```

4. Generate a report:

```bash
uv run python -m graph_invariant.cli report --artifacts artifacts_smoke
```

5. Run multi-seed benchmark:

```bash
cat > /tmp/benchmark_config.json <<'JSON'
{
  "benchmark_seeds": [11, 22, 33],
  "max_generations": 0,
  "run_baselines": true,
  "artifacts_dir": "artifacts_benchmark"
}
JSON
uv run python -m graph_invariant.cli benchmark --config /tmp/benchmark_config.json
```

## CLI Commands

- `uv run python -m graph_invariant.cli phase1 --config <config.json> [--resume <checkpoint.json>]`
- `uv run python -m graph_invariant.cli report --artifacts <artifacts_dir>`
- `uv run python -m graph_invariant.cli benchmark --config <config.json>`

## Architecture Snapshot

- `src/graph_invariant/cli.py`: orchestration and CLI entry points.
- `src/graph_invariant/data.py`: graph generation and dataset splitting.
- `src/graph_invariant/sandbox.py`: static checks + constrained execution pool.
- `src/graph_invariant/scoring.py`: metrics, simplicity, novelty, total score.
- `src/graph_invariant/benchmark.py`: deterministic multi-seed benchmark runner.
- `src/graph_invariant/baselines/`: statistical and PySR baselines.

## Artifacts

Phase 1 run outputs:
- `<artifacts_dir>/logs/events.jsonl`
- `<artifacts_dir>/checkpoints/<experiment_id>/gen_<N>.json`
- `<artifacts_dir>/phase1_summary.json`
- `<artifacts_dir>/baselines_summary.json` (if `run_baselines=true`)
- `<artifacts_dir>/report.md` (from `report` command)

Benchmark outputs:
- `<artifacts_dir>/benchmark_<timestamp>/benchmark_summary.json`
- `<artifacts_dir>/benchmark_<timestamp>/benchmark_report.md`
- `<artifacts_dir>/benchmark_<timestamp>/seed_<N>/...`

## Documentation Map

- `AGENTS.md`: contributor/agent workflow and repository etiquette.
- `PRODUCT.md`: product goals and user-centric context.
- `TECH.md`: technology choices and constraints.
- `STRUCTURE.md`: repository layout and code organization.
- `SPEC.md`: implementation spec (authoritative details).
- `REVIEW.md`: resolved review findings.
- `Research_Plan_Graph_Invariant_Discovery.md`: original proposal (historical context).

## Security and Safety Notes

- The sandbox is best-effort for research. It is not a production security boundary.
- Candidate prompts/responses can be logged if enabled; treat logs as sensitive.
