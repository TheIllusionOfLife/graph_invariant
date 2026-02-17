# graph_invariant

[![Python CI](https://github.com/yuyamukai/graph_invariant/actions/workflows/python-ci.yml/badge.svg)](https://github.com/yuyamukai/graph_invariant/actions/workflows/python-ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

LLM-driven graph invariant discovery for research workflows.
This repository contains a Phase 1 implementation (data generation, candidate search/evaluation, novelty scoring, baselines, and reporting) with MAP-Elites diversity archive, bounds mode optimization, OOD validation, and self-correction.

## What This Project Does

- Generates synthetic graph datasets and target values for configurable targets.
- Uses an island-style evolutionary loop with MAP-Elites diversity archive to search candidate invariants from an LLM.
- Evaluates candidates with a constrained Python sandbox.
- Scores by accuracy, simplicity, and novelty against known invariants.
- Supports bounds mode (upper/lower bound optimization) in addition to correlation fitness.
- Applies self-correction: failed candidates are repaired via LLM feedback loops.
- Validates discovered invariants on out-of-distribution graphs (large-scale, extreme topologies).
- Produces reproducible JSON/JSONL artifacts, checkpoints, and markdown reports.

## Requirements

- Python 3.11+
- `uv` for environment/dependency management
- Optional: local Ollama model for candidate generation (default model: `gpt-oss:20b`)
- Optional: Julia runtime for PySR symbolic regression baseline
- Optional: [tectonic](https://tectonic-typesetting.github.io/) for compiling the paper

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

6. Run OOD validation on experiment results:

```bash
uv run python -m graph_invariant.cli ood-validate \
  --summary artifacts_smoke/phase1_summary.json \
  --output artifacts_smoke/ood
```

## Experiment Automation

Run the full experiment suite (MAP-Elites, algebraic connectivity, upper bound, benchmark, OOD validation):

```bash
# Quick profile (default):
bash run_all_experiments.sh

# Full profile:
PROFILE=full bash run_all_experiments.sh

# To override model or generation count, edit the relevant config file under configs/.
```

Pre-built configs are available under `configs/`.

## CLI Commands

- `uv run python -m graph_invariant.cli phase1 --config <config.json> [--resume <checkpoint.json>]`
- `uv run python -m graph_invariant.cli report --artifacts <artifacts_dir>`
- `uv run python -m graph_invariant.cli benchmark --config <config.json>`
- `uv run python -m graph_invariant.cli ood-validate --summary <summary.json> --output <output_dir> [--seed N] [--num-large N] [--num-extreme N]`

## Architecture Snapshot

- `src/graph_invariant/cli.py`: orchestration and CLI entry points.
- `src/graph_invariant/config.py`: validated runtime config dataclass (`Phase1Config`).
- `src/graph_invariant/data.py`: graph generation and dataset splitting.
- `src/graph_invariant/sandbox.py`: static checks + constrained execution pool.
- `src/graph_invariant/scoring.py`: metrics, simplicity, novelty, total score, bound metrics.
- `src/graph_invariant/targets.py`: target value computation for configurable invariant targets.
- `src/graph_invariant/map_elites.py`: MAP-Elites diversity archive (grid-based).
- `src/graph_invariant/ood_validation.py`: out-of-distribution validation on large/extreme graphs.
- `src/graph_invariant/benchmark.py`: deterministic multi-seed benchmark runner.
- `src/graph_invariant/baselines/`: statistical and PySR baselines.
- `src/graph_invariant/baselines/features.py`: baseline feature extraction (excludes target to prevent leakage).
- `configs/`: pre-built experiment configurations (quick and full profiles).

## Artifacts

Phase 1 run outputs:
- `<artifacts_dir>/logs/events.jsonl`
- `<artifacts_dir>/checkpoints/<experiment_id>/gen_<N>.json`
- `<artifacts_dir>/phase1_summary.json`
- `<artifacts_dir>/baselines_summary.json` (if `run_baselines=true`)
- `<artifacts_dir>/report.md` (from `report` command)

OOD validation outputs:
- `<artifacts_dir>/ood/ood_validation.json`

Benchmark outputs:
- `<artifacts_dir>/benchmark_<timestamp>/benchmark_summary.json`
- `<artifacts_dir>/benchmark_<timestamp>/benchmark_report.md`
- `<artifacts_dir>/benchmark_<timestamp>/seed_<N>/...`

## Documentation Map

- `AGENTS.md`: contributor/agent workflow and repository etiquette.
- `PRODUCT.md`: product goals and user-centric context.
- `TECH.md`: technology choices and constraints.
- `STRUCTURE.md`: repository layout and code organization.
- `docs/SPEC.md`: implementation spec (authoritative details).
- `docs/REVIEW.md`: resolved review findings.
- `docs/Research_Plan_Graph_Invariant_Discovery.md`: original proposal (historical context).

## Paper

This repository accompanies the paper:

> **LLM-Driven Discovery of Interpretable Graph Invariants via Island-Model Evolution**
> Yuya Mukai. 2026.

The paper source is in `paper/` (NeurIPS format). Analysis scripts are in `analysis/`.

To compile the paper (requires [tectonic](https://tectonic-typesetting.github.io/)):

```bash
cd paper && tectonic main.tex
```

To regenerate analysis and figures from experiment artifacts:

```bash
uv run python analysis/analyze_experiments.py --artifacts-root artifacts/ --output analysis/results/
uv run python analysis/generate_figures.py --data analysis/results/ --output paper/figures/
```

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

## Security and Safety Notes

- The sandbox is best-effort for research. It is not a production security boundary.
- Candidate prompts/responses can be logged if enabled; treat logs as sensitive.
