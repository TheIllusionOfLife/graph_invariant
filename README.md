# Harmony — Theory Discovery in Knowledge Graphs

License: MIT

> Codebase accompanying the NeurIPS 2026 anonymous submission
> *"Harmony-Driven Theory Discovery in Knowledge Graphs via LLM-Guided Island Search."*

## What This Project Does

- Defines a four-component **Harmony score** (compressibility, coherence, symmetry, generativity) for typed knowledge graphs.
- Runs **LLM-guided island-model search** with MAP-Elites quality-diversity to propose KG mutations (new edges or entities).
- Validates proposals via a deterministic schema and scores them by Harmony gain; archives top candidates per (simplicity, gain) cell.
- Reproduces all paper results across **seven KG domains** (five hand-curated: linear algebra, periodic table, astronomy, physics, materials science; two Wikidata-sourced: Wikidata Physics, Wikidata Materials).

## Requirements

- Python 3.11+
- [`uv`](https://github.com/astral-sh/uv) for environment and dependency management
- Optional: local [Ollama](https://ollama.com) server for the proposer LLM (default model: `gpt-oss:20b`)
- Optional: Apple Silicon + [`mlx_lm`](https://github.com/ml-explore/mlx-lm) for the factor-decomposition backend (`mlx-community/Qwen3.5-35B-A3B-4bit`)
- Optional: [tectonic](https://tectonic-typesetting.github.io/) for compiling the paper

## Quickstart

```bash
# 1. Install dependencies (creates .venv and resolves the harmony console-script).
uv sync --group dev

# 2. Run the test suite.
uv run --group dev pytest tests/harmony -q

# 3. Run a small Harmony discovery (requires Ollama at localhost:11434).
uv run harmony --domain astronomy --generations 20 --output-dir runs/astronomy

# 4. Generate a markdown report from the run.
uv run python analysis/harmony_report.py --output-dir runs/astronomy --domain astronomy
```

The `harmony` console-script accepts the seven supported domains via `--domain`:
`linear_algebra`, `periodic_table`, `astronomy`, `physics`, `materials`, `wikidata_physics`, `wikidata_materials`.

## Reproducing Paper Results

Multi-seed evaluation (10 seeds across 5 discovery domains, ~1 CPU-hour total):

```bash
uv run python scripts/run_multi_seed.py --domain astronomy
uv run python scripts/run_multi_seed.py --domain physics
uv run python scripts/run_multi_seed.py --domain materials
uv run python scripts/run_multi_seed.py --domain wikidata_physics
uv run python scripts/run_multi_seed.py --domain wikidata_materials
```

Factor decomposition (Apple Silicon, MLX backend):

```bash
# Switch backend by setting HarmonyConfig.backend = "mlx"; the loop dispatches automatically.
uv run python scripts/run_mlx_batch.py --seed 42
```

Regenerate appendix tables and figures from bundled artifacts:

```bash
uv run python analysis/generate_appendix_tables.py \
  --factor-csv data/results/factor_decomposition.csv \
  --stat-tests data/results/statistical_tests.json \
  --output paper/sections/appendix_tables_generated.tex

uv run python analysis/generate_harmony_figures.py \
  --astronomy artifacts/harmony/astronomy \
  --physics artifacts/harmony/physics \
  --materials artifacts/harmony/materials \
  --wikidata_physics artifacts/harmony/wikidata_physics \
  --wikidata_materials artifacts/harmony/wikidata_materials \
  --metrics-csv artifacts/harmony/metrics_table.csv \
  --figures-dir paper/figures/
```

## Architecture

- `src/harmony/types.py` — `Entity`, `TypedEdge` (7 EdgeTypes), `KnowledgeGraph`
- `src/harmony/metric/` — `compressibility.py`, `coherence.py`, `symmetry.py`, `generativity.py`, `harmony.py`
- `src/harmony/proposals/` — proposal schema, validator, LLM proposer (Ollama + MLX backends)
- `src/harmony/harmony_loop.py` — island-model search loop with MAP-Elites archive
- `src/harmony/datasets/` — KG builders for each of the seven supported domains
- `analysis/` — report generation, multi-seed metrics tables, NeurIPS figure generation
- `scripts/` — multi-seed runner, MLX batch runner, supplementary-archive builder

## Repository Layout

- `src/harmony/` — primary implementation (this paper)
- `src/graph_invariant/` — legacy module from a prior project; not used by the Harmony pipeline. Retained only because some shared utilities still live there.
- `tests/harmony/` — Harmony test suite
- `paper/` — NeurIPS 2026 manuscript source (LaTeX)
- `data/results/` — multi-seed evaluation summaries (CSV + JSON)
- `analysis/calibration_gate.md` — pre-registered Harmony metric calibration gate

## Compiling the Paper

```bash
cd paper && tectonic main.tex
```

The compiled PDF is `paper/main.pdf` (≤ 9 pages main body + appendix + checklist).

## License

MIT. See [LICENSE](LICENSE) for details.

## Security and Safety Notes

- The proposer calls a local LLM endpoint by default (`http://localhost:11434/api/generate`). Pass `--allow-remote` only with trusted endpoints.
- Prompts and LLM responses are logged when `--output-dir` is set; treat the resulting JSONL files as sensitive.
- Generated proposals are archived but never auto-applied to the base KG; see Section 6 of the paper for the safety rationale.
