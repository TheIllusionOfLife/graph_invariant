# Final NeurIPS Rerun Manifest (2026-02-20)

This manifest records the exact commands and artifact paths used for the final
analysis/paper refresh pass on branch `feat/final-neurips-rerun-rebuttal-pack`.

## Branch and base

- Base merge commit on `main`: `ab9bcce06765554af84fa65cc3f646850248e069`
- Working branch: `feat/final-neurips-rerun-rebuttal-pack`

## Validation commands

```bash
uv run --group dev ruff check .
uv run --group dev ruff format --check .
uv run --group dev pytest -q
```

## Matrix rerun command (attempted)

```bash
uv run python scripts/run_neurips_matrix.py \
  --configs \
    configs/neurips_matrix/map_elites_aspl_full.json \
    configs/neurips_matrix/algebraic_connectivity_full.json \
    configs/neurips_matrix/upper_bound_aspl_full.json \
    configs/neurips_matrix/small_data_aspl_train20.json \
    configs/neurips_matrix/small_data_aspl_train35.json \
    configs/neurips_matrix/benchmark_aspl_full.json \
  --seeds 11 22 33 44 55 \
  --max-parallel 3 \
  --output-root artifacts/neurips_matrix_2026-02-20_final
```

Status: launched and partially executed, but not fully completed in this pass
because full-profile runs with long LLM timeouts exceeded practical in-session
runtime.

## Analysis and figure refresh commands

```bash
uv run python analysis/analyze_experiments.py \
  --artifacts-root artifacts/ \
  --output analysis/results/ \
  --appendix-tex-output paper/sections/appendix_tables_generated.tex

uv run python analysis/generate_figures.py \
  --data analysis/results/ \
  --output paper/figures/
```

## Paper build

```bash
cd paper && tectonic -r 2 main.tex
```

## Output files updated

- `analysis/results/analysis_report.md`
- `analysis/results/figure_data.json`
- `paper/figures/*.pdf` (refreshed where data available)
- `paper/sections/appendix_tables_generated.tex`
- `paper/main.pdf`
