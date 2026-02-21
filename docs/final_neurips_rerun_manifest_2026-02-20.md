# Final NeurIPS Rerun Manifest (2026-02-20)

This manifest records the exact commands and artifact paths used for the final
analysis/paper refresh pass on branch `chore/final-matrix-results-refresh`.

## Branch and base

- Base commit on `main`: `6debfe3fe69f236bb714b490c1cb2f67b690493a`
- Working branch: `chore/final-matrix-results-refresh`

## Matrix rerun (day-scale matrix)

Command:

```bash
uv run python scripts/run_neurips_matrix.py \
  --configs \
    configs/neurips_day1/map_elites_aspl_medium.json \
    configs/neurips_day1/algebraic_connectivity_medium.json \
    configs/neurips_day1/upper_bound_aspl_medium.json \
    configs/neurips_day1/small_data_aspl_train20_medium.json \
    configs/neurips_day1/small_data_aspl_train35_medium.json \
    configs/neurips_day1/benchmark_aspl_medium.json \
  --seeds 11 22 33 \
  --max-parallel 2 \
  --output-root artifacts/neurips_matrix_day1_2026-02-21
```

Completion timestamp: 2026-02-22 07:47:07 JST

Matrix summary path:

- `artifacts/neurips_matrix_day1_2026-02-21/matrix_summary.json`

Run completion status:

- Completed runs: 17/18
- Incomplete run: `algebraic_connectivity_medium/seed_11` (pathological runtime hang; recorded as failed status in matrix summary)

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

## Output files refreshed

- `analysis/results/*`
- `paper/figures/*`
- `paper/main.pdf`
- `docs/final_neurips_rerun_manifest_2026-02-20.md`

## Zenodo archival handoff (prepared)

- Local archive bundle (ignored by Git, upload this file to Zenodo):
  - `zenodo_staging/neurips_day1_2026-02-22/neurips_matrix_day1_2026-02-21_artifacts.tar.gz`
- Archive SHA256 (tracked):
  - `docs/zenodo_neurips_day1_archive_sha256_2026-02-22.txt`
- Per-file SHA256 manifest for raw artifacts (tracked):
  - `docs/zenodo_neurips_day1_sha256s_2026-02-22.txt`
- Raw artifact source included in the bundle:
  - `artifacts/neurips_matrix_day1_2026-02-21/`
- Published Zenodo record:
  - DOI: `10.5281/zenodo.18727765`
  - URL: `https://zenodo.org/record/18727765`
