# Data Policy (Zenodo)

This project uses Zenodo as the canonical archive for heavy experiment data.

## Scope

- Raw experiment outputs (full checkpoints, logs, seed-level benchmark runs, OOD outputs) are archived on Zenodo.
- The Git repository keeps code, configs, and lightweight summaries only.

## What Must Not Be Committed to Git

- Large raw artifact directories (for example: `artifacts/`, benchmark seed dumps, full JSONL logs from long runs).
- Any data that exceeds practical repository size limits or harms clone/CI performance.

## What Should Be Kept in Git

- Reproducibility code and configs (`src/`, `configs/`, scripts).
- Lightweight derived outputs used by docs/paper (small summary tables/figures).
- References to archived datasets (Zenodo DOI, record URL, and checksums/manifest pointers).

## Zenodo Archival Requirements

For each major experiment release:

1. Upload the raw dataset bundle to Zenodo.
2. Record metadata:
   - title and version tag (matching Git tag/commit),
   - creation date,
   - generator command/config profile,
   - checksums (SHA256 preferred),
   - license and access conditions.
3. Capture the DOI and record URL in repository docs.

## Citation in the Paper

- Cite the Zenodo dataset DOI in the paper's data availability section.
- Add the dataset entry to `paper/references.bib`.
- Ensure the DOI in paper text and repository docs is identical.

## Reproducibility Note

The repository should allow users to:

- rerun experiments from configs, or
- download archived raw data from Zenodo and reproduce figures/tables.
