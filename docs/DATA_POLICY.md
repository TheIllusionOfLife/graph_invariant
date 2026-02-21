# Data Policy (Zenodo)

This project uses Zenodo as the canonical archive for heavy experiment data.

## Publication Split (Mandatory)

For paper submissions, artifact publication is split across two channels:

1. GitHub repository/PR (lightweight, reviewable):
   - paper source and compiled PDF,
   - analysis/figure generation scripts,
   - lightweight derived outputs used in the manuscript,
   - run configs and manifests that document how experiments were executed.
2. Zenodo record (heavy, immutable, citable):
   - raw experiment outputs and run directories under `artifacts/`,
   - seed-level logs/checkpoints and benchmark/OOD raw outputs,
   - matrix summaries and raw per-run summaries needed for independent verification.

This split is required for reproducibility, reviewability, and long-term archival.

## Scope

- Raw experiment outputs (full checkpoints, logs, seed-level benchmark runs, OOD outputs) are archived on Zenodo.
- The Git repository keeps code, configs, and lightweight summaries only.

## What Must Not Be Committed to Git

- Large raw artifact directories (for example: `artifacts/`, benchmark seed dumps, full JSONL logs from long runs).
- Any data that exceeds practical repository size limits or harms clone/CI performance.

## What Should Be Kept in Git

- Reproducibility code and configs (`src/`, `configs/`, scripts).
- Lightweight derived outputs used by docs/paper (small summary tables/figures).
- Paper assets needed for review (`paper/main.tex`, section files, generated figures, `paper/main.pdf`).
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

## Paper-Ready Release Checklist (Required)

Before calling a paper PR submission-ready:

1. Ensure GitHub PR contains the lightweight paper package:
   - updated manuscript source and `paper/main.pdf`,
   - generated paper figures and appendix/table sources,
   - analysis summaries used by the paper text.
2. Ensure Zenodo bundle contains raw experimental evidence:
   - all relevant `artifacts/` directories for reported results,
   - matrix summary and seed-level outputs,
   - checksum manifest for uploaded files.
3. Cross-link artifacts:
   - add Zenodo DOI/URL to repository docs and `paper/references.bib`,
   - ensure commands/configs in docs reproduce the archived run family.

Example implementation for this repository:

- `docs/zenodo_release_neurips_day1_2026-02-22.md`
- `docs/zenodo_neurips_day1_archive_sha256_2026-02-22.txt`
- `docs/zenodo_neurips_day1_sha256s_2026-02-22.txt`

## Citation in the Paper

- Cite the Zenodo dataset DOI in the paper's data availability section.
- Add the dataset entry to `paper/references.bib`.
- Ensure the DOI in paper text and repository docs is identical.

## Reproducibility Note

The repository should allow users to:

- rerun experiments from configs, or
- download archived raw data from Zenodo and reproduce figures/tables.
