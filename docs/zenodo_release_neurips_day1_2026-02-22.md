# Zenodo Release Note: NeurIPS Day1 Matrix (2026-02-22)

This note records the Zenodo upload payload and verification material for the
day-scale NeurIPS matrix rerun used by the paper refresh in PR #38.

## Source Run

- Artifacts root: `artifacts/neurips_matrix_day1_2026-02-21/`
- Matrix summary: `artifacts/neurips_matrix_day1_2026-02-21/matrix_summary.json`
- Completion status: 17/18 runs (missing `algebraic_connectivity_medium/seed_11`,
  recorded as failed in matrix summary)

## Zenodo Upload File

- Upload this archive:
  - `zenodo_staging/neurips_day1_2026-02-22/neurips_matrix_day1_2026-02-21_artifacts.tar.gz`
- Archive size (local): ~308 KB
- Archive SHA256:
  - `2c85883beb111233f5e8b4801eb6cd3b4e4b441aa50634437ab5f203faf939a9`

## Verification Files Committed in Git

- Archive checksum file:
  - `docs/zenodo_neurips_day1_archive_sha256_2026-02-22.txt`
- Full per-file checksum manifest (103 files):
  - `docs/zenodo_neurips_day1_sha256s_2026-02-22.txt`

## Upload Metadata Template

Use the following when publishing the Zenodo record:

- Title: `Graph Invariant Discovery: NeurIPS Day1 Matrix Raw Artifacts`
- Version: `v2026.02.22-day1`
- Resource type: `Dataset`
- Related code commit: `27ace93` (or final merge commit)
- License: `MIT` (for code) and repository data policy for artifacts
- Description: include run command from
  `docs/final_neurips_rerun_manifest_2026-02-20.md`

## Post-Upload Sync (Required)

After Zenodo DOI is assigned:

1. Replace placeholders in `paper/references.bib` dataset entry.
2. Add DOI/URL to `docs/final_neurips_rerun_manifest_2026-02-20.md`.
3. Ensure README and paper data-availability wording matches the DOI exactly.
