# Zenodo Release Note: Harmony Discovery v1 (2026-02-27)

This note records the Zenodo draft upload for the Harmony-Driven Theory
Discovery experiment artifacts used in the NeurIPS 2025 submission.

## Source Runs

- Artifacts root: `artifacts/harmony/{astronomy,physics,materials}/`
- Domains: astronomy, physics, materials science
- Generations: 20 per domain
- Seed: 42
- Model: gpt-oss:20b (local Ollama)

## Zenodo Draft Record

- Record ID: `18795697`
- DOI: `10.5281/zenodo.18795697`
- Draft URL: `https://zenodo.org/deposit/18795697`
- Status: **DRAFT** (do not publish until final submission)

## Upload File

- Archive: `zenodo_staging/harmony_v1/harmony_experiments.tar.gz`
- Archive size: ~11 KB
- Archive SHA256: `eaea52269e00ef3ea8be79d4a3b5f9212fd2d5e3ca86a2710edba66b4c618c27`
- Checksum file: `docs/zenodo_harmony_v1_archive_sha256.txt`

## Upload Metadata

- Title: Harmony-Driven Theory Discovery: Experiment Artifacts
- Version: v1.0
- Resource type: Dataset
- License: MIT
- Keywords: knowledge graphs, theory discovery, harmony metric
- Metadata JSON: `docs/zenodo_harmony_v1_metadata.json`

## Verification Files Committed in Git

- Archive checksum: `docs/zenodo_harmony_v1_archive_sha256.txt`
- Metadata JSON: `docs/zenodo_harmony_v1_metadata.json`

## Next Steps

1. Re-run live gate with entity-grounded prompts for improved results
2. Re-upload updated artifacts to the same draft (or create new version)
3. Review draft at the URL above before publishing
4. Publish only at final submission time (per `docs/DATA_POLICY.md`)
5. After publishing, update `paper/references.bib` with the final DOI
