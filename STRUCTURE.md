# STRUCTURE.md

## Repository Layout

- `src/graph_invariant/`: implementation package
- `tests/`: test suite
- `configs/`: pre-built experiment configurations (quick and full profiles)
- `docs/`: historical and reference documents
- `analysis/`: experiment analysis scripts and notes
  - `analysis/analyze_experiments.py`: cross-experiment analysis and report generation
  - `analysis/generate_figures.py`: publication-quality figure generation (PDF)
- `paper/`: main-conference paper (NeurIPS format)
  - `paper/main.tex`: paper entry point
  - `paper/references.bib`: verified bibliography
  - `paper/neurips_2025.sty`: NeurIPS style file
  - `paper/sections/`: section files (abstract through conclusion)
  - `paper/figures/`: generated figures (from `generate_figures.py`)
- `.github/workflows/`: CI and automation workflows
- `run_all_experiments.sh`: automated experiment suite runner
- `README.md`: human-oriented project entry point
- `AGENTS.md`: agent/contributor operational rules
- `PRODUCT.md`: product goals and user context
- `TECH.md`: stack and constraints

## Source Package Structure

- `src/graph_invariant/cli.py`
  - CLI entry points and phase orchestration
  - Subcommands: `phase1`, `report`, `benchmark`, `ood-validate`
- `src/graph_invariant/config.py`
  - validated runtime config dataclass (`Phase1Config`)
- `src/graph_invariant/data.py`
  - dataset generation for train/validation/test
- `src/graph_invariant/targets.py`
  - target value computation for configurable invariant targets
- `src/graph_invariant/sandbox.py`
  - static checks and constrained candidate execution pool
- `src/graph_invariant/scoring.py`
  - metrics, simplicity, novelty, total score, bound metrics
- `src/graph_invariant/map_elites.py`
  - MAP-Elites diversity archive (grid-based behavioral mapping)
- `src/graph_invariant/ood_validation.py`
  - out-of-distribution validation on large-scale and extreme-topology graphs
- `src/graph_invariant/evolution.py`
  - island migration logic
- `src/graph_invariant/known_invariants.py`
  - known graph invariant calculations
- `src/graph_invariant/logging_io.py`
  - JSON/JSONL writes, checkpoint persistence helpers
- `src/graph_invariant/benchmark.py`
  - multi-seed benchmark orchestration
- `src/graph_invariant/baselines/`
  - baseline feature extraction and baseline runners
- `src/graph_invariant/baselines/features.py`
  - baseline feature extraction (excludes target to prevent leakage)
- `src/graph_invariant/baselines/pysr_baseline.py`
  - PySR symbolic regression baseline
- `src/graph_invariant/baselines/stat_baselines.py`
  - statistical baselines (RandomForest, linear regression)

## Configs Directory

- `configs/phase1_experiment.json`: standard Phase 1 experiment
- `configs/phase1_v2_experiment.json`: v2 experiment with self-correction
- `configs/experiment_map_elites_aspl.json`: MAP-Elites ASPL (full)
- `configs/experiment_algebraic_connectivity.json`: algebraic connectivity target (full)
- `configs/experiment_upper_bound_aspl.json`: upper bound mode (full)
- `configs/benchmark_aspl.json`: multi-seed benchmark (full)
- `configs/quick_*.json`: quick-profile variants of the above

## Naming and Organization Conventions

- Test files: `tests/test_<module>.py`
- Public functions and variables: `snake_case`
- Classes/dataclasses: `PascalCase`
- Keep module responsibilities narrow and explicit.

## Import and Dependency Patterns

- Import within package via relative imports (`from .module import ...`).
- Keep cross-module coupling low:
  - `cli.py` can orchestrate across modules.
  - lower-level modules should avoid depending on CLI.
- Add optional dependency handling in baseline modules where needed.

## Artifact Structure (Runtime Output)

- Phase 1:
  - `<artifacts_dir>/logs/events.jsonl`
  - `<artifacts_dir>/checkpoints/<experiment_id>/gen_<N>.json`
  - `<artifacts_dir>/phase1_summary.json`
  - `<artifacts_dir>/baselines_summary.json` (optional)
- OOD Validation:
  - `<artifacts_dir>/ood/ood_validation.json`
- Benchmark:
  - `<artifacts_dir>/benchmark_<timestamp>/benchmark_summary.json`
  - `<artifacts_dir>/benchmark_<timestamp>/benchmark_report.md`
  - `<artifacts_dir>/benchmark_<timestamp>/seed_<N>/...`

## Historical Documents

- `docs/SPEC.md`: authoritative implementation reference.
- `docs/REVIEW.md`: historical technical review notes.
- `docs/Research_Plan_Graph_Invariant_Discovery.md`: original proposal document.
