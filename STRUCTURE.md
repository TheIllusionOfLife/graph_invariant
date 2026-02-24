# STRUCTURE.md

## Repository Layout

- `src/graph_invariant/`: implementation package
- `tests/`: test suite
- `configs/`: pre-built experiment configurations (quick, full, ablation profiles)
  - `configs/quick/`: short-runtime configurations for development/CI
  - `configs/full/`: production-quality experiment configurations
  - `configs/ablation/`: ablation study configurations
  - `configs/neurips_day1/`: NeurIPS day-1 matrix configurations
  - `configs/neurips_matrix/`: NeurIPS full matrix configurations
- `docs/`: historical and reference documents
  - `docs/DATA_POLICY.md`: Zenodo archival and dataset citation policy
  - `docs/peer_reviews/`: peer review notes and NeurIPS readiness status reports
- `scripts/`: automation scripts
  - `scripts/run_all_experiments.sh`: automated experiment suite runner
  - `scripts/run_neurips_matrix.py`: NeurIPS multi-seed evidence matrix runner
- `analysis/`: experiment analysis scripts and submodules
  - `analysis/analyze_experiments.py`: CLI entry point (re-exports all public names)
  - `analysis/experiment_loader.py`: artifact discovery and JSON loading
  - `analysis/experiment_analysis.py`: aggregation, convergence extraction, comparison tables
  - `analysis/report_writers.py`: markdown, LaTeX, and figure-data JSON output
  - `analysis/generate_figures.py`: publication-quality figure generation (PDF)
- `paper/`: main-conference paper (NeurIPS format)
  - `paper/main.tex`: paper entry point
  - `paper/references.bib`: verified bibliography
  - `paper/neurips_2025.sty`: NeurIPS style file
  - `paper/sections/`: section files (abstract through conclusion)
  - `paper/figures/`: generated figures (from `generate_figures.py`)
- `.github/workflows/`: CI and automation workflows
- `README.md`: human-oriented project entry point
- `AGENTS.md`: agent/contributor operational rules
- `PRODUCT.md`: product goals and user context
- `TECH.md`: stack and constraints

## Source Package Structure

- `src/graph_invariant/cli.py`
  - Thin CLI entry point: `main()` + `write_report()`
  - Subcommands: `phase1`, `report`, `benchmark`, `ood-validate`
  - Delegates to `phase1_loop.run_phase1()` for the main loop
- `src/graph_invariant/candidate_pipeline.py`
  - Per-generation candidate generation, validation, scoring, and repair
  - `_run_one_generation()`: inner loop coordinator
  - `_generate_candidate()`, `_build_repair_prompt()`, `_candidate_prompt()`
  - `_topology_descriptor()`, `_island_strategy()`, `_update_prompt_mode_after_generation()`
  - `_corr_abs()`, `_new_experiment_id()`, `_restore_rng()`, `_state_defaults()`
- `src/graph_invariant/phase1_loop.py`
  - `run_phase1()`: full Phase 1 orchestration (outer loop, checkpointing, summaries)
  - `_collect_baseline_results()`, `_write_baseline_summary()`
  - `_require_model_available()`
- `src/graph_invariant/config.py`
  - Validated runtime config dataclass (`Phase1Config`)
- `src/graph_invariant/data.py`
  - Dataset generation for train/validation/test
- `src/graph_invariant/targets.py`
  - Target value computation for configurable invariant targets
- `src/graph_invariant/sandbox.py`
  - Static checks and constrained candidate execution pool
- `src/graph_invariant/scoring.py`
  - Metrics, simplicity, novelty, total score, bound metrics
- `src/graph_invariant/map_elites.py`
  - MAP-Elites diversity archives (primary + optional topology-behavior mapping)
- `src/graph_invariant/ood_validation.py`
  - Out-of-distribution validation on large-scale and extreme-topology graphs
- `src/graph_invariant/evolution.py`
  - Island migration logic
- `src/graph_invariant/known_invariants.py`
  - Known graph invariant calculations (including optional spectral feature pack)
  - `compute_known_invariant_values()`: returns sorted degree sequence (consistent with `compute_feature_dict()`)
- `src/graph_invariant/logging_io.py`
  - JSON/JSONL writes, checkpoint persistence helpers
- `src/graph_invariant/benchmark.py`
  - Multi-seed benchmark orchestration
- `src/graph_invariant/baselines/`
  - Baseline feature extraction and baseline runners
- `src/graph_invariant/baselines/features.py`
  - Baseline feature extraction (excludes target to prevent leakage)
- `src/graph_invariant/baselines/pysr_baseline.py`
  - PySR symbolic regression baseline (optional: `pysr` extra)
- `src/graph_invariant/baselines/stat_baselines.py`
  - Statistical baselines (RandomForest, linear regression)

## Configs Directory

```text
configs/
  quick/
    phase1_experiment.json          # standard Phase 1 experiment (quick)
    phase1_v2_experiment.json       # v2 with self-correction (quick)
    phase1_v2_benchmark.json        # benchmark (quick)
    algebraic_connectivity.json     # algebraic connectivity target (quick)
    benchmark_aspl.json             # multi-seed benchmark (quick)
    map_elites_aspl.json            # MAP-Elites ASPL (quick)
    upper_bound_aspl.json           # upper bound mode (quick)
  full/
    benchmark_aspl.json                    # multi-seed benchmark (full)
    experiment_algebraic_connectivity.json # algebraic connectivity (full)
    experiment_map_elites_aspl.json        # MAP-Elites ASPL (full)
    experiment_upper_bound_aspl.json       # upper bound mode (full)
  ablation/
    sc_off_seed11.json              # self-correction disabled, seed 11
    sc_off_seed22.json              # self-correction disabled, seed 22
    sc_off_seed33.json              # self-correction disabled, seed 33
  neurips_day1/                     # NeurIPS day-1 matrix configs
  neurips_matrix/                   # NeurIPS full matrix configs
```

## Test Structure

```text
tests/
  test_cli.py                   # main() arg-parsing and benchmark dispatch (~24 LOC)
  test_candidate_pipeline.py    # unit tests for candidate_pipeline functions
  test_phase1_loop.py           # integration tests for run_phase1()
  test_report.py                # tests for write_report()
  test_config.py                # Phase1Config validation tests
  test_data.py                  # dataset generation tests
  test_scoring.py               # scoring / metrics tests
  test_known_invariants.py      # known invariant calculation tests
  test_sandbox.py               # sandbox static check tests
  test_map_elites.py            # MAP-Elites archive tests
  test_ood_validation.py        # OOD validation tests
  test_benchmark.py             # benchmark runner tests
  test_baselines.py             # baseline runner tests
  test_analyze_experiments.py   # analysis pipeline integration tests
  test_experiment_loader.py     # artifact discovery and loading tests
  test_experiment_analysis.py   # aggregation/analysis function tests
  test_report_writers.py        # markdown/LaTeX/JSON output tests
  test_pysr_baseline.py         # PySR baseline tests (requires pysr extra)
```

## Naming and Organization Conventions

- Test files: `tests/test_<module>.py`
- Public functions and variables: `snake_case`
- Classes/dataclasses: `PascalCase`
- Keep module responsibilities narrow and explicit.
- Orchestration lives in `phase1_loop.py`; per-generation logic in `candidate_pipeline.py`; CLI wiring in `cli.py`.

## Import and Dependency Patterns

- Import within package via relative imports (`from .module import ...`).
- Keep cross-module coupling low:
  - `phase1_loop.py` orchestrates across modules.
  - `candidate_pipeline.py` handles inner-loop generation/scoring.
  - `cli.py` is a thin wrapper — no business logic.
  - Lower-level modules should avoid depending on CLI or orchestration layers.
- Add optional dependency handling in baseline modules where needed.
- `pysr` is an optional dependency (`uv sync --extra pysr`); `pysr_baseline.py` uses lazy import.

## Analysis Pipeline

The `analysis/` directory has no `__init__.py`; submodules are loaded via `sys.path` insertion in the entry point.

- `analyze_experiments.py` re-exports all public names from the three submodules so that `importlib`-based test fixtures can access all functions without changes.
- Data flow: `experiment_loader` → `experiment_analysis` → `report_writers`.

## Artifact Structure (Runtime Output)

Heavy/raw outputs are archived on Zenodo for long-term storage and citation.

- Phase 1:
  - `<artifacts_dir>/logs/events.jsonl`
  - `<artifacts_dir>/checkpoints/<experiment_id>/gen_<N>.json`
  - `<artifacts_dir>/phase1_summary.json`
  - `<artifacts_dir>/baselines_summary.json` (optional)
  - Checkpoints may contain both `map_elites_archive` (legacy alias) and `map_elites_archives` (primary/topology)
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
- `docs/peer_reviews/`: peer review notes from NeurIPS submission cycle.
