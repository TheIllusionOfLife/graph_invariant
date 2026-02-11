# STRUCTURE.md

## Repository Layout

- `src/graph_invariant/`: implementation package
- `tests/`: test suite
- `.github/workflows/`: CI and automation workflows
- `README.md`: human-oriented project entry point
- `AGENTS.md`: agent/contributor operational rules
- `PRODUCT.md`: product goals and user context
- `TECH.md`: stack and constraints
- `SPEC.md`: detailed implementation spec
- `REVIEW.md`: historical technical review notes
- `Research_Plan_Graph_Invariant_Discovery.md`: original proposal document

## Source Package Structure

- `src/graph_invariant/cli.py`
  - CLI entry points and phase orchestration
- `src/graph_invariant/config.py`
  - validated runtime config dataclass (`Phase1Config`)
- `src/graph_invariant/data.py`
  - dataset generation for train/validation/test
- `src/graph_invariant/sandbox.py`
  - static checks and constrained candidate execution pool
- `src/graph_invariant/scoring.py`
  - metrics, simplicity, novelty, total score
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
- Benchmark:
  - `<artifacts_dir>/benchmark_<timestamp>/benchmark_summary.json`
  - `<artifacts_dir>/benchmark_<timestamp>/benchmark_report.md`
  - `<artifacts_dir>/benchmark_<timestamp>/seed_<N>/...`

## Historical Documents

- `REVIEW.md` and `Research_Plan_Graph_Invariant_Discovery.md` are retained for historical context.
- `SPEC.md` is the authoritative implementation reference.
