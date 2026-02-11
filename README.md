# graph_invariant

Phase 1 MVP for LLM-driven graph invariant discovery.

## Quickstart

1. Install dependencies with `uv`:

```bash
uv sync --group dev
```

2. Run tests and lint:

```bash
uv run --group dev pytest
uv run --group dev ruff check .
uv run --group dev ruff format --check .
```

These commands mirror the `Python CI` GitHub Actions workflow.

3. Run a small Phase 1 execution:

```bash
cat > /tmp/phase1_config.json <<'JSON'
{"num_train_graphs": 2, "num_val_graphs": 3, "num_test_graphs": 2, "timeout_sec": 0.5, "artifacts_dir": "artifacts_smoke"}
JSON
uv run python -m graph_invariant.cli phase1 --config /tmp/phase1_config.json
```

Artifacts are written under `artifacts*/logs` and `artifacts*/checkpoints`.
Additional summary artifacts:
- `artifacts*/phase1_summary.json`
- `artifacts*/baselines_summary.json` (only when `run_baselines=true`)

4. Generate a markdown report from artifacts:

```bash
uv run python -m graph_invariant.cli report --artifacts artifacts_smoke
```

This writes `artifacts_smoke/report.md`.

5. Run a multi-seed benchmark sweep:

```bash
cat > /tmp/benchmark_config.json <<'JSON'
{"benchmark_seeds": [11, 22, 33], "max_generations": 0, "run_baselines": true, "artifacts_dir": "artifacts_benchmark"}
JSON
uv run python -m graph_invariant.cli benchmark --config /tmp/benchmark_config.json
```

This writes one benchmark directory under `artifacts_benchmark/benchmark_*` with:
- `benchmark_summary.json`
- `benchmark_report.md`
- per-seed subdirectories (`seed_<N>/`) containing regular Phase 1 artifacts.

## Optional Baseline Dependencies

`run_baselines` is disabled by default. If enabled:
- Linear regression baseline uses NumPy only.
- Random forest baseline requires `scikit-learn`.
- PySR baseline requires `pysr` and a Julia runtime.

## Security Note

`src/graph_invariant/sandbox.py` is a best-effort research sandbox. It uses static AST checks plus constrained execution with resource and timeout limits, but it is not a full security boundary. For production-grade untrusted execution, run candidates inside stronger OS/container isolation.
