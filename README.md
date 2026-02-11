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

3. Run a small Phase 1 execution:

```bash
cat > /tmp/phase1_config.json <<'JSON'
{"num_train_graphs": 2, "num_val_graphs": 3, "num_test_graphs": 2, "timeout_sec": 0.5, "artifacts_dir": "artifacts_smoke"}
JSON
uv run python -m graph_invariant.cli phase1 --config /tmp/phase1_config.json
```

Artifacts are written under `artifacts*/logs` and `artifacts*/checkpoints`.

## Security Note

`src/graph_invariant/sandbox.py` is a best-effort research sandbox. It uses static AST checks plus constrained execution with resource and timeout limits, but it is not a full security boundary. For production-grade untrusted execution, run candidates inside stronger OS/container isolation.
