from __future__ import annotations

import networkx as nx


def run_pysr_baseline(
    train_graphs: list[nx.Graph],
    val_graphs: list[nx.Graph],
    test_graphs: list[nx.Graph],
    y_train: list[float],
    y_val: list[float],
    y_test: list[float],
) -> dict[str, object]:
    del train_graphs, val_graphs, test_graphs, y_train, y_val, y_test
    try:
        import pysr  # noqa: F401
    except Exception:
        return {"status": "skipped", "reason": "pysr not installed"}
    return {"status": "skipped", "reason": "pysr baseline not yet configured"}
