import sys
from types import SimpleNamespace

import networkx as nx
import numpy as np

from graph_invariant.baselines.pysr_baseline import run_pysr_baseline


def test_run_pysr_baseline_skips_when_pysr_missing(monkeypatch):
    monkeypatch.setattr(
        "graph_invariant.baselines.pysr_baseline._load_pysr_module",
        lambda: None,
    )
    payload = run_pysr_baseline(
        train_graphs=[nx.path_graph(4)],
        val_graphs=[nx.path_graph(4)],
        test_graphs=[nx.path_graph(4)],
        y_train=[1.0],
        y_val=[1.0],
        y_test=[1.0],
    )
    assert payload == {"status": "skipped", "reason": "pysr not installed"}


def test_run_pysr_baseline_runs_with_fake_module(monkeypatch):
    captured_kwargs: dict[str, object] = {}

    class FakeRegressor:
        def __init__(self, **kwargs):
            captured_kwargs.update(kwargs)
            self._mean = 0.0

        def fit(self, _x, y):
            self._mean = float(np.mean(y))

        def predict(self, x):
            return np.full((x.shape[0],), self._mean, dtype=float)

    fake_module = SimpleNamespace(PySRRegressor=FakeRegressor)
    monkeypatch.setitem(sys.modules, "pysr", fake_module)
    monkeypatch.setattr(
        "graph_invariant.baselines.pysr_baseline._load_pysr_module",
        lambda: fake_module,
    )
    payload = run_pysr_baseline(
        train_graphs=[nx.path_graph(4), nx.path_graph(5)],
        val_graphs=[nx.path_graph(4)],
        test_graphs=[nx.path_graph(5)],
        y_train=[1.0, 2.0],
        y_val=[1.5],
        y_test=[1.5],
        niterations=12,
        populations=3,
        timeout_in_seconds=9.0,
    )
    assert payload["status"] == "ok"
    assert payload["model"] == "pysr"
    assert "val_metrics" in payload
    assert "test_metrics" in payload
    assert captured_kwargs["niterations"] == 12
    assert captured_kwargs["populations"] == 3
    assert captured_kwargs["timeout_in_seconds"] == 9.0


def test_run_pysr_baseline_returns_error_status_on_fit_failure(monkeypatch):
    class FailingRegressor:
        def __init__(self, **kwargs):
            del kwargs

        def fit(self, _x, _y):
            raise RuntimeError("boom")

        def predict(self, _x):
            raise RuntimeError("unreachable")

    fake_module = SimpleNamespace(PySRRegressor=FailingRegressor)
    monkeypatch.setattr(
        "graph_invariant.baselines.pysr_baseline._load_pysr_module",
        lambda: fake_module,
    )
    payload = run_pysr_baseline(
        train_graphs=[nx.path_graph(4)],
        val_graphs=[nx.path_graph(4)],
        test_graphs=[nx.path_graph(4)],
        y_train=[1.0],
        y_val=[1.0],
        y_test=[1.0],
    )
    assert payload["status"] == "error"
    assert payload["error_type"] == "RuntimeError"
