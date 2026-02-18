import sys
from types import SimpleNamespace

import networkx as nx
import numpy as np

from graph_invariant.baselines.features import FEATURE_ORDER, feature_order, features_from_graphs
from graph_invariant.baselines.pysr_baseline import run_pysr_baseline
from graph_invariant.baselines.stat_baselines import run_stat_baselines


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
            return np.full((x.shape[0], 1), self._mean, dtype=float)

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
        procs=2,
        timeout_in_seconds=9.0,
    )
    assert payload["status"] == "ok"
    assert payload["model"] == "pysr"
    assert "val_metrics" in payload
    assert "test_metrics" in payload
    assert captured_kwargs["niterations"] == 12
    assert captured_kwargs["populations"] == 3
    assert captured_kwargs["procs"] == 2
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


def test_run_pysr_baseline_returns_error_on_prediction_shape_mismatch(monkeypatch):
    class BadShapeRegressor:
        def __init__(self, **kwargs):
            del kwargs

        def fit(self, _x, _y):
            return None

        def predict(self, _x):
            return np.asarray([1.0, 2.0, 3.0], dtype=float)

    fake_module = SimpleNamespace(PySRRegressor=BadShapeRegressor)
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
    assert payload["error_type"] == "ValueError"


# ── Feature leakage prevention tests ────────────────────────────────


def test_features_from_graphs_excludes_target_feature():
    """When exclude_features includes a FEATURE_ORDER column, it must be dropped."""
    graphs = [nx.path_graph(5), nx.cycle_graph(6)]
    full = features_from_graphs(graphs)
    filtered = features_from_graphs(graphs, exclude_features=("algebraic_connectivity",))
    assert full.shape[1] == len(FEATURE_ORDER)
    assert filtered.shape[1] == len(FEATURE_ORDER) - 1
    assert filtered.shape[0] == 2


def test_features_from_graphs_excludes_multiple_features():
    graphs = [nx.path_graph(5)]
    filtered = features_from_graphs(
        graphs, exclude_features=("algebraic_connectivity", "spectral_radius", "diameter")
    )
    assert filtered.shape[1] == len(FEATURE_ORDER) - 3


def test_features_from_graphs_all_excluded_returns_2d_empty():
    """Excluding every feature must return shape (n_graphs, 0), not (0,)."""
    graphs = [nx.path_graph(5), nx.cycle_graph(6), nx.star_graph(4)]
    result = features_from_graphs(graphs, exclude_features=FEATURE_ORDER)
    assert result.ndim == 2
    assert result.shape == (3, 0)


def test_features_from_graphs_no_exclusion_returns_all():
    """Default (no exclusion) returns the full feature matrix."""
    graphs = [nx.path_graph(5)]
    full = features_from_graphs(graphs)
    no_exclude = features_from_graphs(graphs, exclude_features=None)
    assert full.shape == no_exclude.shape
    np.testing.assert_allclose(full, no_exclude, atol=1e-12)


def test_features_from_graphs_can_disable_spectral_pack():
    graphs = [nx.path_graph(5), nx.cycle_graph(6)]
    full = features_from_graphs(graphs, enable_spectral_feature_pack=True)
    base_only = features_from_graphs(graphs, enable_spectral_feature_pack=False)
    assert full.shape[1] == len(FEATURE_ORDER)
    assert base_only.shape[1] == len(feature_order(enable_spectral_feature_pack=False))
    assert full.shape[0] == base_only.shape[0]


def test_run_stat_baselines_accepts_target_name():
    """run_stat_baselines with target_name filters leaking features."""
    graphs = [nx.path_graph(5), nx.cycle_graph(6)]
    y = [1.0, 2.0]
    result = run_stat_baselines(
        train_graphs=graphs,
        val_graphs=graphs,
        test_graphs=graphs,
        y_train=y,
        y_val=y,
        y_test=y,
        target_name="algebraic_connectivity",
    )
    assert result["linear_regression"]["status"] == "ok"


def test_run_pysr_baseline_accepts_target_name(monkeypatch):
    """run_pysr_baseline with target_name filters leaking features."""
    captured_x_shapes: list[tuple[int, ...]] = []

    class ShapeCapturingRegressor:
        def __init__(self, **kwargs):
            del kwargs

        def fit(self, x, _y):
            captured_x_shapes.append(x.shape)

        def predict(self, x):
            return np.zeros(x.shape[0])

    fake_module = SimpleNamespace(PySRRegressor=ShapeCapturingRegressor)
    monkeypatch.setattr(
        "graph_invariant.baselines.pysr_baseline._load_pysr_module",
        lambda: fake_module,
    )
    graphs = [nx.path_graph(5), nx.cycle_graph(6)]
    payload = run_pysr_baseline(
        train_graphs=graphs,
        val_graphs=graphs,
        test_graphs=graphs,
        y_train=[1.0, 2.0],
        y_val=[1.0, 2.0],
        y_test=[1.0, 2.0],
        target_name="algebraic_connectivity",
    )
    assert payload["status"] == "ok"
    # Should have 8 features (9 - 1 excluded)
    assert captured_x_shapes[0][1] == len(FEATURE_ORDER) - 1
