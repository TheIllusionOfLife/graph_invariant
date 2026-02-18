import networkx as nx
import pytest

from graph_invariant.config import Phase1Config
from graph_invariant.data import generate_phase1_datasets


def test_generate_phase1_datasets_has_expected_sizes():
    cfg = Phase1Config(num_train_graphs=4, num_val_graphs=5, num_test_graphs=6, seed=11)
    bundle = generate_phase1_datasets(cfg)
    assert len(bundle.train) == 4
    assert len(bundle.val) == 5
    assert len(bundle.test) == 6
    assert len(bundle.sanity) == 3


def test_generate_phase1_datasets_is_deterministic():
    cfg = Phase1Config(num_train_graphs=3, num_val_graphs=3, num_test_graphs=3, seed=123)
    a = generate_phase1_datasets(cfg)
    b = generate_phase1_datasets(cfg)
    a_edges = [g.number_of_edges() for g in a.train]
    b_edges = [g.number_of_edges() for g in b.train]
    assert a_edges == b_edges


def test_generate_phase1_datasets_builds_special_pool_once(monkeypatch):
    calls = {"count": 0}

    def _fake_pool():
        calls["count"] += 1
        return [nx.path_graph(4) for _ in range(8)]

    monkeypatch.setattr("graph_invariant.data.generate_ood_special_topology", _fake_pool)
    cfg = Phase1Config(
        num_train_graphs=10,
        num_val_graphs=10,
        num_test_graphs=2,
        ood_train_special_topology_ratio=0.2,
        ood_val_special_topology_ratio=0.2,
        seed=7,
    )
    _ = generate_phase1_datasets(cfg)
    assert calls["count"] == 1


def test_generate_phase1_datasets_warns_for_high_special_topology_ratio():
    cfg = Phase1Config(
        num_train_graphs=10,
        num_val_graphs=10,
        num_test_graphs=2,
        ood_train_special_topology_ratio=0.3,
        ood_val_special_topology_ratio=0.1,
        seed=7,
    )
    with pytest.warns(UserWarning, match="recommended maximum 0.2"):
        _ = generate_phase1_datasets(cfg)
