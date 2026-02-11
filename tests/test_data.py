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
