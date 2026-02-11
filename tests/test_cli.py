from graph_invariant.cli import run_phase1
from graph_invariant.config import Phase1Config
from graph_invariant.data import DatasetBundle


def test_run_phase1_uses_configured_score_weights(monkeypatch, tmp_path):
    import networkx as nx

    cfg = Phase1Config(
        artifacts_dir=str(tmp_path / "artifacts"),
        alpha=0.9,
        beta=0.05,
        gamma=0.05,
        timeout_sec=0.2,
        num_train_graphs=1,
        num_val_graphs=2,
        num_test_graphs=1,
    )
    bundle = DatasetBundle(
        train=[nx.path_graph(4)],
        val=[nx.path_graph(4), nx.path_graph(5)],
        test=[nx.path_graph(4)],
        sanity=[nx.path_graph(4)],
    )
    captured: dict[str, float] = {}

    def fake_total(abs_spearman, simplicity, novelty_bonus, alpha, beta, gamma):  # noqa: ANN001
        captured["alpha"] = alpha
        captured["beta"] = beta
        captured["gamma"] = gamma
        return 0.5

    monkeypatch.setattr("graph_invariant.cli.generate_phase1_datasets", lambda _cfg: bundle)
    monkeypatch.setattr(
        "graph_invariant.cli.evaluate_candidate_on_graphs", lambda *_args, **_kw: [1.0, 2.0]
    )
    monkeypatch.setattr("graph_invariant.cli.compute_total_score", fake_total)

    assert run_phase1(cfg) == 0
    assert captured == {"alpha": 0.9, "beta": 0.05, "gamma": 0.05}
