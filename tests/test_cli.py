import json
from pathlib import Path

import pytest

from graph_invariant.cli import run_phase1
from graph_invariant.config import Phase1Config
from graph_invariant.data import DatasetBundle
from graph_invariant.types import EvaluationResult


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
        "graph_invariant.cli.list_available_models",
        lambda *_args, **_kwargs: ["gpt-oss:20b"],
    )
    monkeypatch.setattr(
        "graph_invariant.cli.generate_candidate_code",
        lambda *_args, **_kwargs: "def new_invariant(G):\n    return 1.0",
    )
    monkeypatch.setattr(
        "graph_invariant.cli.evaluate_candidate_on_graphs",
        lambda _code, graphs, **_kw: [float(i + 1) for i in range(len(graphs))],
    )
    monkeypatch.setattr(
        "graph_invariant.cli.compute_metrics",
        lambda *_args, **_kwargs: EvaluationResult(0.9, 0.9, 0.1, 0.1, 2, 0),
    )
    monkeypatch.setattr("graph_invariant.cli.compute_total_score", fake_total)
    monkeypatch.setattr("graph_invariant.cli.compute_novelty_bonus", lambda *_args, **_kwargs: 0.7)

    assert run_phase1(cfg) == 0
    assert captured == {"alpha": 0.9, "beta": 0.05, "gamma": 0.05}


def test_run_phase1_fails_fast_when_ollama_model_is_missing(monkeypatch, tmp_path):
    import networkx as nx

    cfg = Phase1Config(
        artifacts_dir=str(tmp_path / "artifacts"),
        max_generations=0,
    )
    bundle = DatasetBundle(
        train=[nx.path_graph(4)],
        val=[nx.path_graph(4)],
        test=[nx.path_graph(4)],
        sanity=[nx.path_graph(4)],
    )
    monkeypatch.setattr("graph_invariant.cli.generate_phase1_datasets", lambda _cfg: bundle)
    monkeypatch.setattr(
        "graph_invariant.cli.list_available_models",
        lambda *_args, **_kwargs: ["gemma3:4b"],
    )

    with pytest.raises(RuntimeError, match="gpt-oss:20b"):
        run_phase1(cfg)


def test_run_phase1_rotates_generation_checkpoints(monkeypatch, tmp_path):
    import networkx as nx

    cfg = Phase1Config(
        artifacts_dir=str(tmp_path / "artifacts"),
        max_generations=4,
        population_size=1,
        migration_interval=10,
        checkpoint_keep_last=3,
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
    monkeypatch.setattr("graph_invariant.cli.generate_phase1_datasets", lambda _cfg: bundle)
    monkeypatch.setattr(
        "graph_invariant.cli.list_available_models",
        lambda *_args, **_kwargs: ["gpt-oss:20b"],
    )
    monkeypatch.setattr(
        "graph_invariant.cli.generate_candidate_code",
        lambda *_args, **_kwargs: "def new_invariant(G):\n    return 1.0",
    )
    monkeypatch.setattr(
        "graph_invariant.cli.evaluate_candidate_on_graphs",
        lambda _code, graphs, **_kw: [float(i + 1) for i in range(len(graphs))],
    )
    monkeypatch.setattr(
        "graph_invariant.cli.compute_metrics",
        lambda *_args, **_kwargs: EvaluationResult(0.9, 0.9, 0.1, 0.1, 2, 0),
    )
    monkeypatch.setattr("graph_invariant.cli.compute_total_score", lambda *_args, **_kwargs: 0.8)
    monkeypatch.setattr("graph_invariant.cli.compute_novelty_bonus", lambda *_args, **_kwargs: 0.4)

    assert run_phase1(cfg) == 0

    ckpt_root = Path(cfg.artifacts_dir) / "checkpoints"
    experiment_dirs = [p for p in ckpt_root.iterdir() if p.is_dir()]
    assert len(experiment_dirs) == 1
    ckpt_files = sorted(experiment_dirs[0].glob("gen_*.json"))
    assert [p.name for p in ckpt_files] == ["gen_2.json", "gen_3.json", "gen_4.json"]


def test_run_phase1_resume_continues_from_saved_generation(monkeypatch, tmp_path):
    import networkx as nx

    cfg = Phase1Config(
        artifacts_dir=str(tmp_path / "artifacts"),
        max_generations=2,
        population_size=1,
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
    monkeypatch.setattr("graph_invariant.cli.generate_phase1_datasets", lambda _cfg: bundle)
    monkeypatch.setattr(
        "graph_invariant.cli.list_available_models",
        lambda *_args, **_kwargs: ["gpt-oss:20b"],
    )
    monkeypatch.setattr(
        "graph_invariant.cli.generate_candidate_code",
        lambda *_args, **_kwargs: "def new_invariant(G):\n    return 1.0",
    )
    monkeypatch.setattr(
        "graph_invariant.cli.evaluate_candidate_on_graphs",
        lambda _code, graphs, **_kw: [float(i + 1) for i in range(len(graphs))],
    )
    monkeypatch.setattr(
        "graph_invariant.cli.compute_metrics",
        lambda *_args, **_kwargs: EvaluationResult(0.9, 0.9, 0.1, 0.1, 2, 0),
    )
    monkeypatch.setattr("graph_invariant.cli.compute_total_score", lambda *_args, **_kwargs: 0.8)
    monkeypatch.setattr("graph_invariant.cli.compute_novelty_bonus", lambda *_args, **_kwargs: 0.4)

    assert run_phase1(cfg) == 0
    ckpt_root = Path(cfg.artifacts_dir) / "checkpoints"
    experiment_dir = next(ckpt_root.iterdir())
    resume_path = experiment_dir / "gen_2.json"

    cfg_resume = Phase1Config(
        artifacts_dir=cfg.artifacts_dir,
        max_generations=3,
        population_size=1,
        num_train_graphs=1,
        num_val_graphs=2,
        num_test_graphs=1,
    )
    assert run_phase1(cfg_resume, resume=str(resume_path)) == 0

    gen3 = json.loads((experiment_dir / "gen_3.json").read_text(encoding="utf-8"))
    assert gen3["generation"] == 3


def test_run_phase1_rejects_invalid_experiment_id(monkeypatch, tmp_path):
    import networkx as nx

    cfg = Phase1Config(
        artifacts_dir=str(tmp_path / "artifacts"),
        max_generations=0,
        experiment_id="../escape",
    )
    bundle = DatasetBundle(
        train=[nx.path_graph(4)],
        val=[nx.path_graph(4)],
        test=[nx.path_graph(4)],
        sanity=[nx.path_graph(4)],
    )
    monkeypatch.setattr("graph_invariant.cli.generate_phase1_datasets", lambda _cfg: bundle)
    monkeypatch.setattr(
        "graph_invariant.cli.list_available_models",
        lambda *_args, **_kwargs: ["gpt-oss:20b"],
    )

    with pytest.raises(ValueError, match="experiment_id"):
        run_phase1(cfg)
