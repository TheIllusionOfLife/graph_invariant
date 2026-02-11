import hashlib
import json
from pathlib import Path

import pytest

from graph_invariant.cli import run_phase1
from graph_invariant.config import Phase1Config
from graph_invariant.data import DatasetBundle
from graph_invariant.types import EvaluationResult


def _patch_sandbox_evaluator(monkeypatch, evaluate_fn):  # noqa: ANN001
    class FakeSandboxEvaluator:
        def __init__(
            self,
            timeout_sec: float,
            memory_mb: int,
            max_workers: int | None = None,
        ):
            self.timeout_sec = timeout_sec
            self.memory_mb = memory_mb
            self.max_workers = max_workers

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            del exc_type, exc, tb
            return False

        def evaluate(self, code, graphs):  # noqa: ANN001
            return evaluate_fn(
                code,
                graphs,
                timeout_sec=self.timeout_sec,
                memory_mb=self.memory_mb,
            )

    monkeypatch.setattr("graph_invariant.cli.SandboxEvaluator", FakeSandboxEvaluator)


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
    _patch_sandbox_evaluator(
        monkeypatch,
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
    _patch_sandbox_evaluator(
        monkeypatch,
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
    _patch_sandbox_evaluator(
        monkeypatch,
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


def test_run_phase1_writes_final_summary_with_test_metrics(monkeypatch, tmp_path):
    import networkx as nx

    cfg = Phase1Config(
        artifacts_dir=str(tmp_path / "artifacts"),
        max_generations=1,
        population_size=1,
        num_train_graphs=2,
        num_val_graphs=2,
        num_test_graphs=2,
        run_baselines=False,
    )
    bundle = DatasetBundle(
        train=[nx.path_graph(4), nx.path_graph(5)],
        val=[nx.path_graph(4), nx.path_graph(5)],
        test=[nx.path_graph(4), nx.path_graph(5)],
        sanity=[nx.path_graph(4)],
    )
    monkeypatch.setattr("graph_invariant.cli.generate_phase1_datasets", lambda _cfg: bundle)
    monkeypatch.setattr(
        "graph_invariant.cli.list_available_models",
        lambda *_args, **_kwargs: ["gpt-oss:20b"],
    )
    monkeypatch.setattr(
        "graph_invariant.cli.generate_candidate_code",
        lambda *_args, **_kwargs: "def new_invariant(G):\n    return float(G.number_of_nodes())",
    )
    _patch_sandbox_evaluator(
        monkeypatch,
        lambda _code, graphs, **_kw: [float(i + 1) for i in range(len(graphs))],
    )
    monkeypatch.setattr(
        "graph_invariant.cli.compute_metrics",
        lambda *_args, **_kwargs: EvaluationResult(0.9, 0.9, 0.1, 0.1, 2, 0),
    )
    monkeypatch.setattr("graph_invariant.cli.compute_total_score", lambda *_args, **_kwargs: 0.8)
    monkeypatch.setattr("graph_invariant.cli.compute_novelty_bonus", lambda *_args, **_kwargs: 0.4)

    assert run_phase1(cfg) == 0

    summary_path = Path(cfg.artifacts_dir) / "phase1_summary.json"
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    expected_code = "def new_invariant(G):\n    return float(G.number_of_nodes())"
    assert "test_metrics" in payload
    assert "val_metrics" in payload
    assert "success" in payload
    assert payload["schema_version"] == 3
    assert "best_candidate_code" not in payload
    assert payload["model_name"] == "gpt-oss:20b"
    assert payload["config"]["num_train_graphs"] == 2
    assert payload["config"]["memory_mb"] == cfg.memory_mb
    assert payload["config"]["island_temperatures"] == list(cfg.island_temperatures)
    assert (
        payload["best_candidate_code_sha256"]
        == hashlib.sha256(expected_code.encode("utf-8")).hexdigest()
    )
    assert payload["stop_reason"] in {"max_generations_reached", "early_stop"}


def test_run_phase1_activates_constrained_prompt_after_stagnation(monkeypatch, tmp_path):
    import networkx as nx

    cfg = Phase1Config(
        artifacts_dir=str(tmp_path / "artifacts"),
        max_generations=3,
        population_size=1,
        num_train_graphs=1,
        num_val_graphs=1,
        num_test_graphs=1,
        stagnation_trigger_generations=2,
        early_stop_patience=10,
        run_baselines=False,
    )
    bundle = DatasetBundle(
        train=[nx.path_graph(4)],
        val=[nx.path_graph(4)],
        test=[nx.path_graph(4)],
        sanity=[nx.path_graph(4)],
    )
    prompts: list[str] = []

    def fake_generate(prompt: str, *_args, **_kwargs) -> str:
        prompts.append(prompt)
        return "def new_invariant(G):\n    return 1.0"

    monkeypatch.setattr("graph_invariant.cli.generate_phase1_datasets", lambda _cfg: bundle)
    monkeypatch.setattr(
        "graph_invariant.cli.list_available_models",
        lambda *_args, **_kwargs: ["gpt-oss:20b"],
    )
    monkeypatch.setattr("graph_invariant.cli.generate_candidate_code", fake_generate)
    _patch_sandbox_evaluator(
        monkeypatch,
        lambda _code, graphs, **_kw: [None for _ in graphs],
    )

    assert run_phase1(cfg) == 0
    assert any("Use only these operators" in prompt for prompt in prompts)


def test_run_phase1_writes_baseline_summary(monkeypatch, tmp_path):
    import networkx as nx

    cfg = Phase1Config(
        artifacts_dir=str(tmp_path / "artifacts"),
        max_generations=0,
        num_train_graphs=1,
        num_val_graphs=1,
        num_test_graphs=1,
        run_baselines=True,
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

    assert run_phase1(cfg) == 0
    summary_path = Path(cfg.artifacts_dir) / "baselines_summary.json"
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == 1
    assert "stat_baselines" in payload
    assert "pysr_baseline" in payload


def test_main_report_command_writes_markdown(monkeypatch, tmp_path, capsys):
    artifacts_dir = tmp_path / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    (artifacts_dir / "phase1_summary.json").write_text(
        json.dumps({"success": True, "best_candidate_id": "c1"}),
        encoding="utf-8",
    )
    (artifacts_dir / "baselines_summary.json").write_text(
        json.dumps({"stat_baselines": {"linear_regression": {"status": "ok"}}}),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "sys.argv",
        ["graph_invariant", "report", "--artifacts", str(artifacts_dir)],
    )
    from graph_invariant import cli

    assert cli.main() == 0
    assert "Report written to" in capsys.readouterr().out
    assert (artifacts_dir / "report.md").exists()


def test_main_report_command_handles_schema_v2_summary(monkeypatch, tmp_path):
    artifacts_dir = tmp_path / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    (artifacts_dir / "phase1_summary.json").write_text(
        json.dumps(
            {
                "schema_version": 2,
                "success": True,
                "best_candidate_id": "c1",
                "stop_reason": "early_stop",
                "val_metrics": {"spearman": 0.9},
                "test_metrics": {"spearman": 0.85},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "sys.argv",
        ["graph_invariant", "report", "--artifacts", str(artifacts_dir)],
    )
    from graph_invariant import cli

    assert cli.main() == 0
    report = (artifacts_dir / "report.md").read_text(encoding="utf-8")
    assert "Validation Spearman: 0.9" in report
    assert "Test Spearman: 0.85" in report


def test_main_report_command_tolerates_malformed_json(monkeypatch, tmp_path):
    artifacts_dir = tmp_path / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    (artifacts_dir / "phase1_summary.json").write_text("{bad json", encoding="utf-8")
    monkeypatch.setattr(
        "sys.argv",
        ["graph_invariant", "report", "--artifacts", str(artifacts_dir)],
    )
    from graph_invariant import cli

    assert cli.main() == 0
    report = (artifacts_dir / "report.md").read_text(encoding="utf-8")
    assert "Phase 1 Report" in report


def test_constrained_mode_allows_late_recovery_by_default():
    from graph_invariant.cli import _update_prompt_mode_after_generation
    from graph_invariant.types import CheckpointState

    cfg = Phase1Config(
        constrained_recovery_generations=2,
        run_baselines=False,
    )
    state = CheckpointState(
        experiment_id="exp",
        generation=0,
        islands={0: []},
        island_prompt_mode={0: "constrained"},
        island_constrained_generations={0: 3},
        island_stagnation={0: 3},
    )

    _update_prompt_mode_after_generation(cfg, state, island_id=0, had_valid_train_candidate=True)
    assert state.island_prompt_mode[0] == "free"


def test_constrained_mode_can_forbid_late_recovery():
    from graph_invariant.cli import _update_prompt_mode_after_generation
    from graph_invariant.types import CheckpointState

    cfg = Phase1Config(
        constrained_recovery_generations=2,
        allow_late_constrained_recovery=False,
        run_baselines=False,
    )
    state = CheckpointState(
        experiment_id="exp",
        generation=0,
        islands={0: []},
        island_prompt_mode={0: "constrained"},
        island_constrained_generations={0: 3},
        island_stagnation={0: 3},
    )

    _update_prompt_mode_after_generation(cfg, state, island_id=0, had_valid_train_candidate=True)
    assert state.island_prompt_mode[0] == "constrained"


def test_target_values_handles_disconnected_graph():
    import networkx as nx

    from graph_invariant.cli import _target_values

    g = nx.Graph()
    g.add_nodes_from([0, 1, 2])
    g.add_edge(0, 1)

    assert _target_values([g], "average_shortest_path_length") == [0.0]
    assert _target_values([g], "diameter") == [0.0]


def test_run_phase1_success_threshold_is_configurable(monkeypatch, tmp_path):
    import networkx as nx

    cfg = Phase1Config(
        artifacts_dir=str(tmp_path / "artifacts"),
        max_generations=1,
        population_size=1,
        num_train_graphs=2,
        num_val_graphs=2,
        num_test_graphs=2,
        run_baselines=False,
        success_spearman_threshold=0.95,
    )
    bundle = DatasetBundle(
        train=[nx.path_graph(4), nx.path_graph(5)],
        val=[nx.path_graph(4), nx.path_graph(5)],
        test=[nx.path_graph(4), nx.path_graph(5)],
        sanity=[nx.path_graph(4)],
    )
    monkeypatch.setattr("graph_invariant.cli.generate_phase1_datasets", lambda _cfg: bundle)
    monkeypatch.setattr(
        "graph_invariant.cli.list_available_models",
        lambda *_args, **_kwargs: ["gpt-oss:20b"],
    )
    monkeypatch.setattr(
        "graph_invariant.cli.generate_candidate_code",
        lambda *_args, **_kwargs: "def new_invariant(G):\n    return float(G.number_of_nodes())",
    )
    _patch_sandbox_evaluator(
        monkeypatch,
        lambda _code, graphs, **_kw: [float(i + 1) for i in range(len(graphs))],
    )
    monkeypatch.setattr(
        "graph_invariant.cli.compute_metrics",
        lambda *_args, **_kwargs: EvaluationResult(0.9, 0.9, 0.1, 0.1, 2, 0),
    )
    monkeypatch.setattr("graph_invariant.cli.compute_total_score", lambda *_args, **_kwargs: 0.8)
    monkeypatch.setattr("graph_invariant.cli.compute_novelty_bonus", lambda *_args, **_kwargs: 0.4)

    assert run_phase1(cfg) == 0
    summary = json.loads((Path(cfg.artifacts_dir) / "phase1_summary.json").read_text("utf-8"))
    assert summary["success"] is False


def test_run_phase1_with_zero_generations_does_not_initialize_sandbox_pool(monkeypatch, tmp_path):
    import networkx as nx

    cfg = Phase1Config(
        artifacts_dir=str(tmp_path / "artifacts"),
        max_generations=0,
        run_baselines=False,
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
    monkeypatch.setattr(
        "graph_invariant.sandbox.mp.get_context",
        lambda: (_ for _ in ()).throw(AssertionError("Pool should not initialize")),
    )

    assert run_phase1(cfg) == 0


def test_dataset_fingerprint_depends_on_graph_counts():
    from graph_invariant.cli import _dataset_fingerprint

    cfg_a = Phase1Config(num_train_graphs=1, num_val_graphs=2, num_test_graphs=3)
    cfg_b = Phase1Config(num_train_graphs=2, num_val_graphs=2, num_test_graphs=3)

    fp_a = _dataset_fingerprint(cfg_a, [1.0], [2.0], [3.0])
    fp_b = _dataset_fingerprint(cfg_b, [1.0], [2.0], [3.0])
    assert fp_a != fp_b


def test_random_forest_baseline_skips_when_inputs_contain_nan():
    import numpy as np

    from graph_invariant.baselines.stat_baselines import _run_random_forest_optional

    x_train = np.asarray([[1.0], [np.nan]], dtype=float)
    y_train = np.asarray([1.0, 2.0], dtype=float)
    x_val = np.asarray([[1.0]], dtype=float)
    y_val = np.asarray([1.0], dtype=float)
    x_test = np.asarray([[1.0]], dtype=float)
    y_test = np.asarray([1.0], dtype=float)

    payload = _run_random_forest_optional(
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val,
        x_test=x_test,
        y_test=y_test,
    )
    assert payload == {"status": "skipped", "reason": "nan in features/targets"}


def test_stat_baseline_feature_matrix_has_stable_empty_shape():
    from graph_invariant.baselines.stat_baselines import _features_from_graphs

    x = _features_from_graphs([])
    assert x.shape == (0, 9)


def test_linear_and_random_forest_skip_on_empty_training_data():
    import numpy as np

    from graph_invariant.baselines.stat_baselines import (
        _run_linear_regression,
        _run_random_forest_optional,
    )

    x_train = np.empty((0, 9), dtype=float)
    y_train = np.empty((0,), dtype=float)
    x_val = np.empty((0, 9), dtype=float)
    y_val = np.empty((0,), dtype=float)
    x_test = np.empty((0, 9), dtype=float)
    y_test = np.empty((0,), dtype=float)

    lr = _run_linear_regression(x_train, y_train, x_val, y_val, x_test, y_test)
    rf = _run_random_forest_optional(x_train, y_train, x_val, y_val, x_test, y_test)
    assert lr == {"status": "skipped", "reason": "empty training data"}
    assert rf == {"status": "skipped", "reason": "empty training data"}


def test_run_phase1_summary_enforces_pysr_parity(monkeypatch, tmp_path):
    import networkx as nx

    cfg = Phase1Config(
        artifacts_dir=str(tmp_path / "artifacts"),
        max_generations=1,
        population_size=1,
        num_train_graphs=2,
        num_val_graphs=2,
        num_test_graphs=2,
        run_baselines=True,
        enforce_pysr_parity_for_success=True,
        require_baselines_for_success=True,
    )
    bundle = DatasetBundle(
        train=[nx.path_graph(4), nx.path_graph(5)],
        val=[nx.path_graph(4), nx.path_graph(5)],
        test=[nx.path_graph(4), nx.path_graph(5)],
        sanity=[nx.path_graph(4)],
    )
    monkeypatch.setattr("graph_invariant.cli.generate_phase1_datasets", lambda _cfg: bundle)
    monkeypatch.setattr(
        "graph_invariant.cli.list_available_models",
        lambda *_args, **_kwargs: ["gpt-oss:20b"],
    )
    monkeypatch.setattr(
        "graph_invariant.cli.generate_candidate_code",
        lambda *_args, **_kwargs: "def new_invariant(G):\n    return float(G.number_of_nodes())",
    )
    _patch_sandbox_evaluator(
        monkeypatch,
        lambda _code, graphs, **_kw: [float(i + 1) for i in range(len(graphs))],
    )
    monkeypatch.setattr(
        "graph_invariant.cli.compute_metrics",
        lambda *_args, **_kwargs: EvaluationResult(0.9, 0.9, 0.1, 0.1, 2, 0),
    )
    monkeypatch.setattr("graph_invariant.cli.compute_total_score", lambda *_args, **_kwargs: 0.8)
    monkeypatch.setattr("graph_invariant.cli.compute_novelty_bonus", lambda *_args, **_kwargs: 0.4)
    monkeypatch.setattr(
        "graph_invariant.cli.run_pysr_baseline",
        lambda **_kwargs: {
            "status": "ok",
            "val_metrics": {"spearman": 0.95},
            "test_metrics": {"spearman": 0.95},
        },
    )

    assert run_phase1(cfg) == 0
    summary = json.loads((Path(cfg.artifacts_dir) / "phase1_summary.json").read_text("utf-8"))
    assert summary["schema_version"] == 3
    assert summary["success"] is False
    assert summary["success_criteria"]["pysr_parity_passed"] is False


def test_run_phase1_pysr_parity_allows_small_epsilon_gap(monkeypatch, tmp_path):
    import networkx as nx

    cfg = Phase1Config(
        artifacts_dir=str(tmp_path / "artifacts"),
        max_generations=1,
        population_size=1,
        num_train_graphs=2,
        num_val_graphs=2,
        num_test_graphs=2,
        run_baselines=True,
        enforce_pysr_parity_for_success=True,
        require_baselines_for_success=True,
        pysr_parity_epsilon=0.001,
    )
    bundle = DatasetBundle(
        train=[nx.path_graph(4), nx.path_graph(5)],
        val=[nx.path_graph(4), nx.path_graph(5)],
        test=[nx.path_graph(4), nx.path_graph(5)],
        sanity=[nx.path_graph(4)],
    )
    monkeypatch.setattr("graph_invariant.cli.generate_phase1_datasets", lambda _cfg: bundle)
    monkeypatch.setattr(
        "graph_invariant.cli.list_available_models",
        lambda *_args, **_kwargs: ["gpt-oss:20b"],
    )
    monkeypatch.setattr(
        "graph_invariant.cli.generate_candidate_code",
        lambda *_args, **_kwargs: "def new_invariant(G):\n    return float(G.number_of_nodes())",
    )
    _patch_sandbox_evaluator(
        monkeypatch,
        lambda _code, graphs, **_kw: [float(i + 1) for i in range(len(graphs))],
    )
    monkeypatch.setattr(
        "graph_invariant.cli.compute_metrics",
        lambda *_args, **_kwargs: EvaluationResult(0.9, 0.9, 0.1, 0.1, 2, 0),
    )
    monkeypatch.setattr("graph_invariant.cli.compute_total_score", lambda *_args, **_kwargs: 0.8)
    monkeypatch.setattr("graph_invariant.cli.compute_novelty_bonus", lambda *_args, **_kwargs: 0.4)
    monkeypatch.setattr(
        "graph_invariant.cli.run_pysr_baseline",
        lambda **_kwargs: {
            "status": "ok",
            "val_metrics": {"spearman": 0.9005},
            "test_metrics": {"spearman": 0.9005},
        },
    )

    assert run_phase1(cfg) == 0
    summary = json.loads((Path(cfg.artifacts_dir) / "phase1_summary.json").read_text("utf-8"))
    assert summary["success_criteria"]["pysr_parity_passed"] is True
