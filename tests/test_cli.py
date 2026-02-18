import hashlib
import json
from pathlib import Path

import pytest

from graph_invariant.cli import run_phase1
from graph_invariant.config import Phase1Config
from graph_invariant.data import DatasetBundle
from graph_invariant.types import CheckpointState, EvaluationResult


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

        def evaluate(self, code, features_list):  # noqa: ANN001
            return evaluate_fn(
                code,
                features_list,
                timeout_sec=self.timeout_sec,
                memory_mb=self.memory_mb,
            )

        def evaluate_detailed(self, code, features_list):  # noqa: ANN001
            values = self.evaluate(code, features_list)
            details = []
            for value in values:
                if value is None:
                    details.append(
                        {
                            "value": None,
                            "error_type": "runtime_exception",
                            "error_detail": "fake evaluator returned None",
                        }
                    )
                else:
                    details.append(
                        {"value": float(value), "error_type": None, "error_detail": None}
                    )
            return details

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

    def fake_total(fitness_signal, simplicity, novelty_bonus, alpha, beta, gamma):  # noqa: ANN001
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
        lambda *_args, **_kwargs: "def new_invariant(s):\n    return 1.0",
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
        lambda *_args, **_kwargs: "def new_invariant(s):\n    return 1.0",
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
        lambda *_args, **_kwargs: "def new_invariant(s):\n    return 1.0",
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
        lambda *_args, **_kwargs: "def new_invariant(s):\n    return float(s['n'])",
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
    expected_code = "def new_invariant(s):\n    return float(s['n'])"
    assert "test_metrics" in payload
    assert "val_metrics" in payload
    assert "success" in payload
    assert payload["schema_version"] == 3
    assert payload["fitness_mode"] == "correlation"
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
        return "def new_invariant(s):\n    return 1.0"

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


def test_run_phase1_self_correction_repairs_failed_candidate_once(monkeypatch, tmp_path):
    import networkx as nx

    cfg = Phase1Config(
        artifacts_dir=str(tmp_path / "artifacts"),
        max_generations=1,
        population_size=1,
        num_train_graphs=2,
        num_val_graphs=2,
        num_test_graphs=2,
        run_baselines=False,
        enable_self_correction=True,
        self_correction_max_retries=1,
    )
    bundle = DatasetBundle(
        train=[nx.path_graph(4), nx.path_graph(5)],
        val=[nx.path_graph(4), nx.path_graph(5)],
        test=[nx.path_graph(4), nx.path_graph(5)],
        sanity=[nx.path_graph(4)],
    )
    prompts: list[str] = []
    calls = {"count": 0}

    def _fake_generate(prompt: str, *_args, **_kwargs) -> str:
        prompts.append(prompt)
        calls["count"] += 1
        if calls["count"] == 1:
            return "def new_invariant(s):\n    BROKEN"
        return "def new_invariant(s):\n    return float(s['n'])"

    def _fake_eval(code, graphs, **_kwargs):  # noqa: ANN001
        if "BROKEN" in code:
            return [None for _ in graphs]
        return [float(i + 1) for i in range(len(graphs))]

    monkeypatch.setattr("graph_invariant.cli.generate_phase1_datasets", lambda _cfg: bundle)
    monkeypatch.setattr(
        "graph_invariant.cli.list_available_models",
        lambda *_args, **_kwargs: ["gpt-oss:20b"],
    )
    monkeypatch.setattr("graph_invariant.cli.generate_candidate_code", _fake_generate)
    _patch_sandbox_evaluator(monkeypatch, _fake_eval)
    monkeypatch.setattr(
        "graph_invariant.cli.compute_metrics",
        lambda *_args, **_kwargs: EvaluationResult(0.9, 0.9, 0.1, 0.1, 2, 0),
    )
    monkeypatch.setattr("graph_invariant.cli.compute_total_score", lambda *_args, **_kwargs: 0.8)
    monkeypatch.setattr("graph_invariant.cli.compute_novelty_bonus", lambda *_args, **_kwargs: 0.4)

    assert run_phase1(cfg) == 0
    assert calls["count"] == 5
    assert any("Repair this candidate" in prompt for prompt in prompts)

    summary = json.loads((Path(cfg.artifacts_dir) / "phase1_summary.json").read_text("utf-8"))
    stats = summary["self_correction_stats"]
    assert stats["attempted_repairs"] == 1
    assert stats["successful_repairs"] == 1
    assert stats["failed_repairs"] == 0


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


def test_main_benchmark_command_invokes_runner(monkeypatch, tmp_path):
    calls: dict[str, object] = {}

    def fake_run_benchmark(cfg):  # noqa: ANN001
        calls["artifacts_dir"] = cfg.artifacts_dir
        return 0

    config_path = tmp_path / "benchmark.json"
    config_path.write_text(json.dumps({"artifacts_dir": "artifacts_bench"}), encoding="utf-8")
    monkeypatch.setattr("graph_invariant.benchmark.run_benchmark", fake_run_benchmark)
    monkeypatch.setattr(
        "sys.argv",
        ["graph_invariant", "benchmark", "--config", str(config_path)],
    )
    from graph_invariant import cli

    assert cli.main() == 0
    assert calls["artifacts_dir"] == "artifacts_bench"


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

    from graph_invariant.targets import target_values

    g = nx.Graph()
    g.add_nodes_from([0, 1, 2])
    g.add_edge(0, 1)

    assert target_values([g], "average_shortest_path_length") == [0.0]
    assert target_values([g], "diameter") == [0.0]


def test_summarize_error_details_uses_detail_for_top_category():
    from graph_invariant.cli import _summarize_error_details

    details = [
        {
            "value": None,
            "error_type": "timeout",
            "error_detail": "candidate evaluation timed out",
        },
        {
            "value": None,
            "error_type": "runtime_exception",
            "error_detail": "ZeroDivisionError: division by zero",
        },
        {
            "value": None,
            "error_type": "runtime_exception",
            "error_detail": "TypeError: bad operand",
        },
    ]
    summary = _summarize_error_details(details)
    assert summary.startswith("runtime_exception:")
    assert "timed out" not in summary


def test_record_recent_failure_deduplicates_and_keeps_recency():
    from graph_invariant.cli import _record_recent_failure

    state = CheckpointState(experiment_id="exp", generation=0, islands={0: []})
    _record_recent_failure(state, island_id=0, failure_text="a", max_items=3)
    _record_recent_failure(state, island_id=0, failure_text="b", max_items=3)
    _record_recent_failure(state, island_id=0, failure_text="a", max_items=3)
    _record_recent_failure(state, island_id=0, failure_text="c", max_items=3)
    assert state.island_recent_failures[0] == ["b", "a", "c"]


def test_topology_descriptor_handles_missing_feature_keys():
    from graph_invariant.cli import _topology_descriptor

    descriptor = _topology_descriptor(
        y_pred_valid=[1.0, 2.0, 3.0],
        features_val=[{"density": 0.2}, {"density": 0.3}, {"density": 0.4}],
        valid_indices=(0, 1, 2),
    )
    assert 0.0 <= descriptor[0] <= 1.0
    assert 0.0 <= descriptor[1] <= 1.0


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
        lambda *_args, **_kwargs: "def new_invariant(s):\n    return float(s['n'])",
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
    from graph_invariant.baselines.stat_baselines import _FEATURE_ORDER, _features_from_graphs

    x = _features_from_graphs([])
    assert x.shape == (0, len(_FEATURE_ORDER))


def test_linear_and_random_forest_skip_on_empty_training_data():
    import numpy as np

    from graph_invariant.baselines.stat_baselines import (
        _FEATURE_ORDER,
        _run_linear_regression,
        _run_random_forest_optional,
    )

    width = len(_FEATURE_ORDER)
    x_train = np.empty((0, width), dtype=float)
    y_train = np.empty((0,), dtype=float)
    x_val = np.empty((0, width), dtype=float)
    y_val = np.empty((0,), dtype=float)
    x_test = np.empty((0, width), dtype=float)
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
        lambda *_args, **_kwargs: "def new_invariant(s):\n    return float(s['n'])",
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
        lambda *_args, **_kwargs: "def new_invariant(s):\n    return float(s['n'])",
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


def test_run_phase1_requires_healthy_baselines_when_configured(monkeypatch, tmp_path):
    import networkx as nx

    cfg = Phase1Config(
        artifacts_dir=str(tmp_path / "artifacts"),
        max_generations=1,
        population_size=1,
        num_train_graphs=2,
        num_val_graphs=2,
        num_test_graphs=2,
        run_baselines=True,
        enforce_pysr_parity_for_success=False,
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
        lambda *_args, **_kwargs: "def new_invariant(s):\n    return float(s['n'])",
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
        lambda **_kwargs: {"status": "error", "reason": "boom"},
    )
    monkeypatch.setattr(
        "graph_invariant.cli.run_stat_baselines",
        lambda **_kwargs: {
            "linear_regression": {"status": "error", "reason": "bad"},
            "random_forest": {"status": "skipped", "reason": "missing"},
        },
    )

    assert run_phase1(cfg) == 0
    summary = json.loads((Path(cfg.artifacts_dir) / "phase1_summary.json").read_text("utf-8"))
    assert summary["success"] is False
    assert summary["success_criteria"]["baselines_passed"] is False


def test_run_phase1_accepts_single_healthy_baseline_when_required(monkeypatch, tmp_path):
    import networkx as nx

    cfg = Phase1Config(
        artifacts_dir=str(tmp_path / "artifacts"),
        max_generations=1,
        population_size=1,
        num_train_graphs=2,
        num_val_graphs=2,
        num_test_graphs=2,
        run_baselines=True,
        enforce_pysr_parity_for_success=False,
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
        lambda *_args, **_kwargs: "def new_invariant(s):\n    return float(s['n'])",
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
        lambda **_kwargs: {"status": "error", "reason": "boom"},
    )
    monkeypatch.setattr(
        "graph_invariant.cli.run_stat_baselines",
        lambda **_kwargs: {
            "linear_regression": {"status": "ok", "val_metrics": {"spearman": 0.2}},
            "random_forest": {"status": "skipped", "reason": "missing"},
        },
    )

    assert run_phase1(cfg) == 0
    summary = json.loads((Path(cfg.artifacts_dir) / "phase1_summary.json").read_text("utf-8"))
    assert summary["success"] is True
    assert summary["success_criteria"]["baselines_healthy"] is True
    assert summary["success_criteria"]["stat_baseline_ok"] is True
    assert summary["success_criteria"]["pysr_ok"] is False
    assert summary["success_criteria"]["baselines_passed"] is True


def test_candidate_prompt_maps_island_0_to_refinement():
    from graph_invariant.cli import _candidate_prompt

    state = CheckpointState(
        experiment_id="exp",
        generation=0,
        islands={0: [], 1: [], 2: [], 3: []},
    )
    prompt = _candidate_prompt(state, island_id=0, target_name="diameter")
    assert any(word in prompt.lower() for word in ("improve", "refine"))


def test_candidate_prompt_maps_island_1_to_combination():
    from graph_invariant.cli import _candidate_prompt

    state = CheckpointState(
        experiment_id="exp",
        generation=0,
        islands={0: [], 1: [], 2: [], 3: []},
    )
    prompt = _candidate_prompt(state, island_id=1, target_name="diameter")
    assert "combine" in prompt.lower()


def test_candidate_prompt_maps_island_3_to_novel():
    from graph_invariant.cli import _candidate_prompt

    state = CheckpointState(
        experiment_id="exp",
        generation=0,
        islands={0: [], 1: [], 2: [], 3: []},
    )
    prompt = _candidate_prompt(state, island_id=3, target_name="diameter")
    assert any(word in prompt.lower() for word in ("new", "novel"))


def test_generation_rejects_candidate_below_novelty_gate(monkeypatch, tmp_path):
    import networkx as nx

    cfg = Phase1Config(
        artifacts_dir=str(tmp_path / "artifacts"),
        max_generations=1,
        population_size=1,
        num_train_graphs=2,
        num_val_graphs=2,
        num_test_graphs=2,
        run_baselines=False,
        novelty_gate_threshold=0.15,
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
        lambda *_args, **_kwargs: "def new_invariant(s):\n    return float(s['n'])",
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
    # Return very low novelty — below the 0.15 gate
    monkeypatch.setattr("graph_invariant.cli.compute_novelty_bonus", lambda *_args, **_kwargs: 0.08)

    assert run_phase1(cfg) == 0
    summary = json.loads((Path(cfg.artifacts_dir) / "phase1_summary.json").read_text("utf-8"))
    # All candidates should have been rejected by novelty gate → no candidates
    assert summary.get("success") is False


def test_novelty_gate_rejection_triggers_self_correction(monkeypatch, tmp_path):
    """When a candidate is novelty-gated, self-correction should be attempted."""
    import networkx as nx

    cfg = Phase1Config(
        artifacts_dir=str(tmp_path / "artifacts"),
        max_generations=1,
        population_size=1,
        num_train_graphs=2,
        num_val_graphs=2,
        num_test_graphs=2,
        run_baselines=False,
        novelty_gate_threshold=0.15,
        enable_self_correction=True,
        self_correction_max_retries=1,
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
    generate_call_count = {"count": 0}

    def _counting_generate(*_args, **_kwargs):
        generate_call_count["count"] += 1
        return "def new_invariant(s):\n    return float(s['n'])"

    monkeypatch.setattr(
        "graph_invariant.cli.generate_candidate_code",
        _counting_generate,
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
    # Novelty below threshold → should trigger repair attempt
    monkeypatch.setattr("graph_invariant.cli.compute_novelty_bonus", lambda *_args, **_kwargs: 0.08)

    assert run_phase1(cfg) == 0
    # 4 islands × 1 candidate × 2 attempts (original + repair) = 8 generate calls.
    # Without the fix, novelty gate has repairable=False so only 4 calls (no repairs).
    assert generate_call_count["count"] == 8


def test_generation_accepts_candidate_above_novelty_gate(monkeypatch, tmp_path):
    import networkx as nx

    cfg = Phase1Config(
        artifacts_dir=str(tmp_path / "artifacts"),
        max_generations=1,
        population_size=1,
        num_train_graphs=2,
        num_val_graphs=2,
        num_test_graphs=2,
        run_baselines=False,
        novelty_gate_threshold=0.15,
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
        lambda *_args, **_kwargs: "def new_invariant(s):\n    return float(s['n'])",
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
    # Return novelty above the 0.15 gate
    monkeypatch.setattr("graph_invariant.cli.compute_novelty_bonus", lambda *_args, **_kwargs: 0.5)

    assert run_phase1(cfg) == 0
    summary = json.loads((Path(cfg.artifacts_dir) / "phase1_summary.json").read_text("utf-8"))
    assert summary.get("best_candidate_id") is not None


def test_novelty_gate_disabled_when_zero(monkeypatch, tmp_path):
    import networkx as nx

    cfg = Phase1Config(
        artifacts_dir=str(tmp_path / "artifacts"),
        max_generations=1,
        population_size=1,
        num_train_graphs=2,
        num_val_graphs=2,
        num_test_graphs=2,
        run_baselines=False,
        novelty_gate_threshold=0.0,
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
        lambda *_args, **_kwargs: "def new_invariant(s):\n    return float(s['n'])",
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
    # Very low novelty, but gate is disabled (threshold=0.0)
    monkeypatch.setattr("graph_invariant.cli.compute_novelty_bonus", lambda *_args, **_kwargs: 0.01)

    assert run_phase1(cfg) == 0
    summary = json.loads((Path(cfg.artifacts_dir) / "phase1_summary.json").read_text("utf-8"))
    # Should still produce candidates since gate is disabled
    assert summary.get("best_candidate_id") is not None


# ── Algebraic connectivity target + bounds mode tests ────────────────


def test_target_values_supports_algebraic_connectivity():
    import networkx as nx

    from graph_invariant.targets import target_values

    g = nx.path_graph(5)
    values = target_values([g], "algebraic_connectivity")
    assert len(values) == 1
    assert values[0] > 0.0


def test_target_values_algebraic_connectivity_handles_disconnected():
    import networkx as nx

    from graph_invariant.targets import target_values

    g = nx.Graph()
    g.add_nodes_from([0, 1, 2])
    g.add_edge(0, 1)
    values = target_values([g], "algebraic_connectivity")
    assert values == [0.0]


def test_target_values_algebraic_connectivity_single_node():
    import networkx as nx

    from graph_invariant.targets import target_values

    g = nx.Graph()
    g.add_node(0)
    values = target_values([g], "algebraic_connectivity")
    assert values == [0.0]


def test_safe_algebraic_connectivity_handles_eigenvalue_error(monkeypatch):
    """_safe_algebraic_connectivity should return 0.0 on eigenvalue computation errors."""
    import networkx as nx

    from graph_invariant.targets import _safe_algebraic_connectivity

    g = nx.path_graph(5)
    monkeypatch.setattr(
        "graph_invariant.targets.nx.algebraic_connectivity",
        lambda *_a, **_kw: (_ for _ in ()).throw(nx.NetworkXError("convergence failed")),
    )
    assert _safe_algebraic_connectivity(g) == 0.0


def test_run_phase1_bounds_mode_uses_bound_score(monkeypatch, tmp_path):
    """In bounds mode, the generation loop should use bound_score instead of spearman."""
    import networkx as nx

    cfg = Phase1Config(
        artifacts_dir=str(tmp_path / "artifacts"),
        max_generations=1,
        population_size=1,
        num_train_graphs=2,
        num_val_graphs=2,
        num_test_graphs=2,
        run_baselines=False,
        fitness_mode="upper_bound",
        train_score_threshold=0.0,
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
        lambda *_args, **_kwargs: "def new_invariant(s):\n    return float(s['n']) * 10",
    )
    _patch_sandbox_evaluator(
        monkeypatch,
        lambda _code, graphs, **_kw: [float(i + 100) for i in range(len(graphs))],
    )
    monkeypatch.setattr("graph_invariant.cli.compute_novelty_bonus", lambda *_args, **_kwargs: 0.5)

    assert run_phase1(cfg) == 0
    summary = json.loads((Path(cfg.artifacts_dir) / "phase1_summary.json").read_text("utf-8"))
    assert summary["schema_version"] == 4
    assert summary["fitness_mode"] == "upper_bound"
    assert "bounds_metrics" in summary
    assert "success_criteria_bounds" in summary


def test_run_phase1_bounds_mode_summary_schema(monkeypatch, tmp_path):
    """Bounds mode summary should include bounds-specific fields and schema v4."""
    import networkx as nx

    cfg = Phase1Config(
        artifacts_dir=str(tmp_path / "artifacts"),
        max_generations=1,
        population_size=1,
        num_train_graphs=2,
        num_val_graphs=2,
        num_test_graphs=2,
        run_baselines=False,
        fitness_mode="lower_bound",
        train_score_threshold=0.0,
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
        lambda *_args, **_kwargs: "def new_invariant(s):\n    return 0.1",
    )
    _patch_sandbox_evaluator(
        monkeypatch,
        lambda _code, graphs, **_kw: [0.1 for _ in range(len(graphs))],
    )
    monkeypatch.setattr("graph_invariant.cli.compute_novelty_bonus", lambda *_args, **_kwargs: 0.5)

    assert run_phase1(cfg) == 0
    summary = json.loads((Path(cfg.artifacts_dir) / "phase1_summary.json").read_text("utf-8"))
    assert summary["schema_version"] == 4
    bounds_metrics = summary["bounds_metrics"]
    assert "val" in bounds_metrics
    assert "test" in bounds_metrics
    criteria = summary["success_criteria_bounds"]
    assert "bound_score_threshold" in criteria
    assert "satisfaction_threshold" in criteria


def test_candidate_prompt_passes_fitness_mode(monkeypatch):
    """_candidate_prompt should forward fitness_mode to build_prompt."""
    import inspect

    from graph_invariant.cli import _candidate_prompt
    from graph_invariant.llm_ollama import build_prompt as original_build

    captured: dict[str, str] = {}
    sig = inspect.signature(original_build)

    def spy_build(*args, **kwargs):
        bound = sig.bind(*args, **kwargs)
        bound.apply_defaults()
        captured["fitness_mode"] = bound.arguments["fitness_mode"]
        return original_build(*args, **kwargs)

    monkeypatch.setattr("graph_invariant.cli.build_prompt", spy_build)

    state = CheckpointState(
        experiment_id="exp",
        generation=0,
        islands={0: [], 1: [], 2: [], 3: []},
    )
    _candidate_prompt(state, island_id=0, target_name="diameter", fitness_mode="upper_bound")
    assert captured["fitness_mode"] == "upper_bound"


def test_run_phase1_persists_prompt_and_response_when_enabled(monkeypatch, tmp_path):
    import networkx as nx

    cfg = Phase1Config(
        artifacts_dir=str(tmp_path / "artifacts"),
        max_generations=1,
        population_size=1,
        num_train_graphs=1,
        num_val_graphs=2,
        num_test_graphs=1,
        run_baselines=False,
        persist_prompt_and_response_logs=True,
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
        "graph_invariant.cli.generate_candidate_payload",
        lambda *_args, **_kwargs: {
            "response": "llm text",
            "code": "def new_invariant(s):\n    return 1.0",
        },
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
    events_path = Path(cfg.artifacts_dir) / "logs" / "events.jsonl"
    records = [json.loads(line) for line in events_path.read_text("utf-8").splitlines()]
    evaluated = [r for r in records if r["event_type"] == "candidate_evaluated"]
    assert evaluated
    payload = evaluated[0]["payload"]
    assert payload["prompt"] is not None
    assert payload["llm_response"] == "llm text"
    assert payload["extracted_code"].startswith("def new_invariant")


# ── MAP-Elites integration tests ─────────────────────────────────────


def test_run_phase1_with_map_elites_populates_archive(monkeypatch, tmp_path):
    import networkx as nx

    cfg = Phase1Config(
        artifacts_dir=str(tmp_path / "artifacts"),
        max_generations=1,
        population_size=1,
        num_train_graphs=2,
        num_val_graphs=2,
        num_test_graphs=2,
        run_baselines=False,
        enable_map_elites=True,
        map_elites_bins=3,
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
        lambda *_args, **_kwargs: "def new_invariant(s):\n    return float(s['n'])",
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
    monkeypatch.setattr("graph_invariant.cli.compute_novelty_bonus", lambda *_args, **_kwargs: 0.5)

    assert run_phase1(cfg) == 0

    # Check that archive stats appear in generation_summary events
    events_path = Path(cfg.artifacts_dir) / "logs" / "events.jsonl"
    records = [json.loads(line) for line in events_path.read_text("utf-8").splitlines()]
    gen_summaries = [r for r in records if r["event_type"] == "generation_summary"]
    assert gen_summaries
    assert "map_elites_stats" in gen_summaries[0]["payload"]
    stats = gen_summaries[0]["payload"]["map_elites_stats"]
    assert stats["coverage"] > 0
    assert "map_elites_stats_primary" in gen_summaries[0]["payload"]
    assert "map_elites_stats_topology" in gen_summaries[0]["payload"]


def test_run_phase1_without_map_elites_omits_archive(monkeypatch, tmp_path):
    import networkx as nx

    cfg = Phase1Config(
        artifacts_dir=str(tmp_path / "artifacts"),
        max_generations=1,
        population_size=1,
        num_train_graphs=2,
        num_val_graphs=2,
        num_test_graphs=2,
        run_baselines=False,
        enable_map_elites=False,
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
        lambda *_args, **_kwargs: "def new_invariant(s):\n    return float(s['n'])",
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
    monkeypatch.setattr("graph_invariant.cli.compute_novelty_bonus", lambda *_args, **_kwargs: 0.5)

    assert run_phase1(cfg) == 0

    # No archive stats in generation_summary
    events_path = Path(cfg.artifacts_dir) / "logs" / "events.jsonl"
    records = [json.loads(line) for line in events_path.read_text("utf-8").splitlines()]
    gen_summaries = [r for r in records if r["event_type"] == "generation_summary"]
    assert gen_summaries
    assert "map_elites_stats" not in gen_summaries[0]["payload"]


def test_run_phase1_map_elites_checkpoint_roundtrip(monkeypatch, tmp_path):
    import networkx as nx

    cfg = Phase1Config(
        artifacts_dir=str(tmp_path / "artifacts"),
        max_generations=1,
        population_size=1,
        num_train_graphs=2,
        num_val_graphs=2,
        num_test_graphs=2,
        run_baselines=False,
        enable_map_elites=True,
        map_elites_bins=3,
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
        lambda *_args, **_kwargs: "def new_invariant(s):\n    return float(s['n'])",
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
    monkeypatch.setattr("graph_invariant.cli.compute_novelty_bonus", lambda *_args, **_kwargs: 0.5)

    assert run_phase1(cfg) == 0

    # Verify checkpoint contains archive
    ckpt_root = Path(cfg.artifacts_dir) / "checkpoints"
    experiment_dir = next(ckpt_root.iterdir())
    ckpt_path = sorted(experiment_dir.glob("gen_*.json"))[-1]
    ckpt_data = json.loads(ckpt_path.read_text("utf-8"))
    assert "map_elites_archive" in ckpt_data
    assert ckpt_data["map_elites_archive"]["num_bins"] == 3
    assert "map_elites_archives" in ckpt_data
    assert ckpt_data["map_elites_archives"]["primary"]["num_bins"] == 3
    assert ckpt_data["map_elites_archives"]["topology"]["num_bins"] == 5


def test_candidate_prompt_includes_archive_exemplars():
    from graph_invariant.cli import _candidate_prompt

    state = CheckpointState(
        experiment_id="exp",
        generation=0,
        islands={0: [], 1: [], 2: [], 3: []},
    )
    exemplar_codes = ["def new_invariant(s): return s['n'] + 1"]
    prompt = _candidate_prompt(
        state,
        island_id=1,
        target_name="diameter",
        archive_exemplars=exemplar_codes,
    )
    assert "s['n'] + 1" in prompt


def test_report_includes_ood_validation(monkeypatch, tmp_path):
    from graph_invariant.cli import write_report

    artifacts_dir = tmp_path / "artifacts"
    artifacts_dir.mkdir()
    (artifacts_dir / "phase1_summary.json").write_text(
        json.dumps({"success": True, "best_candidate_id": "c1"}),
        encoding="utf-8",
    )
    (artifacts_dir / "ood_validation.json").write_text(
        json.dumps(
            {
                "large_random": {"spearman": 0.85, "valid_count": 90, "total_count": 100},
                "extreme_params": {"spearman": 0.72, "valid_count": 45, "total_count": 50},
                "special_topology": {"spearman": 0.91, "valid_count": 8, "total_count": 8},
            }
        ),
        encoding="utf-8",
    )
    report_path = write_report(str(artifacts_dir))
    report = report_path.read_text(encoding="utf-8")
    assert "## OOD Validation" in report
    assert "large_random" in report
    assert "spearman=0.8500" in report


def test_report_includes_map_elites_coverage(monkeypatch, tmp_path):
    from graph_invariant.cli import write_report

    artifacts_dir = tmp_path / "artifacts"
    logs_dir = artifacts_dir / "logs"
    logs_dir.mkdir(parents=True)
    (artifacts_dir / "phase1_summary.json").write_text(
        json.dumps({"success": True, "best_candidate_id": "c1"}),
        encoding="utf-8",
    )
    events = [
        json.dumps(
            {
                "event_type": "generation_summary",
                "payload": {"map_elites_stats_primary": {"coverage": 3}},
            }
        ),
        json.dumps(
            {
                "event_type": "generation_summary",
                "payload": {"map_elites_stats_primary": {"coverage": 7}},
            }
        ),
    ]
    (logs_dir / "events.jsonl").write_text("\n".join(events) + "\n", encoding="utf-8")
    report_path = write_report(str(artifacts_dir))
    report = report_path.read_text(encoding="utf-8")
    assert "## MAP-Elites Archive" in report
    assert "Final coverage: 7 cells" in report
    assert "3 -> 7" in report
