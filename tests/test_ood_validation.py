import json

import networkx as nx

# ── Dataset generation ───────────────────────────────────────────────


def test_generate_ood_datasets_returns_three_categories():
    from graph_invariant.ood_validation import generate_ood_datasets

    bundle = generate_ood_datasets(seed=42, num_large=5, num_extreme=3)
    assert len(bundle["large_random"]) == 5
    assert len(bundle["extreme_params"]) == 3
    assert len(bundle["special_topology"]) > 0


def test_generate_ood_datasets_produces_connected_graphs():
    from graph_invariant.ood_validation import generate_ood_datasets

    bundle = generate_ood_datasets(seed=42, num_large=5, num_extreme=3)
    for category, graphs in bundle.items():
        for g in graphs:
            assert nx.is_connected(g), f"disconnected graph in {category}"


def test_generate_ood_datasets_large_random_has_big_graphs():
    from graph_invariant.ood_validation import generate_ood_datasets

    bundle = generate_ood_datasets(seed=42, num_large=10, num_extreme=3)
    sizes = [len(g) for g in bundle["large_random"]]
    # All should have n >= 100 (we sample from [200, 500] but connected subgraph may shrink)
    assert all(s >= 50 for s in sizes)


def test_generate_ood_datasets_special_topology_includes_known_graphs():
    from graph_invariant.ood_validation import generate_ood_datasets

    bundle = generate_ood_datasets(seed=42, num_large=3, num_extreme=3)
    special = bundle["special_topology"]
    # Should have at least the sanity graphs + some deterministic structures
    assert len(special) >= 5


def test_generate_ood_datasets_is_deterministic():
    from graph_invariant.ood_validation import generate_ood_datasets

    b1 = generate_ood_datasets(seed=99, num_large=5, num_extreme=3)
    b2 = generate_ood_datasets(seed=99, num_large=5, num_extreme=3)
    for cat in ("large_random", "extreme_params"):
        assert len(b1[cat]) == len(b2[cat])
        for g1, g2 in zip(b1[cat], b2[cat], strict=True):
            assert len(g1) == len(g2)
            assert g1.number_of_edges() == g2.number_of_edges()


# ── Evaluation ───────────────────────────────────────────────────────


def test_evaluate_ood_split_returns_metrics():
    from graph_invariant.ood_validation import _evaluate_ood_split

    graphs = [nx.path_graph(5), nx.path_graph(6)]

    class FakeEvaluator:
        def evaluate(self, code, features_list):
            return [float(i + 1) for i in range(len(features_list))]

    result = _evaluate_ood_split(
        code="def new_invariant(s): return float(s['n'])",
        graphs=graphs,
        target_name="average_shortest_path_length",
        evaluator=FakeEvaluator(),
        fitness_mode="correlation",
    )
    assert "spearman" in result or "bound_score" in result
    assert "valid_count" in result


def test_evaluate_ood_split_handles_all_none():
    from graph_invariant.ood_validation import _evaluate_ood_split

    graphs = [nx.path_graph(5)]

    class FakeEvaluator:
        def evaluate(self, code, features_list):
            return [None for _ in features_list]

    result = _evaluate_ood_split(
        code="def broken(): pass",
        graphs=graphs,
        target_name="average_shortest_path_length",
        evaluator=FakeEvaluator(),
        fitness_mode="correlation",
    )
    assert result["valid_count"] == 0


def test_evaluate_ood_split_bounds_mode():
    from graph_invariant.ood_validation import _evaluate_ood_split

    graphs = [nx.path_graph(5), nx.path_graph(6)]

    class FakeEvaluator:
        def evaluate(self, code, features_list):
            return [100.0 for _ in features_list]

    result = _evaluate_ood_split(
        code="def new_invariant(s): return 100.0",
        graphs=graphs,
        target_name="average_shortest_path_length",
        evaluator=FakeEvaluator(),
        fitness_mode="upper_bound",
    )
    assert "bound_score" in result
    assert "satisfaction_rate" in result


# ── End-to-end run_ood_validation ────────────────────────────────────


def test_run_ood_validation_end_to_end(monkeypatch, tmp_path):
    from graph_invariant.ood_validation import run_ood_validation

    summary_dir = tmp_path / "artifacts"
    summary_dir.mkdir()
    summary_path = summary_dir / "phase1_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "best_candidate_code": "def new_invariant(s):\n    return float(s['n'])",
                "config": {
                    "target_name": "average_shortest_path_length",
                    "fitness_mode": "correlation",
                    "timeout_sec": 2.0,
                    "memory_mb": 256,
                },
            }
        ),
        encoding="utf-8",
    )

    class FakeEvaluator:
        def __init__(self, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def evaluate(self, code, features_list):
            return [float(i + 1) for i in range(len(features_list))]

    monkeypatch.setattr("graph_invariant.ood_validation.SandboxEvaluator", FakeEvaluator)

    output_dir = tmp_path / "ood_output"
    rc = run_ood_validation(
        summary_path=str(summary_path),
        output_dir=str(output_dir),
        seed=42,
        num_large=3,
        num_extreme=2,
    )
    assert rc == 0
    result_path = output_dir / "ood_validation.json"
    assert result_path.exists()
    result = json.loads(result_path.read_text("utf-8"))
    assert "large_random" in result
    assert "extreme_params" in result
    assert "special_topology" in result


def test_run_ood_validation_missing_code_fails(tmp_path):
    from graph_invariant.ood_validation import run_ood_validation

    summary_dir = tmp_path / "artifacts"
    summary_dir.mkdir()
    summary_path = summary_dir / "phase1_summary.json"
    summary_path.write_text(
        json.dumps({"config": {"target_name": "average_shortest_path_length"}}),
        encoding="utf-8",
    )
    output_dir = tmp_path / "ood_output"
    rc = run_ood_validation(
        summary_path=str(summary_path),
        output_dir=str(output_dir),
    )
    assert rc == 1
