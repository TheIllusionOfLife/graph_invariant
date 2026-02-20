"""Tests for analysis/analyze_experiments.py functions.

Follows test_ood_validation.py patterns: tmp_path fixtures with minimal mock
JSON data, asserting output shapes and key presence rather than exact values.
"""

import importlib.util
import json
from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def analyze_module():
    analysis_path = Path(__file__).resolve().parent.parent / "analysis" / "analyze_experiments.py"
    spec = importlib.util.spec_from_file_location("analyze_experiments", analysis_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Failed to load analyze_experiments module spec")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ── Fixtures ─────────────────────────────────────────────────────────


@pytest.fixture
def mock_phase1_summary():
    """Minimal phase1_summary.json for correlation-mode experiment."""
    return {
        "schema_version": 3,
        "experiment_id": "test_exp",
        "fitness_mode": "correlation",
        "model_name": "gpt-oss:20b",
        "best_candidate_id": "cand_001",
        "best_val_score": 0.92,
        "stop_reason": "max_generations",
        "success": True,
        "final_generation": 30,
        "train_metrics": {"spearman": 0.95, "valid_count": 100, "total_count": 100},
        "val_metrics": {"spearman": 0.92, "valid_count": 50, "total_count": 50},
        "test_metrics": {"spearman": 0.90, "valid_count": 50, "total_count": 50},
        "sanity_metrics": {"spearman": 1.0, "valid_count": 10, "total_count": 10},
        "novelty_ci": {
            "validation": {"max_ci_upper_abs_rho": 0.7, "novelty_passed": True},
            "test": {"max_ci_upper_abs_rho": 0.65, "novelty_passed": True},
        },
        "self_correction_stats": {
            "enabled": True,
            "attempted_repairs": 5,
            "successful_repairs": 3,
            "failed_repairs": 2,
        },
        "baseline_comparison": {
            "candidate": {"val_spearman": 0.92, "test_spearman": 0.90},
            "pysr": {"status": "ok", "val_spearman": 0.88, "test_spearman": 0.86},
        },
        "config": {
            "target_name": "average_shortest_path_length",
            "num_islands": 4,
            "max_generations": 30,
        },
        "best_candidate_code": (
            "def f(G, feats):\n    return feats['num_edges'] / feats['num_nodes']"
        ),
    }


@pytest.fixture
def mock_bounds_summary():
    """Minimal phase1_summary.json for bounds-mode experiment."""
    return {
        "schema_version": 4,
        "experiment_id": "upper_bound_exp",
        "fitness_mode": "upper_bound",
        "model_name": "gpt-oss:20b",
        "best_candidate_id": "cand_ub_001",
        "best_val_score": 0.85,
        "stop_reason": "max_generations",
        "success": True,
        "final_generation": 20,
        "train_metrics": {"spearman": 0.80},
        "val_metrics": {"spearman": 0.78},
        "test_metrics": {"spearman": 0.75},
        "sanity_metrics": {"spearman": 0.90},
        "novelty_ci": {
            "validation": {"max_ci_upper_abs_rho": 0.5, "novelty_passed": True},
            "test": {"max_ci_upper_abs_rho": 0.5, "novelty_passed": True},
        },
        "self_correction_stats": {"enabled": True, "attempted_repairs": 2},
        "bounds_metrics": {
            "val": {"bound_score": 0.85, "satisfaction_rate": 0.90},
            "test": {"bound_score": 0.82, "satisfaction_rate": 0.88},
        },
        "config": {
            "target_name": "average_shortest_path_length",
            "fitness_mode": "upper_bound",
            "max_generations": 20,
        },
    }


@pytest.fixture
def mock_baselines_summary():
    """Minimal baselines_summary.json."""
    return {
        "stat_baselines": {
            "linear_regression": {
                "status": "ok",
                "val_metrics": {"spearman": 0.75, "r2": 0.55},
                "test_metrics": {"spearman": 0.72, "r2": 0.50},
            },
            "random_forest": {
                "status": "ok",
                "val_metrics": {"spearman": 0.85, "r2": 0.70},
                "test_metrics": {"spearman": 0.83, "r2": 0.68},
            },
        },
        "pysr_baseline": {
            "status": "ok",
            "val_metrics": {"spearman": 0.88, "r2": 0.75},
            "test_metrics": {"spearman": 0.86, "r2": 0.72},
        },
    }


@pytest.fixture
def mock_ood_results():
    """Minimal ood_validation.json."""
    return {
        "large_random": {
            "spearman": 0.80,
            "valid_count": 90,
            "total_count": 100,
        },
        "extreme_params": {
            "spearman": 0.65,
            "valid_count": 40,
            "total_count": 50,
        },
        "special_topology": {
            "spearman": 0.70,
            "valid_count": 8,
            "total_count": 10,
        },
    }


@pytest.fixture
def mock_events():
    """Minimal events.jsonl lines."""
    events = []
    for gen in range(5):
        events.append(
            {
                "event_type": "generation_summary",
                "payload": {
                    "generation": gen,
                    "best_val_score": 0.5 + gen * 0.1,
                    "map_elites_stats": {"coverage": 5 + gen * 3},
                },
            }
        )
    return events


@pytest.fixture
def artifacts_dir(
    tmp_path,
    mock_phase1_summary,
    mock_baselines_summary,
    mock_ood_results,
    mock_events,
):
    """Set up a minimal artifacts directory with all expected files."""
    exp_dir = tmp_path / "experiment_map_elites_aspl"
    exp_dir.mkdir()

    (exp_dir / "phase1_summary.json").write_text(json.dumps(mock_phase1_summary))
    (exp_dir / "baselines_summary.json").write_text(json.dumps(mock_baselines_summary))

    ood_dir = exp_dir / "ood"
    ood_dir.mkdir()
    (ood_dir / "ood_validation.json").write_text(json.dumps(mock_ood_results))

    logs_dir = exp_dir / "logs"
    logs_dir.mkdir()
    with (logs_dir / "events.jsonl").open("w") as f:
        for event in mock_events:
            f.write(json.dumps(event) + "\n")

    return tmp_path


# ── Data loading tests ───────────────────────────────────────────────


def test_load_experiment_summary(artifacts_dir, analyze_module):
    summary = analyze_module.load_experiment_summary(artifacts_dir / "experiment_map_elites_aspl")
    assert isinstance(summary, dict)
    assert summary["experiment_id"] == "test_exp"
    assert "val_metrics" in summary
    assert "test_metrics" in summary


def test_load_experiment_summary_missing_file(tmp_path, analyze_module):
    summary = analyze_module.load_experiment_summary(tmp_path / "nonexistent")
    assert summary == {}


def test_load_baselines_summary(artifacts_dir, analyze_module):
    baselines = analyze_module.load_baselines_summary(artifacts_dir / "experiment_map_elites_aspl")
    assert isinstance(baselines, dict)
    assert "stat_baselines" in baselines
    assert "pysr_baseline" in baselines


def test_load_ood_results(artifacts_dir, analyze_module):
    ood = analyze_module.load_ood_results(artifacts_dir / "experiment_map_elites_aspl")
    assert isinstance(ood, dict)
    assert "large_random" in ood
    assert "extreme_params" in ood
    assert "special_topology" in ood


def test_load_event_log(artifacts_dir, analyze_module):
    events = analyze_module.load_event_log(artifacts_dir / "experiment_map_elites_aspl")
    assert isinstance(events, list)
    assert len(events) == 5
    assert events[0]["event_type"] == "generation_summary"


def test_load_event_log_missing_file(tmp_path, analyze_module):
    events = analyze_module.load_event_log(tmp_path / "nonexistent")
    assert events == []


# ── Analysis function tests ──────────────────────────────────────────


def test_extract_convergence_data(artifacts_dir, analyze_module):
    events = analyze_module.load_event_log(artifacts_dir / "experiment_map_elites_aspl")
    convergence = analyze_module.extract_convergence_data(events)
    assert isinstance(convergence, dict)
    assert "generations" in convergence
    assert "best_scores" in convergence
    assert len(convergence["generations"]) == 5
    assert len(convergence["best_scores"]) == 5
    # Scores should be monotonically non-decreasing in our mock
    for i in range(1, len(convergence["best_scores"])):
        assert convergence["best_scores"][i] >= convergence["best_scores"][i - 1]


def test_extract_convergence_data_includes_coverage(artifacts_dir, analyze_module):
    events = analyze_module.load_event_log(artifacts_dir / "experiment_map_elites_aspl")
    convergence = analyze_module.extract_convergence_data(events)
    assert "map_elites_coverage" in convergence
    assert len(convergence["map_elites_coverage"]) == 5


def test_build_comparison_table(artifacts_dir, mock_phase1_summary, analyze_module):
    experiments = {
        "map_elites_aspl": {
            "summary": mock_phase1_summary,
            "baselines": {
                "stat_baselines": {
                    "linear_regression": {
                        "status": "ok",
                        "val_metrics": {"spearman": 0.75},
                        "test_metrics": {"spearman": 0.72},
                    },
                },
                "pysr_baseline": {
                    "status": "ok",
                    "val_metrics": {"spearman": 0.88},
                    "test_metrics": {"spearman": 0.86},
                },
            },
            "ood": {
                "large_random": {"spearman": 0.80},
                "extreme_params": {"spearman": 0.65},
                "special_topology": {"spearman": 0.70},
            },
        },
    }
    table = analyze_module.build_comparison_table(experiments)
    assert isinstance(table, list)
    assert len(table) == 1
    row = table[0]
    assert row["experiment"] == "map_elites_aspl"
    assert "val_spearman" in row
    assert "test_spearman" in row
    assert "success" in row


def test_build_comparison_table_multiple_experiments(
    mock_phase1_summary, mock_bounds_summary, analyze_module
):
    experiments = {
        "map_elites_aspl": {"summary": mock_phase1_summary, "baselines": {}, "ood": {}},
        "upper_bound_aspl": {"summary": mock_bounds_summary, "baselines": {}, "ood": {}},
    }
    table = analyze_module.build_comparison_table(experiments)
    assert len(table) == 2
    names = {row["experiment"] for row in table}
    assert names == {"map_elites_aspl", "upper_bound_aspl"}


def test_build_comparison_table_empty_experiments(analyze_module):
    table = analyze_module.build_comparison_table({})
    assert table == []


# ── Output tests ─────────────────────────────────────────────────────


def test_write_analysis_report(
    tmp_path, mock_phase1_summary, mock_baselines_summary, analyze_module
):
    experiments = {
        "map_elites_aspl": {
            "summary": mock_phase1_summary,
            "baselines": mock_baselines_summary,
            "ood": {},
            "convergence": {
                "generations": [0, 1, 2],
                "best_scores": [0.5, 0.6, 0.7],
            },
        },
    }
    output_path = tmp_path / "report.md"
    analyze_module.write_analysis_report(experiments, output_path)
    assert output_path.exists()
    content = output_path.read_text()
    assert "map_elites_aspl" in content
    assert "val" in content.lower() or "validation" in content.lower()


def test_write_figure_data_json(tmp_path, mock_phase1_summary, analyze_module):
    experiments = {
        "map_elites_aspl": {
            "summary": mock_phase1_summary,
            "baselines": {},
            "ood": {},
            "convergence": {
                "generations": [0, 1, 2],
                "best_scores": [0.5, 0.6, 0.7],
            },
        },
    }
    output_path = tmp_path / "figure_data.json"
    analyze_module.write_figure_data_json(experiments, output_path)
    assert output_path.exists()
    data = json.loads(output_path.read_text())
    assert isinstance(data, dict)
    assert "map_elites_aspl" in data


def test_extract_convergence_data_from_log_file_streaming(artifacts_dir, analyze_module):
    events_path = artifacts_dir / "experiment_map_elites_aspl" / "logs" / "events.jsonl"
    convergence = analyze_module.extract_convergence_data_from_log_file(events_path)
    assert convergence["generations"] == [0, 1, 2, 3, 4]
    assert len(convergence["best_scores"]) == 5
    assert convergence["map_elites_coverage"] == [5, 8, 11, 14, 17]


def test_build_comparison_table_extracts_pysr_val_spearman(mock_phase1_summary, analyze_module):
    experiments = {
        "map_elites_aspl": {
            "summary": mock_phase1_summary,
            "baselines": {},
            "ood": {},
        }
    }
    table = analyze_module.build_comparison_table(experiments)
    assert table[0]["pysr_val_spearman"] == 0.88


def test_discover_experiments_reads_benchmark_and_streams_convergence(tmp_path, analyze_module):
    exp_dir = tmp_path / "experiment_map_elites_aspl"
    exp_dir.mkdir()
    summary = {"fitness_mode": "correlation", "success": True, "val_metrics": {"spearman": 0.1}}
    (exp_dir / "phase1_summary.json").write_text(json.dumps(summary), encoding="utf-8")
    logs_dir = exp_dir / "logs"
    logs_dir.mkdir()
    (logs_dir / "events.jsonl").write_text(
        json.dumps(
            {
                "event_type": "generation_summary",
                "payload": {
                    "generation": 1,
                    "best_val_score": 0.1,
                    "map_elites_stats": {"coverage": 2},
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )

    bench_dir = tmp_path / "benchmark_aspl" / "benchmark_20260101T000000Z"
    bench_dir.mkdir(parents=True)
    (bench_dir / "benchmark_summary.json").write_text(
        json.dumps({"runs": [{"seed": 1, "val_spearman": 0.2, "test_spearman": 0.3}]}),
        encoding="utf-8",
    )

    experiments = analyze_module.discover_experiments(tmp_path)
    assert "experiment_map_elites_aspl" in experiments
    assert experiments["experiment_map_elites_aspl"]["convergence"]["best_scores"] == [0.1]
    assert "benchmark/benchmark_20260101T000000Z" in experiments


def test_extract_acceptance_funnel(analyze_module):
    events = [
        {"event_type": "candidate_rejected", "payload": {"generation": 0, "reason": "x"}},
        {"event_type": "candidate_evaluated", "payload": {"generation": 0}},
        {"event_type": "candidate_evaluated", "payload": {"generation": 1}},
    ]
    funnel = analyze_module.extract_acceptance_funnel(events)
    assert funnel["generations"] == [0, 1]
    assert funnel["attempted"] == [2, 1]
    assert funnel["evaluated"] == [1, 1]
    assert funnel["rejected"] == [1, 0]


def test_build_seed_aggregates(analyze_module):
    experiments = {
        "neurips_matrix/map_elites_aspl_full/seed_11": {
            "summary": {
                "success": True,
                "val_metrics": {"spearman": 0.9},
                "test_metrics": {"spearman": 0.8},
                "best_candidate_code": "def f(s):\n    return 1",
            }
        },
        "neurips_matrix/map_elites_aspl_full/seed_22": {
            "summary": {
                "success": False,
                "val_metrics": {"spearman": 0.7},
                "test_metrics": {"spearman": 0.6},
                "best_candidate_code": "def f(s):\n    return 2",
            }
        },
    }
    aggregates = analyze_module.build_seed_aggregates(experiments)
    key = "neurips_matrix/map_elites_aspl_full"
    assert key in aggregates
    assert aggregates[key]["seed_count"] == 2
    assert aggregates[key]["val_spearman"]["mean"] == pytest.approx(0.8)


def test_discover_experiments_reads_neurips_matrix_seed_dirs(tmp_path, analyze_module):
    seed_dir = tmp_path / "neurips_matrix_2026_final" / "map_elites_aspl_full" / "seed_11"
    seed_dir.mkdir(parents=True)
    (seed_dir / "phase1_summary.json").write_text(
        json.dumps(
            {
                "fitness_mode": "correlation",
                "success": True,
                "val_metrics": {"spearman": 0.5},
                "test_metrics": {"spearman": 0.4},
            }
        ),
        encoding="utf-8",
    )
    (seed_dir / "logs").mkdir()
    (seed_dir / "logs" / "events.jsonl").write_text("", encoding="utf-8")

    experiments = analyze_module.discover_experiments(tmp_path)
    assert "neurips_matrix_2026_final/map_elites_aspl_full/seed_11" in experiments


def test_write_figure_data_json_includes_aggregates(tmp_path, mock_phase1_summary, analyze_module):
    experiments = {
        "neurips_matrix/map_elites_aspl_full/seed_11": {
            "summary": mock_phase1_summary,
            "baselines": {},
            "ood": {},
            "convergence": {},
            "acceptance_funnel": {},
            "repair_breakdown": {},
            "bounds_diagnostics": {},
        },
    }
    output_path = tmp_path / "figure_data.json"
    analyze_module.write_figure_data_json(experiments, output_path)
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert "__aggregates__" in payload
    assert "appendix_seed_aggregates" in payload
    assert "appendix_small_data_tradeoff" in payload
    assert "appendix_bounds_diagnostics" in payload
    assert "appendix_runtime_summary" in payload


def test_discover_matrix_summaries(tmp_path, analyze_module):
    matrix_root = tmp_path / "neurips_matrix_2026_final"
    matrix_root.mkdir()
    matrix_payload = {
        "runs": [{"experiment": "map_elites_aspl_full", "status": 0, "success": True}]
    }
    (matrix_root / "matrix_summary.json").write_text(
        json.dumps(matrix_payload),
        encoding="utf-8",
    )
    discovered = analyze_module.discover_matrix_summaries(tmp_path)
    assert "neurips_matrix_2026_final" in discovered


def test_write_appendix_tables_tex(tmp_path, analyze_module):
    payload = {
        "appendix_seed_aggregates": {
            "neurips_matrix/map_elites_aspl_full": {
                "seed_count": 2,
                "success_count": 1,
                "val_spearman": {"mean": 0.8, "std": 0.1, "ci95_half_width": 0.2},
                "test_spearman": {"mean": 0.7, "std": 0.1, "ci95_half_width": 0.2},
            }
        },
        "appendix_bounds_diagnostics": {
            "experiment_upper_bound_aspl": {
                "val_bound_score": 0.8,
                "val_satisfaction_rate": 0.9,
                "test_bound_score": 0.7,
                "test_satisfaction_rate": 0.85,
            }
        },
        "appendix_runtime_summary": {
            "map_elites_aspl_full": {
                "duration_sec": {"mean": 100.0, "std": 10.0},
                "total_runs": 5,
                "completed_runs": 5,
                "criteria_success_runs": 2,
            }
        },
    }
    out = tmp_path / "appendix_tables.tex"
    analyze_module.write_appendix_tables_tex(payload, out)
    content = out.read_text(encoding="utf-8")
    assert "tab:appendix_seed_aggregates" in content
    assert "tab:appendix_bounds" in content
    assert "tab:appendix_runtime" in content
