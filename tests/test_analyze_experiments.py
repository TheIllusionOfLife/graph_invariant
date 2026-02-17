"""Tests for analysis/analyze_experiments.py functions.

Follows test_ood_validation.py patterns: tmp_path fixtures with minimal mock
JSON data, asserting output shapes and key presence rather than exact values.
"""

import json
import sys
from pathlib import Path

import pytest

# Make analysis/ importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "analysis"))


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


def test_load_experiment_summary(artifacts_dir):
    from analyze_experiments import load_experiment_summary

    summary = load_experiment_summary(artifacts_dir / "experiment_map_elites_aspl")
    assert isinstance(summary, dict)
    assert summary["experiment_id"] == "test_exp"
    assert "val_metrics" in summary
    assert "test_metrics" in summary


def test_load_experiment_summary_missing_file(tmp_path):
    from analyze_experiments import load_experiment_summary

    summary = load_experiment_summary(tmp_path / "nonexistent")
    assert summary == {}


def test_load_baselines_summary(artifacts_dir):
    from analyze_experiments import load_baselines_summary

    baselines = load_baselines_summary(artifacts_dir / "experiment_map_elites_aspl")
    assert isinstance(baselines, dict)
    assert "stat_baselines" in baselines
    assert "pysr_baseline" in baselines


def test_load_ood_results(artifacts_dir):
    from analyze_experiments import load_ood_results

    ood = load_ood_results(artifacts_dir / "experiment_map_elites_aspl")
    assert isinstance(ood, dict)
    assert "large_random" in ood
    assert "extreme_params" in ood
    assert "special_topology" in ood


def test_load_event_log(artifacts_dir):
    from analyze_experiments import load_event_log

    events = load_event_log(artifacts_dir / "experiment_map_elites_aspl")
    assert isinstance(events, list)
    assert len(events) == 5
    assert events[0]["event_type"] == "generation_summary"


def test_load_event_log_missing_file(tmp_path):
    from analyze_experiments import load_event_log

    events = load_event_log(tmp_path / "nonexistent")
    assert events == []


# ── Analysis function tests ──────────────────────────────────────────


def test_extract_convergence_data(artifacts_dir):
    from analyze_experiments import extract_convergence_data, load_event_log

    events = load_event_log(artifacts_dir / "experiment_map_elites_aspl")
    convergence = extract_convergence_data(events)
    assert isinstance(convergence, dict)
    assert "generations" in convergence
    assert "best_scores" in convergence
    assert len(convergence["generations"]) == 5
    assert len(convergence["best_scores"]) == 5
    # Scores should be monotonically non-decreasing in our mock
    for i in range(1, len(convergence["best_scores"])):
        assert convergence["best_scores"][i] >= convergence["best_scores"][i - 1]


def test_extract_convergence_data_includes_coverage(artifacts_dir):
    from analyze_experiments import extract_convergence_data, load_event_log

    events = load_event_log(artifacts_dir / "experiment_map_elites_aspl")
    convergence = extract_convergence_data(events)
    assert "map_elites_coverage" in convergence
    assert len(convergence["map_elites_coverage"]) == 5


def test_build_comparison_table(artifacts_dir, mock_phase1_summary):
    from analyze_experiments import build_comparison_table

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
    table = build_comparison_table(experiments)
    assert isinstance(table, list)
    assert len(table) == 1
    row = table[0]
    assert row["experiment"] == "map_elites_aspl"
    assert "val_spearman" in row
    assert "test_spearman" in row
    assert "success" in row


def test_build_comparison_table_multiple_experiments(mock_phase1_summary, mock_bounds_summary):
    from analyze_experiments import build_comparison_table

    experiments = {
        "map_elites_aspl": {"summary": mock_phase1_summary, "baselines": {}, "ood": {}},
        "upper_bound_aspl": {"summary": mock_bounds_summary, "baselines": {}, "ood": {}},
    }
    table = build_comparison_table(experiments)
    assert len(table) == 2
    names = {row["experiment"] for row in table}
    assert names == {"map_elites_aspl", "upper_bound_aspl"}


def test_build_comparison_table_empty_experiments():
    from analyze_experiments import build_comparison_table

    table = build_comparison_table({})
    assert table == []


# ── Output tests ─────────────────────────────────────────────────────


def test_write_analysis_report(tmp_path, mock_phase1_summary, mock_baselines_summary):
    from analyze_experiments import write_analysis_report

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
    write_analysis_report(experiments, output_path)
    assert output_path.exists()
    content = output_path.read_text()
    assert "map_elites_aspl" in content
    assert "val" in content.lower() or "validation" in content.lower()


def test_write_figure_data_json(tmp_path, mock_phase1_summary):
    from analyze_experiments import write_figure_data_json

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
    write_figure_data_json(experiments, output_path)
    assert output_path.exists()
    data = json.loads(output_path.read_text())
    assert isinstance(data, dict)
    assert "map_elites_aspl" in data
