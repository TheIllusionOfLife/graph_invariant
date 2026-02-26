"""Tests for analysis/generate_harmony_figures.py — data parsing and transformation.

Focuses on the pure functions that parse JSONL events, build heatmap matrices,
and transform metrics CSVs. Does NOT test matplotlib rendering (tested via
smoke-test that figures are callable without error on fixture data).
"""

from __future__ import annotations

import json
import textwrap
from pathlib import Path
from typing import Any

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Fixtures — synthetic data matching real pipeline formats
# ---------------------------------------------------------------------------


def _make_events_jsonl(generations: int = 5) -> str:
    """Return JSONL string with generation_summary events."""
    lines: list[str] = []
    for g in range(1, generations + 1):
        event = {
            "event": "generation_summary",
            "experiment_id": "test-exp",
            "generation": g,
            "valid_rate": 0.3 + 0.05 * g,  # 0.35 → 0.55
            "pipeline_valid_rate": 0.4 + 0.05 * g,
            "best_harmony_gain": 0.01 * g,
            "no_improve_count": max(0, g - 3),
        }
        lines.append(json.dumps(event))
    # Add a non-generation event that should be ignored
    lines.append(json.dumps({"event": "proposal_accepted", "generation": 1}))
    return "\n".join(lines) + "\n"


def _make_checkpoint(num_bins: int = 5) -> dict[str, Any]:
    """Return a checkpoint dict with a populated archive."""
    cells: dict[str, dict[str, Any]] = {}
    for r in range(num_bins):
        for c in range(num_bins):
            if (r + c) % 3 == 0:  # Sparse: ~8 of 25 cells
                cells[f"{r},{c}"] = {
                    "proposal": {
                        "proposal_type": "ADD_EDGE",
                        "claim": f"claim-{r}-{c}",
                        "justification": "test",
                        "falsification_condition": "test",
                        "kg_domain": "astronomy",
                        "source_entity": "star",
                        "target_entity": "planet",
                        "edge_type": "explains",
                    },
                    "fitness_signal": 0.1 * r + 0.01 * c,
                }
    return {
        "experiment_id": "test-exp",
        "generation": 5,
        "islands": {},
        "rng_seed": 42,
        "archive": {
            "num_bins": num_bins,
            "archive_id": "primary",
            "cells": cells,
        },
        "best_harmony_gain": 0.05,
        "no_improve_count": 2,
    }


def _make_metrics_csv() -> str:
    """Return CSV matching metrics_table.py output format."""
    return textwrap.dedent("""\
        domain,random_hits10,freq_hits10,distmult_hits10,harmony_hits10,mrr_random,mrr_distmult,mrr_harmony
        astronomy,0.12,0.35,0.58,0.67,0.06,0.37,0.43
        physics,0.10,0.32,0.55,0.63,0.05,0.35,0.41
        materials,0.11,0.30,0.52,0.61,0.05,0.32,0.39
    """)


@pytest.fixture()
def events_dir(tmp_path: Path) -> Path:
    """Create a temporary directory with harmony_events.jsonl."""
    logs_dir = tmp_path / "logs"
    logs_dir.mkdir()
    (logs_dir / "harmony_events.jsonl").write_text(_make_events_jsonl())
    return tmp_path


@pytest.fixture()
def checkpoint_dir(tmp_path: Path) -> Path:
    """Create a temporary directory with checkpoint.json."""
    (tmp_path / "checkpoint.json").write_text(json.dumps(_make_checkpoint()))
    return tmp_path


@pytest.fixture()
def full_domain_dir(tmp_path: Path) -> Path:
    """Create a temporary directory with both checkpoint and events."""
    logs_dir = tmp_path / "logs"
    logs_dir.mkdir()
    (logs_dir / "harmony_events.jsonl").write_text(_make_events_jsonl())
    (tmp_path / "checkpoint.json").write_text(json.dumps(_make_checkpoint()))
    return tmp_path


@pytest.fixture()
def metrics_csv(tmp_path: Path) -> Path:
    """Create a temporary metrics CSV."""
    csv_path = tmp_path / "metrics_table.csv"
    csv_path.write_text(_make_metrics_csv())
    return csv_path


# ===========================================================================
# Tests: parse_convergence_data
# ===========================================================================


class TestParseConvergenceData:
    def test_returns_correct_length(self, events_dir: Path) -> None:
        from analysis.generate_harmony_figures import parse_convergence_data

        data = parse_convergence_data(events_dir)
        # 5 generation_summary events, ignoring the proposal_accepted event
        assert len(data["generation"]) == 5

    def test_generations_are_sequential(self, events_dir: Path) -> None:
        from analysis.generate_harmony_figures import parse_convergence_data

        data = parse_convergence_data(events_dir)
        assert data["generation"] == [1, 2, 3, 4, 5]

    def test_valid_rate_values(self, events_dir: Path) -> None:
        from analysis.generate_harmony_figures import parse_convergence_data

        data = parse_convergence_data(events_dir)
        expected = [0.35, 0.40, 0.45, 0.50, 0.55]
        np.testing.assert_allclose(data["valid_rate"], expected, atol=1e-10)

    def test_best_harmony_gain_values(self, events_dir: Path) -> None:
        from analysis.generate_harmony_figures import parse_convergence_data

        data = parse_convergence_data(events_dir)
        expected = [0.01, 0.02, 0.03, 0.04, 0.05]
        np.testing.assert_allclose(data["best_harmony_gain"], expected, atol=1e-10)

    def test_missing_events_file_returns_empty(self, tmp_path: Path) -> None:
        from analysis.generate_harmony_figures import parse_convergence_data

        data = parse_convergence_data(tmp_path)
        assert data["generation"] == []
        assert data["valid_rate"] == []
        assert data["best_harmony_gain"] == []

    def test_empty_events_file(self, tmp_path: Path) -> None:
        from analysis.generate_harmony_figures import parse_convergence_data

        logs_dir = tmp_path / "logs"
        logs_dir.mkdir()
        (logs_dir / "harmony_events.jsonl").write_text("")
        data = parse_convergence_data(tmp_path)
        assert data["generation"] == []

    def test_malformed_json_lines_skipped(self, tmp_path: Path) -> None:
        from analysis.generate_harmony_figures import parse_convergence_data

        logs_dir = tmp_path / "logs"
        logs_dir.mkdir()
        ev1 = json.dumps(
            {
                "event": "generation_summary",
                "generation": 1,
                "valid_rate": 0.5,
                "best_harmony_gain": 0.01,
            }
        )
        ev2 = json.dumps(
            {
                "event": "generation_summary",
                "generation": 2,
                "valid_rate": 0.6,
                "best_harmony_gain": 0.02,
            }
        )
        content = f"{ev1}\nnot valid json\n{ev2}\n"
        (logs_dir / "harmony_events.jsonl").write_text(content)
        data = parse_convergence_data(tmp_path)
        assert len(data["generation"]) == 2


# ===========================================================================
# Tests: build_heatmap_matrix
# ===========================================================================


class TestBuildHeatmapMatrix:
    def test_output_shape(self, checkpoint_dir: Path) -> None:
        from analysis.generate_harmony_figures import build_heatmap_matrix

        matrix, num_bins = build_heatmap_matrix(checkpoint_dir)
        assert matrix.shape == (5, 5)
        assert num_bins == 5

    def test_unoccupied_cells_are_nan(self, checkpoint_dir: Path) -> None:
        from analysis.generate_harmony_figures import build_heatmap_matrix

        matrix, _ = build_heatmap_matrix(checkpoint_dir)
        # Cell (0, 1) has (0+1)%3 == 1, so unoccupied → NaN
        assert np.isnan(matrix[0, 1])

    def test_occupied_cells_have_correct_fitness(self, checkpoint_dir: Path) -> None:
        from analysis.generate_harmony_figures import build_heatmap_matrix

        matrix, _ = build_heatmap_matrix(checkpoint_dir)
        # Cell (0, 0): (0+0)%3 == 0 → fitness = 0.1*0 + 0.01*0 = 0.0
        assert matrix[0, 0] == pytest.approx(0.0)
        # Cell (0, 3): (0+3)%3 == 0 → fitness = 0.1*0 + 0.01*3 = 0.03
        assert matrix[0, 3] == pytest.approx(0.03)
        # Cell (3, 0): (3+0)%3 == 0 → fitness = 0.1*3 + 0.01*0 = 0.3
        assert matrix[3, 0] == pytest.approx(0.3)

    def test_missing_checkpoint_returns_empty(self, tmp_path: Path) -> None:
        from analysis.generate_harmony_figures import build_heatmap_matrix

        matrix, num_bins = build_heatmap_matrix(tmp_path)
        assert matrix.size == 0
        assert num_bins == 0

    def test_checkpoint_without_archive(self, tmp_path: Path) -> None:
        from analysis.generate_harmony_figures import build_heatmap_matrix

        checkpoint = _make_checkpoint()
        checkpoint["archive"] = None
        (tmp_path / "checkpoint.json").write_text(json.dumps(checkpoint))
        matrix, num_bins = build_heatmap_matrix(tmp_path)
        assert matrix.size == 0
        assert num_bins == 0


# ===========================================================================
# Tests: parse_metrics_csv
# ===========================================================================


class TestParseMetricsCsv:
    def test_returns_correct_domains(self, metrics_csv: Path) -> None:
        from analysis.generate_harmony_figures import parse_metrics_csv

        data = parse_metrics_csv(metrics_csv)
        assert data["domains"] == ["astronomy", "physics", "materials"]

    def test_hits10_values(self, metrics_csv: Path) -> None:
        from analysis.generate_harmony_figures import parse_metrics_csv

        data = parse_metrics_csv(metrics_csv)
        np.testing.assert_allclose(data["random"], [0.12, 0.10, 0.11], atol=1e-10)
        np.testing.assert_allclose(data["frequency"], [0.35, 0.32, 0.30], atol=1e-10)
        np.testing.assert_allclose(data["distmult"], [0.58, 0.55, 0.52], atol=1e-10)
        np.testing.assert_allclose(data["harmony"], [0.67, 0.63, 0.61], atol=1e-10)

    def test_missing_csv_returns_empty(self, tmp_path: Path) -> None:
        from analysis.generate_harmony_figures import parse_metrics_csv

        data = parse_metrics_csv(tmp_path / "nonexistent.csv")
        assert data["domains"] == []


# ===========================================================================
# Tests: build_ablation_data
# ===========================================================================


class TestBuildAblationData:
    def test_returns_weight_configs_and_scores(self) -> None:
        from analysis.generate_harmony_figures import build_ablation_data

        data = build_ablation_data()
        # Should have 6 weight configs from WEIGHT_GRID
        assert len(data["labels"]) == 6
        assert len(data["scores"]) == 6
        # All scores should be non-negative
        assert all(s >= 0 for s in data["scores"])

    def test_labels_contain_alpha_beta(self) -> None:
        from analysis.generate_harmony_figures import build_ablation_data

        data = build_ablation_data()
        # Each label should describe the weight config
        for label in data["labels"]:
            assert "α" in label or "a" in label.lower()


# ===========================================================================
# Tests: Plotting smoke tests (no visual assertion, just no crash)
# ===========================================================================


class TestPlottingSmoke:
    """Verify that plot functions run without error on fixture data."""

    def test_plot_convergence(self, full_domain_dir: Path, tmp_path: Path) -> None:
        from analysis.generate_harmony_figures import (
            parse_convergence_data,
            plot_convergence,
        )

        domain_data = {"test_domain": parse_convergence_data(full_domain_dir)}
        out = tmp_path / "convergence.pdf"
        plot_convergence(domain_data, out)
        assert out.exists()
        assert out.stat().st_size > 0

    def test_plot_heatmap(self, full_domain_dir: Path, tmp_path: Path) -> None:
        from analysis.generate_harmony_figures import (
            build_heatmap_matrix,
            plot_archive_heatmap,
        )

        matrix, num_bins = build_heatmap_matrix(full_domain_dir)
        domain_data = {"test_domain": (matrix, num_bins)}
        out = tmp_path / "archive_heatmap.pdf"
        plot_archive_heatmap(domain_data, out)
        assert out.exists()
        assert out.stat().st_size > 0

    def test_plot_baseline_comparison(self, metrics_csv: Path, tmp_path: Path) -> None:
        from analysis.generate_harmony_figures import (
            parse_metrics_csv,
            plot_baseline_comparison,
        )

        data = parse_metrics_csv(metrics_csv)
        out = tmp_path / "baseline_comparison.pdf"
        plot_baseline_comparison(data, out)
        assert out.exists()
        assert out.stat().st_size > 0

    def test_plot_ablation_weights(self, tmp_path: Path) -> None:
        from analysis.generate_harmony_figures import (
            build_ablation_data,
            plot_ablation_weights,
        )

        data = build_ablation_data()
        out = tmp_path / "ablation_weights.pdf"
        plot_ablation_weights(data, out)
        assert out.exists()
        assert out.stat().st_size > 0

    def test_plot_convergence_skips_empty_domain(self, tmp_path: Path) -> None:
        """Empty convergence data should not crash, just produce fewer subplots."""
        from analysis.generate_harmony_figures import (
            parse_convergence_data,
            plot_convergence,
        )

        domain_data = {"empty_domain": parse_convergence_data(tmp_path)}
        out = tmp_path / "convergence_empty.pdf"
        plot_convergence(domain_data, out)
        assert out.exists()

    def test_plot_heatmap_skips_empty(self, tmp_path: Path) -> None:
        """Empty heatmap data should not crash."""
        from analysis.generate_harmony_figures import (
            build_heatmap_matrix,
            plot_archive_heatmap,
        )

        matrix, num_bins = build_heatmap_matrix(tmp_path)
        domain_data = {"empty_domain": (matrix, num_bins)}
        out = tmp_path / "heatmap_empty.pdf"
        plot_archive_heatmap(domain_data, out)
        assert out.exists()
