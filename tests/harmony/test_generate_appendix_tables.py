"""Tests for analysis.generate_appendix_tables — TDD written before implementation.

Verifies:
  - extract_runtime_stats returns correct mean/std/n from run.log files
  - build_factor_decomp_table reads CSV and produces LaTeX rows
  - build_statistical_tests_table reads JSON and produces LaTeX rows
  - generate_appendix_tables produces output with no stale references
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "analysis"))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_run_log(path: Path, real_sec: float) -> None:
    path.write_text(f"Some output\nreal {real_sec:.2f}\nuser 1.00\nsys 0.50\n")


# ---------------------------------------------------------------------------
# Tests for extract_runtime_stats
# ---------------------------------------------------------------------------


class TestExtractRuntimeStats:
    def test_single_seed(self, tmp_path: Path) -> None:
        from generate_appendix_tables import extract_runtime_stats

        domain_dir = tmp_path / "astronomy"
        seed_dir = domain_dir / "seed_42"
        seed_dir.mkdir(parents=True)
        _write_run_log(seed_dir / "run.log", 1500.0)

        stats = extract_runtime_stats(domain_dir)
        assert stats["n"] == 1
        assert stats["mean_sec"] == pytest.approx(1500.0)
        assert stats["std_sec"] == pytest.approx(0.0)

    def test_multiple_seeds(self, tmp_path: Path) -> None:
        from generate_appendix_tables import extract_runtime_stats

        domain_dir = tmp_path / "physics"
        for seed, t in [("seed_42", 1000.0), ("seed_123", 2000.0)]:
            d = domain_dir / seed
            d.mkdir(parents=True)
            _write_run_log(d / "run.log", t)

        stats = extract_runtime_stats(domain_dir)
        assert stats["n"] == 2
        assert stats["mean_sec"] == pytest.approx(1500.0)
        assert stats["std_sec"] == pytest.approx(500.0)

    def test_missing_directory_returns_none(self, tmp_path: Path) -> None:
        from generate_appendix_tables import extract_runtime_stats

        stats = extract_runtime_stats(tmp_path / "nonexistent")
        assert stats is None

    def test_no_run_logs_returns_none(self, tmp_path: Path) -> None:
        from generate_appendix_tables import extract_runtime_stats

        domain_dir = tmp_path / "materials"
        seed_dir = domain_dir / "seed_42"
        seed_dir.mkdir(parents=True)
        # no run.log written

        stats = extract_runtime_stats(domain_dir)
        assert stats is None

    def test_skips_seeds_without_real_line(self, tmp_path: Path) -> None:
        from generate_appendix_tables import extract_runtime_stats

        domain_dir = tmp_path / "physics"
        good_dir = domain_dir / "seed_42"
        good_dir.mkdir(parents=True)
        _write_run_log(good_dir / "run.log", 800.0)

        bad_dir = domain_dir / "seed_99"
        bad_dir.mkdir(parents=True)
        (bad_dir / "run.log").write_text("no timing here\n")

        stats = extract_runtime_stats(domain_dir)
        assert stats["n"] == 1
        assert stats["mean_sec"] == pytest.approx(800.0)


# ---------------------------------------------------------------------------
# Tests for build_factor_decomp_table
# ---------------------------------------------------------------------------


class TestBuildFactorDecompTable:
    def _write_csv(self, path: Path) -> None:
        rows = [
            ["config", "domain", "generations", "best_harmony_gain", "archive_size"],
            ["llm_only", "astronomy", "11", "1.0", "1"],
            ["no_qd", "astronomy", "11", "0.052", "1"],
            ["harmony_only", "astronomy", "20", "0.054", "5"],
            ["full", "astronomy", "20", "0.048", "3"],
        ]
        with path.open("w", newline="") as f:
            csv.writer(f).writerows(rows)

    def test_returns_latex_string(self, tmp_path: Path) -> None:
        from generate_appendix_tables import build_factor_decomp_table

        csv_path = tmp_path / "factor_decomposition.csv"
        self._write_csv(csv_path)

        result = build_factor_decomp_table(csv_path)
        assert isinstance(result, str)
        assert "\\begin{tabular}" in result

    def test_contains_domain_name(self, tmp_path: Path) -> None:
        from generate_appendix_tables import build_factor_decomp_table

        csv_path = tmp_path / "factor_decomposition.csv"
        self._write_csv(csv_path)

        result = build_factor_decomp_table(csv_path)
        assert "astronomy" in result.lower() or "Astronomy" in result

    def test_contains_config_names(self, tmp_path: Path) -> None:
        from generate_appendix_tables import build_factor_decomp_table

        csv_path = tmp_path / "factor_decomposition.csv"
        self._write_csv(csv_path)

        result = build_factor_decomp_table(csv_path)
        assert "harmony" in result.lower()
        assert "no" in result.lower()

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        from generate_appendix_tables import build_factor_decomp_table

        with pytest.raises(FileNotFoundError):
            build_factor_decomp_table(tmp_path / "nonexistent.csv")


# ---------------------------------------------------------------------------
# Tests for build_statistical_tests_table
# ---------------------------------------------------------------------------


class TestBuildStatisticalTestsTable:
    def _write_json(self, path: Path) -> None:
        data = {
            "astronomy": {
                "hits10_bootstrap": {
                    "mean_diff": 0.0,
                    "ci_low": 0.0,
                    "ci_high": 0.0,
                    "p_value": 1.0,
                },
                "hits10_cliffs_delta": 0.0,
            },
            "wikidata_materials": {
                "hits10_bootstrap": {
                    "mean_diff": 0.012,
                    "ci_low": -0.019,
                    "ci_high": 0.039,
                    "p_value": 0.378,
                },
                "hits10_cliffs_delta": 0.12,
            },
        }
        path.write_text(json.dumps(data))

    def test_returns_latex_string(self, tmp_path: Path) -> None:
        from generate_appendix_tables import build_statistical_tests_table

        json_path = tmp_path / "statistical_tests.json"
        self._write_json(json_path)

        result = build_statistical_tests_table(json_path)
        assert isinstance(result, str)
        assert "\\begin{tabular}" in result

    def test_contains_domain_names(self, tmp_path: Path) -> None:
        from generate_appendix_tables import build_statistical_tests_table

        json_path = tmp_path / "statistical_tests.json"
        self._write_json(json_path)

        result = build_statistical_tests_table(json_path)
        assert "astronomy" in result.lower() or "Astronomy" in result

    def test_contains_pvalue(self, tmp_path: Path) -> None:
        from generate_appendix_tables import build_statistical_tests_table

        json_path = tmp_path / "statistical_tests.json"
        self._write_json(json_path)

        result = build_statistical_tests_table(json_path)
        assert "0.378" in result or "1.0" in result

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        from generate_appendix_tables import build_statistical_tests_table

        with pytest.raises(FileNotFoundError):
            build_statistical_tests_table(tmp_path / "nonexistent.json")


# ---------------------------------------------------------------------------
# Tests for generate_appendix_tables (integration)
# ---------------------------------------------------------------------------


class TestGenerateAppendixTablesOutput:
    def test_no_stale_references(self, tmp_path: Path) -> None:
        """Output must contain no old graph-invariant project terms."""
        from generate_appendix_tables import generate_appendix_tables

        # Minimal inputs
        csv_path = tmp_path / "factor_decomposition.csv"
        rows = [
            ["config", "domain", "generations", "best_harmony_gain", "archive_size"],
            ["harmony_only", "astronomy", "20", "0.054", "5"],
        ]
        with csv_path.open("w", newline="") as f:
            csv.writer(f).writerows(rows)

        json_path = tmp_path / "statistical_tests.json"
        json_path.write_text(
            json.dumps(
                {
                    "astronomy": {
                        "hits10_bootstrap": {"p_value": 1.0},
                        "hits10_cliffs_delta": 0.0,
                    }
                }
            )
        )

        out_path = tmp_path / "appendix_tables_generated.tex"
        generate_appendix_tables(
            factor_csv=csv_path,
            stat_tests_json=json_path,
            mlx_base_dir=tmp_path / "mlx_runs",  # no data → skip runtime table
            output_path=out_path,
        )

        content = out_path.read_text()
        stale_terms = ["PySR", "aspl", "algebraic_connectivity", "self-correction", "SC attempted"]
        for term in stale_terms:
            assert term not in content, f"Stale term '{term}' found in output"

    def test_build_runtime_table_warns_incomplete_seeds(
        self, tmp_path: Path, recwarn: pytest.WarningsChecker
    ) -> None:
        from generate_appendix_tables import build_runtime_table

        # Build a domain dir with only 3 seed logs (< EXPECTED_SEEDS = 10)
        domain_dir = tmp_path / "astronomy"
        for i in range(3):
            seed_dir = domain_dir / f"seed_{i}"
            seed_dir.mkdir(parents=True)
            _write_run_log(seed_dir / "run.log", float(1000 + i * 100))

        build_runtime_table(tmp_path)

        warning_msgs = [str(w.message) for w in recwarn.list]
        assert any("EXPECTED_SEEDS" in m or "incomplete" in m.lower() for m in warning_msgs), (
            f"Expected incomplete-seeds warning; got: {warning_msgs}"
        )

    def test_output_file_created(self, tmp_path: Path) -> None:
        from generate_appendix_tables import generate_appendix_tables

        csv_path = tmp_path / "factor_decomposition.csv"
        rows = [
            ["config", "domain", "generations", "best_harmony_gain", "archive_size"],
            ["harmony_only", "astronomy", "20", "0.054", "5"],
        ]
        with csv_path.open("w", newline="") as f:
            csv.writer(f).writerows(rows)

        json_path = tmp_path / "statistical_tests.json"
        json_path.write_text(
            json.dumps(
                {
                    "astronomy": {
                        "hits10_bootstrap": {"p_value": 1.0},
                        "hits10_cliffs_delta": 0.0,
                    }
                }
            )
        )

        out_path = tmp_path / "out.tex"
        generate_appendix_tables(
            factor_csv=csv_path,
            stat_tests_json=json_path,
            mlx_base_dir=None,
            output_path=out_path,
        )
        assert out_path.exists()
        assert out_path.stat().st_size > 0
