"""Tests for analysis.backtesting — hold-out precision/recall.

TDD: written BEFORE implementation. Verifies:
  - recall_at_n returns correct values for empty/full/partial cases
  - BacktestResult has expected fields
  - backtest_proposals correctly matches proposals to hidden edges
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "analysis"))


class TestRecallAtN:
    def test_empty_proposals(self):
        from precision_at_n import recall_at_n

        result = recall_at_n([], [("a", "b", "DEPENDS_ON")], n=5)
        assert result == pytest.approx(0.0)

    def test_empty_withheld(self):
        from precision_at_n import recall_at_n

        result = recall_at_n([("a", "b", "DEPENDS_ON")], [], n=5)
        assert result == pytest.approx(0.0)

    def test_full_recall(self):
        from precision_at_n import recall_at_n

        withheld = [("a", "b", "DEPENDS_ON"), ("c", "d", "DERIVES")]
        proposals = [("a", "b", "DEPENDS_ON"), ("c", "d", "DERIVES"), ("e", "f", "EXPLAINS")]
        result = recall_at_n(proposals, withheld, n=3)
        assert result == pytest.approx(1.0)

    def test_partial_recall(self):
        from precision_at_n import recall_at_n

        withheld = [("a", "b", "DEPENDS_ON"), ("c", "d", "DERIVES")]
        proposals = [("a", "b", "DEPENDS_ON"), ("e", "f", "EXPLAINS")]
        result = recall_at_n(proposals, withheld, n=2)
        assert result == pytest.approx(0.5)

    def test_recall_respects_n(self):
        from precision_at_n import recall_at_n

        withheld = [("a", "b", "DEPENDS_ON"), ("c", "d", "DERIVES")]
        # Both matches are at position 1 and 3, but n=1 only looks at top 1
        proposals = [("a", "b", "DEPENDS_ON"), ("e", "f", "EXPLAINS"), ("c", "d", "DERIVES")]
        result = recall_at_n(proposals, withheld, n=1)
        assert result == pytest.approx(0.5)  # 1 out of 2 withheld found


class TestBacktestProposals:
    def test_returns_backtest_result(self):
        from backtesting import BacktestResult, backtest_proposals

        proposals_ranked = [("a", "b", "DEPENDS_ON"), ("c", "d", "DERIVES")]
        hidden_edges = [("a", "b", "DEPENDS_ON"), ("e", "f", "EXPLAINS")]
        result = backtest_proposals(proposals_ranked, hidden_edges)
        assert isinstance(result, BacktestResult)

    def test_result_has_expected_fields(self):
        from backtesting import backtest_proposals

        result = backtest_proposals(
            [("a", "b", "DEPENDS_ON")],
            [("a", "b", "DEPENDS_ON")],
        )
        assert hasattr(result, "n_proposals")
        assert hasattr(result, "n_hidden")
        assert hasattr(result, "precision_at_5")
        assert hasattr(result, "precision_at_10")
        assert hasattr(result, "precision_at_20")
        assert hasattr(result, "recall_at_5")
        assert hasattr(result, "recall_at_10")
        assert hasattr(result, "recall_at_20")
        assert hasattr(result, "matched_triples")

    def test_empty_archive(self):
        from backtesting import backtest_proposals

        result = backtest_proposals([], [("a", "b", "DEPENDS_ON")])
        assert result.n_proposals == 0
        assert result.precision_at_5 == pytest.approx(0.0)
        assert result.recall_at_5 == pytest.approx(0.0)

    def test_perfect_match(self):
        from backtesting import backtest_proposals

        triples = [("a", "b", "DEPENDS_ON"), ("c", "d", "DERIVES")]
        result = backtest_proposals(triples, triples)
        assert result.precision_at_5 == pytest.approx(1.0)
        assert result.recall_at_5 == pytest.approx(1.0)
        assert len(result.matched_triples) == 2

    def test_no_match(self):
        from backtesting import backtest_proposals

        result = backtest_proposals(
            [("a", "b", "DEPENDS_ON")],
            [("x", "y", "EXPLAINS")],
        )
        assert result.precision_at_5 == pytest.approx(0.0)
        assert result.recall_at_5 == pytest.approx(0.0)
        assert len(result.matched_triples) == 0


class TestSoftBacktestProposals:
    def test_returns_soft_backtest_result(self):
        from backtesting import SoftBacktestResult, soft_backtest_proposals

        result = soft_backtest_proposals(
            [("a", "b", "DEPENDS_ON")],
            [("a", "c", "DEPENDS_ON")],  # same (source, edge_type) → source-type match
        )
        assert isinstance(result, SoftBacktestResult)

    def test_result_has_expected_fields(self):
        from backtesting import soft_backtest_proposals

        result = soft_backtest_proposals(
            [("a", "b", "DEPENDS_ON")],
            [("a", "c", "DEPENDS_ON")],
        )
        for attr in [
            "n_proposals",
            "n_hidden",
            "soft_precision_at_5",
            "soft_precision_at_10",
            "soft_precision_at_20",
            "soft_recall_at_5",
            "soft_recall_at_10",
            "soft_recall_at_20",
        ]:
            assert hasattr(result, attr), f"Missing field: {attr}"

    def test_source_type_match(self):
        """Proposal shares (source, edge_type) with hidden → soft hit."""
        from backtesting import soft_backtest_proposals

        proposals = [("a", "b", "DEPENDS_ON")]
        hidden = [("a", "x", "DEPENDS_ON")]  # same source + edge_type, different target
        result = soft_backtest_proposals(proposals, hidden)
        assert result.soft_precision_at_5 == pytest.approx(1.0)
        assert result.soft_recall_at_5 == pytest.approx(1.0)

    def test_type_target_match(self):
        """Proposal shares (edge_type, target) with hidden → soft hit."""
        from backtesting import soft_backtest_proposals

        proposals = [("x", "b", "DERIVES")]
        hidden = [("y", "b", "DERIVES")]  # same edge_type + target, different source
        result = soft_backtest_proposals(proposals, hidden)
        assert result.soft_precision_at_5 == pytest.approx(1.0)
        assert result.soft_recall_at_5 == pytest.approx(1.0)

    def test_no_soft_match(self):
        """Completely different (source, edge_type, target) → zero."""
        from backtesting import soft_backtest_proposals

        proposals = [("a", "b", "DEPENDS_ON")]
        hidden = [("x", "y", "EXPLAINS")]
        result = soft_backtest_proposals(proposals, hidden)
        assert result.soft_precision_at_5 == pytest.approx(0.0)
        assert result.soft_recall_at_5 == pytest.approx(0.0)

    def test_empty_proposals(self):
        from backtesting import soft_backtest_proposals

        result = soft_backtest_proposals([], [("a", "b", "DEPENDS_ON")])
        assert result.n_proposals == 0
        assert result.soft_precision_at_5 == pytest.approx(0.0)
        assert result.soft_recall_at_5 == pytest.approx(0.0)

    def test_empty_hidden(self):
        from backtesting import soft_backtest_proposals

        result = soft_backtest_proposals([("a", "b", "DEPENDS_ON")], [])
        assert result.n_hidden == 0
        assert result.soft_precision_at_5 == pytest.approx(0.0)
        assert result.soft_recall_at_5 == pytest.approx(0.0)
