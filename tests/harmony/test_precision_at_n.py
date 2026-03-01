"""Tests for analysis.precision_at_n — precision against withheld edges."""

from __future__ import annotations


def _withheld_edges() -> list[tuple[str, str, str]]:
    """Ground-truth edges held out for evaluation."""
    return [
        ("A", "D", "DEPENDS_ON"),
        ("B", "D", "MAPS_TO"),
    ]


class TestPrecisionAtN:
    def test_perfect_precision_when_all_match(self) -> None:
        from analysis.precision_at_n import precision_at_n

        # Proposals that exactly match withheld edges
        proposals_ranked = [
            ("A", "D", "DEPENDS_ON"),
            ("B", "D", "MAPS_TO"),
        ]
        withheld = _withheld_edges()
        assert precision_at_n(proposals_ranked, withheld, n=2) == 1.0

    def test_zero_precision_when_none_match(self) -> None:
        from analysis.precision_at_n import precision_at_n

        proposals_ranked = [
            ("A", "B", "EXPLAINS"),
            ("C", "D", "CONTRADICTS"),
        ]
        withheld = _withheld_edges()
        assert precision_at_n(proposals_ranked, withheld, n=2) == 0.0

    def test_partial_precision(self) -> None:
        from analysis.precision_at_n import precision_at_n

        proposals_ranked = [
            ("A", "D", "DEPENDS_ON"),  # match
            ("A", "B", "EXPLAINS"),  # no match
        ]
        withheld = _withheld_edges()
        assert precision_at_n(proposals_ranked, withheld, n=2) == 0.5

    def test_n_limits_evaluation(self) -> None:
        from analysis.precision_at_n import precision_at_n

        proposals_ranked = [
            ("A", "D", "DEPENDS_ON"),  # match
            ("A", "B", "EXPLAINS"),  # no match
            ("B", "D", "MAPS_TO"),  # match (but n=2, so ignored)
        ]
        withheld = _withheld_edges()
        assert precision_at_n(proposals_ranked, withheld, n=2) == 0.5

    def test_empty_proposals_returns_zero(self) -> None:
        from analysis.precision_at_n import precision_at_n

        withheld = _withheld_edges()
        assert precision_at_n([], withheld, n=5) == 0.0

    def test_duplicate_predictions_not_double_counted(self) -> None:
        from analysis.precision_at_n import precision_at_n

        proposals_ranked = [
            ("A", "D", "DEPENDS_ON"),  # match
            ("A", "D", "DEPENDS_ON"),  # duplicate — should be ignored
            ("A", "B", "EXPLAINS"),  # no match (fills slot 2)
        ]
        withheld = _withheld_edges()
        # Only 2 unique proposals evaluated: 1 match + 1 miss = 0.5
        assert precision_at_n(proposals_ranked, withheld, n=2) == 0.5

    def test_returns_float_in_unit_interval(self) -> None:
        from analysis.precision_at_n import precision_at_n

        proposals_ranked = [("A", "D", "DEPENDS_ON")]
        withheld = _withheld_edges()
        result = precision_at_n(proposals_ranked, withheld, n=1)
        assert isinstance(result, float)
        assert 0.0 <= result <= 1.0
