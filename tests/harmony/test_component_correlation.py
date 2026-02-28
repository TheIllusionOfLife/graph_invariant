"""Tests for analysis.component_correlation — pairwise Spearman of 4 Harmony components."""

from __future__ import annotations


class TestComponentCorrelation:
    def test_returns_4x4_matrix(self) -> None:
        from analysis.component_correlation import compute_correlation_matrix

        # 5 observations × 4 components
        scores = [
            {"compressibility": 0.8, "coherence": 0.6, "symmetry": 0.7, "generativity": 0.5},
            {"compressibility": 0.7, "coherence": 0.5, "symmetry": 0.6, "generativity": 0.4},
            {"compressibility": 0.9, "coherence": 0.8, "symmetry": 0.9, "generativity": 0.7},
            {"compressibility": 0.6, "coherence": 0.4, "symmetry": 0.5, "generativity": 0.3},
            {"compressibility": 0.5, "coherence": 0.3, "symmetry": 0.4, "generativity": 0.2},
        ]
        matrix = compute_correlation_matrix(scores)
        assert len(matrix) == 4
        for row in matrix.values():
            assert len(row) == 4

    def test_diagonal_is_one(self) -> None:
        from analysis.component_correlation import compute_correlation_matrix

        scores = [
            {"compressibility": 0.8, "coherence": 0.6, "symmetry": 0.7, "generativity": 0.5},
            {"compressibility": 0.7, "coherence": 0.5, "symmetry": 0.6, "generativity": 0.4},
            {"compressibility": 0.9, "coherence": 0.8, "symmetry": 0.9, "generativity": 0.7},
        ]
        matrix = compute_correlation_matrix(scores)
        for comp in ("compressibility", "coherence", "symmetry", "generativity"):
            assert matrix[comp][comp] == 1.0

    def test_symmetric(self) -> None:
        from analysis.component_correlation import compute_correlation_matrix

        scores = [
            {"compressibility": 0.8, "coherence": 0.6, "symmetry": 0.7, "generativity": 0.5},
            {"compressibility": 0.7, "coherence": 0.5, "symmetry": 0.6, "generativity": 0.4},
            {"compressibility": 0.9, "coherence": 0.8, "symmetry": 0.9, "generativity": 0.7},
        ]
        matrix = compute_correlation_matrix(scores)
        components = list(matrix.keys())
        for i, a in enumerate(components):
            for b in components[i + 1 :]:
                assert abs(matrix[a][b] - matrix[b][a]) < 1e-10

    def test_values_in_valid_range(self) -> None:
        from analysis.component_correlation import compute_correlation_matrix

        scores = [
            {"compressibility": 0.8, "coherence": 0.6, "symmetry": 0.7, "generativity": 0.5},
            {"compressibility": 0.7, "coherence": 0.5, "symmetry": 0.6, "generativity": 0.4},
            {"compressibility": 0.9, "coherence": 0.8, "symmetry": 0.9, "generativity": 0.7},
        ]
        matrix = compute_correlation_matrix(scores)
        for row in matrix.values():
            for val in row.values():
                assert -1.0 <= val <= 1.0

    def test_perfectly_correlated(self) -> None:
        from analysis.component_correlation import compute_correlation_matrix

        # All components have identical ranking
        scores = [
            {"compressibility": 0.1, "coherence": 0.1, "symmetry": 0.1, "generativity": 0.1},
            {"compressibility": 0.5, "coherence": 0.5, "symmetry": 0.5, "generativity": 0.5},
            {"compressibility": 0.9, "coherence": 0.9, "symmetry": 0.9, "generativity": 0.9},
        ]
        matrix = compute_correlation_matrix(scores)
        assert matrix["compressibility"]["coherence"] == 1.0

    def test_summary_string(self) -> None:
        from analysis.component_correlation import correlation_summary

        scores = [
            {"compressibility": 0.8, "coherence": 0.6, "symmetry": 0.7, "generativity": 0.5},
            {"compressibility": 0.7, "coherence": 0.5, "symmetry": 0.6, "generativity": 0.4},
            {"compressibility": 0.9, "coherence": 0.8, "symmetry": 0.9, "generativity": 0.7},
        ]
        summary = correlation_summary(scores)
        assert isinstance(summary, str)
        assert "compressibility" in summary
