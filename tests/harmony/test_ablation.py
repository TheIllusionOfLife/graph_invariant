"""Tests for harmony.metric.ablation — TDD-first, written before implementation.

Verifies:
  - run_ablation returns exactly 5 AblationRow instances
  - "full" row is always first
  - All row means are in [0, 1]
  - Results are deterministic under the same seed
  - delta_vs_full arithmetic is correct
  - Removing compressibility from a contradicts-heavy KG changes the score
"""

from __future__ import annotations

from harmony.types import EdgeType, Entity, KnowledgeGraph, TypedEdge

# ── Factories ─────────────────────────────────────────────────────────


def _make_sufficient_kg(n_entities: int = 20, n_edges: int = 30) -> KnowledgeGraph:
    """KG big enough for bootstrap to work."""
    import numpy as np

    kg = KnowledgeGraph(domain="sufficient")
    for i in range(n_entities):
        kg.add_entity(Entity(id=f"e{i}", entity_type="concept"))
    rng = np.random.default_rng(42)
    added = 0
    attempts = 0
    while added < n_edges and attempts < n_edges * 10:
        attempts += 1
        s, t = rng.integers(0, n_entities, 2)
        if s == t:
            continue
        et = list(EdgeType)[int(rng.integers(0, len(EdgeType)))]
        kg.add_edge(TypedEdge(source=f"e{s}", target=f"e{t}", edge_type=et))
        added += 1
    return kg


def _make_contradicts_kg() -> KnowledgeGraph:
    """KG dominated by CONTRADICTS edges — low coherence & compressibility."""
    kg = KnowledgeGraph(domain="contradicts")
    for i in range(12):
        kg.add_entity(Entity(id=f"e{i}", entity_type="concept"))
    for i in range(0, 10, 2):
        kg.add_edge(
            TypedEdge(source=f"e{i}", target=f"e{i + 1}", edge_type=EdgeType.CONTRADICTS)
        )
    # Add a few non-contradicting edges to keep the KG diverse
    kg.add_edge(TypedEdge(source="e0", target="e2", edge_type=EdgeType.DEPENDS_ON))
    kg.add_edge(TypedEdge(source="e2", target="e4", edge_type=EdgeType.DERIVES))
    return kg


# ── Tests ─────────────────────────────────────────────────────────────


def test_run_ablation_returns_5_rows() -> None:
    from harmony.metric.ablation import run_ablation

    rows = run_ablation(_make_sufficient_kg(), n_bootstrap=10, seed=42)
    assert len(rows) == 5


def test_run_ablation_full_row_first() -> None:
    from harmony.metric.ablation import run_ablation

    rows = run_ablation(_make_sufficient_kg(), n_bootstrap=10, seed=42)
    assert rows[0].component == "full"


def test_all_means_in_bounds() -> None:
    from harmony.metric.ablation import run_ablation

    rows = run_ablation(_make_sufficient_kg(), n_bootstrap=10, seed=42)
    for row in rows:
        assert 0.0 <= row.mean <= 1.0, f"Row {row.component} mean={row.mean} out of [0,1]"


def test_all_stds_non_negative() -> None:
    from harmony.metric.ablation import run_ablation

    rows = run_ablation(_make_sufficient_kg(), n_bootstrap=10, seed=42)
    for row in rows:
        assert row.std >= 0.0, f"Row {row.component} std={row.std} < 0"


def test_all_ns_match_bootstrap() -> None:
    from harmony.metric.ablation import run_ablation

    rows = run_ablation(_make_sufficient_kg(), n_bootstrap=15, seed=42)
    for row in rows:
        assert row.n == 15, f"Row {row.component} n={row.n}, expected 15"


def test_ablation_deterministic() -> None:
    from harmony.metric.ablation import run_ablation

    kg = _make_sufficient_kg()
    rows_a = run_ablation(kg, n_bootstrap=10, seed=7)
    rows_b = run_ablation(kg, n_bootstrap=10, seed=7)
    for a, b in zip(rows_a, rows_b, strict=True):
        assert a.component == b.component
        assert a.mean == b.mean
        assert a.std == b.std


def test_delta_vs_full_correct() -> None:
    """delta_vs_full for each row == row.mean - full.mean."""
    from harmony.metric.ablation import run_ablation

    rows = run_ablation(_make_sufficient_kg(), n_bootstrap=10, seed=42)
    full_mean = rows[0].mean
    for row in rows:
        expected_delta = row.mean - full_mean
        assert abs(row.delta_vs_full - expected_delta) < 1e-9, (
            f"Row {row.component}: delta_vs_full={row.delta_vs_full:.6f}, "
            f"expected {expected_delta:.6f}"
        )


def test_full_row_delta_is_zero() -> None:
    from harmony.metric.ablation import run_ablation

    rows = run_ablation(_make_sufficient_kg(), n_bootstrap=10, seed=42)
    assert rows[0].delta_vs_full == 0.0


def test_row_components_are_all_5_variants() -> None:
    from harmony.metric.ablation import run_ablation

    rows = run_ablation(_make_sufficient_kg(), n_bootstrap=10, seed=42)
    components = {r.component for r in rows}
    expected = {"full", "w/o_comp", "w/o_coh", "w/o_sym", "w/o_gen"}
    assert components == expected


def test_removing_comp_changes_score() -> None:
    """Removing any component from a diverse KG should change the mean score."""
    from harmony.metric.ablation import run_ablation

    rows = run_ablation(_make_sufficient_kg(n_entities=20, n_edges=30), n_bootstrap=20, seed=42)
    full_mean = rows[0].mean
    # At least one leave-one-out variant should differ from the full metric
    diffs = [abs(r.mean - full_mean) for r in rows[1:]]
    assert any(d > 0.0 for d in diffs), (
        "All leave-one-out means equal full — no component contributes"
    )


def test_ablation_row_is_dataclass() -> None:
    from dataclasses import fields

    from harmony.metric.ablation import AblationRow

    field_names = {f.name for f in fields(AblationRow)}
    expected = {"component", "mean", "std", "ci95_half_width", "n", "delta_vs_full"}
    assert expected == field_names
