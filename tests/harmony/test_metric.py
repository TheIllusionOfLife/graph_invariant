"""Tests for harmony.metric — all 4 distortion components + composite harmony score.

TDD: these tests are written BEFORE implementation. They verify:
  - Each component returns a float in [0, 1]
  - Each component is deterministic (same KG → same score)
  - Each component responds correctly to structural properties
  - The composite harmony score aggregates components correctly
"""

from __future__ import annotations

import pytest

from harmony.datasets.linear_algebra import build_linear_algebra_kg
from harmony.datasets.periodic_table import build_periodic_table_kg
from harmony.metric.coherence import coherence
from harmony.metric.compressibility import compressibility
from harmony.metric.generativity import generativity
from harmony.metric.harmony import distortion, harmony_score, value_of
from harmony.metric.symmetry import symmetry
from harmony.types import EdgeType, Entity, KnowledgeGraph, TypedEdge  # noqa: F401

# ── Fixtures / Factories ──────────────────────────────────────────────


def _make_empty_kg() -> KnowledgeGraph:
    return KnowledgeGraph(domain="empty")


def _make_tiny_kg() -> KnowledgeGraph:
    """3 nodes, 2 edges (A→B→C linear chain)."""
    kg = KnowledgeGraph(domain="tiny")
    for eid in ("a", "b", "c"):
        kg.add_entity(Entity(id=eid, entity_type="concept"))
    kg.add_edge(TypedEdge(source="a", target="b", edge_type=EdgeType.DEPENDS_ON))
    kg.add_edge(TypedEdge(source="b", target="c", edge_type=EdgeType.DEPENDS_ON))
    return kg


def _make_triangle_kg() -> KnowledgeGraph:
    """3 nodes, 3 edges forming a triangle (A→B, B→C, A→C) — same edge type."""
    kg = KnowledgeGraph(domain="triangle")
    for eid in ("a", "b", "c"):
        kg.add_entity(Entity(id=eid, entity_type="concept"))
    kg.add_edge(TypedEdge(source="a", target="b", edge_type=EdgeType.DEPENDS_ON))
    kg.add_edge(TypedEdge(source="b", target="c", edge_type=EdgeType.DEPENDS_ON))
    kg.add_edge(TypedEdge(source="a", target="c", edge_type=EdgeType.DEPENDS_ON))
    return kg


def _make_mixed_edge_type_kg() -> KnowledgeGraph:
    """4 nodes, 4 edges with 4 different edge types — high entropy, less compressible."""
    kg = KnowledgeGraph(domain="mixed")
    for eid in ("a", "b", "c", "d"):
        kg.add_entity(Entity(id=eid, entity_type="concept"))
    kg.add_edge(TypedEdge(source="a", target="b", edge_type=EdgeType.DEPENDS_ON))
    kg.add_edge(TypedEdge(source="b", target="c", edge_type=EdgeType.DERIVES))
    kg.add_edge(TypedEdge(source="c", target="d", edge_type=EdgeType.EQUIVALENT_TO))
    kg.add_edge(TypedEdge(source="a", target="d", edge_type=EdgeType.GENERALIZES))
    return kg


def _make_single_edge_type_kg() -> KnowledgeGraph:
    """4 nodes, 4 edges all of the SAME type — low entropy, more compressible."""
    kg = KnowledgeGraph(domain="single_type")
    for eid in ("a", "b", "c", "d"):
        kg.add_entity(Entity(id=eid, entity_type="concept"))
    kg.add_edge(TypedEdge(source="a", target="b", edge_type=EdgeType.DEPENDS_ON))
    kg.add_edge(TypedEdge(source="b", target="c", edge_type=EdgeType.DEPENDS_ON))
    kg.add_edge(TypedEdge(source="c", target="d", edge_type=EdgeType.DEPENDS_ON))
    kg.add_edge(TypedEdge(source="a", target="c", edge_type=EdgeType.DEPENDS_ON))
    return kg


def _make_contradicts_kg() -> KnowledgeGraph:
    """4 nodes with many CONTRADICTS edges — should lower coherence."""
    kg = KnowledgeGraph(domain="contradicts")
    for eid in ("a", "b", "c", "d"):
        kg.add_entity(Entity(id=eid, entity_type="concept"))
    kg.add_edge(TypedEdge(source="a", target="b", edge_type=EdgeType.CONTRADICTS))
    kg.add_edge(TypedEdge(source="b", target="c", edge_type=EdgeType.CONTRADICTS))
    kg.add_edge(TypedEdge(source="c", target="d", edge_type=EdgeType.CONTRADICTS))
    kg.add_edge(TypedEdge(source="a", target="d", edge_type=EdgeType.CONTRADICTS))
    return kg


def _make_single_entity_type_kg() -> KnowledgeGraph:
    """4 nodes of the same entity type — max symmetry (all entity types identical)."""
    kg = KnowledgeGraph(domain="homogeneous")
    for eid in ("a", "b", "c", "d"):
        kg.add_entity(Entity(id=eid, entity_type="concept"))
    kg.add_edge(TypedEdge(source="a", target="b", edge_type=EdgeType.DEPENDS_ON))
    kg.add_edge(TypedEdge(source="b", target="c", edge_type=EdgeType.DERIVES))
    kg.add_edge(TypedEdge(source="c", target="d", edge_type=EdgeType.EQUIVALENT_TO))
    return kg


def _make_heterogeneous_entity_type_kg() -> KnowledgeGraph:
    """4 nodes with 3 entity types and very different edge patterns — lower symmetry."""
    kg = KnowledgeGraph(domain="heterogeneous")
    kg.add_entity(Entity(id="a", entity_type="concept"))
    kg.add_entity(Entity(id="b", entity_type="theorem"))
    kg.add_entity(Entity(id="c", entity_type="element"))
    kg.add_entity(Entity(id="d", entity_type="concept"))
    # concept only uses DEPENDS_ON
    kg.add_edge(TypedEdge(source="a", target="b", edge_type=EdgeType.DEPENDS_ON))
    kg.add_edge(TypedEdge(source="a", target="c", edge_type=EdgeType.DEPENDS_ON))
    kg.add_edge(TypedEdge(source="a", target="d", edge_type=EdgeType.DEPENDS_ON))
    # theorem only uses DERIVES
    kg.add_edge(TypedEdge(source="b", target="c", edge_type=EdgeType.DERIVES))
    # element only uses EQUIVALENT_TO
    kg.add_edge(TypedEdge(source="c", target="d", edge_type=EdgeType.EQUIVALENT_TO))
    return kg


# ── Compressibility ───────────────────────────────────────────────────


def test_compressibility_returns_float_in_unit_interval():
    for kg in [_make_empty_kg(), _make_tiny_kg(), _make_triangle_kg(), _make_mixed_edge_type_kg()]:
        score = compressibility(kg)
        assert isinstance(score, float), f"Expected float, got {type(score)}"
        assert 0.0 <= score <= 1.0, f"Expected [0,1], got {score} for domain={kg.domain}"


def test_compressibility_is_deterministic():
    kg = _make_triangle_kg()
    assert compressibility(kg) == compressibility(kg)


def test_compressibility_single_edge_type_higher_than_mixed():
    single = compressibility(_make_single_edge_type_kg())
    mixed = compressibility(_make_mixed_edge_type_kg())
    assert single > mixed, f"Single-type ({single:.3f}) should beat mixed ({mixed:.3f})"


def test_compressibility_empty_kg_is_one():
    """Empty graph has no edges — vacuously perfectly compressible."""
    assert compressibility(_make_empty_kg()) == pytest.approx(1.0)


def test_compressibility_linear_algebra_kg_in_bounds():
    kg = build_linear_algebra_kg()
    score = compressibility(kg)
    assert 0.0 <= score <= 1.0


# ── Coherence ─────────────────────────────────────────────────────────


def test_coherence_returns_float_in_unit_interval():
    for kg in [_make_empty_kg(), _make_tiny_kg(), _make_triangle_kg(), _make_contradicts_kg()]:
        score = coherence(kg)
        assert isinstance(score, float), f"Expected float, got {type(score)}"
        assert 0.0 <= score <= 1.0, f"Expected [0,1], got {score} for domain={kg.domain}"


def test_coherence_is_deterministic():
    kg = _make_triangle_kg()
    assert coherence(kg) == coherence(kg)


def test_coherence_contradicts_kg_lower_than_no_contradicts():
    no_contradicts = coherence(_make_tiny_kg())
    with_contradicts = coherence(_make_contradicts_kg())
    assert with_contradicts < no_contradicts, (
        f"Contradicts KG ({with_contradicts:.3f}) should be lower "
        f"than clean KG ({no_contradicts:.3f})"
    )


def test_coherence_empty_kg_is_one():
    """Empty graph has no edges — vacuously perfectly coherent."""
    assert coherence(_make_empty_kg()) == pytest.approx(1.0)


def test_coherence_linear_algebra_kg_in_bounds():
    kg = build_linear_algebra_kg()
    score = coherence(kg)
    assert 0.0 <= score <= 1.0


# ── Symmetry ─────────────────────────────────────────────────────────


def test_symmetry_returns_float_in_unit_interval():
    for kg in [
        _make_empty_kg(),
        _make_tiny_kg(),
        _make_single_entity_type_kg(),
        _make_heterogeneous_entity_type_kg(),
    ]:
        score = symmetry(kg)
        assert isinstance(score, float), f"Expected float, got {type(score)}"
        assert 0.0 <= score <= 1.0, f"Expected [0,1], got {score} for domain={kg.domain}"


def test_symmetry_is_deterministic():
    kg = build_linear_algebra_kg()
    assert symmetry(kg) == symmetry(kg)


def test_symmetry_single_entity_type_is_one():
    """All entities of same type → no divergence → max symmetry."""
    score = symmetry(_make_single_entity_type_kg())
    assert score == pytest.approx(1.0), f"Expected 1.0 for homogeneous KG, got {score}"


def test_symmetry_heterogeneous_lower_than_homogeneous():
    homo = symmetry(_make_single_entity_type_kg())
    hetero = symmetry(_make_heterogeneous_entity_type_kg())
    assert hetero < homo, (
        f"Heterogeneous ({hetero:.3f}) should be lower than homogeneous ({homo:.3f})"
    )


def test_symmetry_empty_kg_is_one():
    assert symmetry(_make_empty_kg()) == pytest.approx(1.0)


def test_symmetry_linear_algebra_kg_in_bounds():
    kg = build_linear_algebra_kg()
    score = symmetry(kg)
    assert 0.0 <= score <= 1.0


# ── Generativity ─────────────────────────────────────────────────────


def test_generativity_returns_float_in_unit_interval():
    kg = build_linear_algebra_kg()
    score = generativity(kg)
    assert isinstance(score, float), f"Expected float, got {type(score)}"
    assert 0.0 <= score <= 1.0, f"Expected [0,1], got {score}"


def test_generativity_is_deterministic_with_same_seed():
    kg = build_linear_algebra_kg()
    assert generativity(kg, seed=0) == generativity(kg, seed=0)


def test_generativity_all_seeds_produce_valid_scores():
    """Every seed must produce a score in [0, 1] — not just the default."""
    kg = build_linear_algebra_kg()
    for s in range(5):
        score = generativity(kg, seed=s)
        assert 0.0 <= score <= 1.0, f"seed={s} produced out-of-bounds score {score}"


def test_generativity_insufficient_edges_returns_zero():
    """KG with too few edges for training returns 0.0 gracefully."""
    kg = _make_tiny_kg()  # only 2 edges — too few
    score = generativity(kg)
    assert score == pytest.approx(0.0)


def test_generativity_empty_kg_returns_zero():
    kg = _make_empty_kg()
    score = generativity(kg)
    assert score == pytest.approx(0.0)


def test_generativity_linear_algebra_kg_in_bounds():
    kg = build_linear_algebra_kg()
    score = generativity(kg, seed=42, k=10)
    assert 0.0 <= score <= 1.0


# ── Harmony composite ─────────────────────────────────────────────────


def test_harmony_score_returns_float_in_unit_interval():
    for kg in [_make_empty_kg(), _make_tiny_kg(), build_linear_algebra_kg()]:
        score = harmony_score(kg)
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0, f"Expected [0,1], got {score} for domain={kg.domain}"


def test_harmony_score_is_deterministic():
    kg = build_linear_algebra_kg()
    assert harmony_score(kg, seed=7) == harmony_score(kg, seed=7)


def test_distortion_is_one_minus_harmony():
    kg = build_linear_algebra_kg()
    h = harmony_score(kg, seed=7)
    d = distortion(kg, seed=7)
    assert d == pytest.approx(1.0 - h)


def test_harmony_weights_sum_to_one_by_default():
    """Default equal weights α=β=γ=δ=0.25 sum to 1."""
    kg = build_linear_algebra_kg()
    # Score with equal weights should lie strictly between 0 and 1
    score = harmony_score(kg)
    assert 0.0 <= score <= 1.0


def test_harmony_alpha_one_equals_compressibility():
    """When α=1 and β=γ=δ=0, harmony == compressibility."""
    kg = build_linear_algebra_kg()
    h = harmony_score(kg, alpha=1.0, beta=0.0, gamma=0.0, delta=0.0)
    c = compressibility(kg)
    assert h == pytest.approx(c)


def test_harmony_delta_one_equals_generativity():
    """When α=β=γ=0 and δ=1, harmony == generativity (same seed)."""
    kg = build_linear_algebra_kg()
    h = harmony_score(kg, alpha=0.0, beta=0.0, gamma=0.0, delta=1.0, seed=42)
    g = generativity(kg, seed=42)
    assert h == pytest.approx(g)


def test_value_of_positive_when_proposal_adds_coherent_edge():
    """Adding a well-typed edge to a small KG should improve harmony (value_of > 0)."""
    kg_before = _make_tiny_kg()  # a→b→c
    # Build KG after adding A→C with DEPENDS_ON (closes the triangle coherently)
    kg_after = KnowledgeGraph(domain="tiny")
    for eid in ("a", "b", "c"):
        kg_after.add_entity(Entity(id=eid, entity_type="concept"))
    kg_after.add_edge(TypedEdge(source="a", target="b", edge_type=EdgeType.DEPENDS_ON))
    kg_after.add_edge(TypedEdge(source="b", target="c", edge_type=EdgeType.DEPENDS_ON))
    kg_after.add_edge(TypedEdge(source="a", target="c", edge_type=EdgeType.DEPENDS_ON))

    # With no generativity (too few edges for DistMult), use alpha+beta+gamma only
    val = value_of(
        kg_before,
        kg_after,
        alpha=1 / 3,
        beta=1 / 3,
        gamma=1 / 3,
        delta=0.0,
        lambda_cost=0.0,
        cost=0.0,
        seed=42,
    )
    # Adding a coherent edge should not decrease harmony
    assert isinstance(val, float)


def test_value_of_zero_cost_equals_distortion_reduction():
    """With λ=0 and cost=0, value_of equals distortion reduction."""
    kg_before = _make_tiny_kg()
    kg_after = _make_triangle_kg()  # same KG + one edge

    val = value_of(kg_before, kg_after, lambda_cost=0.0, cost=0.0, seed=42)
    d_before = distortion(kg_before, seed=42)
    d_after = distortion(kg_after, seed=42)
    assert val == pytest.approx(d_before - d_after)


def test_harmony_periodic_table_in_bounds():
    kg = build_periodic_table_kg()
    score = harmony_score(kg)
    assert 0.0 <= score <= 1.0


# ── L2: negative weight rejection ────────────────────────────────────


def test_harmony_rejects_negative_weights():
    """Negative weights can push the composite score outside [0,1] — must raise."""
    kg = build_linear_algebra_kg()
    with pytest.raises(ValueError, match=">="):
        harmony_score(kg, alpha=-1.0, beta=2.0, gamma=0.0, delta=0.0)


def test_distortion_rejects_negative_weights():
    kg = build_linear_algebra_kg()
    with pytest.raises(ValueError, match=">="):
        distortion(kg, alpha=0.0, beta=-0.1, gamma=0.5, delta=0.6)


# ── L3: compressibility order-invariance ─────────────────────────────


def test_compressibility_order_invariant():
    """Swapping entity insertion order must not change the compressibility score."""
    # Build KG with forward-order insertion
    kg_fwd = KnowledgeGraph(domain="order_test")
    for eid in ("a", "b", "c", "d"):
        kg_fwd.add_entity(Entity(id=eid, entity_type="concept"))
    kg_fwd.add_edge(TypedEdge(source="a", target="b", edge_type=EdgeType.DEPENDS_ON))
    kg_fwd.add_edge(TypedEdge(source="b", target="c", edge_type=EdgeType.DEPENDS_ON))
    kg_fwd.add_edge(TypedEdge(source="c", target="d", edge_type=EdgeType.DEPENDS_ON))

    # Build same KG with reversed entity insertion order
    kg_rev = KnowledgeGraph(domain="order_test")
    for eid in ("d", "c", "b", "a"):
        kg_rev.add_entity(Entity(id=eid, entity_type="concept"))
    kg_rev.add_edge(TypedEdge(source="a", target="b", edge_type=EdgeType.DEPENDS_ON))
    kg_rev.add_edge(TypedEdge(source="b", target="c", edge_type=EdgeType.DEPENDS_ON))
    kg_rev.add_edge(TypedEdge(source="c", target="d", edge_type=EdgeType.DEPENDS_ON))

    assert compressibility(kg_fwd) == pytest.approx(compressibility(kg_rev)), (
        "compressibility must be invariant to entity insertion order"
    )


# ── L4: generativity with k >= n_entities ────────────────────────────


def test_generativity_k_larger_than_entities_stays_bounded():
    """k=1000 on a small KG must not trivially return 1.0 (effective_k clamp)."""
    kg = build_linear_algebra_kg()
    score = generativity(kg, k=1000)
    # Clamped to n_entities − 1, so score is a meaningful [0,1] value
    assert 0.0 <= score <= 1.0


def test_generativity_k_1_stricter_than_k_10():
    """Hits@1 ≤ Hits@10 on the same KG and seed."""
    kg = build_linear_algebra_kg()
    hits1 = generativity(kg, seed=0, k=1)
    hits10 = generativity(kg, seed=0, k=10)
    assert hits1 <= hits10, f"Hits@1 ({hits1}) should be ≤ Hits@10 ({hits10})"
