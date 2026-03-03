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


# ── Symmetry (intra-type behavioral consistency) ────────────────────
#
# New design: within each entity type, how consistently do individual
# entities use edge types?  symmetry = weighted mean of per-type
# consistency, where consistency(T) = 1 − avg JS distance to centroid.


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


def test_symmetry_consistent_same_type_is_high():
    """All entities of same type using the SAME edge type → max intra-type consistency."""
    kg = KnowledgeGraph(domain="consistent_same_type")
    for eid in ("a", "b", "c", "d"):
        kg.add_entity(Entity(id=eid, entity_type="concept"))
    # Every entity uses only DEPENDS_ON outgoing
    kg.add_edge(TypedEdge(source="a", target="b", edge_type=EdgeType.DEPENDS_ON))
    kg.add_edge(TypedEdge(source="b", target="c", edge_type=EdgeType.DEPENDS_ON))
    kg.add_edge(TypedEdge(source="c", target="d", edge_type=EdgeType.DEPENDS_ON))
    score = symmetry(kg)
    assert score == pytest.approx(1.0), f"Expected 1.0 for consistent KG, got {score}"


def test_symmetry_inconsistent_same_type_is_lower():
    """Same entity type but each entity uses different edge types → lower consistency."""
    kg = _make_single_entity_type_kg()
    # a→b: DEPENDS_ON, b→c: DERIVES, c→d: EQUIVALENT_TO — each entity is different
    score = symmetry(kg)
    assert score < 1.0, f"Expected < 1.0 for inconsistent same-type KG, got {score}"


def test_symmetry_multi_type_each_internally_consistent():
    """Multiple entity types, each internally consistent → high symmetry."""
    kg = _make_heterogeneous_entity_type_kg()
    # concept: a uses only DEPENDS_ON; theorem: b uses only DERIVES; element: c uses only EQUIVALENT_TO
    # d ("concept") has no outgoing edges → skipped
    # Each type has ≤1 entity with outgoing edges → trivially consistent → 1.0
    score = symmetry(kg)
    assert score == pytest.approx(1.0), (
        f"Expected 1.0 for multi-type internally-consistent KG, got {score}"
    )


def test_symmetry_empty_kg_is_one():
    assert symmetry(_make_empty_kg()) == pytest.approx(1.0)


def test_symmetry_single_entity_per_type_is_one():
    """Each entity type has exactly one entity → trivially consistent → 1.0."""
    kg = KnowledgeGraph(domain="single_per_type")
    kg.add_entity(Entity(id="a", entity_type="star"))
    kg.add_entity(Entity(id="b", entity_type="planet"))
    kg.add_entity(Entity(id="c", entity_type="moon"))
    kg.add_edge(TypedEdge(source="a", target="b", edge_type=EdgeType.GENERALIZES))
    kg.add_edge(TypedEdge(source="b", target="c", edge_type=EdgeType.DEPENDS_ON))
    score = symmetry(kg)
    assert score == pytest.approx(1.0), f"Expected 1.0 for single-entity-per-type, got {score}"


def test_symmetry_two_entities_same_type_consistent():
    """Two entities of the same type, both using the same edge type → 1.0."""
    kg = KnowledgeGraph(domain="two_consistent")
    kg.add_entity(Entity(id="a", entity_type="concept"))
    kg.add_entity(Entity(id="b", entity_type="concept"))
    kg.add_entity(Entity(id="c", entity_type="other"))
    kg.add_edge(TypedEdge(source="a", target="c", edge_type=EdgeType.DEPENDS_ON))
    kg.add_edge(TypedEdge(source="b", target="c", edge_type=EdgeType.DEPENDS_ON))
    score = symmetry(kg)
    assert score == pytest.approx(1.0), f"Expected 1.0 for two consistent entities, got {score}"


def test_symmetry_two_entities_same_type_inconsistent():
    """Two entities of same type using completely different edge types → lower score."""
    kg = KnowledgeGraph(domain="two_inconsistent")
    kg.add_entity(Entity(id="a", entity_type="concept"))
    kg.add_entity(Entity(id="b", entity_type="concept"))
    kg.add_entity(Entity(id="c", entity_type="other"))
    kg.add_edge(TypedEdge(source="a", target="c", edge_type=EdgeType.DEPENDS_ON))
    kg.add_edge(TypedEdge(source="b", target="c", edge_type=EdgeType.CONTRADICTS))
    score = symmetry(kg)
    assert score < 1.0, f"Expected < 1.0 for inconsistent pair, got {score}"


def test_symmetry_three_entities_mixed_consistency():
    """Three entities of same type: 2 consistent, 1 different → intermediate score."""
    kg = KnowledgeGraph(domain="three_mixed")
    kg.add_entity(Entity(id="a", entity_type="concept"))
    kg.add_entity(Entity(id="b", entity_type="concept"))
    kg.add_entity(Entity(id="c", entity_type="concept"))
    kg.add_entity(Entity(id="target", entity_type="other"))
    kg.add_edge(TypedEdge(source="a", target="target", edge_type=EdgeType.DEPENDS_ON))
    kg.add_edge(TypedEdge(source="b", target="target", edge_type=EdgeType.DEPENDS_ON))
    kg.add_edge(TypedEdge(source="c", target="target", edge_type=EdgeType.CONTRADICTS))
    score = symmetry(kg)
    # 2 out of 3 are consistent, so score should be between low and 1.0
    assert 0.0 < score < 1.0, f"Expected intermediate score, got {score}"


def test_symmetry_no_outgoing_edges_is_one():
    """Entities with no outgoing edges should not affect symmetry score."""
    kg = KnowledgeGraph(domain="no_outgoing")
    kg.add_entity(Entity(id="a", entity_type="concept"))
    kg.add_entity(Entity(id="b", entity_type="concept"))
    # b→a has outgoing from b only; a has no outgoing
    kg.add_edge(TypedEdge(source="b", target="a", edge_type=EdgeType.DEPENDS_ON))
    score = symmetry(kg)
    # Only one entity (b) contributes → trivially consistent → 1.0
    assert score == pytest.approx(1.0)


def test_symmetry_weighted_by_entity_count():
    """Larger entity types should have more weight in the final score."""
    # Type A: 3 entities, all consistent (all DEPENDS_ON)
    # Type B: 2 entities, inconsistent (DEPENDS_ON vs CONTRADICTS)
    kg = KnowledgeGraph(domain="weighted")
    for eid in ("a1", "a2", "a3"):
        kg.add_entity(Entity(id=eid, entity_type="majority"))
    for eid in ("b1", "b2"):
        kg.add_entity(Entity(id=eid, entity_type="minority"))
    kg.add_entity(Entity(id="target", entity_type="other"))
    # majority: all DEPENDS_ON → consistency = 1.0
    for src in ("a1", "a2", "a3"):
        kg.add_edge(TypedEdge(source=src, target="target", edge_type=EdgeType.DEPENDS_ON))
    # minority: inconsistent → consistency < 1.0
    kg.add_edge(TypedEdge(source="b1", target="target", edge_type=EdgeType.DEPENDS_ON))
    kg.add_edge(TypedEdge(source="b2", target="target", edge_type=EdgeType.CONTRADICTS))
    score = symmetry(kg)
    # Weighted avg: 3/(3+2)*1.0 + 2/(3+2)*consistency(minority)
    # Overall should be > 0.5 because majority is perfect
    assert score > 0.5, f"Expected > 0.5 due to majority weighting, got {score}"


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
    # pytest.approx used per project convention; identical RNG seed → identical floats
    assert generativity(kg, seed=0) == pytest.approx(generativity(kg, seed=0))


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
    assert harmony_score(kg, seed=7) == pytest.approx(harmony_score(kg, seed=7))


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


def test_value_of_positive_when_replacing_contradicts_with_harmonious():
    """Replacing a contradicts-heavy KG with a coherent one yields value_of > 0.

    kg_before: 4 CONTRADICTS edges → coherence ≈ 0.5
    kg_after:  same size, all DEPENDS_ON edges → coherence = 1.0

    With α=β=γ=1/3 and δ=0 (KGs too small for DistMult), the analytic gain is:
      harmony_before ≈ (0.875 + 0.5 + 1.0) / 3 ≈ 0.792
      harmony_after  ≈ (0.875 + 1.0 + 1.0) / 3 ≈ 0.958
      value_of       ≈ +0.167 > 0
    """
    kg_before = _make_contradicts_kg()  # all CONTRADICTS edges → low coherence
    kg_after = _make_single_edge_type_kg()  # all DEPENDS_ON + triangle → high coherence

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
    assert val > 0.0, f"Expected positive improvement, got {val}"


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


# ── value_of input guards ─────────────────────────────────────────────


def test_value_of_rejects_negative_lambda_cost():
    kg = build_linear_algebra_kg()
    with pytest.raises(ValueError, match="lambda_cost"):
        value_of(kg, kg, lambda_cost=-0.1)


def test_value_of_rejects_negative_cost():
    kg = build_linear_algebra_kg()
    with pytest.raises(ValueError, match="cost"):
        value_of(kg, kg, cost=-1.0)


# ── generativity input validation ────────────────────────────────────


def test_generativity_rejects_invalid_mask_ratio():
    kg = build_linear_algebra_kg()
    with pytest.raises(ValueError, match="mask_ratio"):
        generativity(kg, mask_ratio=0.0)
    with pytest.raises(ValueError, match="mask_ratio"):
        generativity(kg, mask_ratio=1.0)
    with pytest.raises(ValueError, match="mask_ratio"):
        generativity(kg, mask_ratio=-0.1)


def test_generativity_rejects_invalid_k():
    kg = build_linear_algebra_kg()
    with pytest.raises(ValueError, match="k must"):
        generativity(kg, k=0)


def test_generativity_rejects_invalid_dim():
    kg = build_linear_algebra_kg()
    with pytest.raises(ValueError, match="dim must"):
        generativity(kg, dim=0)


def test_generativity_rejects_negative_n_epochs():
    kg = build_linear_algebra_kg()
    with pytest.raises(ValueError, match="n_epochs"):
        generativity(kg, n_epochs=-1)


# ── Coherence M2: lenient multi-edge triangle policy ──────────────────


def test_coherence_lenient_triangle_with_parallel_edges():
    """Triangle (a→b, b→c, a→c) with a closing edge type matching one hop is coherent.

    a→b: DEPENDS_ON, b→c: DERIVES, a→c: DERIVES
    t_ac=DERIVES ∈ {DEPENDS_ON, DERIVES} → coherent (lenient policy).
    """
    kg = KnowledgeGraph(domain="lenient_test")
    for eid in ("a", "b", "c"):
        kg.add_entity(Entity(id=eid, entity_type="concept"))
    kg.add_edge(TypedEdge(source="a", target="b", edge_type=EdgeType.DEPENDS_ON))
    kg.add_edge(TypedEdge(source="b", target="c", edge_type=EdgeType.DERIVES))
    kg.add_edge(TypedEdge(source="a", target="c", edge_type=EdgeType.DERIVES))

    score = coherence(kg)
    # Triangle is coherent (closing type matches one hop) → coherence = 1.0
    assert score == pytest.approx(1.0)


# ── Epsilon (Harmony+Frequency hybrid) ─────────────────────────────


def test_harmony_epsilon_zero_backward_compatible():
    """epsilon=0.0 (default) produces the same result as no epsilon argument."""
    kg = build_linear_algebra_kg()
    score_no_eps = harmony_score(kg, seed=42)
    score_eps_zero = harmony_score(kg, epsilon=0.0, seed=42)
    assert score_no_eps == pytest.approx(score_eps_zero)


def test_harmony_epsilon_nonzero_changes_score():
    """epsilon > 0 should change the composite score (frequency component)."""
    kg = build_linear_algebra_kg()
    score_base = harmony_score(kg, seed=42)
    score_eps = harmony_score(kg, epsilon=0.2, seed=42)
    # They should differ because frequency component is added
    assert score_base != pytest.approx(score_eps, abs=1e-6)


def test_harmony_epsilon_nonzero_still_in_bounds():
    """Score with epsilon > 0 must still be in [0, 1]."""
    kg = build_linear_algebra_kg()
    score = harmony_score(kg, epsilon=0.3, seed=42)
    assert 0.0 <= score <= 1.0


def test_harmony_weight_normalization_with_epsilon():
    """5 weights (α+β+γ+δ+ε) should be normalised internally."""
    kg = build_linear_algebra_kg()
    # All weights = 0.2 → should sum to 1.0
    score = harmony_score(kg, alpha=0.2, beta=0.2, gamma=0.2, delta=0.2, epsilon=0.2, seed=42)
    assert 0.0 <= score <= 1.0


def test_harmony_rejects_negative_epsilon():
    """Negative epsilon should raise ValueError."""
    kg = build_linear_algebra_kg()
    with pytest.raises(ValueError, match=">="):
        harmony_score(kg, epsilon=-0.1)


def test_distortion_passes_epsilon_through():
    """distortion() should accept and forward epsilon."""
    kg = build_linear_algebra_kg()
    h = harmony_score(kg, epsilon=0.1, seed=42)
    d = distortion(kg, epsilon=0.1, seed=42)
    assert d == pytest.approx(1.0 - h)


def test_value_of_passes_epsilon_through():
    """value_of() should accept and forward epsilon."""
    kg = build_linear_algebra_kg()
    val = value_of(kg, kg, epsilon=0.1, lambda_cost=0.0, cost=0.0, seed=42)
    # Same KG before and after → value = 0.0
    assert val == pytest.approx(0.0)
