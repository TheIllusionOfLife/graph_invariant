"""Tests for medium-scale Wikidata KG datasets (physics + materials)."""

from __future__ import annotations

import pytest

from harmony.types import EdgeType, KnowledgeGraph  # noqa: I001

# ── Shared helpers ──────────────────────────────────────────────────


def _build_undirected_adjacency(kg: KnowledgeGraph) -> dict[str, set[str]]:
    """Build undirected adjacency dict from KG edges."""
    adj: dict[str, set[str]] = {eid: set() for eid in kg.entities}
    for e in kg.edges:
        adj[e.source].add(e.target)
        adj[e.target].add(e.source)
    return adj


def _assert_medium_scale(kg: KnowledgeGraph, min_entities: int = 200, min_edges: int = 800) -> None:
    """Assert KG meets medium-scale thresholds."""
    assert kg.num_entities >= min_entities, (
        f"Expected ≥{min_entities} entities, got {kg.num_entities}"
    )
    assert kg.num_edges >= min_edges, f"Expected ≥{min_edges} edges, got {kg.num_edges}"


def _assert_all_edge_types_present(kg: KnowledgeGraph) -> None:
    """Assert all 7 edge types appear at least once."""
    present = {e.edge_type for e in kg.edges}
    missing = set(EdgeType) - present
    assert not missing, f"Missing edge types: {missing}"


def _assert_connected(kg: KnowledgeGraph) -> None:
    """Assert the KG is weakly connected (single component)."""
    if kg.num_entities == 0:
        return
    adj = _build_undirected_adjacency(kg)

    visited: set[str] = set()
    stack = [next(iter(kg.entities))]
    while stack:
        node = stack.pop()
        if node in visited:
            continue
        visited.add(node)
        stack.extend(adj.get(node, set()) - visited)

    assert len(visited) == kg.num_entities, (
        f"Graph has {kg.num_entities} entities but only {len(visited)} reachable "
        f"(disconnected: {kg.num_entities - len(visited)})"
    )


def _assert_triangle_density(kg: KnowledgeGraph, min_triangles: int = 10) -> None:
    """Assert minimum triangle count for meaningful coherence computation."""
    adj = _build_undirected_adjacency(kg)

    triangles = 0
    for a in adj:
        for b in adj[a]:
            if b <= a:
                continue
            common = adj[a] & adj[b]
            triangles += sum(1 for c in common if c > b)

    assert triangles >= min_triangles, f"Expected ≥{min_triangles} triangles, got {triangles}"


def _assert_no_duplicate_typed_edges(kg: KnowledgeGraph) -> None:
    """Assert no duplicate (source, target, edge_type) triples."""
    seen: set[tuple[str, str, EdgeType]] = set()
    for e in kg.edges:
        key = (e.source, e.target, e.edge_type)
        assert key not in seen, f"Duplicate edge: {key}"
        seen.add(key)


def _assert_entity_type_diversity(kg: KnowledgeGraph, min_types: int = 5) -> None:
    """Assert at least min_types distinct entity types."""
    types = {e.entity_type for e in kg.entities.values()}
    assert len(types) >= min_types, f"Expected ≥{min_types} entity types, got {len(types)}: {types}"


# ── Wikidata Physics Tests ──────────────────────────────────────────


class TestWikidataPhysics:
    @pytest.fixture()
    def kg(self) -> KnowledgeGraph:
        from harmony.datasets.wikidata_physics import build_wikidata_physics_kg

        return build_wikidata_physics_kg()

    def test_domain(self, kg: KnowledgeGraph) -> None:
        assert kg.domain == "wikidata_physics"

    def test_medium_scale(self, kg: KnowledgeGraph) -> None:
        _assert_medium_scale(kg, min_entities=200, min_edges=800)

    def test_all_edge_types_present(self, kg: KnowledgeGraph) -> None:
        _assert_all_edge_types_present(kg)

    def test_connected(self, kg: KnowledgeGraph) -> None:
        _assert_connected(kg)

    def test_triangle_density(self, kg: KnowledgeGraph) -> None:
        _assert_triangle_density(kg, min_triangles=50)

    def test_entity_type_diversity(self, kg: KnowledgeGraph) -> None:
        _assert_entity_type_diversity(kg, min_types=8)

    def test_no_self_loops(self, kg: KnowledgeGraph) -> None:
        for e in kg.edges:
            assert e.source != e.target, f"Self-loop: {e.source}"

    def test_edge_type_distribution_not_degenerate(self, kg: KnowledgeGraph) -> None:
        """No single edge type should dominate >70% of all edges."""
        from collections import Counter

        counts = Counter(e.edge_type for e in kg.edges)
        for etype, count in counts.items():
            ratio = count / kg.num_edges
            assert ratio < 0.70, f"{etype.name} has {count}/{kg.num_edges} = {ratio:.0%} of edges"

    def test_no_duplicate_typed_edges(self, kg: KnowledgeGraph) -> None:
        _assert_no_duplicate_typed_edges(kg)

    def test_deterministic(self, kg: KnowledgeGraph) -> None:
        from harmony.datasets.wikidata_physics import build_wikidata_physics_kg

        kg2 = build_wikidata_physics_kg()
        assert kg.num_entities == kg2.num_entities
        assert kg.num_edges == kg2.num_edges


# ── Wikidata Materials Tests ────────────────────────────────────────


class TestWikidataMaterials:
    @pytest.fixture()
    def kg(self) -> KnowledgeGraph:
        from harmony.datasets.wikidata_materials import build_wikidata_materials_kg

        return build_wikidata_materials_kg()

    def test_domain(self, kg: KnowledgeGraph) -> None:
        assert kg.domain == "wikidata_materials"

    def test_medium_scale(self, kg: KnowledgeGraph) -> None:
        _assert_medium_scale(kg, min_entities=200, min_edges=800)

    def test_all_edge_types_present(self, kg: KnowledgeGraph) -> None:
        _assert_all_edge_types_present(kg)

    def test_connected(self, kg: KnowledgeGraph) -> None:
        _assert_connected(kg)

    def test_triangle_density(self, kg: KnowledgeGraph) -> None:
        _assert_triangle_density(kg, min_triangles=50)

    def test_entity_type_diversity(self, kg: KnowledgeGraph) -> None:
        _assert_entity_type_diversity(kg, min_types=8)

    def test_no_self_loops(self, kg: KnowledgeGraph) -> None:
        for e in kg.edges:
            assert e.source != e.target, f"Self-loop: {e.source}"

    def test_edge_type_distribution_not_degenerate(self, kg: KnowledgeGraph) -> None:
        from collections import Counter

        counts = Counter(e.edge_type for e in kg.edges)
        for etype, count in counts.items():
            ratio = count / kg.num_edges
            assert ratio < 0.70, f"{etype.name} has {count}/{kg.num_edges} = {ratio:.0%} of edges"

    def test_no_duplicate_typed_edges(self, kg: KnowledgeGraph) -> None:
        _assert_no_duplicate_typed_edges(kg)

    def test_deterministic(self, kg: KnowledgeGraph) -> None:
        from harmony.datasets.wikidata_materials import build_wikidata_materials_kg

        kg2 = build_wikidata_materials_kg()
        assert kg.num_entities == kg2.num_entities
        assert kg.num_edges == kg2.num_edges


# ── Cross-dataset Tests ─────────────────────────────────────────────


class TestCrossDataset:
    def test_minimal_entity_overlap(self) -> None:
        """Physics and materials KGs should have mostly independent entity namespaces.

        A small number of cross-domain concepts (e.g. superconductivity, band_gap)
        legitimately appear in both Wikidata subgraphs. We assert overlap is <5%
        of the smaller dataset to guard against accidental namespace collision.
        """
        from harmony.datasets.wikidata_materials import build_wikidata_materials_kg
        from harmony.datasets.wikidata_physics import build_wikidata_physics_kg

        physics = build_wikidata_physics_kg()
        materials = build_wikidata_materials_kg()
        # Domains are different
        assert physics.domain != materials.domain
        # Overlap should be small (< 5% of smaller KG)
        physics_ids = set(physics.entities.keys())
        materials_ids = set(materials.entities.keys())
        overlap = physics_ids & materials_ids
        smaller = min(len(physics_ids), len(materials_ids))
        ratio = len(overlap) / smaller if smaller > 0 else 0.0
        assert ratio < 0.05, (
            f"Entity overlap too large: {len(overlap)}/{smaller} = {ratio:.1%}. "
            f"Overlapping: {overlap}"
        )

    def test_both_substantially_larger_than_small_datasets(self) -> None:
        """Medium-scale KGs should be ≥4x larger than existing small KGs."""
        from harmony.datasets.materials import build_materials_kg
        from harmony.datasets.physics import build_physics_kg
        from harmony.datasets.wikidata_materials import build_wikidata_materials_kg
        from harmony.datasets.wikidata_physics import build_wikidata_physics_kg

        small_phys = build_physics_kg()
        small_mat = build_materials_kg()
        big_phys = build_wikidata_physics_kg()
        big_mat = build_wikidata_materials_kg()

        assert big_phys.num_entities >= 4 * small_phys.num_entities
        assert big_mat.num_entities >= 4 * small_mat.num_entities
        assert big_phys.num_edges >= 4 * small_phys.num_edges
        assert big_mat.num_edges >= 4 * small_mat.num_edges
