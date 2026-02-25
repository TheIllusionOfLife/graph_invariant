"""Tests for harmony.types — Entity, TypedEdge, KnowledgeGraph."""
import json

import pytest

from harmony.types import EdgeType, Entity, KnowledgeGraph, TypedEdge

# ── Entity ────────────────────────────────────────────────────────────


def test_entity_creation_with_defaults():
    e = Entity(id="vector_space", entity_type="concept")
    assert e.id == "vector_space"
    assert e.entity_type == "concept"
    assert e.properties == {}


def test_entity_creation_with_properties():
    e = Entity(id="h", entity_type="element", properties={"atomic_number": 1, "period": 1})
    assert e.properties["atomic_number"] == 1


# ── EdgeType ──────────────────────────────────────────────────────────


def test_all_seven_edge_types_exist():
    expected = {
        "depends_on",
        "derives",
        "equivalent_to",
        "maps_to",
        "explains",
        "contradicts",
        "generalizes",
    }
    actual = {e.value for e in EdgeType}
    assert actual == expected


def test_edge_type_string_values_are_snake_case():
    for et in EdgeType:
        assert et.value == et.value.lower()
        assert " " not in et.value


# ── TypedEdge ─────────────────────────────────────────────────────────


def test_typed_edge_creation():
    edge = TypedEdge(source="a", target="b", edge_type=EdgeType.DEPENDS_ON)
    assert edge.source == "a"
    assert edge.target == "b"
    assert edge.edge_type == EdgeType.DEPENDS_ON
    assert edge.properties == {}


def test_typed_edge_all_edge_types():
    for et in EdgeType:
        edge = TypedEdge(source="x", target="y", edge_type=et)
        assert edge.edge_type == et


def test_typed_edge_with_properties():
    edge = TypedEdge(
        source="a", target="b", edge_type=EdgeType.EXPLAINS, properties={"confidence": 0.9}
    )
    assert edge.properties["confidence"] == 0.9


# ── KnowledgeGraph ────────────────────────────────────────────────────


def _make_simple_kg() -> KnowledgeGraph:
    kg = KnowledgeGraph(domain="test")
    kg.add_entity(Entity(id="a", entity_type="concept"))
    kg.add_entity(Entity(id="b", entity_type="theorem"))
    kg.add_entity(Entity(id="c", entity_type="concept"))
    return kg


def test_knowledge_graph_add_entity():
    kg = KnowledgeGraph(domain="test")
    kg.add_entity(Entity(id="vec", entity_type="concept"))
    assert kg.num_entities == 1


def test_knowledge_graph_add_edge_valid():
    kg = _make_simple_kg()
    kg.add_edge(TypedEdge(source="a", target="b", edge_type=EdgeType.DERIVES))
    assert kg.num_edges == 1


def test_knowledge_graph_num_entities_and_num_edges():
    kg = _make_simple_kg()
    kg.add_edge(TypedEdge(source="a", target="b", edge_type=EdgeType.DERIVES))
    kg.add_edge(TypedEdge(source="b", target="c", edge_type=EdgeType.EXPLAINS))
    assert kg.num_entities == 3
    assert kg.num_edges == 2


def test_knowledge_graph_rejects_edge_with_unknown_source():
    kg = _make_simple_kg()
    with pytest.raises(ValueError, match="source"):
        kg.add_edge(TypedEdge(source="UNKNOWN", target="b", edge_type=EdgeType.DERIVES))


def test_knowledge_graph_rejects_edge_with_unknown_target():
    kg = _make_simple_kg()
    with pytest.raises(ValueError, match="target"):
        kg.add_edge(TypedEdge(source="a", target="UNKNOWN", edge_type=EdgeType.DERIVES))


def test_knowledge_graph_rejects_duplicate_entity_id():
    kg = KnowledgeGraph(domain="test")
    kg.add_entity(Entity(id="a", entity_type="concept"))
    with pytest.raises(ValueError, match="already exists"):
        kg.add_entity(Entity(id="a", entity_type="theorem"))


def test_knowledge_graph_entity_lookup():
    kg = _make_simple_kg()
    entity = kg.entities["a"]
    assert entity.entity_type == "concept"


def test_knowledge_graph_edges_is_list():
    kg = _make_simple_kg()
    kg.add_edge(TypedEdge(source="a", target="b", edge_type=EdgeType.MAPS_TO))
    assert isinstance(kg.edges, list)


# ── Serialization ─────────────────────────────────────────────────────


def test_knowledge_graph_to_dict_round_trip():
    kg = _make_simple_kg()
    kg.add_edge(TypedEdge(source="a", target="b", edge_type=EdgeType.DERIVES))
    kg.add_edge(TypedEdge(source="b", target="c", edge_type=EdgeType.EXPLAINS))

    d = kg.to_dict()
    kg2 = KnowledgeGraph.from_dict(d)

    assert kg2.domain == kg.domain
    assert kg2.num_entities == kg.num_entities
    assert kg2.num_edges == kg.num_edges
    assert kg2.entities["a"].entity_type == "concept"


def test_knowledge_graph_to_dict_is_json_serializable():
    kg = _make_simple_kg()
    kg.add_edge(TypedEdge(source="a", target="b", edge_type=EdgeType.GENERALIZES))
    d = kg.to_dict()
    json_str = json.dumps(d)
    assert isinstance(json_str, str)


def test_knowledge_graph_from_dict_restores_edge_types():
    kg = _make_simple_kg()
    for et in EdgeType:
        kg.add_edge(TypedEdge(source="a", target="b", edge_type=et))
    d = kg.to_dict()
    kg2 = KnowledgeGraph.from_dict(d)
    edge_types_restored = {e.edge_type for e in kg2.edges}
    assert edge_types_restored == set(EdgeType)


def test_knowledge_graph_entity_types_preserved_in_round_trip():
    kg = _make_simple_kg()
    d = kg.to_dict()
    kg2 = KnowledgeGraph.from_dict(d)
    for eid, entity in kg.entities.items():
        assert kg2.entities[eid].entity_type == entity.entity_type


def test_knowledge_graph_properties_preserved_in_round_trip():
    kg = KnowledgeGraph(domain="test")
    kg.add_entity(Entity(id="h", entity_type="element", properties={"atomic_number": 1}))
    d = kg.to_dict()
    kg2 = KnowledgeGraph.from_dict(d)
    assert kg2.entities["h"].properties["atomic_number"] == 1
