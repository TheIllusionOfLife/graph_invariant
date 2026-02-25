"""Tests for harmony.dataset (KGDataset splits/masking) and all 5 domain KGs."""
import json

import pytest

from harmony.dataset import KGDataset
from harmony.datasets.astronomy import build_astronomy_kg
from harmony.datasets.linear_algebra import build_linear_algebra_kg
from harmony.datasets.materials import build_materials_kg
from harmony.datasets.periodic_table import build_periodic_table_kg
from harmony.datasets.physics import build_physics_kg
from harmony.types import EdgeType, KnowledgeGraph

# ── Domain KG construction ─────────────────────────────────────────────


def test_linear_algebra_kg_has_minimum_entities():
    kg = build_linear_algebra_kg()
    assert kg.num_entities >= 30


def test_linear_algebra_kg_has_minimum_edges():
    kg = build_linear_algebra_kg()
    assert kg.num_edges >= 40


def test_linear_algebra_kg_domain_name():
    kg = build_linear_algebra_kg()
    assert kg.domain == "linear_algebra"


def test_periodic_table_kg_has_all_118_elements():
    kg = build_periodic_table_kg()
    assert kg.num_entities >= 118


def test_periodic_table_kg_has_period_and_group_entities():
    kg = build_periodic_table_kg()
    assert kg.num_entities >= 118 + 7 + 18  # elements + periods + groups


def test_periodic_table_kg_has_edges():
    kg = build_periodic_table_kg()
    assert kg.num_edges >= 118  # at least one edge per element


def test_periodic_table_kg_domain_name():
    kg = build_periodic_table_kg()
    assert kg.domain == "periodic_table"


def test_astronomy_kg_has_minimum_entities():
    kg = build_astronomy_kg()
    assert kg.num_entities >= 40


def test_astronomy_kg_has_minimum_edges():
    kg = build_astronomy_kg()
    assert kg.num_edges >= 30


def test_astronomy_kg_domain_name():
    kg = build_astronomy_kg()
    assert kg.domain == "astronomy"


def test_physics_kg_has_minimum_entities():
    kg = build_physics_kg()
    assert kg.num_entities >= 40


def test_physics_kg_has_minimum_edges():
    kg = build_physics_kg()
    assert kg.num_edges >= 30


def test_physics_kg_domain_name():
    kg = build_physics_kg()
    assert kg.domain == "physics"


def test_materials_kg_has_minimum_entities():
    kg = build_materials_kg()
    assert kg.num_entities >= 40


def test_materials_kg_has_minimum_edges():
    kg = build_materials_kg()
    assert kg.num_edges >= 30


def test_materials_kg_domain_name():
    kg = build_materials_kg()
    assert kg.domain == "materials"


# ── Edge type constraint validation ───────────────────────────────────


ALL_BUILDERS = [
    build_linear_algebra_kg,
    build_periodic_table_kg,
    build_astronomy_kg,
    build_physics_kg,
    build_materials_kg,
]


@pytest.mark.parametrize("builder", ALL_BUILDERS)
def test_kg_all_edges_reference_existing_entities(builder):
    kg = builder()
    for edge in kg.edges:
        assert edge.source in kg.entities, f"Source '{edge.source}' not in entities"
        assert edge.target in kg.entities, f"Target '{edge.target}' not in entities"


@pytest.mark.parametrize("builder", ALL_BUILDERS)
def test_kg_all_edge_types_are_valid(builder):
    kg = builder()
    valid_types = set(EdgeType)
    for edge in kg.edges:
        assert edge.edge_type in valid_types


@pytest.mark.parametrize("builder", ALL_BUILDERS)
def test_kg_no_duplicate_entity_ids(builder):
    kg = builder()
    assert len(kg.entities) == len(set(kg.entities.keys()))


# ── KGDataset splits ──────────────────────────────────────────────────


@pytest.fixture
def small_kg() -> KnowledgeGraph:
    return build_linear_algebra_kg()


def test_kg_dataset_split_sizes_sum_to_total(small_kg):
    dataset = KGDataset.from_kg(small_kg, seed=42)
    total = (
        len(dataset.train_edges)
        + len(dataset.val_edges)
        + len(dataset.test_edges)
        + len(dataset.hidden_edges)
    )
    assert total == small_kg.num_edges


def test_kg_dataset_hidden_edges_are_non_empty(small_kg):
    dataset = KGDataset.from_kg(small_kg, seed=42)
    assert len(dataset.hidden_edges) > 0


def test_kg_dataset_train_edges_are_largest_split(small_kg):
    dataset = KGDataset.from_kg(small_kg, seed=42)
    assert len(dataset.train_edges) > len(dataset.val_edges)
    assert len(dataset.train_edges) > len(dataset.test_edges)
    assert len(dataset.train_edges) > len(dataset.hidden_edges)


def test_kg_dataset_hidden_ratio_respected(small_kg):
    dataset = KGDataset.from_kg(small_kg, seed=42, hidden_ratio=0.1)
    expected_hidden = max(1, round(small_kg.num_edges * 0.1))
    # Allow ±1 for rounding
    assert abs(len(dataset.hidden_edges) - expected_hidden) <= 1


def test_kg_dataset_is_deterministic(small_kg):
    d1 = KGDataset.from_kg(small_kg, seed=99)
    d2 = KGDataset.from_kg(small_kg, seed=99)
    assert len(d1.train_edges) == len(d2.train_edges)
    assert len(d1.hidden_edges) == len(d2.hidden_edges)
    # Same edges in same order
    for e1, e2 in zip(d1.hidden_edges, d2.hidden_edges, strict=True):
        assert e1.source == e2.source
        assert e1.target == e2.target
        assert e1.edge_type == e2.edge_type


def test_kg_dataset_different_seeds_give_different_splits(small_kg):
    d1 = KGDataset.from_kg(small_kg, seed=1)
    d2 = KGDataset.from_kg(small_kg, seed=2)
    sources_1 = [e.source for e in d1.hidden_edges]
    sources_2 = [e.source for e in d2.hidden_edges]
    # Different seeds should produce different hidden sets (very likely with > 5 edges)
    assert sources_1 != sources_2


def test_kg_dataset_hidden_edges_are_valid_kg_edges(small_kg):
    dataset = KGDataset.from_kg(small_kg, seed=42)
    original_edge_set = {(e.source, e.target, e.edge_type) for e in small_kg.edges}
    for edge in dataset.hidden_edges:
        assert (edge.source, edge.target, edge.edge_type) in original_edge_set


def test_kg_dataset_no_overlap_between_splits(small_kg):
    dataset = KGDataset.from_kg(small_kg, seed=42)

    def edge_key(e):
        return (e.source, e.target, e.edge_type)

    train_set = {edge_key(e) for e in dataset.train_edges}
    val_set = {edge_key(e) for e in dataset.val_edges}
    test_set = {edge_key(e) for e in dataset.test_edges}
    hidden_set = {edge_key(e) for e in dataset.hidden_edges}

    assert train_set.isdisjoint(val_set)
    assert train_set.isdisjoint(test_set)
    assert train_set.isdisjoint(hidden_set)
    assert val_set.isdisjoint(test_set)
    assert val_set.isdisjoint(hidden_set)
    assert test_set.isdisjoint(hidden_set)


# ── KG masking ────────────────────────────────────────────────────────


def test_masked_kg_excludes_hidden_edges(small_kg):
    dataset = KGDataset.from_kg(small_kg, seed=42)
    masked = dataset.train_kg()
    hidden_edge_set = {(e.source, e.target, e.edge_type) for e in dataset.hidden_edges}
    for edge in masked.edges:
        assert (edge.source, edge.target, edge.edge_type) not in hidden_edge_set


def test_masked_kg_contains_all_entities(small_kg):
    dataset = KGDataset.from_kg(small_kg, seed=42)
    masked = dataset.train_kg()
    assert masked.num_entities == small_kg.num_entities


def test_masked_kg_has_fewer_edges_than_original(small_kg):
    dataset = KGDataset.from_kg(small_kg, seed=42)
    masked = dataset.train_kg()
    assert masked.num_edges < small_kg.num_edges


# ── Serialization round-trip ──────────────────────────────────────────


def test_kg_dataset_to_dict_is_json_serializable(small_kg):
    dataset = KGDataset.from_kg(small_kg, seed=42)
    d = dataset.to_dict()
    json_str = json.dumps(d)
    assert isinstance(json_str, str)


def test_kg_dataset_from_dict_round_trip(small_kg):
    dataset = KGDataset.from_kg(small_kg, seed=42)
    d = dataset.to_dict()
    dataset2 = KGDataset.from_dict(d)

    assert len(dataset2.train_edges) == len(dataset.train_edges)
    assert len(dataset2.val_edges) == len(dataset.val_edges)
    assert len(dataset2.test_edges) == len(dataset.test_edges)
    assert len(dataset2.hidden_edges) == len(dataset.hidden_edges)


# ── Backtesting split non-empty across all domains ────────────────────


@pytest.mark.parametrize("builder", ALL_BUILDERS)
def test_backtesting_split_non_empty_for_all_domains(builder):
    kg = builder()
    dataset = KGDataset.from_kg(kg, seed=42, hidden_ratio=0.1)
    assert len(dataset.hidden_edges) >= 1


@pytest.mark.parametrize("builder", ALL_BUILDERS)
def test_train_edges_non_empty_for_all_domains(builder):
    kg = builder()
    dataset = KGDataset.from_kg(kg, seed=42, hidden_ratio=0.1)
    assert len(dataset.train_edges) >= 1
