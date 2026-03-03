"""Symmetry component — intra-type behavioral consistency.

Measures how consistently entities *within the same type* use edge types
when connecting to their neighbours.  Higher score = more symmetric.

Approach — per-entity edge-type distributions compared to type centroid:
  1. For each entity type T, collect per-entity outgoing edge-type
     distributions (normalised probability vectors).
  2. Compute the centroid distribution for type T.
  3. within_type_consistency(T) = 1 − mean JS distance to centroid.
  4. symmetry = weighted mean of consistency(T), weighted by the number
     of contributing entities per type.

This design avoids penalising natural functional specialisation between
entity types (e.g. stars ≠ planets), which the previous inter-type JS
divergence approach incorrectly penalised.

Special cases:
  - Empty graph or no edges → 1.0  (vacuously consistent)
  - Single entity with outgoing edges per type → 1.0  (trivially consistent)
  - Entity with no outgoing edges → skipped (does not contribute)
"""

from __future__ import annotations

from collections import Counter, defaultdict

import numpy as np
from scipy.spatial.distance import jensenshannon

from harmony.types import EdgeType, KnowledgeGraph

_ALL_EDGE_TYPES: list[EdgeType] = list(EdgeType)
_N_EDGE_TYPES: int = len(_ALL_EDGE_TYPES)
_ET_TO_IDX: dict[EdgeType, int] = {et: i for i, et in enumerate(_ALL_EDGE_TYPES)}


def _per_entity_distributions(
    kg: KnowledgeGraph,
) -> dict[str, list[np.ndarray]]:
    """Return per-entity outgoing edge-type distributions grouped by entity type.

    Returns a dict mapping entity_type → list of normalised probability
    vectors (one per entity that has ≥1 outgoing edge).
    """
    # Collect raw edge-type counts per entity
    entity_counts: dict[str, Counter[EdgeType]] = defaultdict(Counter)
    for edge in kg.edges:
        entity_counts[edge.source][edge.edge_type] += 1

    # Group by entity type
    type_distributions: dict[str, list[np.ndarray]] = defaultdict(list)
    for entity_id, edge_counts in entity_counts.items():
        entity_type = kg.entities[entity_id].entity_type
        total = sum(edge_counts.values())
        vec = np.zeros(_N_EDGE_TYPES, dtype=float)
        for et, c in edge_counts.items():
            vec[_ET_TO_IDX[et]] = c / total
        type_distributions[entity_type].append(vec)

    return dict(type_distributions)


def symmetry(kg: KnowledgeGraph) -> float:
    """Symmetry score ∈ [0,1]; higher = more consistent intra-type behaviour.

    Returns 1.0 for empty graphs, KGs with no outgoing edges, or KGs
    where every entity type has at most one entity with outgoing edges.
    """
    if kg.num_edges == 0:
        return 1.0

    type_dists = _per_entity_distributions(kg)

    if not type_dists:
        return 1.0  # No outgoing edges from any entity

    total_weight = 0
    weighted_consistency = 0.0

    for _entity_type, distributions in type_dists.items():
        n_entities = len(distributions)

        if n_entities <= 1:
            # Single entity → trivially consistent → consistency = 1.0
            weighted_consistency += n_entities * 1.0
            total_weight += n_entities
            continue

        # Compute centroid distribution
        centroid = np.mean(distributions, axis=0)

        # Average JS distance from each entity to centroid
        total_js = 0.0
        for dist in distributions:
            js = float(jensenshannon(dist, centroid, base=2))
            # Guard against numerical noise
            js = min(max(js, 0.0), 1.0)
            total_js += js

        avg_js = total_js / n_entities
        consistency = 1.0 - avg_js

        weighted_consistency += n_entities * consistency
        total_weight += n_entities

    if total_weight == 0:
        return 1.0

    return weighted_consistency / total_weight
