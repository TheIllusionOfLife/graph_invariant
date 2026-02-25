"""Symmetry distortion component.

Measures how consistently different *entity types* use edge types when
connecting to their neighbours.  Higher score = more symmetric.

Approach — Jensen-Shannon divergence on typed-neighbor distributions:
  For each entity type, build a probability vector over the 7 EdgeTypes
  based on outgoing edges from entities of that type.  Then compute the
  average pairwise JS divergence between entity-type distributions.

  scipy.spatial.distance.jensenshannon(p, q, base=2) returns √(JS),
  which lies in [0, 1] when the distributions are normalised.  A value of
  0 means identical distributions (maximally symmetric); 1 means entirely
  disjoint.

  symmetry = 1 − mean_pairwise_jsd  ∈ [0,1]

Special cases:
  - Empty graph or no edges → 1.0  (vacuously symmetric)
  - Single entity type → 1.0  (no pairs to compare)
"""

from __future__ import annotations

from collections import Counter, defaultdict

import numpy as np
from scipy.spatial.distance import jensenshannon

from harmony.types import EdgeType, KnowledgeGraph

_ALL_EDGE_TYPES: list[EdgeType] = list(EdgeType)
_N_EDGE_TYPES: int = len(_ALL_EDGE_TYPES)
_ET_TO_IDX: dict[EdgeType, int] = {et: i for i, et in enumerate(_ALL_EDGE_TYPES)}


def _entity_type_distributions(kg: KnowledgeGraph) -> dict[str, np.ndarray]:
    """Return a probability distribution over EdgeType for each entity type.

    Only entities with at least one outgoing edge contribute to their type's
    distribution.  Entity types with no outgoing edges are omitted.
    """
    counts: dict[str, Counter[EdgeType]] = defaultdict(Counter)
    for edge in kg.edges:
        source_entity_type = kg.entities[edge.source].entity_type
        counts[source_entity_type][edge.edge_type] += 1

    distributions: dict[str, np.ndarray] = {}
    for etype, edge_counts in counts.items():
        total = sum(edge_counts.values())
        vec = np.zeros(_N_EDGE_TYPES, dtype=float)
        for et, c in edge_counts.items():
            vec[_ET_TO_IDX[et]] = c / total
        distributions[etype] = vec

    return distributions


def symmetry(kg: KnowledgeGraph) -> float:
    """Symmetry score ∈ [0,1]; higher = more symmetric entity-type behaviour.

    Returns 1.0 for empty graphs or KGs with ≤1 entity type (no divergence
    to measure).
    """
    if kg.num_edges == 0:
        return 1.0

    dists = _entity_type_distributions(kg)
    entity_types = list(dists.keys())

    if len(entity_types) <= 1:
        return 1.0  # Single entity type → identical distributions → JS = 0

    total_js = 0.0
    n_pairs = 0
    for i in range(len(entity_types)):
        for j in range(i + 1, len(entity_types)):
            p = dists[entity_types[i]]
            q = dists[entity_types[j]]
            js = float(jensenshannon(p, q, base=2))
            # Guard against numerical noise pushing JS slightly above 1
            js = min(max(js, 0.0), 1.0)
            total_js += js
            n_pairs += 1

    avg_js = total_js / n_pairs
    return 1.0 - avg_js
