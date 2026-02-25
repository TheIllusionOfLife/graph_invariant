"""Compressibility distortion component.

MDL (Minimum Description Length) proxy: how much the KG's edge structure
can be compressed.  Higher score = more compressible (more structured).

Two signals are averaged:
  1. Edge-type entropy: a KG that uses only one edge type is maximally
     compressible; uniform distribution over all 7 types is least.
  2. BFS spanning fraction: ratio of BFS-spanning-tree edges to total edges.
     A pure tree (=1.0) is maximally compressible; dense multi-edges lower it.

Both signals are normalised to [0,1] individually, then averaged.
"""

from __future__ import annotations

import math
from collections import Counter, deque

from harmony.types import EdgeType, KnowledgeGraph

_MAX_ENTROPY = math.log2(len(EdgeType))  # log2(7) ≈ 2.807


def _edge_type_entropy_score(kg: KnowledgeGraph) -> float:
    """1 − (H / log2(n_types)) ∈ [0,1]; 1 = single edge type used."""
    if kg.num_edges == 0:
        return 1.0
    counts = Counter(e.edge_type for e in kg.edges)
    n = kg.num_edges
    entropy = -sum((c / n) * math.log2(c / n) for c in counts.values() if c > 0)
    return 1.0 - entropy / _MAX_ENTROPY


def _bfs_spanning_fraction(kg: KnowledgeGraph) -> float:
    """Ratio of BFS spanning-tree edges to total edges ∈ [0,1].

    Runs BFS from each unvisited entity (handles disconnected graphs).
    Every edge that first discovers a new node is a spanning edge.
    For a tree this equals 1.0; adding cross-edges lowers it.

    Uses an *undirected* adjacency view so that the score is invariant to
    entity insertion order and edge direction.  A purely directed chain
    a→b→c (where b is inserted first) would otherwise yield 0 spanning
    edges from root b.
    """
    if kg.num_edges == 0 or kg.num_entities == 0:
        return 1.0

    # Build undirected adjacency (both directions) to ensure order-invariance
    adj: dict[str, list[str]] = {eid: [] for eid in kg.entities}
    for e in kg.edges:
        adj[e.source].append(e.target)
        adj[e.target].append(e.source)

    visited: set[str] = set()
    spanning_edges = 0

    for root in kg.entities:
        if root in visited:
            continue
        queue: deque[str] = deque([root])
        visited.add(root)
        while queue:
            node = queue.popleft()
            for neighbor in adj[node]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    spanning_edges += 1
                    queue.append(neighbor)

    return spanning_edges / kg.num_edges


def compressibility(kg: KnowledgeGraph) -> float:
    """MDL-proxy compressibility score ∈ [0,1]; higher = more compressible.

    Returns 1.0 for empty graphs (vacuously compressible).
    """
    if kg.num_edges == 0:
        return 1.0
    entropy_score = _edge_type_entropy_score(kg)
    spanning_score = _bfs_spanning_fraction(kg)
    return (entropy_score + spanning_score) / 2.0
