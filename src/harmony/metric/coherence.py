"""Coherence distortion component.

Measures how internally consistent the KG's path semantics are.
Higher score = more coherent (paths agree, fewer contradictions).

Two signals are averaged:
  1. Triangle coherence: for every closed triangle (A→B, B→C, A→C),
     check whether the closing edge type is compatible with the path.
     "Compatible" = A→C type matches either the A→B type or the B→C
     type.  Returns 1.0 when no triangles exist (vacuously coherent).
  2. Contradiction-free rate: 1 − (|CONTRADICTS edges| / |total edges|).
     Some contradictions are meaningful (invertible vs singular), but a
     high density suggests structural noise.

Both signals are in [0,1]; their mean is returned.
"""

from __future__ import annotations

from collections import defaultdict

from harmony.types import EdgeType, KnowledgeGraph


def _triangle_coherence(kg: KnowledgeGraph) -> float:
    """Fraction of closed triangles whose closing edge is type-compatible.

    Builds a multi-adjacency dict (source → target → list[EdgeType]) to
    handle multiple parallel edges correctly.  A triangle (A→B type t_ab,
    B→C type t_bc, A→C type t_ac) is coherent if t_ac ∈ {t_ab, t_bc}.
    """
    if kg.num_edges == 0:
        return 1.0

    # Build adjacency: source → target → set of edge types
    adj: dict[str, dict[str, set[EdgeType]]] = defaultdict(lambda: defaultdict(set))
    for e in kg.edges:
        adj[e.source][e.target].add(e.edge_type)

    n_triangles = 0
    n_coherent = 0

    for a, a_neighbors in adj.items():
        for b, types_ab in a_neighbors.items():
            if b not in adj:
                continue
            for c, types_bc in adj[b].items():
                if c == a:
                    continue
                types_ac = adj[a].get(c)
                if types_ac is None:
                    continue
                # One triangle counted per (a, b, c) triple.
                # Lenient multi-edge policy: if *any* closing edge type
                # matches either hop type, the triangle is coherent.  This
                # avoids penalising KGs that carry multiple edge-type
                # interpretations between the same pair.
                n_triangles += 1
                if types_ac & (types_ab | types_bc):
                    n_coherent += 1

    return n_coherent / n_triangles if n_triangles > 0 else 1.0


def _contradiction_free_rate(kg: KnowledgeGraph) -> float:
    """1 − density of CONTRADICTS edges ∈ [0,1]."""
    if kg.num_edges == 0:
        return 1.0
    n_contradicts = sum(1 for e in kg.edges if e.edge_type == EdgeType.CONTRADICTS)
    return 1.0 - n_contradicts / kg.num_edges


def coherence(kg: KnowledgeGraph) -> float:
    """Coherence score ∈ [0,1]; higher = more coherent.

    Returns 1.0 for empty graphs (vacuously coherent).
    """
    if kg.num_edges == 0:
        return 1.0
    return (_triangle_coherence(kg) + _contradiction_free_rate(kg)) / 2.0
