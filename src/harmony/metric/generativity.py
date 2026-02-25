"""Generativity distortion component.

Measures how well a shallow DistMult model can recover masked edges from the
remaining graph structure.  Higher score = more generative (the KG's patterns
are learnable and predictive).

Protocol:
  1. Mask `mask_ratio` (default 20%) of edges uniformly at random.
  2. Train a DistMult bilinear model on the remaining edges.
  3. For each masked edge (s, r, t), rank all entities as potential targets.
  4. Return Hits@K (fraction of masked edges where true target is in top K).

DistMult scoring: score(s, r, t) = E[s] ⊙ R[r] · E[t]
Training: max-margin loss with random target corruption, SGD updates.

Returns 0.0 when there are too few edges (< MIN_TRAIN_EDGES + 1 masked edges)
to train meaningfully.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from harmony.types import EdgeType, KnowledgeGraph, TypedEdge

if TYPE_CHECKING:
    pass

_MIN_TRAIN_EDGES = 10
_ALL_EDGE_TYPES: list[EdgeType] = list(EdgeType)
_ET_TO_IDX: dict[EdgeType, int] = {et: i for i, et in enumerate(_ALL_EDGE_TYPES)}


class _DistMult:
    """Shallow DistMult link-prediction model for small KGs.

    Score(s, r, t) = (E[s] ⊙ R[r]) · E[t]
    where E ∈ R^(n_entities × dim) and R ∈ R^(n_relations × dim).
    """

    def __init__(self, entity_ids: list[str], dim: int = 50, seed: int = 42) -> None:
        self.entity_to_idx: dict[str, int] = {eid: i for i, eid in enumerate(entity_ids)}
        self.n_entities = len(entity_ids)
        self.n_relations = len(EdgeType)
        self.dim = dim

        rng = np.random.default_rng(seed)
        self.E: np.ndarray = rng.normal(0.0, 0.1, (self.n_entities, dim))
        self.R: np.ndarray = rng.normal(0.0, 0.1, (self.n_relations, dim))

    def _score(self, s: int, r: int, t: int) -> float:
        return float(np.dot(self.E[s] * self.R[r], self.E[t]))

    def score_all_targets(self, s: int, r: int) -> np.ndarray:
        """Return scores for all entities as targets. Shape: (n_entities,)."""
        h = self.E[s] * self.R[r]  # (dim,)
        return self.E @ h  # (n_entities,)

    def train(
        self,
        triples: list[tuple[int, int, int]],
        n_epochs: int = 100,
        lr: float = 0.01,
        margin: float = 1.0,
        n_neg: int = 5,
        seed: int = 42,
    ) -> None:
        """Train with max-margin loss and random target corruption."""
        if not triples:
            return

        rng = np.random.default_rng(seed)
        triples_arr = list(triples)

        for _ in range(n_epochs):
            rng.shuffle(triples_arr)  # type: ignore[arg-type]
            for s, r, t in triples_arr:
                pos_score = self._score(s, r, t)
                h = self.E[s] * self.R[r]  # shared head vector

                # Gradient accumulator for E[s] and R[r]
                grad_s = np.zeros(self.dim)
                grad_r = np.zeros(self.dim)

                # Positive gradient
                loss_pos_margin = 0.0

                neg_targets = rng.integers(0, self.n_entities, n_neg)
                for neg_t in neg_targets:
                    if int(neg_t) == t:
                        continue
                    neg_score = self._score(s, r, int(neg_t))
                    loss = margin - pos_score + neg_score
                    if loss > 0:
                        loss_pos_margin = 1.0  # flag: positive gradient needed
                        # ∂loss/∂E[neg_t] = +h
                        self.E[int(neg_t)] -= lr * h
                        # ∂loss/∂E[s] accumulates: −E[t]⊙R[r] + E[neg_t]⊙R[r]
                        grad_s += -self.E[t] * self.R[r] + self.E[int(neg_t)] * self.R[r]
                        grad_r += -self.E[t] * self.E[s] + self.E[int(neg_t)] * self.E[s]

                if loss_pos_margin > 0:
                    # ∂loss/∂E[t] = −h (from positive pair)
                    self.E[t] += lr * h
                    self.E[s] -= lr * grad_s
                    self.R[r] -= lr * grad_r

            # L2 normalise entity embeddings after each epoch to prevent collapse
            norms = np.linalg.norm(self.E, axis=1, keepdims=True)
            self.E /= np.maximum(norms, 1e-8)


def generativity(
    kg: KnowledgeGraph,
    seed: int = 42,
    mask_ratio: float = 0.2,
    k: int = 10,
    dim: int = 50,
    n_epochs: int = 100,
) -> float:
    """Link-prediction generativity score ∈ [0,1]; higher = more generative.

    Returns 0.0 when the KG has too few edges to train meaningfully
    (< _MIN_TRAIN_EDGES training edges or < 1 masked edge).
    """
    if kg.num_edges == 0 or kg.num_entities == 0:
        return 0.0

    edges = kg.edges
    n_mask = max(1, int(len(edges) * mask_ratio))
    n_train = len(edges) - n_mask

    if n_train < _MIN_TRAIN_EDGES:
        return 0.0

    # Split edges into train / test (masked)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(edges))
    mask_idx: set[int] = set(int(i) for i in perm[:n_mask])
    train_edges: list[TypedEdge] = [e for i, e in enumerate(edges) if i not in mask_idx]
    test_edges: list[TypedEdge] = [e for i, e in enumerate(edges) if i in mask_idx]

    if not test_edges:
        return 0.0

    # Build model
    entity_ids = list(kg.entities.keys())
    model = _DistMult(entity_ids=entity_ids, dim=dim, seed=seed)
    entity_to_idx = model.entity_to_idx

    # Convert training edges to integer triples
    triples: list[tuple[int, int, int]] = []
    for e in train_edges:
        s = entity_to_idx.get(e.source)
        t = entity_to_idx.get(e.target)
        if s is None or t is None:
            continue
        r = _ET_TO_IDX[e.edge_type]
        triples.append((s, r, t))

    model.train(triples, n_epochs=n_epochs, seed=seed)

    # Evaluate Hits@K on masked edges
    hits = 0
    for test_edge in test_edges:
        s = entity_to_idx.get(test_edge.source)
        t = entity_to_idx.get(test_edge.target)
        if s is None or t is None:
            continue
        r = _ET_TO_IDX[test_edge.edge_type]
        scores = model.score_all_targets(s, r)
        top_k_indices = np.argpartition(-scores, min(k, len(scores) - 1))[:k]
        if t in top_k_indices:
            hits += 1

    return hits / len(test_edges)
