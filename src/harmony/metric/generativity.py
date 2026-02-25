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


def _split_edges(
    edges: list[TypedEdge],
    mask_ratio: float,
    seed: int,
) -> tuple[list[TypedEdge], list[TypedEdge]]:
    """Return (train_edges, test_edges) by masking mask_ratio of edges at random.

    Uses np.random.default_rng(seed).permutation for reproducibility.
    Shared by generativity() and baselines to guarantee identical splits.
    """
    n_mask = max(1, int(len(edges) * mask_ratio))
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(edges))
    mask_idx: set[int] = set(int(i) for i in perm[:n_mask])
    train = [e for i, e in enumerate(edges) if i not in mask_idx]
    test = [e for i, e in enumerate(edges) if i in mask_idx]
    return train, test


_ET_TO_IDX: dict[EdgeType, int] = {et: i for i, et in enumerate(_ALL_EDGE_TYPES)}


class _DistMult:
    """Shallow DistMult link-prediction model for small KGs.

    Score(s, r, t) = (E[s] ⊙ R[r]) · E[t]
    where E ∈ R^(n_entities × dim) and R ∈ R^(n_relations × dim).
    """

    def __init__(self, entity_ids: list[str], dim: int = 50, seed: int = 42) -> None:
        """Initialise embeddings with small Gaussian noise.

        Parameters
        ----------
        entity_ids: ordered list of all entity IDs in the KG.
        dim: embedding dimension.
        seed: RNG seed for reproducibility.
        """
        self.entity_to_idx: dict[str, int] = {eid: i for i, eid in enumerate(entity_ids)}
        self.n_entities = len(entity_ids)
        self.n_relations = len(EdgeType)
        self.dim = dim

        rng = np.random.default_rng(seed)
        self.E: np.ndarray = rng.normal(0.0, 0.1, (self.n_entities, dim))
        self.R: np.ndarray = rng.normal(0.0, 0.1, (self.n_relations, dim))

    def _score(self, s: int, r: int, t: int) -> float:
        """Element-wise product of head/relation then dot-product with tail."""
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
        """Train with max-margin loss and random target corruption.

        Gradients for all parameters are accumulated using the *current*
        (pre-update) embeddings, then applied together.  This avoids the
        stale-gradient bug where updating E[neg_t] in-place before
        computing ∂loss/∂E[s] causes the accumulator to use post-update
        values.
        """
        if not triples:
            return

        rng = np.random.default_rng(seed)
        triples_arr = list(triples)

        for _ in range(n_epochs):
            rng.shuffle(triples_arr)  # type: ignore[arg-type]  # numpy stubs require ndarray; list is valid at runtime
            for s, r, t in triples_arr:
                pos_score = self._score(s, r, t)
                h_sr = self.E[s] * self.R[r]  # head vector (pre-update snapshot)

                # --- Collect violating negatives using current embeddings ---
                neg_candidates = [
                    int(x) for x in rng.integers(0, self.n_entities, n_neg) if int(x) != t
                ]
                violating: list[int] = []
                for neg_t in neg_candidates:
                    neg_score = float(np.dot(h_sr, self.E[neg_t]))
                    if margin - pos_score + neg_score > 0:
                        violating.append(neg_t)

                if not violating:
                    continue

                n_v = len(violating)

                # --- Accumulate all gradients before touching any embedding ---
                # ∂loss/∂E[t] = −n_v · h_sr  (push positive target up)
                grad_Et = n_v * h_sr

                # ∂loss/∂E[s] and ∂loss/∂R[r] from positive + each negative
                grad_Es = np.zeros(self.dim)
                grad_Er = np.zeros(self.dim)
                # Accumulate per-negative gradients (keyed to avoid duplicate index)
                grad_neg: dict[int, np.ndarray] = {}
                for neg_t in violating:
                    # ∂loss/∂E[neg_t] = +h_sr  (push negative target down)
                    if neg_t not in grad_neg:
                        grad_neg[neg_t] = np.zeros(self.dim)
                    grad_neg[neg_t] += h_sr
                    # ∂loss/∂E[s]: (E[neg_t] − E[t]) ⊙ R[r]
                    grad_Es += (self.E[neg_t] - self.E[t]) * self.R[r]
                    # ∂loss/∂R[r]: (E[neg_t] − E[t]) ⊙ E[s]
                    grad_Er += (self.E[neg_t] - self.E[t]) * self.E[s]

                # --- Apply all updates atomically ---
                self.E[t] += lr * grad_Et
                self.E[s] -= lr * grad_Es
                self.R[r] -= lr * grad_Er
                for neg_t, grad in grad_neg.items():
                    self.E[neg_t] -= lr * grad

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

    Parameters
    ----------
    kg: knowledge graph to evaluate.
    seed: RNG seed for edge masking and model initialisation.
    mask_ratio: fraction of edges to mask for link prediction; must be in (0, 1).
    k: Hits@K cutoff; clamped to min(k, n_entities − 1) to avoid trivial scores.
    dim: DistMult embedding dimension; must be > 0.
    n_epochs: training epochs; must be ≥ 0.
    """
    if not (0.0 < mask_ratio < 1.0):
        raise ValueError(f"mask_ratio must be in (0, 1), got {mask_ratio}")
    if k <= 0:
        raise ValueError(f"k must be > 0, got {k}")
    if dim <= 0:
        raise ValueError(f"dim must be > 0, got {dim}")
    if n_epochs < 0:
        raise ValueError(f"n_epochs must be >= 0, got {n_epochs}")

    if kg.num_edges == 0 or kg.num_entities == 0:
        return 0.0

    edges = kg.edges
    train_edges, test_edges = _split_edges(edges, mask_ratio, seed)

    if len(train_edges) < _MIN_TRAIN_EDGES or not test_edges:
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

    # Clamp k so that k < n_entities; otherwise every entity is in the top-k
    # and the score is trivially 1.0 regardless of model quality.
    effective_k = min(k, model.n_entities - 1)
    if effective_k <= 0:
        return 0.0

    # Evaluate Hits@K on masked edges
    hits = 0
    for test_edge in test_edges:
        s = entity_to_idx.get(test_edge.source)
        t = entity_to_idx.get(test_edge.target)
        if s is None or t is None:
            continue
        r = _ET_TO_IDX[test_edge.edge_type]
        scores = model.score_all_targets(s, r)
        # np.argsort is O(n log n) but correct for all array sizes;
        # argpartition can yield wrong results when n_entities ≤ effective_k.
        top_k_indices = np.argsort(-scores)[:effective_k]
        if t in top_k_indices:
            hits += 1

    return hits / len(test_edges)
