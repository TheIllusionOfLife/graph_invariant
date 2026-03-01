"""TransE link-prediction model for small KGs.

Score(s, r, t) = -||e_s + r - e_t||_1
where e_s, e_t, r ∈ R^d (real-valued embeddings).

Training: max-margin loss with random target corruption, SGD updates.
Returns 0.0 when there are too few edges (< MIN_TRAIN_EDGES + 1 masked edges).
"""

from __future__ import annotations

import numpy as np

from harmony.metric.generativity import _MIN_TRAIN_EDGES, _split_edges
from harmony.types import EdgeType, KnowledgeGraph

_ALL_EDGE_TYPES: list[EdgeType] = list(EdgeType)
_ET_TO_IDX: dict[EdgeType, int] = {et: i for i, et in enumerate(_ALL_EDGE_TYPES)}


class _TransE:
    """TransE link-prediction model.

    Score(s, r, t) = -||e_s + r - e_t||_1
    """

    def __init__(
        self,
        entity_ids: list[str],
        dim: int = 50,
        seed: int = 42,
    ) -> None:
        self.entity_to_idx: dict[str, int] = {eid: i for i, eid in enumerate(entity_ids)}
        self.n_entities = len(entity_ids)
        self.n_relations = len(EdgeType)
        self.dim = dim

        rng = np.random.default_rng(seed)
        self.E: np.ndarray = rng.normal(0.0, 0.1, (self.n_entities, dim))
        self.R: np.ndarray = rng.normal(0.0, 0.1, (self.n_relations, dim))

    def _score(self, s: int, r: int, t: int) -> float:
        """TransE score: -||e_s + r - e_t||_1."""
        diff = self.E[s] + self.R[r] - self.E[t]
        return -float(np.sum(np.abs(diff)))

    def score_all_targets(self, s: int, r: int) -> np.ndarray:
        """Return scores for all entities as targets. Shape: (n_entities,)."""
        h = self.E[s] + self.R[r]  # (dim,)
        # -||h - e_t||_1 for each target
        diffs = h[np.newaxis, :] - self.E  # (n_entities, dim)
        return -np.sum(np.abs(diffs), axis=1)  # (n_entities,)

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
                h = self.E[s] + self.R[r]  # snapshot before updates

                neg_candidates = [
                    int(x) for x in rng.integers(0, self.n_entities, n_neg) if int(x) != t
                ]

                # Accumulate gradients before applying
                grad_Et = np.zeros(self.dim)
                grad_Es = np.zeros(self.dim)
                grad_Rr = np.zeros(self.dim)
                grad_neg: dict[int, np.ndarray] = {}

                for neg_t in neg_candidates:
                    neg_score = self._score(s, r, neg_t)
                    loss = margin - pos_score + neg_score
                    if loss <= 0:
                        continue

                    # ∂L/∂E[t] = sign(h - E[t])  (push t closer to h)
                    sign_pos = np.sign(h - self.E[t])
                    # ∂L/∂E[neg_t] = -sign(h - E[neg_t])  (push neg_t away)
                    sign_neg = np.sign(h - self.E[neg_t])

                    grad_Et += sign_pos
                    if neg_t not in grad_neg:
                        grad_neg[neg_t] = np.zeros(self.dim)
                    grad_neg[neg_t] -= sign_neg

                    # ∂L/∂E[s] and ∂L/∂R[r]: same as ∂L/∂h
                    grad_h = sign_neg - sign_pos
                    grad_Es += grad_h
                    grad_Rr += grad_h

                # Apply accumulated updates (grad_Es/grad_Rr are already
                # negative gradients, so += performs gradient descent)
                self.E[t] += lr * grad_Et
                self.E[s] += lr * grad_Es
                self.R[r] += lr * grad_Rr
                for neg_t, grad in grad_neg.items():
                    self.E[neg_t] += lr * grad

            # L2 normalise entity embeddings
            norms = np.linalg.norm(self.E, axis=1, keepdims=True)
            self.E /= np.maximum(norms, 1e-8)


def transe_hits_at_k(
    kg: KnowledgeGraph,
    seed: int = 42,
    mask_ratio: float = 0.2,
    k: int = 10,
    dim: int = 50,
    n_epochs: int = 100,
) -> float:
    """Hits@K using TransE link prediction.

    Returns 0.0 when the KG has too few edges.
    """
    if not (0.0 < mask_ratio < 1.0):
        raise ValueError(f"mask_ratio must be in (0, 1), got {mask_ratio}")
    if k <= 0:
        raise ValueError(f"k must be > 0, got {k}")

    if kg.num_edges == 0 or kg.num_entities == 0:
        return 0.0

    edges = kg.edges
    train_edges, test_edges = _split_edges(edges, mask_ratio, seed)

    if len(train_edges) < _MIN_TRAIN_EDGES or not test_edges:
        return 0.0

    entity_ids = list(kg.entities.keys())
    model = _TransE(entity_ids=entity_ids, dim=dim, seed=seed)
    entity_to_idx = model.entity_to_idx

    triples: list[tuple[int, int, int]] = []
    for e in train_edges:
        s = entity_to_idx.get(e.source)
        t = entity_to_idx.get(e.target)
        if s is None or t is None:
            continue
        r = _ET_TO_IDX[e.edge_type]
        triples.append((s, r, t))

    model.train(triples, n_epochs=n_epochs, seed=seed)

    effective_k = min(k, model.n_entities - 1)
    if effective_k <= 0:
        return 0.0

    hits = 0
    for test_edge in test_edges:
        s = entity_to_idx.get(test_edge.source)
        t = entity_to_idx.get(test_edge.target)
        if s is None or t is None:
            continue
        r = _ET_TO_IDX[test_edge.edge_type]
        scores = model.score_all_targets(s, r)
        top_k_indices = np.argsort(-scores)[:effective_k]
        if t in top_k_indices:
            hits += 1

    return hits / len(test_edges)
