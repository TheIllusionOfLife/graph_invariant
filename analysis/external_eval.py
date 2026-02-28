"""External evaluation of KG proposals using RotatE and ComplEx.

Breaks the evaluation circularity concern (Reviewer A-Q1): Harmony uses DistMult
internally for generativity, so we evaluate proposals with *different* embedding
models (RotatE, ComplEx) to show improvements transfer across architectures.

Each model follows the same protocol as generativity():
  1. Mask 20% of edges uniformly at random (same _split_edges)
  2. Train the embedding model on remaining edges
  3. Evaluate Hits@K on masked edges

Models:
  - RotatE: score(s, r, t) = -||e_s ∘ r - e_t|| where r is a complex rotation
  - ComplEx: score(s, r, t) = Re(⟨e_s, r, ē_t⟩)  (Hermitian dot product)
"""

from __future__ import annotations

import numpy as np

from harmony.metric.generativity import _MIN_TRAIN_EDGES, _split_edges
from harmony.types import EdgeType, KnowledgeGraph

_ALL_EDGE_TYPES: list[EdgeType] = list(EdgeType)
_ET_TO_IDX: dict[EdgeType, int] = {et: i for i, et in enumerate(_ALL_EDGE_TYPES)}


# ---------------------------------------------------------------------------
# RotatE
# ---------------------------------------------------------------------------


class _RotatE:
    """RotatE link-prediction model for small KGs.

    Score(s, r, t) = -||e_s ∘ r - e_t||_1
    where e_s, e_t ∈ C^d and r ∈ C^d with |r_i| = 1 (unit-modulus rotation).

    Implementation uses 2*dim reals to represent dim complex numbers:
      [re_0, im_0, re_1, im_1, ...] for entities
      [θ_0, θ_1, ...] for relations (phase angles)
    """

    def __init__(self, entity_ids: list[str], dim: int = 50, seed: int = 42) -> None:
        self.entity_to_idx: dict[str, int] = {eid: i for i, eid in enumerate(entity_ids)}
        self.n_entities = len(entity_ids)
        self.n_relations = len(EdgeType)
        self.dim = dim

        rng = np.random.default_rng(seed)
        # Entity embeddings: complex, stored as (n_entities, 2*dim) [re, im interleaved]
        self.E: np.ndarray = rng.normal(0.0, 0.1, (self.n_entities, 2 * dim))
        # Relation embeddings: phase angles in [0, 2π)
        self.phase: np.ndarray = rng.uniform(0.0, 2 * np.pi, (self.n_relations, dim))

    def _complex_mul(
        self,
        a_re: np.ndarray,
        a_im: np.ndarray,
        b_re: np.ndarray,
        b_im: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Complex element-wise multiplication: (a_re + j*a_im) * (b_re + j*b_im)."""
        return a_re * b_re - a_im * b_im, a_re * b_im + a_im * b_re

    def _get_complex(self, idx: int, is_entity: bool = True) -> tuple[np.ndarray, np.ndarray]:
        """Extract real and imaginary parts from embedding."""
        if is_entity:
            vec = self.E[idx]
            return vec[0::2], vec[1::2]  # re, im (each dim,)
        # Relation: convert phase to unit complex number
        theta = self.phase[idx]
        return np.cos(theta), np.sin(theta)

    def _score(self, s: int, r: int, t: int) -> float:
        """RotatE score: -||e_s ∘ r - e_t||_1."""
        s_re, s_im = self._get_complex(s, is_entity=True)
        r_re, r_im = self._get_complex(r, is_entity=False)
        t_re, t_im = self._get_complex(t, is_entity=True)
        # e_s ∘ r (complex multiplication)
        rot_re, rot_im = self._complex_mul(s_re, s_im, r_re, r_im)
        # L1 distance to target
        diff_re = rot_re - t_re
        diff_im = rot_im - t_im
        return -float(np.sum(np.abs(diff_re) + np.abs(diff_im)))

    def score_all_targets(self, s: int, r: int) -> np.ndarray:
        """Return scores for all entities as targets. Shape: (n_entities,)."""
        s_re, s_im = self._get_complex(s, is_entity=True)
        r_re, r_im = self._get_complex(r, is_entity=False)
        rot_re, rot_im = self._complex_mul(s_re, s_im, r_re, r_im)

        scores = np.empty(self.n_entities)
        for t in range(self.n_entities):
            t_re, t_im = self._get_complex(t, is_entity=True)
            diff_re = rot_re - t_re
            diff_im = rot_im - t_im
            scores[t] = -float(np.sum(np.abs(diff_re) + np.abs(diff_im)))
        return scores

    def train(
        self,
        triples: list[tuple[int, int, int]],
        n_epochs: int = 100,
        lr: float = 0.01,
        margin: float = 1.0,
        n_neg: int = 5,
        seed: int = 42,
    ) -> None:
        """Train with self-adversarial negative sampling and max-margin loss."""
        if not triples:
            return
        rng = np.random.default_rng(seed)
        triples_arr = list(triples)

        for _ in range(n_epochs):
            rng.shuffle(triples_arr)  # type: ignore[arg-type]
            for s, r, t in triples_arr:
                pos_score = self._score(s, r, t)
                s_re, s_im = self._get_complex(s, is_entity=True)
                r_re, r_im = self._get_complex(r, is_entity=False)
                t_re, t_im = self._get_complex(t, is_entity=True)

                neg_candidates = [
                    int(x) for x in rng.integers(0, self.n_entities, n_neg) if int(x) != t
                ]
                for neg_t in neg_candidates:
                    neg_score = self._score(s, r, neg_t)
                    loss = margin - pos_score + neg_score
                    if loss <= 0:
                        continue

                    nt_re, nt_im = self._get_complex(neg_t, is_entity=True)
                    rot_re, rot_im = self._complex_mul(s_re, s_im, r_re, r_im)

                    # Gradient: push positive target closer, negative further
                    # L1 distance gradient: sign of (rot - target)
                    rot_re, rot_im = self._complex_mul(
                        s_re,
                        s_im,
                        r_re,
                        r_im,
                    )
                    sign_pos = np.sign(rot_re - t_re)
                    sign_pos_im = np.sign(rot_im - t_im)
                    sign_neg = np.sign(rot_re - nt_re)
                    sign_neg_im = np.sign(rot_im - nt_im)

                    # Update target entities (push t closer, neg_t farther)
                    self.E[t, 0::2] += lr * sign_pos
                    self.E[t, 1::2] += lr * sign_pos_im
                    self.E[neg_t, 0::2] -= lr * sign_neg
                    self.E[neg_t, 1::2] -= lr * sign_neg_im

                    # Update source entity: ∂L/∂E[s] via chain rule
                    # through complex multiply (pos - neg contribution)
                    grad_s_re = (sign_pos - sign_neg) * r_re + (sign_pos_im - sign_neg_im) * (-r_im)
                    grad_s_im = (sign_pos - sign_neg) * r_im + (sign_pos_im - sign_neg_im) * r_re
                    self.E[s, 0::2] -= lr * grad_s_re
                    self.E[s, 1::2] -= lr * grad_s_im

                    # Update relation phase: ∂L/∂θ via chain rule
                    # d(cos θ)/dθ = -sin θ, d(sin θ)/dθ = cos θ
                    theta = self.phase[r]
                    grad_theta = (sign_pos - sign_neg) * (
                        s_re * (-np.sin(theta)) - s_im * np.cos(theta)
                    ) + (sign_pos_im - sign_neg_im) * (
                        s_re * np.cos(theta) + s_im * (-np.sin(theta))
                    )
                    self.phase[r] -= lr * grad_theta

            # L2 normalise entity embeddings
            norms = np.linalg.norm(self.E, axis=1, keepdims=True)
            self.E /= np.maximum(norms, 1e-8)


# ---------------------------------------------------------------------------
# ComplEx
# ---------------------------------------------------------------------------


class _ComplEx:
    """ComplEx link-prediction model for small KGs.

    Score(s, r, t) = Re(⟨e_s, r, ē_t⟩) = Re(Σ e_s_i * r_i * conj(e_t_i))
    where e_s, e_t, r ∈ C^d.

    Stored as (n, 2*dim) reals: [re_0, im_0, re_1, im_1, ...].
    """

    def __init__(self, entity_ids: list[str], dim: int = 50, seed: int = 42) -> None:
        self.entity_to_idx: dict[str, int] = {eid: i for i, eid in enumerate(entity_ids)}
        self.n_entities = len(entity_ids)
        self.n_relations = len(EdgeType)
        self.dim = dim

        rng = np.random.default_rng(seed)
        self.E: np.ndarray = rng.normal(0.0, 0.1, (self.n_entities, 2 * dim))
        self.R: np.ndarray = rng.normal(0.0, 0.1, (self.n_relations, 2 * dim))

    def _score(self, s: int, r: int, t: int) -> float:
        """ComplEx score: Re(⟨e_s, r, ē_t⟩)."""
        s_re, s_im = self.E[s, 0::2], self.E[s, 1::2]
        r_re, r_im = self.R[r, 0::2], self.R[r, 1::2]
        t_re, t_im = self.E[t, 0::2], self.E[t, 1::2]

        # ⟨e_s, r, ē_t⟩ = Σ (s * r * conj(t))
        # Real part of (s_re + j*s_im)(r_re + j*r_im)(t_re - j*t_im)
        # = s_re*r_re*t_re + s_re*r_im*t_im + s_im*r_re*t_im - s_im*r_im*t_re
        score = float(
            np.sum(
                s_re * r_re * t_re + s_re * r_im * t_im + s_im * r_re * t_im - s_im * r_im * t_re
            )
        )
        return score

    def score_all_targets(self, s: int, r: int) -> np.ndarray:
        """Return scores for all entities as targets. Shape: (n_entities,)."""
        s_re, s_im = self.E[s, 0::2], self.E[s, 1::2]
        r_re, r_im = self.R[r, 0::2], self.R[r, 1::2]

        # For each target t, compute Re(⟨s, r, ē_t⟩)
        # = (s_re*r_re - s_im*r_im) * t_re + (s_re*r_im + s_im*r_re) * t_im
        sr_re = s_re * r_re - s_im * r_im  # (dim,)
        sr_im = s_re * r_im + s_im * r_re  # (dim,)

        all_t_re = self.E[:, 0::2]  # (n_entities, dim)
        all_t_im = self.E[:, 1::2]  # (n_entities, dim)

        scores = all_t_re @ sr_re + all_t_im @ sr_im  # (n_entities,)
        return scores

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
                sr_re = self.E[s, 0::2] * self.R[r, 0::2] - self.E[s, 1::2] * self.R[r, 1::2]
                sr_im = self.E[s, 0::2] * self.R[r, 1::2] + self.E[s, 1::2] * self.R[r, 0::2]

                neg_candidates = [
                    int(x) for x in rng.integers(0, self.n_entities, n_neg) if int(x) != t
                ]
                for neg_t in neg_candidates:
                    neg_score = self._score(s, r, neg_t)
                    loss = margin - pos_score + neg_score
                    if loss <= 0:
                        continue

                    # Push positive target score up, negative target score down
                    # grad w.r.t. E[t]: sr_re (real part), sr_im (imag part)
                    self.E[t, 0::2] += lr * sr_re
                    self.E[t, 1::2] += lr * sr_im
                    self.E[neg_t, 0::2] -= lr * sr_re
                    self.E[neg_t, 1::2] -= lr * sr_im

                    # grad w.r.t. E[s]: (t_re - neg_t_re, t_im - neg_t_im)
                    t_re = self.E[t, 0::2]
                    t_im = self.E[t, 1::2]
                    nt_re = self.E[neg_t, 0::2]
                    nt_im = self.E[neg_t, 1::2]
                    r_re = self.R[r, 0::2]
                    r_im = self.R[r, 1::2]
                    diff_re = t_re - nt_re
                    diff_im = t_im - nt_im
                    # ∂score/∂s_re = r_re*t_re + r_im*t_im
                    grad_s_re = r_re * diff_re + r_im * diff_im
                    grad_s_im = r_re * diff_im - r_im * diff_re
                    self.E[s, 0::2] += lr * grad_s_re
                    self.E[s, 1::2] += lr * grad_s_im

                    # grad w.r.t. R[r]
                    s_re = self.E[s, 0::2]
                    s_im = self.E[s, 1::2]
                    grad_r_re = s_re * diff_re + s_im * diff_im
                    grad_r_im = s_re * diff_im - s_im * diff_re
                    self.R[r, 0::2] += lr * grad_r_re
                    self.R[r, 1::2] += lr * grad_r_im

            # L2 normalise entity embeddings
            norms = np.linalg.norm(self.E, axis=1, keepdims=True)
            self.E /= np.maximum(norms, 1e-8)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def _evaluate_model(
    model_class: type,
    kg: KnowledgeGraph,
    seed: int = 42,
    mask_ratio: float = 0.2,
    k: int = 10,
    dim: int = 50,
    n_epochs: int = 100,
) -> float:
    """Generic Hits@K evaluation using the same protocol as generativity().

    Returns 0.0 when the KG has too few edges.
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

    entity_ids = list(kg.entities.keys())
    model = model_class(entity_ids=entity_ids, dim=dim, seed=seed)
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


def evaluate_rotate(
    kg: KnowledgeGraph,
    seed: int = 42,
    mask_ratio: float = 0.2,
    k: int = 10,
    dim: int = 50,
    n_epochs: int = 100,
) -> float:
    """Hits@K using RotatE — external evaluation (not used in Harmony scoring)."""
    return _evaluate_model(
        _RotatE,
        kg,
        seed=seed,
        mask_ratio=mask_ratio,
        k=k,
        dim=dim,
        n_epochs=n_epochs,
    )


def evaluate_complex(
    kg: KnowledgeGraph,
    seed: int = 42,
    mask_ratio: float = 0.2,
    k: int = 10,
    dim: int = 50,
    n_epochs: int = 100,
) -> float:
    """Hits@K using ComplEx — external evaluation (not used in Harmony scoring)."""
    return _evaluate_model(
        _ComplEx,
        kg,
        seed=seed,
        mask_ratio=mask_ratio,
        k=k,
        dim=dim,
        n_epochs=n_epochs,
    )


def evaluate_external(
    kg: KnowledgeGraph,
    seed: int = 42,
    mask_ratio: float = 0.2,
    k: int = 10,
    dim: int = 50,
    n_epochs: int = 100,
) -> dict[str, float]:
    """Evaluate KG with all three embedding models (DistMult + RotatE + ComplEx).

    Returns dict mapping model name → Hits@K score.
    DistMult is included for comparison (same as generativity component).
    """
    from harmony.metric.generativity import generativity

    kw = dict(seed=seed, mask_ratio=mask_ratio, k=k, dim=dim, n_epochs=n_epochs)
    return {
        "distmult": generativity(kg, **kw),
        "rotate": evaluate_rotate(kg, **kw),
        "complex": evaluate_complex(kg, **kw),
    }
