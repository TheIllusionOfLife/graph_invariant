"""Composite Harmony metric and proposal value function.

Harmony(D) = α·comp + β·coh + γ·sym + δ·gen  ∈ [0,1]

where comp, coh, sym, gen are each normalised to [0,1]:
  - comp: compressibility  (MDL proxy)
  - coh : coherence        (path agreement + contradiction-free rate)
  - sym : symmetry         (1 − avg pairwise JS divergence)
  - gen : generativity     (DistMult Hits@K on masked edges)

Distortion(D) = 1 − Harmony(D)  ∈ [0,1]

Proposal value function:
  value_of(D, D⊕Δ) = Distortion(D) − Distortion(D⊕Δ) − λ·Cost(Δ)
                    = Harmony(D⊕Δ) − Harmony(D) − λ·Cost(Δ)

A positive value means the proposal improves harmony net of cost.

Default weights are equal (α=β=γ=δ=0.25).  Custom weights are accepted
but must be non-negative; they are normalised internally if they do not
sum to 1.0.
"""

from __future__ import annotations

from harmony.metric.coherence import coherence
from harmony.metric.compressibility import compressibility
from harmony.metric.generativity import generativity
from harmony.metric.symmetry import symmetry
from harmony.types import KnowledgeGraph


def harmony_score(
    kg: KnowledgeGraph,
    alpha: float = 0.25,
    beta: float = 0.25,
    gamma: float = 0.25,
    delta: float = 0.25,
    seed: int = 42,
    mask_ratio: float = 0.2,
    k: int = 10,
) -> float:
    """Composite Harmony score ∈ [0,1]; higher = more harmonious.

    Parameters
    ----------
    kg:
        The knowledge graph to score.
    alpha, beta, gamma, delta:
        Non-negative weights for compressibility, coherence, symmetry,
        and generativity.  Raw (unnormalized) values are accepted and
        divided by their sum internally — callers do NOT need to pre-
        normalise.  All weights must be >= 0 and at least one must be > 0.
    seed:
        Random seed forwarded to the generativity component.
    mask_ratio:
        Fraction of edges masked for the link-prediction task (generativity).
    k:
        Hits@K cutoff for the generativity component.
    """
    if any(w < 0.0 for w in (alpha, beta, gamma, delta)):
        raise ValueError("All weights must be >= 0.0")
    total_w = alpha + beta + gamma + delta
    if total_w <= 0.0:
        raise ValueError("At least one weight must be > 0")
    alpha, beta, gamma, delta = (
        alpha / total_w,
        beta / total_w,
        gamma / total_w,
        delta / total_w,
    )

    comp = compressibility(kg)
    coh = coherence(kg)
    sym = symmetry(kg)
    gen = generativity(kg, seed=seed, mask_ratio=mask_ratio, k=k)

    return alpha * comp + beta * coh + gamma * sym + delta * gen


def distortion(
    kg: KnowledgeGraph,
    alpha: float = 0.25,
    beta: float = 0.25,
    gamma: float = 0.25,
    delta: float = 0.25,
    seed: int = 42,
    mask_ratio: float = 0.2,
    k: int = 10,
) -> float:
    """Distortion score ∈ [0,1]; lower = more harmonious (1 − harmony_score)."""
    return 1.0 - harmony_score(
        kg,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        delta=delta,
        seed=seed,
        mask_ratio=mask_ratio,
        k=k,
    )


def value_of(
    kg_before: KnowledgeGraph,
    kg_after: KnowledgeGraph,
    alpha: float = 0.25,
    beta: float = 0.25,
    gamma: float = 0.25,
    delta: float = 0.25,
    lambda_cost: float = 0.1,
    cost: float = 0.0,
    seed: int = 42,
    mask_ratio: float = 0.2,
    k: int = 10,
) -> float:
    """Proposal value: harmony gain after applying proposal Δ, minus cost.

    value_of(D, D⊕Δ) = Harmony(D⊕Δ) − Harmony(D) − λ·Cost(Δ)

    A positive return value indicates the proposal improves the KG's
    harmony net of its structural complexity cost.

    Parameters
    ----------
    kg_before:
        Knowledge graph before applying the proposal.
    kg_after:
        Knowledge graph after applying the proposal (D ⊕ Δ).
    lambda_cost:
        Cost penalty weight; must be >= 0.  A negative value would turn
        the penalty into a bonus, inverting the function's semantics.
    cost:
        Normalised complexity cost of the proposal (e.g. number of new
        edges added, divided by |E|); must be >= 0.
    """
    if lambda_cost < 0.0:
        raise ValueError(f"lambda_cost must be >= 0.0, got {lambda_cost}")
    if cost < 0.0:
        raise ValueError(f"cost must be >= 0.0, got {cost}")
    h_after = harmony_score(
        kg_after,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        delta=delta,
        seed=seed,
        mask_ratio=mask_ratio,
        k=k,
    )
    h_before = harmony_score(
        kg_before,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        delta=delta,
        seed=seed,
        mask_ratio=mask_ratio,
        k=k,
    )
    return h_after - h_before - lambda_cost * cost
