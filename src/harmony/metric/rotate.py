"""RotatE link-prediction model for small KGs.

Score(s, r, t) = -||e_s ∘ r - e_t||_1
where e_s, e_t ∈ C^d and r ∈ C^d with |r_i| = 1 (unit-modulus rotation).

Delegates to analysis.external_eval._RotatE for the model implementation.
"""

from __future__ import annotations

from harmony.types import KnowledgeGraph


def rotate_hits_at_k(
    kg: KnowledgeGraph,
    seed: int = 42,
    mask_ratio: float = 0.2,
    k: int = 10,
    dim: int = 50,
    n_epochs: int = 100,
) -> float:
    """Hits@K using RotatE link prediction.

    Thin wrapper over analysis.external_eval.evaluate_rotate for import
    consistency with the other metric modules.
    """
    from analysis.external_eval import evaluate_rotate

    return evaluate_rotate(
        kg, seed=seed, mask_ratio=mask_ratio,
        k=k, dim=dim, n_epochs=n_epochs,
    )
