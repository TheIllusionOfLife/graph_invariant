"""ComplEx link-prediction model for small KGs.

Score(s, r, t) = Re(⟨e_s, r, ē_t⟩)  (Hermitian dot product)
where e_s, e_t, r ∈ C^d.

Delegates to analysis.external_eval._ComplEx for the model implementation.
"""

from __future__ import annotations

from harmony.types import KnowledgeGraph


def complex_hits_at_k(
    kg: KnowledgeGraph,
    seed: int = 42,
    mask_ratio: float = 0.2,
    k: int = 10,
    dim: int = 50,
    n_epochs: int = 100,
) -> float:
    """Hits@K using ComplEx link prediction.

    Thin wrapper over analysis.external_eval.evaluate_complex for import
    consistency with the other metric modules.
    """
    from analysis.external_eval import evaluate_complex

    return evaluate_complex(
        kg,
        seed=seed,
        mask_ratio=mask_ratio,
        k=k,
        dim=dim,
        n_epochs=n_epochs,
    )
