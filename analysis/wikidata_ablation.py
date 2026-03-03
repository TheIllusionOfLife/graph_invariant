"""Wikidata ablation: per-component contribution analysis.

Measures each Harmony component's score per domain using bootstrap
resampling for confidence intervals.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from harmony.metric.coherence import coherence
from harmony.metric.compressibility import compressibility
from harmony.metric.generativity import generativity
from harmony.metric.symmetry import symmetry
from harmony.types import KnowledgeGraph

_COMPONENT_FUNCS = {
    "compressibility": lambda kg, seed: compressibility(kg),
    "coherence": lambda kg, seed: coherence(kg),
    "symmetry": lambda kg, seed: symmetry(kg),
    "generativity": lambda kg, seed: generativity(kg, seed=seed),
}


def run_wikidata_ablation(
    kgs: dict[str, KnowledgeGraph],
    n_bootstrap: int = 200,
    seed: int = 42,
) -> pd.DataFrame:
    """Run per-component ablation analysis on provided KGs.

    Parameters
    ----------
    kgs:
        Mapping of domain name → KnowledgeGraph.
    n_bootstrap:
        Number of bootstrap resamples for CI computation.
    seed:
        Random seed.

    Returns
    -------
    DataFrame with columns: domain, component, mean, std, ci95_low, ci95_high.
    """
    if n_bootstrap <= 0:
        raise ValueError(f"n_bootstrap must be > 0, got {n_bootstrap}")
    rows: list[dict[str, object]] = []
    rng = np.random.default_rng(seed)

    for domain, kg in kgs.items():
        for comp_name, comp_func in _COMPONENT_FUNCS.items():
            # Bootstrap: resample seeds for generativity, use base for deterministic components
            boot_scores = np.empty(n_bootstrap)
            for i in range(n_bootstrap):
                boot_seed = int(rng.integers(0, 2**31))
                boot_scores[i] = comp_func(kg, boot_seed)

            mean = float(np.mean(boot_scores))
            std = float(np.std(boot_scores, ddof=1)) if n_bootstrap > 1 else 0.0
            ci95_low = float(np.percentile(boot_scores, 2.5))
            ci95_high = float(np.percentile(boot_scores, 97.5))

            rows.append(
                {
                    "domain": domain,
                    "component": comp_name,
                    "mean": mean,
                    "std": std,
                    "ci95_low": ci95_low,
                    "ci95_high": ci95_high,
                }
            )

    return pd.DataFrame(rows)
