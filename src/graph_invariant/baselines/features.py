from __future__ import annotations

import networkx as nx
import numpy as np

from ..known_invariants import compute_known_invariant_values

FEATURE_ORDER = (
    "density",
    "clustering_coefficient",
    "degree_assortativity",
    "transitivity",
    "average_degree",
    "max_degree",
    "spectral_radius",
    "diameter",
    "algebraic_connectivity",
)


def features_from_graphs(
    graphs: list[nx.Graph],
    exclude_features: tuple[str, ...] | None = None,
) -> np.ndarray:
    selected = (
        tuple(f for f in FEATURE_ORDER if f not in exclude_features)
        if exclude_features
        else FEATURE_ORDER
    )
    if not graphs:
        return np.empty((0, len(selected)), dtype=float)
    values = compute_known_invariant_values(graphs)
    cols = [values[name] for name in selected]
    return np.asarray(list(zip(*cols, strict=True)), dtype=float)
