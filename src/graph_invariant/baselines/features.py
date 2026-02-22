from __future__ import annotations

import networkx as nx
import numpy as np

from ..known_invariants import compute_known_invariant_values

_BASE_FEATURE_ORDER = (
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

_SPECTRAL_FEATURE_ORDER = (
    "laplacian_lambda2",
    "laplacian_lambda_max",
    "laplacian_spectral_gap",
    "normalized_laplacian_lambda2",
    "laplacian_energy_ratio",
)

FEATURE_ORDER = _BASE_FEATURE_ORDER + _SPECTRAL_FEATURE_ORDER


def feature_order(enable_spectral_feature_pack: bool = True) -> tuple[str, ...]:
    return FEATURE_ORDER if enable_spectral_feature_pack else _BASE_FEATURE_ORDER


def features_from_graphs(
    graphs: list[nx.Graph],
    exclude_features: tuple[str, ...] | None = None,
    enable_spectral_feature_pack: bool = True,
) -> np.ndarray:
    order = feature_order(enable_spectral_feature_pack=enable_spectral_feature_pack)
    if exclude_features:
        exclude_set = frozenset(exclude_features)
        selected = tuple(f for f in order if f not in exclude_set)
    else:
        selected = order
    if not graphs or not selected:
        return np.empty((len(graphs), len(selected)), dtype=float)
    values = compute_known_invariant_values(
        graphs,
        include_spectral_feature_pack=enable_spectral_feature_pack,
    )
    return features_from_dict(
        values,
        num_graphs=len(graphs),
        exclude_features=exclude_features,
        enable_spectral_feature_pack=enable_spectral_feature_pack,
    )


def features_from_dict(
    values: dict[str, list[float]],
    num_graphs: int,
    exclude_features: tuple[str, ...] | None = None,
    enable_spectral_feature_pack: bool = True,
) -> np.ndarray:
    order = feature_order(enable_spectral_feature_pack=enable_spectral_feature_pack)
    if exclude_features:
        exclude_set = frozenset(exclude_features)
        selected = tuple(f for f in order if f not in exclude_set)
    else:
        selected = order

    if not selected or num_graphs == 0:
        return np.empty((num_graphs, len(selected)), dtype=float)

    cols = [values[name] for name in selected]
    return np.asarray(list(zip(*cols, strict=True)), dtype=float)
