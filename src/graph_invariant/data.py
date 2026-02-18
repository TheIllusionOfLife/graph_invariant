import warnings
from dataclasses import dataclass

import networkx as nx
import numpy as np

from .config import Phase1Config

_SPECIAL_TOPOLOGY_POOL_RECOMMENDED_MAX_RATIO = 0.2


@dataclass(slots=True)
class DatasetBundle:
    train: list[nx.Graph]
    val: list[nx.Graph]
    test: list[nx.Graph]
    sanity: list[nx.Graph]


def connected_subgraph(graph: nx.Graph) -> nx.Graph:
    if len(graph) == 0:
        return nx.path_graph(2)
    if nx.is_connected(graph):
        return graph
    largest = max(nx.connected_components(graph), key=len)
    return nx.convert_node_labels_to_integers(graph.subgraph(largest).copy())


def generate_graph(rng: np.random.Generator, n: int) -> nx.Graph:
    kind = rng.choice(["ER", "BA", "WS", "RGG", "SBM"])
    if kind == "ER":
        g = nx.erdos_renyi_graph(n, float(rng.uniform(0.05, 0.3)), seed=rng)
    elif kind == "BA":
        m = int(rng.integers(1, min(4, n - 1) + 1))
        g = nx.barabasi_albert_graph(n, m, seed=rng)
    elif kind == "WS":
        k = int(rng.integers(4, 9))
        if k >= n:
            k = n - 1 if (n - 1) % 2 == 0 else n - 2
        if k < 2:
            k = 2
        if k % 2 == 1:
            k += 1
        g = nx.watts_strogatz_graph(n, k, float(rng.uniform(0.1, 0.5)), seed=rng)
    elif kind == "RGG":
        g = nx.random_geometric_graph(n, float(rng.uniform(0.1, 0.4)), seed=rng)
    else:
        sizes = [n // 3, n // 3, n - 2 * (n // 3)]
        probs = [[0.25, 0.05, 0.02], [0.05, 0.25, 0.05], [0.02, 0.05, 0.25]]
        g = nx.stochastic_block_model(sizes, probs, seed=rng)
    return connected_subgraph(g)


def _sample_graphs(rng: np.random.Generator, count: int) -> list[nx.Graph]:
    out: list[nx.Graph] = []
    for _ in range(count):
        n = int(rng.integers(30, 101))
        out.append(generate_graph(rng, n))
    return out


def _sample_special_topology_graphs(
    rng: np.random.Generator,
    pool: list[nx.Graph],
    count: int,
) -> list[nx.Graph]:
    """Sample from a fixed special-topology pool (with replacement)."""
    if count <= 0:
        return []
    duplication_factor = count / float(len(pool))
    if duplication_factor > 1.0:
        warnings.warn(
            (
                f"special-topology sampling requests {count} graphs from a pool of {len(pool)}; "
                f"expected duplication factor is {duplication_factor:.2f}. "
                "Consider using ood_*_special_topology_ratio <= 0.2 for diversity."
            ),
            stacklevel=2,
        )
    indices = rng.integers(0, len(pool), size=count)
    return [pool[int(idx)].copy() for idx in indices]


# ── OOD graph generation ──────────────────────────────────────────────


def generate_ood_large_random(rng: np.random.Generator, count: int) -> list[nx.Graph]:
    """Same graph types as training but with n in [200, 500]."""
    graphs: list[nx.Graph] = []
    for _ in range(count):
        n = int(rng.integers(200, 501))
        graphs.append(generate_graph(rng, n))
    return graphs


def generate_ood_extreme_params(rng: np.random.Generator, count: int) -> list[nx.Graph]:
    """Graphs with extreme densities/degrees, n in [50, 200]."""
    graphs: list[nx.Graph] = []
    for _ in range(count):
        n = int(rng.integers(50, 201))
        kind = rng.choice(["dense_er", "sparse_er", "high_ba", "low_ws"])
        if kind == "dense_er":
            g = nx.erdos_renyi_graph(n, float(rng.uniform(0.4, 0.7)), seed=rng)
        elif kind == "sparse_er":
            g = nx.erdos_renyi_graph(n, float(rng.uniform(0.01, 0.04)), seed=rng)
        elif kind == "high_ba":
            m = int(rng.integers(8, min(15, n - 1) + 1))
            g = nx.barabasi_albert_graph(n, m, seed=rng)
        else:  # low_ws
            k = 2
            g = nx.watts_strogatz_graph(n, k, float(rng.uniform(0.01, 0.1)), seed=rng)
        graphs.append(connected_subgraph(g))
    return graphs


def generate_ood_special_topology() -> list[nx.Graph]:
    """Deterministic structures for structural generalization.

    Returns a fixed pool of 8 connected, integer-labeled topology archetypes.
    Phase1 train/val injection samples from this pool with replacement; in
    practice, keep `ood_*_special_topology_ratio <= 0.2` to limit duplication.
    """
    graphs: list[nx.Graph] = []
    graphs.append(nx.barbell_graph(20, 5))
    graphs.append(nx.grid_2d_graph(8, 8))
    graphs.append(nx.circular_ladder_graph(30))
    graphs.append(nx.random_regular_graph(4, 50, seed=0))
    graphs.append(nx.powerlaw_cluster_graph(100, 3, 0.5, seed=0))
    graphs.append(nx.karate_club_graph())
    graphs.append(nx.les_miserables_graph())
    graphs.append(nx.florentine_families_graph())
    # Ensure all connected and integer-labeled (e.g. grid_2d_graph uses tuple labels)
    return [nx.convert_node_labels_to_integers(connected_subgraph(g)) for g in graphs]


def generate_phase1_datasets(cfg: Phase1Config) -> DatasetBundle:
    rng = np.random.default_rng(cfg.seed)
    train = _sample_graphs(rng, cfg.num_train_graphs)
    val = _sample_graphs(rng, cfg.num_val_graphs)
    test = _sample_graphs(rng, cfg.num_test_graphs)
    special_pool = generate_ood_special_topology()

    if cfg.ood_train_special_topology_ratio > _SPECIAL_TOPOLOGY_POOL_RECOMMENDED_MAX_RATIO:
        warnings.warn(
            (
                "ood_train_special_topology_ratio exceeds recommended maximum 0.2; "
                "expect higher duplicate special-topology samples."
            ),
            stacklevel=2,
        )
    if cfg.ood_val_special_topology_ratio > _SPECIAL_TOPOLOGY_POOL_RECOMMENDED_MAX_RATIO:
        warnings.warn(
            (
                "ood_val_special_topology_ratio exceeds recommended maximum 0.2; "
                "expect higher duplicate special-topology samples."
            ),
            stacklevel=2,
        )

    train_special_count = min(
        cfg.num_train_graphs,
        int(round(cfg.ood_train_special_topology_ratio * cfg.num_train_graphs)),
    )
    if train_special_count > 0:
        train[:train_special_count] = _sample_special_topology_graphs(
            rng=rng,
            pool=special_pool,
            count=train_special_count,
        )

    val_special_count = min(
        cfg.num_val_graphs,
        int(round(cfg.ood_val_special_topology_ratio * cfg.num_val_graphs)),
    )
    if val_special_count > 0:
        val[:val_special_count] = _sample_special_topology_graphs(
            rng=rng,
            pool=special_pool,
            count=val_special_count,
        )

    sanity = [
        nx.karate_club_graph(),
        nx.les_miserables_graph(),
        nx.florentine_families_graph(),
    ]
    return DatasetBundle(train=train, val=val, test=test, sanity=sanity)
