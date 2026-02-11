from dataclasses import dataclass

import networkx as nx
import numpy as np

from .config import Phase1Config


@dataclass(slots=True)
class DatasetBundle:
    train: list[nx.Graph]
    val: list[nx.Graph]
    test: list[nx.Graph]
    sanity: list[nx.Graph]


def _connected_subgraph(graph: nx.Graph) -> nx.Graph:
    if len(graph) == 0:
        return nx.path_graph(2)
    if nx.is_connected(graph):
        return graph
    largest = max(nx.connected_components(graph), key=len)
    return nx.convert_node_labels_to_integers(graph.subgraph(largest).copy())


def _generate_graph(rng: np.random.Generator, n: int) -> nx.Graph:
    kind = rng.choice(["ER", "BA", "WS", "RGG", "SBM"])
    if kind == "ER":
        g = nx.erdos_renyi_graph(n, float(rng.uniform(0.05, 0.3)), seed=int(rng.integers(1e9)))
    elif kind == "BA":
        m = int(rng.integers(1, min(4, n - 1) + 1))
        g = nx.barabasi_albert_graph(n, m, seed=int(rng.integers(1e9)))
    elif kind == "WS":
        k = int(rng.integers(4, 9))
        if k >= n:
            k = n - 1 if (n - 1) % 2 == 0 else n - 2
        if k < 2:
            k = 2
        if k % 2 == 1:
            k += 1
        g = nx.watts_strogatz_graph(n, k, float(rng.uniform(0.1, 0.5)), seed=int(rng.integers(1e9)))
    elif kind == "RGG":
        g = nx.random_geometric_graph(n, float(rng.uniform(0.1, 0.4)), seed=int(rng.integers(1e9)))
    else:
        sizes = [n // 3, n // 3, n - 2 * (n // 3)]
        probs = [[0.25, 0.05, 0.02], [0.05, 0.25, 0.05], [0.02, 0.05, 0.25]]
        g = nx.stochastic_block_model(sizes, probs, seed=int(rng.integers(1e9)))
    return _connected_subgraph(g)


def _sample_graphs(rng: np.random.Generator, count: int) -> list[nx.Graph]:
    out: list[nx.Graph] = []
    for _ in range(count):
        n = int(rng.integers(30, 101))
        out.append(_generate_graph(rng, n))
    return out


def generate_phase1_datasets(cfg: Phase1Config) -> DatasetBundle:
    rng = np.random.default_rng(cfg.seed)
    train = _sample_graphs(rng, cfg.num_train_graphs)
    val = _sample_graphs(rng, cfg.num_val_graphs)
    test = _sample_graphs(rng, cfg.num_test_graphs)
    sanity = [
        nx.karate_club_graph(),
        nx.les_miserables_graph(),
        nx.florentine_families_graph(),
    ]
    return DatasetBundle(train=train, val=val, test=test, sanity=sanity)
