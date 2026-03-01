#!/usr/bin/env python
"""MRR + Hits@K comparison table across discovery domains.

Computes Harmony vs three baselines (random, frequency, DistMult) using
the same masked-edge evaluation protocol as harmony/metric/baselines.py.
Archive proposals are applied to the KG before measuring Harmony metrics
so that the table reflects the post-search KG quality.

Usage:
    python analysis/metrics_table.py \
        --astronomy artifacts/harmony/astronomy \
        --physics   artifacts/harmony/physics \
        --materials artifacts/harmony/materials \
        --output    artifacts/harmony/metrics_table.csv
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from harmony.map_elites import deserialize_archive
from harmony.metric.baselines import baseline_distmult_alone, baseline_frequency, baseline_random
from harmony.metric.generativity import (
    _ET_TO_IDX,
    _MIN_TRAIN_EDGES,
    _DistMult,
    _split_edges,
    generativity,
)
from harmony.proposals.types import Proposal, ProposalType
from harmony.state import load_state
from harmony.types import EdgeType, KnowledgeGraph, TypedEdge

# ---------------------------------------------------------------------------
# Domain → KG builder mapping (used when kgs= not provided)
# ---------------------------------------------------------------------------

_DOMAIN_BUILDERS: dict[str, str] = {
    "linear_algebra": "harmony.datasets.linear_algebra.build_linear_algebra_kg",
    "periodic_table": "harmony.datasets.periodic_table.build_periodic_table_kg",
    "astronomy": "harmony.datasets.astronomy.build_astronomy_kg",
    "physics": "harmony.datasets.physics.build_physics_kg",
    "materials": "harmony.datasets.materials.build_materials_kg",
    "wikidata_physics": "harmony.datasets.wikidata_physics.build_wikidata_physics_kg",
    "wikidata_materials": "harmony.datasets.wikidata_materials.build_wikidata_materials_kg",
}


def _load_kg_for_domain(domain: str) -> KnowledgeGraph:
    """Import and call the builder function for a named domain."""
    if domain not in _DOMAIN_BUILDERS:
        raise ValueError(f"Unknown domain '{domain}'. Known domains: {sorted(_DOMAIN_BUILDERS)}")
    module_path, func_name = _DOMAIN_BUILDERS[domain].rsplit(".", 1)
    import importlib

    module = importlib.import_module(module_path)
    return getattr(module, func_name)()


# ---------------------------------------------------------------------------
# KG mutation helpers
# ---------------------------------------------------------------------------


def _apply_proposals_to_kg(
    kg: KnowledgeGraph,
    proposals: list[Proposal],
) -> KnowledgeGraph:
    """Return a deep copy of kg with valid ADD_EDGE proposals applied.

    Only ADD_EDGE proposals whose source and target entities both exist in kg
    and whose edge_type is a valid EdgeType name are applied. All others are
    silently skipped to tolerate stale/invalid proposals in the archive.
    """
    kg_copy = copy.deepcopy(kg)
    seen_edges: set[tuple[str, str, EdgeType]] = {
        (edge.source, edge.target, edge.edge_type) for edge in kg_copy.edges
    }
    for proposal in proposals:
        if proposal.proposal_type != ProposalType.ADD_EDGE:
            continue
        if not (proposal.source_entity and proposal.target_entity and proposal.edge_type):
            continue
        if proposal.source_entity not in kg_copy.entities:
            continue
        if proposal.target_entity not in kg_copy.entities:
            continue
        try:
            edge_type = EdgeType[proposal.edge_type]
        except KeyError:
            continue
        edge_key = (proposal.source_entity, proposal.target_entity, edge_type)
        if edge_key in seen_edges:
            continue
        try:
            kg_copy.add_edge(
                TypedEdge(
                    source=proposal.source_entity,
                    target=proposal.target_entity,
                    edge_type=edge_type,
                )
            )
            seen_edges.add(edge_key)
        except ValueError:
            continue
    return kg_copy


# ---------------------------------------------------------------------------
# MRR computation
# ---------------------------------------------------------------------------


def _mean_reciprocal_rank(
    kg: KnowledgeGraph,
    seed: int = 42,
    mask_ratio: float = 0.2,
    dim: int = 50,
    n_epochs: int = 100,
) -> float:
    """Mean Reciprocal Rank via DistMult on masked edges.

    For each masked test edge (s, r, t), rank all entities as targets via
    DistMult scores, find the 0-based rank of the true target, and accumulate
    1/(rank+1). Returns mean over all test edges.

    Returns 0.0 when the KG has too few edges to train meaningfully
    (same threshold as generativity: _MIN_TRAIN_EDGES).
    """
    if kg.num_edges == 0 or kg.num_entities == 0:
        return 0.0

    edges = kg.edges
    train_edges, test_edges = _split_edges(edges, mask_ratio, seed)

    if len(train_edges) < _MIN_TRAIN_EDGES or not test_edges:
        return 0.0

    entity_ids = list(kg.entities.keys())
    model = _DistMult(entity_ids=entity_ids, dim=dim, seed=seed)
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

    reciprocal_ranks: list[float] = []
    for test_edge in test_edges:
        s = entity_to_idx.get(test_edge.source)
        t = entity_to_idx.get(test_edge.target)
        if s is None or t is None:
            continue
        r = _ET_TO_IDX[test_edge.edge_type]
        scores = model.score_all_targets(s, r)
        sorted_indices = np.argsort(-scores)
        rank_positions = np.where(sorted_indices == t)[0]
        if len(rank_positions) == 0:
            continue
        rank = int(rank_positions[0])  # 0-based
        reciprocal_ranks.append(1.0 / (rank + 1))

    if not reciprocal_ranks:
        return 0.0
    return sum(reciprocal_ranks) / len(reciprocal_ranks)


def _mrr_random(
    kg: KnowledgeGraph,
    seed: int = 42,
    mask_ratio: float = 0.2,
) -> float:
    """MRR under uniform random target ranking."""
    if kg.num_edges == 0 or kg.num_entities == 0:
        return 0.0

    edges = kg.edges
    train_edges, test_edges = _split_edges(edges, mask_ratio, seed)
    if len(train_edges) < _MIN_TRAIN_EDGES or not test_edges:
        return 0.0

    entity_ids = list(kg.entities.keys())
    n_entities = len(entity_ids)
    entity_to_idx = {eid: i for i, eid in enumerate(entity_ids)}

    rng = np.random.default_rng(seed)
    reciprocal_ranks: list[float] = []
    for edge in test_edges:
        t_idx = entity_to_idx.get(edge.target)
        if t_idx is None:
            continue
        shuffled = rng.permutation(n_entities)
        rank = int(np.where(shuffled == t_idx)[0][0])
        reciprocal_ranks.append(1.0 / (rank + 1))

    if not reciprocal_ranks:
        return 0.0
    return sum(reciprocal_ranks) / len(reciprocal_ranks)


# ---------------------------------------------------------------------------
# Main table computation
# ---------------------------------------------------------------------------


def compute_metrics_table(
    domain_checkpoints: dict[str, Path],
    seed: int = 42,
    kgs: dict[str, KnowledgeGraph] | None = None,
) -> pd.DataFrame:
    """Compute MRR + Hits@K comparison table across domains.

    Parameters
    ----------
    domain_checkpoints:
        Mapping of domain label → output_dir containing checkpoint.json.
    seed:
        RNG seed passed to all metric functions for reproducibility.
    kgs:
        Optional pre-loaded KGs (for testing). When None, KGs are loaded
        via the domain-name-to-builder mapping.

    Returns
    -------
    DataFrame with one row per domain and columns:
        random_hits10, freq_hits10, distmult_hits10, harmony_hits10,
        mrr_random, mrr_distmult, mrr_harmony
    """
    rows: dict[str, dict[str, float]] = {}

    for domain, output_dir in domain_checkpoints.items():
        output_dir = Path(output_dir)

        # Load KG
        if kgs is not None:
            kg = kgs[domain]
        else:
            kg = _load_kg_for_domain(domain)

        # Load checkpoint and extract archive proposals
        state = load_state(output_dir / "checkpoint.json")
        proposals: list[Proposal] = []
        if state.archive is not None:
            try:
                archive = deserialize_archive(state.archive)
                proposals = [cell.proposal for cell in archive.cells.values()]
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(
                    f"Failed to deserialize archive for domain '{domain}' "
                    f"at '{output_dir / 'checkpoint.json'}'"
                ) from exc

        # Apply best proposals to get post-search KG
        augmented_kg = _apply_proposals_to_kg(kg, proposals)

        # Baselines on original KG
        random_hits = baseline_random(kg, seed=seed)
        freq_hits = baseline_frequency(kg, seed=seed)
        distmult_hits = baseline_distmult_alone(kg, seed=seed)

        # Harmony metrics on augmented KG
        harmony_hits = generativity(augmented_kg, seed=seed)
        harmony_mrr = _mean_reciprocal_rank(augmented_kg, seed=seed)

        # MRR for baseline comparisons
        mrr_rand = _mrr_random(kg, seed=seed)
        mrr_dm = _mean_reciprocal_rank(kg, seed=seed)

        rows[domain] = {
            "random_hits10": float(random_hits),
            "freq_hits10": float(freq_hits),
            "distmult_hits10": float(distmult_hits),
            "harmony_hits10": float(harmony_hits),
            "mrr_random": float(mrr_rand),
            "mrr_distmult": float(mrr_dm),
            "mrr_harmony": float(harmony_mrr),
        }

    df = pd.DataFrame.from_dict(rows, orient="index")
    df.index.name = "domain"
    return df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    import argparse

    known_domains = ["astronomy", "physics", "materials", "linear_algebra", "periodic_table"]
    parser = argparse.ArgumentParser(description="Compute Harmony metrics table")
    for domain in known_domains:
        parser.add_argument(f"--{domain}", type=Path, metavar="DIR", dest=domain)
    parser.add_argument("--output", type=Path, default=Path("metrics_table.csv"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    domain_checkpoints: dict[str, Path] = {}
    for domain in known_domains:
        d = getattr(args, domain, None)
        if d is not None:
            domain_checkpoints[domain] = d

    if not domain_checkpoints:
        parser.error("Specify at least one domain checkpoint directory.")

    df = compute_metrics_table(domain_checkpoints, seed=args.seed)
    df.to_csv(args.output)
    print(f"Metrics table written to {args.output}")
    print(df.to_string())


if __name__ == "__main__":
    main()
