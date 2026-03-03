"""Random proposer — deterministic KG mutations without LLM.

Generates ADD_EDGE proposals by randomly sampling source/target entities
and edge types. Used as the Harmony-only baseline (no LLM) in factor
decomposition experiments.
"""

from __future__ import annotations

import numpy as np

from harmony.proposals.types import Proposal, ProposalType
from harmony.types import EdgeType, KnowledgeGraph

_EDGE_TYPE_NAMES: list[str] = [et.name for et in EdgeType]


def generate_random_proposals(
    kg: KnowledgeGraph,
    n: int,
    seed: int = 42,
) -> list[Proposal]:
    """Generate n random ADD_EDGE proposals for the given KG.

    Parameters
    ----------
    kg:
        Knowledge graph providing entity pool and edge types.
    n:
        Number of proposals to generate.
    seed:
        Random seed for deterministic generation.

    Returns
    -------
    List of n Proposal objects with random source/target/edge_type.
    """
    entity_ids = list(kg.entities.keys())
    if len(entity_ids) < 2:
        return []

    rng = np.random.default_rng(seed)
    proposals: list[Proposal] = []

    for i in range(n):
        src_idx, tgt_idx = rng.choice(len(entity_ids), size=2, replace=False)
        edge_type_idx = rng.integers(0, len(_EDGE_TYPE_NAMES))

        proposals.append(
            Proposal(
                id=f"rand-{seed}-{i:06d}",
                proposal_type=ProposalType.ADD_EDGE,
                claim="Randomly generated edge for ablation baseline.",
                justification="Random mutation — no LLM reasoning involved.",
                falsification_condition="N/A — random baseline proposal.",
                kg_domain=kg.domain,
                source_entity=entity_ids[int(src_idx)],
                target_entity=entity_ids[int(tgt_idx)],
                edge_type=_EDGE_TYPE_NAMES[int(edge_type_idx)],
            )
        )

    return proposals
