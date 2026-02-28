"""Classify proposals as 'rediscovery' (existing edge) or 'novel' (new hypothesis).

Addresses reviewer I9: known/novel split for evaluating discovery quality.
"""

from __future__ import annotations

from harmony.proposals.types import Proposal
from harmony.types import KnowledgeGraph


def classify_proposal(proposal: Proposal, kg: KnowledgeGraph) -> str:
    """Classify a single proposal against a KG.

    Returns "rediscovery" if the exact (source, target, edge_type) triple
    already exists in the KG, otherwise "novel".
    """
    for edge in kg.edges:
        if (
            edge.source == proposal.source_entity
            and edge.target == proposal.target_entity
            and edge.edge_type.name == proposal.edge_type
        ):
            return "rediscovery"
    return "novel"


def classify_batch(
    proposals: list[Proposal],
    kg: KnowledgeGraph,
) -> dict[str, int]:
    """Classify a batch of proposals and return counts.

    Returns dict with keys: "rediscovery", "novel", "total".
    """
    counts = {"rediscovery": 0, "novel": 0, "total": len(proposals)}
    for p in proposals:
        label = classify_proposal(p, kg)
        counts[label] += 1
    return counts
