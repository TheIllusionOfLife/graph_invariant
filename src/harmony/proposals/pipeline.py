"""Deterministic proposal pipeline: validate → mutate → score → archive."""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass

from harmony.map_elites import HarmonyMapElites, try_insert
from harmony.metric.harmony import harmony_score
from harmony.proposals.types import Proposal, ProposalType, ValidationResult
from harmony.proposals.validator import validate
from harmony.types import EdgeType, Entity, KnowledgeGraph, TypedEdge

logger = logging.getLogger(__name__)


@dataclass
class ProposalResult:
    """Outcome of running a single proposal through the pipeline."""

    proposal: Proposal
    validation: ValidationResult
    harmony_gain: float | None  # None when validation failed
    inserted_to_archive: bool


@dataclass
class PipelineResult:
    """Aggregate result of running a batch of proposals."""

    results: list[ProposalResult]
    valid_rate: float  # fraction of proposals that passed validation
    archive: HarmonyMapElites


def _apply_mutation(kg: KnowledgeGraph, proposal: Proposal) -> KnowledgeGraph:
    """Return a deep copy of kg with the proposal mutation applied."""
    kg_copy = copy.deepcopy(kg)

    if proposal.proposal_type == ProposalType.ADD_EDGE:
        edge = TypedEdge(
            source=proposal.source_entity,
            target=proposal.target_entity,
            edge_type=EdgeType[proposal.edge_type],
        )
        kg_copy.add_edge(edge)

    elif proposal.proposal_type == ProposalType.REMOVE_EDGE:
        kg_copy.edges = [
            e
            for e in kg_copy.edges
            if not (
                e.source == proposal.source_entity
                and e.target == proposal.target_entity
                and e.edge_type.name == proposal.edge_type
            )
        ]

    elif proposal.proposal_type == ProposalType.ADD_ENTITY:
        kg_copy.add_entity(Entity(id=proposal.entity_id, entity_type=proposal.entity_type))

    elif proposal.proposal_type == ProposalType.REMOVE_ENTITY:
        kg_copy.entities.pop(proposal.entity_id, None)
        kg_copy.edges = [
            e
            for e in kg_copy.edges
            if e.source != proposal.entity_id and e.target != proposal.entity_id
        ]

    return kg_copy


def run_pipeline(
    kg: KnowledgeGraph,
    proposals: list[Proposal],
    seed: int = 42,
    archive_bins: int = 5,
) -> PipelineResult:
    """Validate → apply mutation → score → archive each proposal.

    harmony_gain = harmony_score(kg_after) − harmony_score(kg_before)
    simplicity   = 1 / (1 + len(claim) + len(justification))  ∈ (0, 1]

    Descriptor (simplicity, gain_norm) is inserted into the MAP-Elites archive,
    where gain_norm is min-max normalised across valid proposals in this batch.

    Logs valid_rate on every call.
    """
    if not proposals:
        logger.info("Pipeline valid_rate=0.000 (0/0)")
        print("Pipeline valid_rate=0.000 (0/0)")
        empty_archive = HarmonyMapElites(num_bins=archive_bins)
        return PipelineResult(results=[], valid_rate=0.0, archive=empty_archive)

    archive = HarmonyMapElites(num_bins=archive_bins)
    h_before = harmony_score(kg, seed=seed)

    # First pass: validate and compute gains
    results: list[ProposalResult] = []
    valid_count = 0

    for proposal in proposals:
        validation = validate(proposal)
        if not validation.is_valid:
            results.append(
                ProposalResult(
                    proposal=proposal,
                    validation=validation,
                    harmony_gain=None,
                    inserted_to_archive=False,
                )
            )
            continue

        valid_count += 1
        try:
            kg_after = _apply_mutation(kg, proposal)
            gain = harmony_score(kg_after, seed=seed) - h_before
        except Exception:
            gain = 0.0

        results.append(
            ProposalResult(
                proposal=proposal,
                validation=validation,
                harmony_gain=gain,
                inserted_to_archive=False,
            )
        )

    valid_rate = valid_count / len(proposals)
    logger.info("Pipeline valid_rate=%.3f (%d/%d)", valid_rate, valid_count, len(proposals))
    print(f"Pipeline valid_rate={valid_rate:.3f} ({valid_count}/{len(proposals)})")

    # Second pass: normalize gains and insert into archive
    gains = [r.harmony_gain for r in results if r.harmony_gain is not None]
    if gains:
        gain_min = min(gains)
        gain_max = max(gains)
        gain_range = gain_max - gain_min if gain_max != gain_min else 1.0

        for result in results:
            if result.harmony_gain is None:
                continue
            claim_len = len(result.proposal.claim) + len(result.proposal.justification)
            simplicity = 1.0 / (1.0 + claim_len)
            gain_norm = (result.harmony_gain - gain_min) / gain_range
            result.inserted_to_archive = try_insert(
                archive,
                result.proposal,
                fitness_signal=result.harmony_gain,
                descriptor=(simplicity, gain_norm),
            )

    return PipelineResult(results=results, valid_rate=valid_rate, archive=archive)
