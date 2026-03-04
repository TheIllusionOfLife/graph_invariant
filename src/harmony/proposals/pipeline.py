"""Deterministic proposal pipeline: validate → mutate → score → archive."""

from __future__ import annotations

import logging
from dataclasses import dataclass

from harmony.map_elites import HarmonyMapElites, try_insert
from harmony.metric.harmony import value_of
from harmony.proposals.types import Proposal, ProposalType, ValidationResult
from harmony.proposals.validator import validate
from harmony.types import EdgeType, Entity, KnowledgeGraph, TypedEdge

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class ProposalResult:
    """Outcome of running a single proposal through the pipeline."""

    proposal: Proposal
    validation: ValidationResult
    harmony_gain: float | None  # None when validation failed
    inserted_to_archive: bool


@dataclass(slots=True)
class PipelineResult:
    """Aggregate result of running a batch of proposals."""

    results: list[ProposalResult]
    valid_rate: float  # fraction of proposals that passed validation
    archive: HarmonyMapElites


def _apply_mutation(kg: KnowledgeGraph, proposal: Proposal) -> KnowledgeGraph:
    """Return a copy of kg with the proposal mutation applied."""
    kg_copy = KnowledgeGraph(
        domain=kg.domain,
        entities=dict(kg.entities),
        edges=list(kg.edges),
    )

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


def _grounding_violations(proposal: Proposal, kg: KnowledgeGraph) -> list[str]:
    """Return KG-grounding validation violations for a schema-valid proposal."""
    violations: list[str] = []
    if proposal.kg_domain != kg.domain:
        violations.append(
            f"'kg_domain' must match current KG domain '{kg.domain}', got '{proposal.kg_domain}'"
        )

    if proposal.proposal_type in (ProposalType.ADD_EDGE, ProposalType.REMOVE_EDGE):
        if proposal.source_entity not in kg.entities:
            violations.append(
                f"unknown source_entity '{proposal.source_entity}' for domain '{kg.domain}'"
            )
        if proposal.target_entity not in kg.entities:
            violations.append(
                f"unknown target_entity '{proposal.target_entity}' for domain '{kg.domain}'"
            )
    elif proposal.proposal_type == ProposalType.ADD_ENTITY:
        if proposal.entity_id in kg.entities:
            violations.append(
                f"entity_id '{proposal.entity_id}' already exists in domain '{kg.domain}'"
            )
    elif proposal.proposal_type == ProposalType.REMOVE_ENTITY:
        if proposal.entity_id not in kg.entities:
            violations.append(f"unknown entity_id '{proposal.entity_id}' for domain '{kg.domain}'")
    return violations


def _proposal_cost(kg: KnowledgeGraph, proposal: Proposal, cost_mode: str) -> float:
    if cost_mode == "none":
        return 0.0
    if cost_mode != "normalized_mutation_size":
        raise ValueError("cost_mode must be 'normalized_mutation_size' or 'none'")
    if proposal.proposal_type in (ProposalType.ADD_EDGE, ProposalType.REMOVE_EDGE):
        return 1.0 / max(1, kg.num_edges)
    return 1.0 / max(1, kg.num_entities)


def run_pipeline(
    kg: KnowledgeGraph,
    proposals: list[Proposal],
    seed: int = 42,
    archive_bins: int = 5,
    accept_all_valid: bool = False,
    archive: HarmonyMapElites | None = None,
    lambda_cost: float = 0.1,
    cost_mode: str = "normalized_mutation_size",
    alpha: float = 0.25,
    beta: float = 0.25,
    gamma: float = 0.25,
    delta: float = 0.25,
    epsilon: float = 0.0,
) -> PipelineResult:
    """Validate → apply mutation → score → archive each proposal.

    harmony_gain = value_of(kg_before, kg_after)
    simplicity   = 1 / (1 + len(claim) + len(justification))  ∈ (0, 1]

    Descriptor (simplicity, gain_norm) is inserted into the MAP-Elites archive,
    where gain_norm is min-max normalised across valid proposals in this batch.

    When accept_all_valid=True (LLM-only baseline), harmony_score is skipped
    and all valid proposals receive gain=1.0.

    Logs valid_rate on every call.
    """
    if not proposals:
        logger.info("Pipeline valid_rate=0.000 (0/0)")
        current_archive = (
            archive if archive is not None else HarmonyMapElites(num_bins=archive_bins)
        )
        return PipelineResult(results=[], valid_rate=0.0, archive=current_archive)

    archive_obj = archive if archive is not None else HarmonyMapElites(num_bins=archive_bins)

    # First pass: validate and compute gains
    results: list[ProposalResult] = []
    valid_count = 0

    for proposal in proposals:
        schema_validation = validate(proposal)
        if not schema_validation.is_valid:
            results.append(
                ProposalResult(
                    proposal=proposal,
                    validation=schema_validation,
                    harmony_gain=None,
                    inserted_to_archive=False,
                )
            )
            continue

        grounding_violations = _grounding_violations(proposal, kg)
        if grounding_violations:
            results.append(
                ProposalResult(
                    proposal=proposal,
                    validation=ValidationResult(is_valid=False, violations=grounding_violations),
                    harmony_gain=None,
                    inserted_to_archive=False,
                )
            )
            continue

        valid_count += 1

        if accept_all_valid:
            gain: float | None = 1.0
        else:
            try:
                kg_after = _apply_mutation(kg, proposal)
                gain = value_of(
                    kg_before=kg,
                    kg_after=kg_after,
                    alpha=alpha,
                    beta=beta,
                    gamma=gamma,
                    delta=delta,
                    epsilon=epsilon,
                    lambda_cost=lambda_cost,
                    cost=_proposal_cost(kg, proposal, cost_mode),
                    seed=seed,
                )
            except Exception as e:
                logger.warning("Error processing valid proposal %s: %s", proposal.id, e)
                gain = None

        results.append(
            ProposalResult(
                proposal=proposal,
                validation=ValidationResult(is_valid=True, violations=[]),
                harmony_gain=gain,
                inserted_to_archive=False,
            )
        )

    valid_rate = valid_count / len(proposals)
    logger.info("Pipeline valid_rate=%.3f (%d/%d)", valid_rate, valid_count, len(proposals))

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
                archive_obj,
                result.proposal,
                fitness_signal=result.harmony_gain,
                descriptor=(simplicity, gain_norm),
            )

    return PipelineResult(results=results, valid_rate=valid_rate, archive=archive_obj)
