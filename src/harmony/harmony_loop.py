"""Harmony island search loop — outer generation orchestration.

Wires together:
  - LLM proposal generation (llm_proposer.py)
  - Schema validation (proposals/validator.py)
  - Scoring and archiving (proposals/pipeline.py + map_elites.py)
  - State checkpointing (state.py)

Mirrors the structure of graph_invariant/phase1_loop.py for the KG-theory domain.
"""

from __future__ import annotations

import json
import logging
import uuid
from pathlib import Path
from typing import Any

from harmony.config import HarmonyConfig
from harmony.map_elites import deserialize_archive, sample_diverse_exemplars, serialize_archive
from harmony.proposals.llm_proposer import (
    build_proposal_prompt,
    generate_proposal_payload,  # imported here so tests can patch at harmony.harmony_loop
    island_strategy,
)
from harmony.proposals.pipeline import run_pipeline
from harmony.proposals.types import Proposal
from harmony.proposals.validator import validate
from harmony.state import HarmonySearchState, load_state, save_state
from harmony.types import KnowledgeGraph

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _new_experiment_id() -> str:
    return f"harmony-{uuid.uuid4().hex[:8]}"


def _append_jsonl(payload: dict[str, Any], path: Path) -> None:
    """Append one JSON line to a JSONL log file (creates parent dirs on first call)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload) + "\n")


def _init_island_state(state: HarmonySearchState, n_islands: int) -> None:
    """Ensure all scheduler state dicts have an entry for every island."""
    for i in range(n_islands):
        state.island_prompt_mode.setdefault(i, "free")
        state.island_stagnation.setdefault(i, 0)
        state.island_recent_failures.setdefault(i, [])
        state.island_constrained_generations.setdefault(i, 0)
        if i not in state.islands:
            state.islands[i] = []


# ---------------------------------------------------------------------------
# Island generation
# ---------------------------------------------------------------------------


def _run_island_generation(
    cfg: HarmonyConfig,
    state: HarmonySearchState,
    kg: KnowledgeGraph,
    island_id: int,
    archive_exemplars: list[Proposal],
) -> tuple[list[Proposal], int]:
    """Generate *cfg.population_size* proposals for one island.

    Returns
    -------
    (new_proposals, attempts)
        new_proposals: list of valid Proposal objects collected this generation.
        attempts: total LLM call attempts made (for valid_rate computation).
    """
    strategy = island_strategy(island_id)
    constrained = state.island_prompt_mode.get(island_id, "free") == "constrained"
    temp = cfg.island_temperatures[island_id % len(cfg.island_temperatures)]

    # Top proposals context (from previous best for this island)
    top_dicts = state.islands.get(island_id, [])[:3]
    top_json = [json.dumps(d) for d in top_dicts]

    # In constrained mode, seed from archive exemplars instead of island history
    if constrained and archive_exemplars:
        top_json = [json.dumps(p.to_dict()) for p in archive_exemplars[:3]]

    failures = list(state.island_recent_failures.get(island_id, []))
    new_proposals: list[Proposal] = []
    attempts = 0

    for _ in range(cfg.population_size):
        attempts += 1
        prompt = build_proposal_prompt(
            kg=kg,
            strategy=strategy,
            top_proposals=top_json,
            recent_failures=failures,
            constrained=constrained,
        )

        try:
            payload = generate_proposal_payload(
                prompt=prompt,
                model=cfg.model_name,
                temperature=temp,
                url=cfg.ollama_url,
                allow_remote=cfg.allow_remote_ollama,
                timeout_sec=cfg.llm_timeout_sec,
            )
        except Exception as exc:
            logger.debug("Island %d LLM call failed: %s", island_id, exc)
            continue

        proposal_dict = payload.get("proposal_dict")
        if proposal_dict is None:
            failures.append("JSON extraction failed — LLM did not return valid JSON object")
            continue

        # Attempt deserialization + validation
        try:
            proposal = Proposal.from_dict(proposal_dict)
        except (KeyError, ValueError, TypeError) as exc:
            failures.append(f"Proposal deserialization error: {exc}")
            continue

        validation = validate(proposal)
        if validation.is_valid:
            new_proposals.append(proposal)
        else:
            for v in validation.violations:
                failures.append(v)

            # Self-correction: retry with violations fed back into prompt
            if cfg.enable_self_correction:
                for _ in range(cfg.self_correction_max_retries):
                    repair_prompt = build_proposal_prompt(
                        kg=kg,
                        strategy=strategy,
                        top_proposals=top_json,
                        recent_failures=validation.violations,
                        constrained=True,  # always constrained for repairs
                    )
                    try:
                        repair_payload = generate_proposal_payload(
                            prompt=repair_prompt,
                            model=cfg.model_name,
                            temperature=0.3,  # low temp for repair
                            url=cfg.ollama_url,
                            allow_remote=cfg.allow_remote_ollama,
                            timeout_sec=cfg.llm_timeout_sec,
                        )
                    except Exception:
                        break
                    rd = repair_payload.get("proposal_dict")
                    if rd is None:
                        break
                    try:
                        repaired = Proposal.from_dict(rd)
                        rv = validate(repaired)
                        if rv.is_valid:
                            new_proposals.append(repaired)
                            break
                    except (KeyError, ValueError, TypeError):
                        break

    # Keep failure list bounded to avoid bloating prompts
    state.island_recent_failures[island_id] = failures[-10:]

    return new_proposals, attempts


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def run_harmony_loop(
    cfg: HarmonyConfig,
    kg: KnowledgeGraph,
    output_dir: Path = Path("harmony_output"),
    resume: str | None = None,
) -> HarmonySearchState:
    """Run the Harmony island search loop.

    Parameters
    ----------
    cfg:
        Full experiment configuration.
    kg:
        Knowledge graph to run discovery on.
    output_dir:
        Directory for checkpoint and log files.
    resume:
        Path to a JSON checkpoint file to resume from. If None, starts fresh.

    Returns
    -------
    Final HarmonySearchState after the loop completes (or early-stops).
    """
    output_dir = Path(output_dir)
    log_path = output_dir / "logs" / "harmony_events.jsonl"
    n_islands = len(cfg.island_temperatures)

    # --- State initialization ---
    if resume:
        state = load_state(Path(resume))
    else:
        state = HarmonySearchState(
            experiment_id=_new_experiment_id(),
            generation=0,
            islands={i: [] for i in range(n_islands)},
            rng_seed=cfg.seed,
        )

    _init_island_state(state, n_islands)

    _append_jsonl(
        {
            "event": "harmony_loop_started",
            "experiment_id": state.experiment_id,
            "resume": resume,
            "model_name": cfg.model_name,
            "domain": kg.domain,
        },
        log_path,
    )

    # --- Main generation loop ---
    for _gen in range(state.generation, cfg.max_generations):
        # Reconstruct archive for sampling exemplars (may be None on gen 0)
        archive = None
        if state.archive is not None:
            try:
                archive = deserialize_archive(state.archive)
            except Exception:
                archive = None

        archive_exemplars: list[Proposal] = []
        if archive is not None:
            import numpy as np

            rng = np.random.default_rng(cfg.seed + state.generation)
            archive_exemplars = sample_diverse_exemplars(archive, rng, count=3)

        # --- Per-island generation ---
        all_new_proposals: list[Proposal] = []
        per_island_any_valid: dict[int, bool] = {}
        total_attempts = 0
        total_valid = 0

        for island_id in sorted(state.islands):
            new_proposals, attempts = _run_island_generation(
                cfg=cfg,
                state=state,
                kg=kg,
                island_id=island_id,
                archive_exemplars=archive_exemplars,
            )
            per_island_any_valid[island_id] = len(new_proposals) > 0
            total_attempts += attempts
            total_valid += len(new_proposals)
            all_new_proposals.extend(new_proposals)

        # --- Score and archive via pipeline ---
        pipeline_result = run_pipeline(
            kg=kg,
            proposals=all_new_proposals,
            seed=cfg.seed,
            archive_bins=cfg.map_elites_bins,
        )

        # --- Update island context with best proposals ---
        valid_results = sorted(
            (r for r in pipeline_result.results if r.harmony_gain is not None),
            key=lambda r: r.harmony_gain or 0.0,
            reverse=True,
        )
        if valid_results:
            # Distribute top proposals across islands for next-gen context
            for rank, result in enumerate(valid_results[:n_islands]):
                target_island = rank % n_islands
                existing = state.islands[target_island]
                state.islands[target_island] = [result.proposal.to_dict()] + existing[:4]

        # --- Update archive checkpoint ---
        state.archive = serialize_archive(pipeline_result.archive)

        # --- Stagnation / prompt-mode scheduler ---
        for island_id in sorted(state.islands):
            if per_island_any_valid.get(island_id, False):
                state.island_stagnation[island_id] = 0
                state.island_prompt_mode[island_id] = "free"
                state.island_constrained_generations[island_id] = 0
            else:
                state.island_stagnation[island_id] += 1
                if state.island_stagnation[island_id] >= cfg.stagnation_trigger_generations:
                    state.island_prompt_mode[island_id] = "constrained"
                    state.island_constrained_generations[island_id] += 1

        # --- Improvement tracking ---
        best_gain = max(
            (r.harmony_gain for r in pipeline_result.results if r.harmony_gain is not None),
            default=0.0,
        )
        if best_gain > state.best_harmony_gain + 1e-12:
            state.best_harmony_gain = best_gain
            state.no_improve_count = 0
        else:
            state.no_improve_count += 1

        state.generation += 1

        # --- Checkpoint ---
        save_state(state, output_dir / "checkpoint.json")

        # --- Logging ---
        gen_valid_rate = total_valid / total_attempts if total_attempts > 0 else 0.0
        _append_jsonl(
            {
                "event": "generation_summary",
                "experiment_id": state.experiment_id,
                "generation": state.generation,
                "valid_rate": gen_valid_rate,
                "pipeline_valid_rate": pipeline_result.valid_rate,
                "best_harmony_gain": state.best_harmony_gain,
                "no_improve_count": state.no_improve_count,
            },
            log_path,
        )
        logger.info(
            "gen=%d valid_rate=%.3f best_gain=%.4f",
            state.generation,
            gen_valid_rate,
            state.best_harmony_gain,
        )

        if state.no_improve_count >= cfg.early_stop_patience:
            logger.info("Early stopping: no improvement for %d generations", state.no_improve_count)
            break

    return state
