"""Per-generation candidate generation, validation, scoring, and repair logic.

Extracted from cli.py to isolate the evolutionary search inner-loop.
"""

from __future__ import annotations

import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from .config import Phase1Config
from .evolution import migrate_ring_top1
from .llm_ollama import (
    IslandStrategy,
    build_prompt,
    generate_candidate_code,
    generate_candidate_payload,
)
from .logging_io import append_jsonl
from .map_elites import (
    MapElitesArchive,
    sample_diverse_exemplars,
    try_insert,
)
from .sandbox import SandboxEvaluator
from .scoring import (
    compute_bound_metrics,
    compute_metrics,
    compute_novelty_bonus,
    compute_simplicity_score,
    compute_total_score,
)
from .types import Candidate, CheckpointState

_CONSTRAINED_SUFFIX = (
    "\nConstrained mode is active. Use only these operators: +, -, *, /, log, sqrt, **, "
    "sum, mean, max, min. Follow this template: "
    "def new_invariant(s): n = s['n']; m = s['m']; "
    "degrees = s['degrees']; return <expression using n,m,degrees>."
)


def _new_experiment_id() -> str:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"phase1_{stamp}"


def _restore_rng(state: CheckpointState) -> np.random.Generator:
    rng = np.random.default_rng(state.rng_seed)
    if state.rng_state is not None:
        rng.bit_generator.state = state.rng_state
    return rng


def _state_defaults(state: CheckpointState) -> None:
    island_ids = sorted(state.islands.keys())
    for island_id in island_ids:
        state.island_stagnation.setdefault(island_id, 0)
        state.island_prompt_mode.setdefault(island_id, "free")
        state.island_constrained_generations.setdefault(island_id, 0)
        state.island_recent_failures.setdefault(island_id, [])


_STRATEGY_CYCLE = [
    IslandStrategy.REFINEMENT,
    IslandStrategy.COMBINATION,
    IslandStrategy.REFINEMENT,
    IslandStrategy.NOVEL,
]


def _island_strategy(island_id: int) -> IslandStrategy:
    """Return the strategy for a given island, cycling through the pattern."""
    return _STRATEGY_CYCLE[island_id % len(_STRATEGY_CYCLE)]


def _candidate_prompt(
    state: CheckpointState,
    island_id: int,
    target_name: str,
    fitness_mode: str = "correlation",
    enable_spectral_feature_pack: bool = True,
    archive_exemplars: list[str] | None = None,
) -> str:
    """Build the LLM prompt for a candidate, applying the island's strategy."""
    top_candidates = [
        candidate.code
        for candidate in sorted(
            state.islands.get(island_id, []), key=lambda c: c.val_score, reverse=True
        )
    ]
    strategy = _island_strategy(island_id)
    if archive_exemplars and strategy in (IslandStrategy.COMBINATION, IslandStrategy.NOVEL):
        top_candidates = top_candidates + archive_exemplars
    prompt = build_prompt(
        island_mode=f"island_{island_id}_{state.island_prompt_mode.get(island_id, 'free')}",
        top_candidates=top_candidates,
        failures=state.island_recent_failures.get(island_id, []),
        target_name=target_name,
        strategy=strategy,
        fitness_mode=fitness_mode,
        include_spectral_feature_pack=enable_spectral_feature_pack,
    )
    if state.island_prompt_mode.get(island_id, "free") == "constrained":
        return prompt + _CONSTRAINED_SUFFIX
    return prompt


def _record_recent_failure(
    state: CheckpointState,
    island_id: int,
    failure_text: str,
    max_items: int,
) -> None:
    items = state.island_recent_failures.setdefault(island_id, [])
    if failure_text in items:
        items.remove(failure_text)
    items.append(failure_text)
    if len(items) > max_items:
        del items[:-max_items]


def _summarize_error_details(details: list[dict[str, Any]]) -> str:
    if not details:
        return "unknown"
    categories: dict[str, int] = {}
    for payload in details:
        error_type = payload.get("error_type")
        if not isinstance(error_type, str) or not error_type:
            continue
        categories[error_type] = categories.get(error_type, 0) + 1
    if not categories:
        return "unknown"
    top_category = max(categories.items(), key=lambda item: item[1])[0]
    for payload in details:
        if payload.get("error_type") != top_category:
            continue
        raw_detail = payload.get("error_detail")
        if raw_detail is not None:
            return f"{top_category}: {raw_detail}"
    return top_category


def _build_repair_prompt(
    original_prompt: str,
    candidate_code: str,
    failure_feedback: str,
) -> str:
    return (
        f"{original_prompt}\n\n"
        "Repair this candidate. It failed evaluation.\n"
        f"Failure: {failure_feedback}\n"
        "Previous candidate code:\n"
        f"```python\n{candidate_code}\n```\n"
        "Return only corrected python code defining `def new_invariant(s):`."
    )


def _generate_candidate(
    cfg: Phase1Config,
    prompt: str,
    temperature: float,
) -> tuple[str, str | None]:
    if cfg.persist_prompt_and_response_logs:
        payload = generate_candidate_payload(
            prompt=prompt,
            model=cfg.model_name,
            temperature=temperature,
            url=cfg.ollama_url,
            allow_remote=cfg.allow_remote_ollama,
            timeout_sec=cfg.llm_timeout_sec,
        )
        return payload["code"], payload["response"]
    code = generate_candidate_code(
        prompt=prompt,
        model=cfg.model_name,
        temperature=temperature,
        url=cfg.ollama_url,
        allow_remote=cfg.allow_remote_ollama,
        timeout_sec=cfg.llm_timeout_sec,
    )
    return code, None


def _validate_experiment_id(value: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9._-]+", value):
        raise ValueError("experiment_id must match [A-Za-z0-9._-]+")
    return value


def _checkpoint_dir_for_experiment(artifacts_dir: str | Path, experiment_id: str) -> Path:
    safe_experiment_id = _validate_experiment_id(experiment_id)
    root = Path(artifacts_dir) / "checkpoints"
    checkpoint_dir = root / safe_experiment_id
    resolved_root = root.resolve()
    resolved_target = checkpoint_dir.resolve()
    if resolved_root not in resolved_target.parents and resolved_target != resolved_root:
        raise ValueError("checkpoint path escapes artifacts_dir/checkpoints")
    return checkpoint_dir


def _update_prompt_mode_after_generation(
    cfg: Phase1Config,
    state: CheckpointState,
    island_id: int,
    had_valid_train_candidate: bool,
) -> None:
    mode = state.island_prompt_mode.get(island_id, "free")
    if not cfg.enable_constrained_fallback:
        return

    if had_valid_train_candidate:
        state.island_stagnation[island_id] = 0
        if mode == "constrained":
            constrained_span = state.island_constrained_generations.get(island_id, 0)
            if (
                constrained_span <= cfg.constrained_recovery_generations
                or cfg.allow_late_constrained_recovery
            ):
                state.island_prompt_mode[island_id] = "free"
                state.island_constrained_generations[island_id] = 0
        return

    state.island_stagnation[island_id] = state.island_stagnation.get(island_id, 0) + 1
    if mode == "free" and state.island_stagnation[island_id] >= cfg.stagnation_trigger_generations:
        state.island_prompt_mode[island_id] = "constrained"
        state.island_constrained_generations[island_id] = 0
        return

    if mode == "constrained":
        state.island_constrained_generations[island_id] = (
            state.island_constrained_generations.get(island_id, 0) + 1
        )


def _corr_abs(x: list[float], y: list[float]) -> float:
    if len(x) < 2 or len(y) < 2:
        return 0.0
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    if x_arr.shape != y_arr.shape:
        return 0.0
    corr = np.corrcoef(x_arr, y_arr)[0, 1]
    if np.isnan(corr) or np.isinf(corr):
        return 0.0
    return float(abs(corr))


def _topology_descriptor(
    y_pred_valid: list[float],
    features_val: list[dict[str, Any]],
    valid_indices: tuple[int, ...],
) -> tuple[float, float]:
    density = [float(features_val[idx].get("density", 0.0)) for idx in valid_indices]
    clustering = [float(features_val[idx].get("avg_clustering", 0.0)) for idx in valid_indices]
    transitivity = [float(features_val[idx].get("transitivity", 0.0)) for idx in valid_indices]
    axis_density = _corr_abs(y_pred_valid, density)
    axis_cluster = (
        _corr_abs(y_pred_valid, clustering) + _corr_abs(y_pred_valid, transitivity)
    ) / 2.0
    return max(0.0, min(1.0, axis_density)), max(0.0, min(1.0, axis_cluster))


def _run_one_generation(
    cfg: Phase1Config,
    state: CheckpointState,
    evaluator: SandboxEvaluator,
    features_train: list[dict[str, Any]],
    features_val: list[dict[str, Any]],
    y_true_train: list[float],
    y_true_val: list[float],
    known_invariants_val: dict[str, list[float]],
    rng: np.random.Generator,
    log_path: str | Path,
    self_correction_stats: dict[str, Any],
    primary_archive: MapElitesArchive | None = None,
    topology_archive: MapElitesArchive | None = None,
) -> None:
    failure_categories = self_correction_stats.setdefault("failure_categories", {})

    def _count_failure(reason: str) -> None:
        failure_categories[reason] = int(failure_categories.get(reason, 0)) + 1

    def _handle_rejection(
        *,
        island_id: int,
        pop_idx: int,
        attempt_idx: int,
        rejection_reason: str,
        failure_feedback: str | None,
        base_prompt: str,
        code: str,
        max_attempts: int,
        repairable: bool,
        train_signal: float | None = None,
    ) -> tuple[bool, str | None]:
        _count_failure(rejection_reason)
        _record_recent_failure(
            state,
            island_id=island_id,
            failure_text=f"{rejection_reason}: {failure_feedback}",
            max_items=cfg.self_correction_feedback_window,
        )
        rejected_payload: dict[str, Any] = {
            "generation": state.generation,
            "island_id": island_id,
            "population_idx": pop_idx,
            "attempt_index": attempt_idx,
            "reason": rejection_reason,
            "failure_feedback": failure_feedback,
            "model_name": cfg.model_name,
        }
        if train_signal is not None:
            rejected_payload["train_signal"] = train_signal
        append_jsonl("candidate_rejected", rejected_payload, log_path)

        if repairable and attempt_idx + 1 < max_attempts:
            self_correction_stats["attempted_repairs"] = (
                int(self_correction_stats.get("attempted_repairs", 0)) + 1
            )
            append_jsonl(
                "candidate_repair_attempted",
                {
                    "generation": state.generation,
                    "island_id": island_id,
                    "population_idx": pop_idx,
                    "attempt_index": attempt_idx,
                    "reason": rejection_reason,
                    "failure_feedback": failure_feedback,
                    "model_name": cfg.model_name,
                },
                log_path,
            )
            next_prompt = _build_repair_prompt(
                original_prompt=base_prompt,
                candidate_code=code,
                failure_feedback=failure_feedback or rejection_reason,
            )
            return True, next_prompt

        if attempt_idx > 0:
            self_correction_stats["failed_repairs"] = (
                int(self_correction_stats.get("failed_repairs", 0)) + 1
            )
            append_jsonl(
                "candidate_repair_result",
                {
                    "generation": state.generation,
                    "island_id": island_id,
                    "population_idx": pop_idx,
                    "status": "failed",
                    "reason": rejection_reason,
                    "model_name": cfg.model_name,
                },
                log_path,
            )
        return False, None

    island_ids = sorted(state.islands.keys())
    for island_id in island_ids:
        archive_exemplar_codes: list[str] | None = None
        exemplar_pool: list[Candidate] = []
        if primary_archive is not None:
            exemplar_pool.extend(
                sample_diverse_exemplars(primary_archive, rng, count=2, exclude_island=island_id)
            )
        if topology_archive is not None:
            exemplar_pool.extend(
                sample_diverse_exemplars(topology_archive, rng, count=2, exclude_island=island_id)
            )
        if exemplar_pool:
            seen_ids: set[str] = set()
            deduped: list[Candidate] = []
            for exemplar in exemplar_pool:
                if exemplar.id in seen_ids:
                    continue
                seen_ids.add(exemplar.id)
                deduped.append(exemplar)
            archive_exemplar_codes = [candidate.code for candidate in deduped]
        new_candidates: list[Candidate] = []
        had_valid_train_candidate = False
        for pop_idx in range(cfg.population_size):
            base_prompt = _candidate_prompt(
                state,
                island_id,
                cfg.target_name,
                fitness_mode=cfg.fitness_mode,
                enable_spectral_feature_pack=cfg.enable_spectral_feature_pack,
                archive_exemplars=archive_exemplar_codes,
            )
            current_prompt = base_prompt
            max_attempts = 1 + (
                cfg.self_correction_max_retries if cfg.enable_self_correction else 0
            )
            for attempt_idx in range(max_attempts):
                code, llm_response = _generate_candidate(
                    cfg=cfg,
                    prompt=current_prompt,
                    temperature=cfg.island_temperatures[island_id % len(cfg.island_temperatures)],
                )
                train_details = evaluator.evaluate_detailed(code, features_train)
                y_pred_train_raw = [payload.get("value") for payload in train_details]
                train_pairs = [
                    (idx, yt, yp)
                    for idx, (yt, yp) in enumerate(zip(y_true_train, y_pred_train_raw, strict=True))
                    if isinstance(yp, float)
                ]
                rejection_reason: str | None = None
                failure_feedback: str | None = None
                train_signal: float | None = None

                is_bounds = cfg.fitness_mode in ("upper_bound", "lower_bound")

                if not train_pairs:
                    rejection_reason = "no_valid_train_predictions"
                    failure_feedback = _summarize_error_details(train_details)
                else:
                    _, y_t_train, y_p_train = zip(*train_pairs, strict=True)
                    if is_bounds:
                        train_bound = compute_bound_metrics(
                            list(y_t_train),
                            list(y_p_train),
                            mode=cfg.fitness_mode,
                            tolerance=cfg.bound_tolerance,
                        )
                        train_signal = train_bound.bound_score
                    else:
                        train_metrics = compute_metrics(list(y_t_train), list(y_p_train))
                        train_signal = abs(train_metrics.rho_spearman)
                    if train_signal <= cfg.train_score_threshold:
                        rejection_reason = "below_train_threshold"
                        failure_feedback = f"train_signal={train_signal:.6f}"

                if rejection_reason is not None:
                    should_retry, next_prompt = _handle_rejection(
                        island_id=island_id,
                        pop_idx=pop_idx,
                        attempt_idx=attempt_idx,
                        rejection_reason=rejection_reason,
                        failure_feedback=failure_feedback,
                        base_prompt=base_prompt,
                        code=code,
                        max_attempts=max_attempts,
                        repairable=rejection_reason == "no_valid_train_predictions",
                        train_signal=train_signal,
                    )
                    if should_retry and next_prompt is not None:
                        current_prompt = next_prompt
                        continue
                    break

                had_valid_train_candidate = True
                val_details = evaluator.evaluate_detailed(code, features_val)
                y_pred_val_raw = [payload.get("value") for payload in val_details]
                val_pairs = [
                    (idx, yt, yp)
                    for idx, (yt, yp) in enumerate(zip(y_true_val, y_pred_val_raw, strict=True))
                    if isinstance(yp, float)
                ]
                if not val_pairs:
                    rejection_reason = "no_valid_val_predictions"
                    failure_feedback = _summarize_error_details(val_details)
                    should_retry, next_prompt = _handle_rejection(
                        island_id=island_id,
                        pop_idx=pop_idx,
                        attempt_idx=attempt_idx,
                        rejection_reason=rejection_reason,
                        failure_feedback=failure_feedback,
                        base_prompt=base_prompt,
                        code=code,
                        max_attempts=max_attempts,
                        repairable=True,
                    )
                    if should_retry and next_prompt is not None:
                        current_prompt = next_prompt
                        continue
                    break

                valid_indices, y_t_val, y_p_val = zip(*val_pairs, strict=True)
                simplicity = compute_simplicity_score(code)
                known_subset = {
                    name: [values[idx] for idx in valid_indices]
                    for name, values in known_invariants_val.items()
                }
                novelty_bonus = compute_novelty_bonus(list(y_p_val), known_subset)
                if cfg.novelty_gate_threshold > 0 and novelty_bonus < cfg.novelty_gate_threshold:
                    should_retry, next_prompt = _handle_rejection(
                        island_id=island_id,
                        pop_idx=pop_idx,
                        attempt_idx=attempt_idx,
                        rejection_reason="below_novelty_threshold",
                        failure_feedback=f"novelty_bonus={novelty_bonus:.6f}",
                        base_prompt=base_prompt,
                        code=code,
                        max_attempts=max_attempts,
                        repairable=True,
                    )
                    if should_retry and next_prompt is not None:
                        current_prompt = next_prompt
                        continue
                    break
                if is_bounds:
                    val_bound = compute_bound_metrics(
                        list(y_t_val),
                        list(y_p_val),
                        mode=cfg.fitness_mode,
                        tolerance=cfg.bound_tolerance,
                    )
                    fitness_signal = val_bound.bound_score
                else:
                    val_metrics = compute_metrics(list(y_t_val), list(y_p_val))
                    fitness_signal = abs(val_metrics.rho_spearman)
                total = compute_total_score(
                    fitness_signal,
                    simplicity,
                    novelty_bonus=novelty_bonus,
                    alpha=cfg.alpha,
                    beta=cfg.beta,
                    gamma=cfg.gamma,
                )
                candidate = Candidate(
                    id=f"g{state.generation}_i{island_id}_p{pop_idx}_{int(rng.integers(1_000_000_000))}",
                    code=code,
                    island_id=island_id,
                    generation=state.generation,
                    train_score=float(train_signal or 0.0),
                    val_score=total,
                    simplicity_score=simplicity,
                    novelty_bonus=novelty_bonus,
                )
                new_candidates.append(candidate)
                if primary_archive is not None:
                    try_insert(primary_archive, candidate, fitness_signal)
                if topology_archive is not None:
                    descriptor = _topology_descriptor(
                        y_pred_valid=list(y_p_val),
                        features_val=features_val,
                        valid_indices=valid_indices,
                    )
                    try_insert(
                        topology_archive,
                        candidate,
                        fitness_signal,
                        descriptor=descriptor,
                    )
                log_payload: dict[str, Any] = {
                    "candidate_id": candidate.id,
                    "generation": state.generation,
                    "island_id": island_id,
                    "model_name": cfg.model_name,
                    "train_signal": train_signal,
                    "fitness_mode": cfg.fitness_mode,
                    "simplicity_score": simplicity,
                    "novelty_bonus": novelty_bonus,
                    "total_score": total,
                    "prompt": current_prompt if cfg.persist_prompt_and_response_logs else None,
                    "llm_response": llm_response if cfg.persist_prompt_and_response_logs else None,
                    "extracted_code": code if cfg.persist_prompt_and_response_logs else None,
                }
                if is_bounds:
                    log_payload["bound_score"] = val_bound.bound_score
                    log_payload["satisfaction_rate"] = val_bound.satisfaction_rate
                    log_payload["mean_gap"] = val_bound.mean_gap
                else:
                    log_payload["spearman"] = val_metrics.rho_spearman
                    log_payload["pearson"] = val_metrics.r_pearson
                    log_payload["rmse"] = val_metrics.rmse
                    log_payload["mae"] = val_metrics.mae
                append_jsonl("candidate_evaluated", log_payload, log_path)
                if attempt_idx > 0:
                    self_correction_stats["successful_repairs"] = (
                        int(self_correction_stats.get("successful_repairs", 0)) + 1
                    )
                    append_jsonl(
                        "candidate_repair_result",
                        {
                            "generation": state.generation,
                            "island_id": island_id,
                            "population_idx": pop_idx,
                            "status": "success",
                            "model_name": cfg.model_name,
                        },
                        log_path,
                    )
                break

        merged = state.islands.get(island_id, []) + new_candidates
        merged.sort(key=lambda candidate: candidate.val_score, reverse=True)
        state.islands[island_id] = merged[: cfg.population_size]
        _update_prompt_mode_after_generation(cfg, state, island_id, had_valid_train_candidate)

    if cfg.migration_interval > 0 and (state.generation + 1) % cfg.migration_interval == 0:
        migrated = migrate_ring_top1(state)
        state.islands = migrated.islands
        append_jsonl(
            "generation_migration",
            {
                "generation": state.generation,
                "model_name": cfg.model_name,
            },
            log_path,
        )
