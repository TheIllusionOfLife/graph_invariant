import argparse
import json
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np

from .baselines import run_pysr_baseline, run_stat_baselines
from .config import Phase1Config
from .data import generate_phase1_datasets
from .evaluation import (
    dataset_fingerprint,
    global_best_score,
)
from .evolution import migrate_ring_top1
from .known_invariants import compute_feature_dicts, compute_known_invariant_values
from .llm_ollama import (
    IslandStrategy,
    build_prompt,
    generate_candidate_code,
    generate_candidate_payload,
    list_available_models,
)
from .logging_io import (
    append_jsonl,
    load_checkpoint,
    rotate_generation_checkpoints,
    save_checkpoint,
    write_json,
)
from .map_elites import (
    MapElitesArchive,
    archive_stats,
    deserialize_archive,
    sample_diverse_exemplars,
    serialize_archive,
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
from .summary import write_phase1_summary
from .targets import target_values
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


def _require_model_available(cfg: Phase1Config) -> None:
    available_models = list_available_models(cfg.ollama_url, allow_remote=cfg.allow_remote_ollama)
    if cfg.model_name not in available_models:
        if available_models:
            available = ", ".join(available_models)
            raise RuntimeError(
                f"ollama model '{cfg.model_name}' is not installed; available: {available}"
            )
        raise RuntimeError(
            f"ollama model '{cfg.model_name}' is not installed and no models were listed"
        )


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


def _checkpoint_dir_for_experiment(artifacts_dir: Path, experiment_id: str) -> Path:
    safe_experiment_id = _validate_experiment_id(experiment_id)
    root = artifacts_dir / "checkpoints"
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
    log_path: Path,
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
                    temperature=cfg.island_temperatures[island_id],
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




def _write_baseline_summary(
    payload: dict[str, object] | None,
    artifacts_dir: Path,
) -> None:
    if payload is None:
        return

    write_json(payload, artifacts_dir / "baselines_summary.json")


def _collect_baseline_results(
    cfg: Phase1Config,
    datasets_train: list[nx.Graph],
    datasets_val: list[nx.Graph],
    datasets_test: list[nx.Graph],
    y_true_train: list[float],
    y_true_val: list[float],
    y_true_test: list[float],
    enable_spectral_feature_pack: bool,
) -> dict[str, object] | None:
    if not cfg.run_baselines:
        return None

    stat = run_stat_baselines(
        train_graphs=datasets_train,
        val_graphs=datasets_val,
        test_graphs=datasets_test,
        y_train=y_true_train,
        y_val=y_true_val,
        y_test=y_true_test,
        target_name=cfg.target_name,
        enable_spectral_feature_pack=enable_spectral_feature_pack,
    )
    pysr = run_pysr_baseline(
        train_graphs=datasets_train,
        val_graphs=datasets_val,
        test_graphs=datasets_test,
        y_train=y_true_train,
        y_val=y_true_val,
        y_test=y_true_test,
        niterations=cfg.pysr_niterations,
        populations=cfg.pysr_populations,
        procs=cfg.pysr_procs,
        timeout_in_seconds=cfg.pysr_timeout_in_seconds,
        target_name=cfg.target_name,
        enable_spectral_feature_pack=enable_spectral_feature_pack,
    )
    return {
        "schema_version": 1,
        "target_name": cfg.target_name,
        "stat_baselines": stat,
        "pysr_baseline": pysr,
    }


def run_phase1(cfg: Phase1Config, resume: str | None = None) -> int:
    _require_model_available(cfg)
    datasets = generate_phase1_datasets(cfg)
    artifacts_dir = Path(cfg.artifacts_dir)
    log_path = artifacts_dir / "logs" / "events.jsonl"

    if resume:
        state = load_checkpoint(resume)
        resume_experiment_id = _validate_experiment_id(state.experiment_id)
        if cfg.experiment_id:
            resume_experiment_id = _validate_experiment_id(cfg.experiment_id)
            state.experiment_id = resume_experiment_id
        experiment_id = resume_experiment_id
    elif cfg.experiment_id:
        experiment_id = _validate_experiment_id(cfg.experiment_id)
        state = CheckpointState(
            experiment_id=experiment_id,
            generation=0,
            islands={i: [] for i in range(len(cfg.island_temperatures))},
            rng_seed=cfg.seed,
            rng_state=None,
            best_val_score=0.0,
            no_improve_count=0,
        )
    else:
        experiment_id = _new_experiment_id()
        state = CheckpointState(
            experiment_id=experiment_id,
            generation=0,
            islands={i: [] for i in range(len(cfg.island_temperatures))},
            rng_seed=cfg.seed,
            rng_state=None,
            best_val_score=0.0,
            no_improve_count=0,
        )

    _state_defaults(state)
    checkpoint_dir = _checkpoint_dir_for_experiment(artifacts_dir, experiment_id)
    rng = _restore_rng(state)
    y_true_train = target_values(datasets.train, cfg.target_name)
    y_true_val = target_values(datasets.val, cfg.target_name)
    y_true_test = target_values(datasets.test, cfg.target_name)
    known_invariants_val = compute_known_invariant_values(
        datasets.val,
        include_spectral_feature_pack=cfg.enable_spectral_feature_pack,
    )
    features_train = compute_feature_dicts(
        datasets.train,
        include_spectral_feature_pack=cfg.enable_spectral_feature_pack,
    )
    features_val = compute_feature_dicts(
        datasets.val,
        include_spectral_feature_pack=cfg.enable_spectral_feature_pack,
    )
    features_test = compute_feature_dicts(
        datasets.test,
        include_spectral_feature_pack=cfg.enable_spectral_feature_pack,
    )
    features_sanity = compute_feature_dicts(
        datasets.sanity,
        include_spectral_feature_pack=cfg.enable_spectral_feature_pack,
    )

    append_jsonl(
        "phase1_started",
        {
            "experiment_id": experiment_id,
            "resume": resume,
            "model_name": cfg.model_name,
            "dataset_fingerprint": dataset_fingerprint(cfg, y_true_train, y_true_val, y_true_test),
        },
        log_path,
    )

    stop_reason = "max_generations_reached"
    baseline_results = _collect_baseline_results(
        cfg=cfg,
        datasets_train=datasets.train,
        datasets_val=datasets.val,
        datasets_test=datasets.test,
        y_true_train=y_true_train,
        y_true_val=y_true_val,
        y_true_test=y_true_test,
        enable_spectral_feature_pack=cfg.enable_spectral_feature_pack,
    )
    self_correction_stats: dict[str, Any] = {
        "enabled": cfg.enable_self_correction,
        "max_retries": cfg.self_correction_max_retries,
        "feedback_window": cfg.self_correction_feedback_window,
        "attempted_repairs": 0,
        "successful_repairs": 0,
        "failed_repairs": 0,
        "failure_categories": {},
    }
    primary_archive: MapElitesArchive | None = None
    topology_archive: MapElitesArchive | None = None
    if cfg.enable_map_elites:
        primary_raw = None
        topology_raw = None
        if state.map_elites_archives is not None:
            primary_raw = state.map_elites_archives.get("primary")
            topology_raw = state.map_elites_archives.get("topology")
        if primary_raw is None and state.map_elites_archive is not None:
            primary_raw = state.map_elites_archive

        if isinstance(primary_raw, dict):
            primary_archive = deserialize_archive(primary_raw)
            primary_archive.archive_id = "primary"
        else:
            primary_archive = MapElitesArchive(
                num_bins=cfg.map_elites_bins,
                archive_id="primary",
                cells={},
            )
        if cfg.enable_dual_map_elites:
            if isinstance(topology_raw, dict):
                topology_archive = deserialize_archive(topology_raw)
                topology_archive.archive_id = "topology"
            else:
                topology_archive = MapElitesArchive(
                    num_bins=cfg.map_elites_topology_bins,
                    archive_id="topology",
                    cells={},
                )
    with SandboxEvaluator(
        timeout_sec=cfg.timeout_sec,
        memory_mb=cfg.memory_mb,
        max_workers=cfg.sandbox_max_workers,
    ) as evaluator:
        for _ in range(state.generation, cfg.max_generations):
            _run_one_generation(
                cfg=cfg,
                state=state,
                evaluator=evaluator,
                features_train=features_train,
                features_val=features_val,
                y_true_train=y_true_train,
                y_true_val=y_true_val,
                known_invariants_val=known_invariants_val,
                rng=rng,
                log_path=log_path,
                self_correction_stats=self_correction_stats,
                primary_archive=primary_archive,
                topology_archive=topology_archive,
            )
            current_best = global_best_score(state)
            improved = current_best > state.best_val_score + 1e-12
            if improved:
                state.best_val_score = current_best
                state.no_improve_count = 0
            else:
                state.no_improve_count += 1

            state.generation += 1
            state.rng_state = rng.bit_generator.state
            if primary_archive is not None:
                serialized_primary = serialize_archive(primary_archive)
                serialized_archives = {"primary": serialized_primary}
                if topology_archive is not None:
                    serialized_archives["topology"] = serialize_archive(topology_archive)
                state.map_elites_archives = serialized_archives
                # Keep legacy field for backward-compatible readers.
                state.map_elites_archive = serialized_primary
            checkpoint_path = checkpoint_dir / f"gen_{state.generation}.json"
            save_checkpoint(state, checkpoint_path)
            rotate_generation_checkpoints(checkpoint_dir, cfg.checkpoint_keep_last)
            gen_summary_payload: dict[str, Any] = {
                "experiment_id": experiment_id,
                "generation": state.generation,
                "model_name": cfg.model_name,
                "best_val_score": state.best_val_score,
                "no_improve_count": state.no_improve_count,
            }
            if primary_archive is not None:
                primary_stats = archive_stats(primary_archive)
                gen_summary_payload["map_elites_stats"] = primary_stats
                gen_summary_payload["map_elites_stats_primary"] = primary_stats
                if topology_archive is not None:
                    gen_summary_payload["map_elites_stats_topology"] = archive_stats(
                        topology_archive
                    )
            append_jsonl("generation_summary", gen_summary_payload, log_path)
            if state.no_improve_count >= cfg.early_stop_patience:
                stop_reason = "early_stop"
                break

        write_phase1_summary(
            cfg=cfg,
            state=state,
            evaluator=evaluator,
            artifacts_dir=artifacts_dir,
            stop_reason=stop_reason,
            features_train=features_train,
            features_val=features_val,
            features_test=features_test,
            features_sanity=features_sanity,
            datasets_val=datasets.val,
            datasets_test=datasets.test,
            datasets_sanity=datasets.sanity,
            y_true_train=y_true_train,
            y_true_val=y_true_val,
            y_true_test=y_true_test,
            baseline_results=baseline_results,
            self_correction_stats=self_correction_stats,
        )
    _write_baseline_summary(
        payload=baseline_results,
        artifacts_dir=artifacts_dir,
    )
    return 0


def write_report(artifacts_dir: str | Path) -> Path:
    def _load_json_or_default(path: Path) -> dict:
        if not path.exists():
            return {}
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return {}

    root = Path(artifacts_dir)
    phase1_summary_path = root / "phase1_summary.json"
    baseline_summary_path = root / "baselines_summary.json"
    phase1 = _load_json_or_default(phase1_summary_path)
    baselines = _load_json_or_default(baseline_summary_path)

    lines = ["# Phase 1 Report", "", "## Run Summary", ""]
    lines.append(f"- Success: {phase1.get('success', False)}")
    lines.append(f"- Best candidate: {phase1.get('best_candidate_id', 'n/a')}")
    lines.append(f"- Stop reason: {phase1.get('stop_reason', 'n/a')}")

    val_metrics = phase1.get("val_metrics", {})
    test_metrics = phase1.get("test_metrics", {})
    if val_metrics:
        lines.extend(
            [
                "",
                "## Candidate Metrics",
                "",
                f"- Validation Spearman: {val_metrics.get('spearman', 0.0)}",
                f"- Test Spearman: {test_metrics.get('spearman', 0.0)}",
            ]
        )

    stat = baselines.get("stat_baselines", {})
    if stat:
        lines.extend(["", "## Baselines", ""])
        for name, payload in stat.items():
            lines.append(f"- {name}: {payload.get('status', 'unknown')}")

    self_correction = phase1.get("self_correction_stats", {})
    if isinstance(self_correction, dict) and self_correction:
        lines.extend(["", "## Self-Correction", ""])
        lines.append(f"- Enabled: {self_correction.get('enabled', False)}")
        lines.append(f"- Attempted repairs: {self_correction.get('attempted_repairs', 0)}")
        lines.append(f"- Successful repairs: {self_correction.get('successful_repairs', 0)}")
        lines.append(f"- Failed repairs: {self_correction.get('failed_repairs', 0)}")

    ood_path = root / "ood_validation.json"
    ood = _load_json_or_default(ood_path)
    if ood:
        lines.extend(["", "## OOD Validation", ""])
        for category, metrics in ood.items():
            if not isinstance(metrics, dict):
                continue
            valid = metrics.get("valid_count", 0)
            total = metrics.get("total_count", 0)
            spearman = metrics.get("spearman")
            bound_score = metrics.get("bound_score")
            if spearman is not None:
                score_str = f"spearman={spearman:.4f}"
            elif bound_score is not None:
                score_str = f"bound_score={bound_score:.4f}"
            else:
                score_str = "no valid predictions"
            lines.append(f"- {category}: {score_str} ({valid}/{total} valid)")

    # MAP-Elites archive coverage from generation logs
    events_path = root / "logs" / "events.jsonl"
    if events_path.exists():
        try:
            primary_coverages: list[int] = []
            topology_coverages: list[int] = []
            with events_path.open("r", encoding="utf-8") as f:
                for line in f:
                    if '"generation_summary"' not in line:
                        continue
                    event = json.loads(line)
                    payload = event.get("payload", {})
                    primary_stats = payload.get("map_elites_stats_primary") or payload.get(
                        "map_elites_stats"
                    )
                    topology_stats = payload.get("map_elites_stats_topology")
                    if isinstance(primary_stats, dict) and "coverage" in primary_stats:
                        primary_coverages.append(int(primary_stats["coverage"]))
                    if isinstance(topology_stats, dict) and "coverage" in topology_stats:
                        topology_coverages.append(int(topology_stats["coverage"]))
            if primary_coverages:
                lines.extend(["", "## MAP-Elites Archive", ""])
                lines.append(f"- Final coverage: {primary_coverages[-1]} cells")
                lines.append(
                    f"- Coverage progression: {' -> '.join(str(c) for c in primary_coverages)}"
                )
                if topology_coverages:
                    lines.append(f"- Topology final coverage: {topology_coverages[-1]} cells")
                    lines.append(
                        f"- Topology progression: {' -> '.join(str(c) for c in topology_coverages)}"
                    )
        except (json.JSONDecodeError, OSError, KeyError):
            pass

    report_path = root / "report.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Graph invariant discovery CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    phase1 = sub.add_parser("phase1")
    phase1.add_argument("--config", type=str, default=None)
    phase1.add_argument("--resume", type=str, default=None)

    report = sub.add_parser("report")
    report.add_argument("--artifacts", type=str, required=True)

    benchmark = sub.add_parser("benchmark")
    benchmark.add_argument("--config", type=str, default=None)

    ood = sub.add_parser("ood-validate")
    ood.add_argument("--summary", type=str, required=True)
    ood.add_argument("--output", type=str, required=True)
    ood.add_argument("--seed", type=int, default=42)
    ood.add_argument("--num-large", type=int, default=100)
    ood.add_argument("--num-extreme", type=int, default=50)

    args = parser.parse_args()

    if args.command == "phase1":
        cfg = Phase1Config.from_json(args.config) if args.config else Phase1Config()
        return run_phase1(cfg, resume=args.resume)
    if args.command == "report":
        report_path = write_report(args.artifacts)
        print(f"Report written to {report_path}")
        return 0
    if args.command == "benchmark":
        from .benchmark import run_benchmark

        cfg = Phase1Config.from_json(args.config) if args.config else Phase1Config()
        return run_benchmark(cfg)
    if args.command == "ood-validate":
        from .ood_validation import run_ood_validation

        return run_ood_validation(
            summary_path=args.summary,
            output_dir=args.output,
            seed=args.seed,
            num_large=args.num_large,
            num_extreme=args.num_extreme,
        )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
