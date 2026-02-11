import argparse
import re
from datetime import UTC, datetime
from pathlib import Path

import networkx as nx
import numpy as np

from .config import Phase1Config
from .data import generate_phase1_datasets
from .evolution import migrate_ring_top1
from .known_invariants import compute_known_invariant_values
from .llm_ollama import build_prompt, generate_candidate_code, list_available_models
from .logging_io import (
    append_jsonl,
    load_checkpoint,
    rotate_generation_checkpoints,
    save_checkpoint,
)
from .sandbox import evaluate_candidate_on_graphs
from .scoring import (
    compute_metrics,
    compute_novelty_bonus,
    compute_simplicity_score,
    compute_total_score,
)
from .types import Candidate, CheckpointState

TARGET_FUNCTIONS = {
    "average_shortest_path_length": nx.average_shortest_path_length,
    "diameter": lambda graph: float(nx.diameter(graph)),
}


def _target_values(graphs: list[nx.Graph], target_name: str) -> list[float]:
    target_fn = TARGET_FUNCTIONS.get(target_name)
    if target_fn is None:
        raise ValueError(f"unsupported target: {target_name}")
    return [target_fn(graph) for graph in graphs]


def _new_experiment_id() -> str:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"phase1_{stamp}"


def _global_best_score(state: CheckpointState) -> float:
    scores = [
        candidate.val_score for candidates in state.islands.values() for candidate in candidates
    ]
    if not scores:
        return 0.0
    return max(scores)


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


def _candidate_prompt(state: CheckpointState, island_id: int, target_name: str) -> str:
    top_candidates = [
        candidate.code
        for candidate in sorted(
            state.islands.get(island_id, []), key=lambda c: c.val_score, reverse=True
        )
    ]
    return build_prompt(
        island_mode=f"island_{island_id}",
        top_candidates=top_candidates,
        failures=[],
        target_name=target_name,
    )


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


def _run_one_generation(
    cfg: Phase1Config,
    state: CheckpointState,
    datasets_train: list[nx.Graph],
    datasets_val: list[nx.Graph],
    y_true_train: list[float],
    y_true_val: list[float],
    known_invariants_val: dict[str, list[float]],
    rng: np.random.Generator,
    log_path: Path,
) -> None:
    island_ids = sorted(state.islands.keys())
    for island_id in island_ids:
        new_candidates: list[Candidate] = []
        for pop_idx in range(cfg.population_size):
            prompt = _candidate_prompt(state, island_id, cfg.target_name)
            code = generate_candidate_code(
                prompt=prompt,
                model=cfg.model_name,
                temperature=cfg.island_temperatures[island_id],
                url=cfg.ollama_url,
                allow_remote=cfg.allow_remote_ollama,
            )
            y_pred_train_raw = evaluate_candidate_on_graphs(
                code, datasets_train, timeout_sec=cfg.timeout_sec, memory_mb=cfg.memory_mb
            )
            train_pairs = [
                (idx, yt, yp)
                for idx, (yt, yp) in enumerate(zip(y_true_train, y_pred_train_raw, strict=True))
                if yp is not None
            ]
            if not train_pairs:
                append_jsonl(
                    "candidate_rejected",
                    {
                        "generation": state.generation,
                        "island_id": island_id,
                        "population_idx": pop_idx,
                        "reason": "no_valid_train_predictions",
                        "model_name": cfg.model_name,
                    },
                    log_path,
                )
                continue

            _, y_t_train, y_p_train = zip(*train_pairs, strict=True)
            train_metrics = compute_metrics(list(y_t_train), list(y_p_train))
            train_signal = abs(train_metrics.rho_spearman)
            if train_signal <= cfg.train_score_threshold:
                append_jsonl(
                    "candidate_rejected",
                    {
                        "generation": state.generation,
                        "island_id": island_id,
                        "population_idx": pop_idx,
                        "reason": "below_train_threshold",
                        "train_signal": train_signal,
                        "model_name": cfg.model_name,
                    },
                    log_path,
                )
                continue

            y_pred_val_raw = evaluate_candidate_on_graphs(
                code, datasets_val, timeout_sec=cfg.timeout_sec, memory_mb=cfg.memory_mb
            )
            val_pairs = [
                (idx, yt, yp)
                for idx, (yt, yp) in enumerate(zip(y_true_val, y_pred_val_raw, strict=True))
                if yp is not None
            ]
            if not val_pairs:
                append_jsonl(
                    "candidate_rejected",
                    {
                        "generation": state.generation,
                        "island_id": island_id,
                        "population_idx": pop_idx,
                        "reason": "no_valid_val_predictions",
                        "model_name": cfg.model_name,
                    },
                    log_path,
                )
                continue

            valid_indices, y_t_val, y_p_val = zip(*val_pairs, strict=True)
            val_metrics = compute_metrics(list(y_t_val), list(y_p_val))
            simplicity = compute_simplicity_score(code)
            known_subset = {
                name: [values[idx] for idx in valid_indices]
                for name, values in known_invariants_val.items()
            }
            novelty_bonus = compute_novelty_bonus(list(y_p_val), known_subset)
            total = compute_total_score(
                abs(val_metrics.rho_spearman),
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
                train_score=train_signal,
                val_score=total,
                simplicity_score=simplicity,
                novelty_bonus=novelty_bonus,
            )
            new_candidates.append(candidate)
            append_jsonl(
                "candidate_evaluated",
                {
                    "candidate_id": candidate.id,
                    "generation": state.generation,
                    "island_id": island_id,
                    "model_name": cfg.model_name,
                    "train_signal": train_signal,
                    "spearman": val_metrics.rho_spearman,
                    "pearson": val_metrics.r_pearson,
                    "rmse": val_metrics.rmse,
                    "mae": val_metrics.mae,
                    "simplicity_score": simplicity,
                    "novelty_bonus": novelty_bonus,
                    "total_score": total,
                },
                log_path,
            )

        merged = state.islands.get(island_id, []) + new_candidates
        merged.sort(key=lambda candidate: candidate.val_score, reverse=True)
        state.islands[island_id] = merged[: cfg.population_size]

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
            islands={i: [] for i in range(4)},
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
            islands={i: [] for i in range(4)},
            rng_seed=cfg.seed,
            rng_state=None,
            best_val_score=0.0,
            no_improve_count=0,
        )

    checkpoint_dir = _checkpoint_dir_for_experiment(artifacts_dir, experiment_id)
    rng = _restore_rng(state)
    y_true_train = _target_values(datasets.train, cfg.target_name)
    y_true_val = _target_values(datasets.val, cfg.target_name)
    known_invariants_val = compute_known_invariant_values(datasets.val)

    append_jsonl(
        "phase1_started",
        {
            "experiment_id": experiment_id,
            "resume": resume,
            "model_name": cfg.model_name,
        },
        log_path,
    )

    for _ in range(state.generation, cfg.max_generations):
        _run_one_generation(
            cfg=cfg,
            state=state,
            datasets_train=datasets.train,
            datasets_val=datasets.val,
            y_true_train=y_true_train,
            y_true_val=y_true_val,
            known_invariants_val=known_invariants_val,
            rng=rng,
            log_path=log_path,
        )
        current_best = _global_best_score(state)
        improved = current_best > state.best_val_score + 1e-12
        if improved:
            state.best_val_score = current_best
            state.no_improve_count = 0
        else:
            state.no_improve_count += 1

        state.generation += 1
        state.rng_state = rng.bit_generator.state
        checkpoint_path = checkpoint_dir / f"gen_{state.generation}.json"
        save_checkpoint(state, checkpoint_path)
        rotate_generation_checkpoints(checkpoint_dir, cfg.checkpoint_keep_last)
        append_jsonl(
            "generation_summary",
            {
                "experiment_id": experiment_id,
                "generation": state.generation,
                "model_name": cfg.model_name,
                "best_val_score": state.best_val_score,
                "no_improve_count": state.no_improve_count,
            },
            log_path,
        )
        if state.no_improve_count >= cfg.early_stop_patience:
            break

    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Graph invariant discovery CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    phase1 = sub.add_parser("phase1")
    phase1.add_argument("--config", type=str, default=None)
    phase1.add_argument("--resume", type=str, default=None)

    args = parser.parse_args()
    cfg = Phase1Config.from_json(args.config) if args.config else Phase1Config()

    if args.command == "phase1":
        return run_phase1(cfg, resume=args.resume)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
