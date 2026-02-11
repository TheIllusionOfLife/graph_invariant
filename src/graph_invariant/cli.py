import argparse
import json
import re
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path

import networkx as nx
import numpy as np

from .baselines import run_pysr_baseline, run_stat_baselines
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
    write_json,
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

_CONSTRAINED_SUFFIX = (
    "\nConstrained mode is active. Use only these operators: +, -, *, /, log, sqrt, **, "
    "sum, mean, max, min. Follow this template: "
    "def new_invariant(G): n = G.number_of_nodes(); m = G.number_of_edges(); "
    "degrees = [d for _, d in G.degree()]; return <expression using n,m,degrees>."
)


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


def _best_candidate(state: CheckpointState) -> Candidate | None:
    all_candidates = [candidate for island in state.islands.values() for candidate in island]
    if not all_candidates:
        return None
    return max(all_candidates, key=lambda c: c.val_score)


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


def _candidate_prompt(state: CheckpointState, island_id: int, target_name: str) -> str:
    top_candidates = [
        candidate.code
        for candidate in sorted(
            state.islands.get(island_id, []), key=lambda c: c.val_score, reverse=True
        )
    ]
    prompt = build_prompt(
        island_mode=f"island_{island_id}_{state.island_prompt_mode.get(island_id, 'free')}",
        top_candidates=top_candidates,
        failures=[],
        target_name=target_name,
    )
    if state.island_prompt_mode.get(island_id, "free") == "constrained":
        return prompt + _CONSTRAINED_SUFFIX
    return prompt


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
            if constrained_span <= cfg.constrained_recovery_generations:
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
        had_valid_train_candidate = False
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

            had_valid_train_candidate = True
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


def _evaluate_split(
    code: str,
    graphs: list[nx.Graph],
    y_true: list[float],
    cfg: Phase1Config,
    known_invariants: dict[str, list[float]] | None = None,
) -> dict[str, float | int | None]:
    y_pred_raw = evaluate_candidate_on_graphs(
        code,
        graphs,
        timeout_sec=cfg.timeout_sec,
        memory_mb=cfg.memory_mb,
    )
    valid_pairs = [
        (idx, yt, yp)
        for idx, (yt, yp) in enumerate(zip(y_true, y_pred_raw, strict=True))
        if yp is not None
    ]
    if not valid_pairs:
        return {
            "spearman": 0.0,
            "pearson": 0.0,
            "rmse": 0.0,
            "mae": 0.0,
            "valid_count": 0,
            "novelty_bonus": None,
        }

    valid_indices, y_true_valid, y_pred_valid = zip(*valid_pairs, strict=True)
    metrics = compute_metrics(list(y_true_valid), list(y_pred_valid))
    novelty = None
    if known_invariants is not None:
        known_subset = {
            name: [values[idx] for idx in valid_indices]
            for name, values in known_invariants.items()
        }
        novelty = compute_novelty_bonus(list(y_pred_valid), known_subset)

    return {
        "spearman": metrics.rho_spearman,
        "pearson": metrics.r_pearson,
        "rmse": metrics.rmse,
        "mae": metrics.mae,
        "valid_count": metrics.valid_count,
        "novelty_bonus": novelty,
    }


def _dataset_fingerprint(
    cfg: Phase1Config, y_train: list[float], y_val: list[float], y_test: list[float]
) -> str:
    payload = {
        "seed": cfg.seed,
        "target_name": cfg.target_name,
        "train": y_train,
        "val": y_val,
        "test": y_test,
    }
    text = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return sha256(text.encode("utf-8")).hexdigest()


def _write_phase1_summary(
    cfg: Phase1Config,
    state: CheckpointState,
    artifacts_dir: Path,
    stop_reason: str,
    datasets_train: list[nx.Graph],
    datasets_val: list[nx.Graph],
    datasets_test: list[nx.Graph],
    datasets_sanity: list[nx.Graph],
    y_true_train: list[float],
    y_true_val: list[float],
    y_true_test: list[float],
) -> None:
    best = _best_candidate(state)
    summary_path = artifacts_dir / "phase1_summary.json"
    if best is None:
        payload = {
            "experiment_id": state.experiment_id,
            "success": False,
            "reason": "no_candidates",
            "stop_reason": stop_reason,
        }
        write_json(payload, summary_path)
        return

    known_val = compute_known_invariant_values(datasets_val)
    known_test = compute_known_invariant_values(datasets_test)
    y_sanity = _target_values(datasets_sanity, cfg.target_name)
    val_metrics = _evaluate_split(best.code, datasets_val, y_true_val, cfg, known_val)
    test_metrics = _evaluate_split(best.code, datasets_test, y_true_test, cfg, known_test)
    train_metrics = _evaluate_split(best.code, datasets_train, y_true_train, cfg)
    sanity_metrics = _evaluate_split(best.code, datasets_sanity, y_sanity, cfg)

    payload = {
        "schema_version": 1,
        "experiment_id": state.experiment_id,
        "best_candidate_id": best.id,
        "best_candidate_code_sha256": sha256(best.code.encode("utf-8")).hexdigest(),
        "best_val_score": state.best_val_score,
        "stop_reason": stop_reason,
        "success": abs(float(val_metrics["spearman"])) >= 0.85,
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "sanity_metrics": sanity_metrics,
        "dataset_fingerprint": _dataset_fingerprint(cfg, y_true_train, y_true_val, y_true_test),
    }
    if cfg.persist_candidate_code_in_summary:
        payload["best_candidate_code"] = best.code
    write_json(payload, summary_path)


def _write_baseline_summary(
    cfg: Phase1Config,
    artifacts_dir: Path,
    datasets_train: list[nx.Graph],
    datasets_val: list[nx.Graph],
    datasets_test: list[nx.Graph],
    y_true_train: list[float],
    y_true_val: list[float],
    y_true_test: list[float],
) -> None:
    if not cfg.run_baselines:
        return

    stat = run_stat_baselines(
        train_graphs=datasets_train,
        val_graphs=datasets_val,
        test_graphs=datasets_test,
        y_train=y_true_train,
        y_val=y_true_val,
        y_test=y_true_test,
    )
    pysr = run_pysr_baseline(
        train_graphs=datasets_train,
        val_graphs=datasets_val,
        test_graphs=datasets_test,
        y_train=y_true_train,
        y_val=y_true_val,
        y_test=y_true_test,
    )
    write_json(
        {
            "schema_version": 1,
            "target_name": cfg.target_name,
            "stat_baselines": stat,
            "pysr_baseline": pysr,
        },
        artifacts_dir / "baselines_summary.json",
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

    _state_defaults(state)
    checkpoint_dir = _checkpoint_dir_for_experiment(artifacts_dir, experiment_id)
    rng = _restore_rng(state)
    y_true_train = _target_values(datasets.train, cfg.target_name)
    y_true_val = _target_values(datasets.val, cfg.target_name)
    y_true_test = _target_values(datasets.test, cfg.target_name)
    known_invariants_val = compute_known_invariant_values(datasets.val)

    append_jsonl(
        "phase1_started",
        {
            "experiment_id": experiment_id,
            "resume": resume,
            "model_name": cfg.model_name,
            "dataset_fingerprint": _dataset_fingerprint(cfg, y_true_train, y_true_val, y_true_test),
        },
        log_path,
    )

    stop_reason = "max_generations_reached"
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
            stop_reason = "early_stop"
            break

    _write_phase1_summary(
        cfg=cfg,
        state=state,
        artifacts_dir=artifacts_dir,
        stop_reason=stop_reason,
        datasets_train=datasets.train,
        datasets_val=datasets.val,
        datasets_test=datasets.test,
        datasets_sanity=datasets.sanity,
        y_true_train=y_true_train,
        y_true_val=y_true_val,
        y_true_test=y_true_test,
    )
    _write_baseline_summary(
        cfg=cfg,
        artifacts_dir=artifacts_dir,
        datasets_train=datasets.train,
        datasets_val=datasets.val,
        datasets_test=datasets.test,
        y_true_train=y_true_train,
        y_true_val=y_true_val,
        y_true_test=y_true_test,
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

    args = parser.parse_args()

    if args.command == "phase1":
        cfg = Phase1Config.from_json(args.config) if args.config else Phase1Config()
        return run_phase1(cfg, resume=args.resume)
    if args.command == "report":
        write_report(args.artifacts)
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
