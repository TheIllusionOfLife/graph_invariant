import argparse
import json
from pathlib import Path
from typing import Any

import networkx as nx

from .baselines import run_pysr_baseline, run_stat_baselines
from .candidate_pipeline import (
    _checkpoint_dir_for_experiment,
    _new_experiment_id,
    _restore_rng,
    _run_one_generation,
    _state_defaults,
    _validate_experiment_id,
)
from .config import Phase1Config
from .data import generate_phase1_datasets
from .evaluation import (
    dataset_fingerprint,
    global_best_score,
)
from .known_invariants import compute_feature_dicts, compute_known_invariant_values
from .llm_ollama import (
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
    serialize_archive,
)
from .sandbox import SandboxEvaluator
from .summary import write_phase1_summary
from .targets import target_values
from .types import CheckpointState


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
