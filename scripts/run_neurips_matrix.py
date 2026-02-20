"""Run a multi-seed NeurIPS evidence matrix across multiple configs.

Usage:
    uv run python scripts/run_neurips_matrix.py \
      --configs configs/neurips_matrix/map_elites_aspl_full.json \
                configs/neurips_matrix/algebraic_connectivity_full.json \
      --seeds 11 22 33 44 55 \
      --output-root artifacts/neurips_matrix
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import time
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from graph_invariant.cli import run_phase1
from graph_invariant.config import Phase1Config
from graph_invariant.logging_io import write_json
from graph_invariant.stats_utils import mean_std_ci95, safe_float


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_json_or_default(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def _summarize_runs(runs: list[dict[str, Any]]) -> dict[str, Any]:
    completed = [run for run in runs if run.get("status") == 0]
    criteria_success = [run for run in completed if bool(run.get("success", False))]
    val_scores = [run["val_spearman"] for run in completed if run.get("val_spearman") is not None]
    test_scores = [
        run["test_spearman"] for run in completed if run.get("test_spearman") is not None
    ]
    durations = [run["duration_sec"] for run in runs if run.get("duration_sec") is not None]
    return {
        "total_runs": len(runs),
        "completed_runs": len(completed),
        "criteria_success_runs": len(criteria_success),
        "failed_runs": len(runs) - len(completed),
        # Backward-compat alias for existing readers that interpreted this as status==0.
        "successful_runs": len(completed),
        "val_spearman": mean_std_ci95(val_scores),
        "test_spearman": mean_std_ci95(test_scores),
        "duration_sec": mean_std_ci95(durations),
    }


def _run_one(config_path: Path, seed: int, output_root: Path) -> dict[str, Any]:
    base_cfg = _load_json(config_path)
    exp_name = config_path.stem
    run_root = output_root / exp_name / f"seed_{seed}"
    run_root.mkdir(parents=True, exist_ok=True)

    if base_cfg.get("enable_dual_map_elites") and not base_cfg.get("enable_map_elites"):
        base_cfg["enable_map_elites"] = True

    run_cfg = Phase1Config.from_dict(
        {
            **base_cfg,
            "seed": seed,
            "artifacts_dir": str(run_root),
            "experiment_id": f"{exp_name}_seed_{seed}",
        }
    )

    started = datetime.now(UTC)
    t0 = time.perf_counter()
    status = run_phase1(run_cfg)
    duration_sec = time.perf_counter() - t0
    ended = datetime.now(UTC)

    summary = _load_json_or_default(run_root / "phase1_summary.json")
    val_metrics = summary.get("val_metrics", {}) if isinstance(summary, dict) else {}
    test_metrics = summary.get("test_metrics", {}) if isinstance(summary, dict) else {}

    result: dict[str, Any] = {
        "experiment": exp_name,
        "seed": seed,
        "status": status,
        "success": bool(summary.get("success", False)),
        "fitness_mode": summary.get("fitness_mode", run_cfg.fitness_mode),
        "artifacts_dir": str(run_root),
        "start_time_utc": started.isoformat(),
        "end_time_utc": ended.isoformat(),
        "duration_sec": round(duration_sec, 3),
        "val_spearman": safe_float(val_metrics.get("spearman")),
        "test_spearman": safe_float(test_metrics.get("spearman")),
    }

    bounds_metrics = summary.get("bounds_metrics", {}) if isinstance(summary, dict) else {}
    if isinstance(bounds_metrics, dict):
        val_bounds = bounds_metrics.get("val", {})
        test_bounds = bounds_metrics.get("test", {})
        if isinstance(val_bounds, dict):
            result["val_bound_score"] = safe_float(val_bounds.get("bound_score"))
            result["val_satisfaction_rate"] = safe_float(val_bounds.get("satisfaction_rate"))
        if isinstance(test_bounds, dict):
            result["test_bound_score"] = safe_float(test_bounds.get("bound_score"))
            result["test_satisfaction_rate"] = safe_float(test_bounds.get("satisfaction_rate"))

    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run NeurIPS experiment matrix")
    parser.add_argument(
        "--configs",
        nargs="+",
        required=True,
        help="List of config JSON paths to run",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[11, 22, 33, 44, 55],
        help="Seeds to evaluate for each config",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default="artifacts/neurips_matrix",
        help="Output root for matrix artifacts and summary",
    )
    parser.add_argument(
        "--max-parallel",
        type=int,
        default=1,
        help="Maximum number of runs to execute in parallel",
    )
    return parser.parse_args()


def _run_one_job(job: tuple[Path, int, Path]) -> dict[str, Any]:
    config_path, seed, output_root = job
    return _run_one(config_path=config_path, seed=seed, output_root=output_root)


def main() -> None:
    args = parse_args()
    config_paths = [Path(p) for p in args.configs]
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    runs: list[dict[str, Any]] = []
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)

    launched_at = datetime.now(UTC)
    jobs: list[tuple[Path, int, Path]] = []
    for config_path in config_paths:
        if not config_path.exists():
            raise FileNotFoundError(f"Config not found: {config_path}")
        for seed in args.seeds:
            jobs.append((config_path, seed, output_root))

    if args.max_parallel < 1:
        raise ValueError("--max-parallel must be >= 1")

    run_results: list[dict[str, Any]]
    if args.max_parallel == 1:
        run_results = [_run_one_job(job) for job in jobs]
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.max_parallel) as pool:
            run_results = list(pool.map(_run_one_job, jobs))

    for run in run_results:
        runs.append(run)
        grouped[run["experiment"]].append(run)

    payload = {
        "schema_version": 1,
        "matrix_id": f"neurips_matrix_{launched_at.strftime('%Y%m%dT%H%M%SZ')}",
        "launched_at_utc": launched_at.isoformat(),
        "seeds": args.seeds,
        "configs": [str(path) for path in config_paths],
        "total_runs": len(runs),
        "runs": runs,
        "experiments": {
            name: {
                "summary": _summarize_runs(exp_runs),
                "runs": exp_runs,
            }
            for name, exp_runs in sorted(grouped.items())
        },
    }

    write_json(payload, output_root / "matrix_summary.json")
    print(f"Wrote matrix summary to {output_root / 'matrix_summary.json'}")


if __name__ == "__main__":
    main()
