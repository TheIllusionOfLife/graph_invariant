from __future__ import annotations

import json
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

from .config import Phase1Config
from .logging_io import write_json
from .phase1_loop import run_phase1


def _load_json_or_default(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def run_benchmark(cfg: Phase1Config) -> int:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    root = Path(cfg.artifacts_dir) / f"benchmark_{stamp}"
    runs: list[dict[str, object]] = []
    success_count = 0

    for seed in cfg.benchmark_seeds:
        run_root = root / f"seed_{seed}"
        run_cfg = replace(
            cfg,
            seed=seed,
            artifacts_dir=str(run_root),
            experiment_id=f"seed_{seed}",
        )
        status = run_phase1(run_cfg)
        summary = _load_json_or_default(run_root / "phase1_summary.json")
        has_baselines = (run_root / "baselines_summary.json").exists()
        run_success = bool(summary.get("success", False))
        if run_success:
            success_count += 1
        runs.append(
            {
                "seed": seed,
                "status": status,
                "success": run_success,
                "artifacts_dir": str(run_root),
                "val_spearman": summary.get("val_metrics", {}).get("spearman"),
                "test_spearman": summary.get("test_metrics", {}).get("spearman"),
                "has_baselines": has_baselines,
            }
        )

    payload = {
        "schema_version": 1,
        "benchmark_id": root.name,
        "total_runs": len(runs),
        "success_count": success_count,
        "success_rate": float(success_count) / len(runs) if runs else 0.0,
        "failed_runs": sum(1 for run in runs if int(run["status"]) != 0),
        "runs": runs,
    }
    write_json(payload, root / "benchmark_summary.json")

    report_lines = [
        "# Benchmark Report",
        "",
        f"- Benchmark ID: {root.name}",
        f"- Total runs: {payload['total_runs']}",
        f"- Success count: {payload['success_count']}",
        f"- Success rate: {payload['success_rate']:.3f}",
        "",
        "## Runs",
        "",
    ]
    for run in runs:
        report_lines.append(
            "- Seed {seed}: success={success}, val_spearman={val}, test_spearman={test}".format(
                seed=run["seed"],
                success=run["success"],
                val=run["val_spearman"],
                test=run["test_spearman"],
            )
        )
    (root / "benchmark_report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    return 1 if payload["failed_runs"] > 0 else 0
