"""Re-evaluate benchmark success criterion from existing per-seed summaries.

Reads all per-seed phase1_summary.json files under artifacts/benchmark_aspl/
and re-applies the updated success criterion: test Spearman rho >= 0.85
(no PySR parity gate). Regenerates benchmark_summary.json in each
benchmark subdirectory.

Usage:
    uv run python scripts/reeval_benchmark_success.py [--artifacts-root artifacts/]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def reeval_benchmark_summary(bench_dir: Path, threshold: float = 0.85) -> dict:
    """Re-evaluate success for all seeds in a benchmark directory.

    Reads the existing benchmark_summary.json and re-applies the
    updated success criterion (rho >= threshold only, no PySR parity).
    Writes the updated summary back and returns it.
    """
    summary_path = bench_dir / "benchmark_summary.json"
    if not summary_path.exists():
        print(f"  Skipping {bench_dir.name}: no benchmark_summary.json found")
        return {}

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    runs = summary.get("runs", [])
    if not isinstance(runs, list):
        print(f"  Skipping {bench_dir.name}: no 'runs' list in summary")
        return summary

    updated_runs = []
    success_count = 0
    for run in runs:
        if not isinstance(run, dict):
            updated_runs.append(run)
            continue
        test_rho = run.get("test_spearman")
        new_success = isinstance(test_rho, (int, float)) and float(test_rho) >= threshold
        updated_run = dict(run)
        updated_run["success"] = new_success
        updated_run["success_criterion"] = f"test_spearman >= {threshold}"
        updated_runs.append(updated_run)
        if new_success:
            success_count += 1

    summary["runs"] = updated_runs
    summary["success_count"] = success_count
    summary["success_rate"] = success_count / len(runs) if runs else 0.0
    summary["success_criterion"] = f"test_spearman >= {threshold} (no PySR parity gate)"

    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(
        f"  {bench_dir.name}: {success_count}/{len(runs)} seeds successful "
        f"(threshold={threshold})"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Re-evaluate benchmark success from existing summaries"
    )
    parser.add_argument(
        "--artifacts-root",
        type=str,
        default="artifacts",
        help="Root directory containing experiment artifacts",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.85,
        help="Spearman rho threshold for success (default: 0.85)",
    )
    args = parser.parse_args()

    artifacts_root = Path(args.artifacts_root)
    bench_root = artifacts_root / "benchmark_aspl"

    if not bench_root.exists():
        print(f"No benchmark directory found at {bench_root}")
        return

    bench_dirs = sorted(d for d in bench_root.iterdir() if d.is_dir())
    if not bench_dirs:
        print(f"No subdirectories found in {bench_root}")
        return

    print(f"Re-evaluating benchmark success in {bench_root}...")
    for bench_dir in bench_dirs:
        reeval_benchmark_summary(bench_dir, threshold=args.threshold)

    print("Done.")


if __name__ == "__main__":
    main()
