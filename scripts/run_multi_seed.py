"""Multi-seed experiment orchestrator for Harmony evaluation.

Runs compute_metrics_table across 10 seeds and aggregates results
into mean±std for each metric. Supports --dry-run for validation.

Usage:
    uv run python scripts/run_multi_seed.py \\
        --astronomy artifacts/harmony/astronomy \\
        --physics   artifacts/harmony/physics \\
        --output    data/results/multi_seed/

    uv run python scripts/run_multi_seed.py --dry-run
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Seed configuration (plan §Phase 7)
# ---------------------------------------------------------------------------
DEFAULT_SEEDS: list[int] = [42, 123, 456, 789, 1024, 1337, 2048, 3141, 4096, 5000]


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def aggregate_seed_results(
    seed_dfs: dict[int, pd.DataFrame],
) -> pd.DataFrame:
    """Aggregate per-seed DataFrames into mean±std summary.

    Parameters
    ----------
    seed_dfs:
        Mapping of seed → DataFrame (output of compute_metrics_table).
        All DataFrames must share the same index (domains) and columns.

    Returns
    -------
    DataFrame with columns: {metric}_mean, {metric}_std, n_seeds
    for each original metric column.
    """
    if not seed_dfs:
        return pd.DataFrame()

    seeds = sorted(seed_dfs.keys())
    reference_df = seed_dfs[seeds[0]]
    domains = reference_df.index
    metrics = list(reference_df.columns)

    # Validate all DataFrames share the same shape
    for s in seeds[1:]:
        df = seed_dfs[s]
        if list(df.index) != list(domains) or list(df.columns) != list(metrics):
            msg = f"Seed {s} DataFrame shape mismatch: expected {list(domains)}/{metrics}"
            raise ValueError(msg)

    result_rows: dict[str, dict[str, float | int]] = {}
    for domain in domains:
        row: dict[str, float | int] = {}
        for metric in metrics:
            values = [float(seed_dfs[s].loc[domain, metric]) for s in seeds]
            row[f"{metric}_mean"] = float(np.mean(values))
            row[f"{metric}_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
        row["n_seeds"] = len(seeds)
        result_rows[domain] = row

    df = pd.DataFrame.from_dict(result_rows, orient="index")
    df.index.name = "domain"
    return df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    import argparse

    sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
    sys.path.insert(0, str(Path(__file__).parent.parent))

    parser = argparse.ArgumentParser(description="Multi-seed Harmony experiment runner")
    known_domains = [
        "linear_algebra",
        "periodic_table",
        "astronomy",
        "physics",
        "materials",
        "wikidata_physics",
        "wikidata_materials",
    ]
    for domain in known_domains:
        parser.add_argument(f"--{domain}", type=Path, metavar="DIR", dest=domain)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/results/multi_seed"),
        help="Output directory",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=DEFAULT_SEEDS,
        help="Seeds to evaluate (default: 10 standard seeds)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate configuration without running experiments",
    )
    args = parser.parse_args()

    # Deduplicate seeds (preserve order, first occurrence wins)
    seen_seeds: set[int] = set()
    unique_seeds: list[int] = []
    for s in args.seeds:
        if s not in seen_seeds:
            seen_seeds.add(s)
            unique_seeds.append(s)
    args.seeds = unique_seeds

    domain_checkpoints: dict[str, Path] = {}
    for domain in known_domains:
        d = getattr(args, domain, None)
        if d is not None:
            domain_checkpoints[domain] = d

    if args.dry_run:
        print("=== DRY RUN ===")
        print(f"Seeds: {args.seeds} ({len(args.seeds)} total)")
        print(f"Domains: {list(domain_checkpoints.keys()) or '(none specified)'}")
        print(f"Output: {args.output}")
        print(f"Total runs: {len(args.seeds)} seeds × {len(domain_checkpoints)} domains")
        if not domain_checkpoints:
            print("WARNING: No domains specified.")
        print("Dry run passed — configuration valid.")
        return

    if not domain_checkpoints:
        parser.error("Specify at least one domain checkpoint directory.")

    # Validate checkpoint paths exist
    for domain, cp_dir in domain_checkpoints.items():
        cp_file = cp_dir / "checkpoint.json"
        if not cp_file.exists():
            parser.error(f"Checkpoint not found for domain '{domain}': {cp_file}")

    from analysis.metrics_table import compute_metrics_table

    seed_dfs: dict[int, pd.DataFrame] = {}
    for i, seed in enumerate(args.seeds):
        print(f"[{i + 1}/{len(args.seeds)}] Running seed {seed}...")
        df = compute_metrics_table(domain_checkpoints, seed=seed)
        seed_dfs[seed] = df
        print(f"  Done. Domains: {list(df.index)}")

    # Aggregate
    summary = aggregate_seed_results(seed_dfs)

    # Write output
    args.output.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.output / "summary.csv")

    # Save per-seed results
    for seed, df in seed_dfs.items():
        df.to_csv(args.output / f"seed_{seed}.csv")

    # Save metadata
    meta = {
        "seeds": list(seed_dfs.keys()),
        "n_seeds": len(seed_dfs),
        "domains": list(domain_checkpoints.keys()),
    }
    (args.output / "metadata.json").write_text(json.dumps(meta, indent=2))

    print(f"\nResults written to {args.output}/")
    print("\n=== Summary (mean ± std) ===")
    print(summary.to_string())


if __name__ == "__main__":
    main()
