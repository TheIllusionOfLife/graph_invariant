#!/usr/bin/env python
"""Block A orchestrator: all non-LLM experiments.

Runs sequentially:
  1. Multi-seed re-evaluation (10 seeds × N domains × 8 models)
  2. Statistical tests (paired bootstrap CI + Cliff's delta)
  3. Backtesting against hidden edges
  4. Downstream analysis (correlations, failure modes, regimes)
  5. Frequency dominance analysis
  6. Wikidata ablation (200 bootstrap resamples)
  7. Reproducibility table

Usage:
    uv run python scripts/run_block_a.py \
        --astronomy artifacts/harmony/astronomy \
        --physics   artifacts/harmony/physics \
        --materials artifacts/harmony/materials \
        --wikidata-physics   artifacts/harmony/wikidata_physics \
        --wikidata-materials artifacts/harmony/wikidata_materials
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent))

_ALL_DOMAINS = [
    "astronomy",
    "physics",
    "materials",
    "wikidata_physics",
    "wikidata_materials",
]

_DOMAIN_BUILDERS: dict[str, str] = {
    "astronomy": "harmony.datasets.astronomy.build_astronomy_kg",
    "physics": "harmony.datasets.physics.build_physics_kg",
    "materials": "harmony.datasets.materials.build_materials_kg",
    "wikidata_physics": "harmony.datasets.wikidata_physics.build_wikidata_physics_kg",
    "wikidata_materials": "harmony.datasets.wikidata_materials.build_wikidata_materials_kg",
}


def _load_kg(builder_path: str):
    """Import and call a KG builder function by dotted path."""
    import importlib

    module_path, func_name = builder_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, func_name)()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Block A: non-LLM experiments")
    parser.add_argument("--astronomy", type=Path, dest="astronomy", metavar="DIR")
    parser.add_argument("--physics", type=Path, dest="physics", metavar="DIR")
    parser.add_argument("--materials", type=Path, dest="materials", metavar="DIR")
    parser.add_argument("--wikidata-physics", type=Path, dest="wikidata_physics", metavar="DIR")
    parser.add_argument("--wikidata-materials", type=Path, dest="wikidata_materials", metavar="DIR")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/results"),
        help="Root output directory (default: data/results)",
    )
    parser.add_argument(
        "--skip",
        nargs="*",
        default=[],
        choices=[
            "multi_seed",
            "stat_tests",
            "backtest",
            "downstream",
            "frequency",
            "wikidata_ablation",
            "reproducibility",
        ],
        help="Steps to skip (space-separated).",
    )
    return parser


def step_multi_seed(domain_checkpoints: dict[str, Path], output_dir: Path) -> None:
    """Step 1: Multi-seed evaluation."""
    import pandas as pd

    from analysis.metrics_table import compute_metrics_table
    from scripts.run_multi_seed import DEFAULT_SEEDS, aggregate_seed_results

    print("=== Step 1: Multi-seed evaluation ===")
    ms_dir = output_dir / "multi_seed"
    ms_dir.mkdir(parents=True, exist_ok=True)

    seed_dfs: dict[int, pd.DataFrame] = {}
    for i, seed in enumerate(DEFAULT_SEEDS):
        print(f"  [{i + 1}/{len(DEFAULT_SEEDS)}] seed={seed}")
        df = compute_metrics_table(domain_checkpoints, seed=seed)
        seed_dfs[seed] = df
        df.to_csv(ms_dir / f"seed_{seed}.csv")

    summary = aggregate_seed_results(seed_dfs)
    summary.to_csv(ms_dir / "summary.csv")

    meta = {
        "seeds": DEFAULT_SEEDS,
        "n_seeds": len(DEFAULT_SEEDS),
        "domains": list(domain_checkpoints.keys()),
    }
    (ms_dir / "metadata.json").write_text(json.dumps(meta, indent=2))
    print(f"  Written to {ms_dir}/")


def step_statistical_tests(output_dir: Path) -> None:
    """Step 2: Statistical tests (paired bootstrap + Cliff's delta)."""
    import pandas as pd

    from analysis.statistical_tests import cliffs_delta, paired_bootstrap_ci

    print("=== Step 2: Statistical tests ===")
    ms_dir = output_dir / "multi_seed"

    # Load per-seed CSVs
    seed_files = sorted(ms_dir.glob("seed_*.csv"))
    if not seed_files:
        print("  WARNING: No per-seed CSVs found. Skipping.")
        return

    seed_dfs = {int(f.stem.split("_")[1]): pd.read_csv(f, index_col=0) for f in seed_files}
    domains = list(next(iter(seed_dfs.values())).index)

    results: dict[str, dict[str, object]] = {}
    for domain in domains:
        harmony_hits = [float(seed_dfs[s].loc[domain, "harmony_hits10"]) for s in sorted(seed_dfs)]
        distmult_hits = [
            float(seed_dfs[s].loc[domain, "distmult_hits10"]) for s in sorted(seed_dfs)
        ]
        harmony_mrr = [float(seed_dfs[s].loc[domain, "mrr_harmony"]) for s in sorted(seed_dfs)]
        distmult_mrr = [float(seed_dfs[s].loc[domain, "mrr_distmult"]) for s in sorted(seed_dfs)]

        results[domain] = {
            "hits10_bootstrap": paired_bootstrap_ci(harmony_hits, distmult_hits),
            "hits10_cliffs_delta": cliffs_delta(harmony_hits, distmult_hits),
            "mrr_bootstrap": paired_bootstrap_ci(harmony_mrr, distmult_mrr),
            "mrr_cliffs_delta": cliffs_delta(harmony_mrr, distmult_mrr),
        }

    out_path = output_dir / "statistical_tests.json"
    out_path.write_text(json.dumps(results, indent=2, default=float))
    print(f"  Written to {out_path}")


def step_backtesting(domain_checkpoints: dict[str, Path], output_dir: Path) -> None:
    """Step 3: Backtest archive proposals against hidden edges."""
    from analysis.backtesting import backtest_proposals
    from harmony.dataset import KGDataset
    from harmony.map_elites import deserialize_archive
    from harmony.proposals.types import ProposalType
    from harmony.state import load_state

    print("=== Step 3: Backtesting ===")
    results: dict[str, dict[str, object]] = {}

    for domain, cp_dir in domain_checkpoints.items():
        kg = _load_kg(_DOMAIN_BUILDERS[domain])
        dataset = KGDataset.from_kg(kg, seed=42)

        # Extract hidden edges as triples
        hidden_triples = [(e.source, e.target, e.edge_type.name) for e in dataset.hidden_edges]

        # Load archive proposals ranked by fitness
        state = load_state(cp_dir / "checkpoint.json")
        proposals_ranked: list[tuple[str, str, str]] = []
        if state.archive is not None:
            archive = deserialize_archive(state.archive)
            cells = sorted(
                archive.cells.values(),
                key=lambda c: c.fitness_signal,
                reverse=True,
            )
            for cell in cells:
                p = cell.proposal
                if p.proposal_type == ProposalType.ADD_EDGE:
                    proposals_ranked.append((p.source_entity, p.target_entity, p.edge_type))

        result = backtest_proposals(proposals_ranked, hidden_triples)
        results[domain] = {
            "n_proposals": result.n_proposals,
            "n_hidden": result.n_hidden,
            "precision_at_5": result.precision_at_5,
            "precision_at_10": result.precision_at_10,
            "precision_at_20": result.precision_at_20,
            "recall_at_5": result.recall_at_5,
            "recall_at_10": result.recall_at_10,
            "recall_at_20": result.recall_at_20,
            "n_matched": len(result.matched_triples),
        }
        print(f"  {domain}: P@10={result.precision_at_10:.3f} R@10={result.recall_at_10:.3f}")

    out_path = output_dir / "backtesting.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"  Written to {out_path}")


def step_downstream(output_dir: Path) -> None:
    """Step 4: Downstream analysis (placeholder data from pipeline results)."""
    print("=== Step 4: Downstream analysis ===")
    # This step requires per-proposal component deltas from pipeline results.
    # If pipeline result files exist, load them; otherwise write an empty stub.
    out_path = output_dir / "downstream_analysis.json"
    if out_path.exists():
        print(f"  {out_path} already exists, skipping.")
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"status": "requires_pipeline_results"}, indent=2))
    print(f"  Stub written to {out_path}")


def step_frequency(domain_checkpoints: dict[str, Path], output_dir: Path) -> None:
    """Step 5: Frequency dominance analysis."""
    from analysis.frequency_analysis import frequency_dominance_analysis

    print("=== Step 5: Frequency analysis ===")
    results: dict[str, dict[str, float]] = {}
    for domain in domain_checkpoints:
        kg = _load_kg(_DOMAIN_BUILDERS[domain])
        results[domain] = frequency_dominance_analysis(kg)
        print(f"  {domain}: entropy={results[domain]['edge_type_entropy']:.3f}")

    out_path = output_dir / "frequency_analysis.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"  Written to {out_path}")


def step_wikidata_ablation(domain_checkpoints: dict[str, Path], output_dir: Path) -> None:
    """Step 6: Wikidata per-component ablation."""
    from analysis.wikidata_ablation import run_wikidata_ablation

    print("=== Step 6: Wikidata ablation ===")
    kgs = {domain: _load_kg(_DOMAIN_BUILDERS[domain]) for domain in domain_checkpoints}
    df = run_wikidata_ablation(kgs, n_bootstrap=200, seed=42)
    out_path = output_dir / "wikidata_ablation.csv"
    df.to_csv(out_path, index=False)
    print(f"  Written to {out_path}")


def step_reproducibility(output_dir: Path) -> None:
    """Step 7: Reproducibility table."""
    from analysis.reproducibility_table import generate_reproducibility_table

    print("=== Step 7: Reproducibility table ===")
    df = generate_reproducibility_table()
    out_path = output_dir / "reproducibility_table.csv"
    df.to_csv(out_path, index=False)
    print(f"  Written to {out_path}")


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    domain_checkpoints: dict[str, Path] = {}
    for domain in _ALL_DOMAINS:
        d = getattr(args, domain, None)
        if d is not None:
            domain_checkpoints[domain] = Path(d)

    if not domain_checkpoints:
        parser.error("Specify at least one domain checkpoint directory.")

    output_dir = args.output
    output_dir.mkdir(parents=True, exist_ok=True)
    skip = set(args.skip)

    if "multi_seed" not in skip:
        step_multi_seed(domain_checkpoints, output_dir)
    if "stat_tests" not in skip:
        step_statistical_tests(output_dir)
    if "backtest" not in skip:
        step_backtesting(domain_checkpoints, output_dir)
    if "downstream" not in skip:
        step_downstream(output_dir)
    if "frequency" not in skip:
        step_frequency(domain_checkpoints, output_dir)
    if "wikidata_ablation" not in skip:
        step_wikidata_ablation(domain_checkpoints, output_dir)
    if "reproducibility" not in skip:
        step_reproducibility(output_dir)

    print("\n=== Block A complete ===")


if __name__ == "__main__":
    main()
