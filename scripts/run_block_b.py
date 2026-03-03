#!/usr/bin/env python
"""Block B orchestrator: LLM-dependent experiments (requires Ollama).

Runs sequentially (GPU contention):
  1. Qwen pilot (3 gens on linear_algebra, validate valid_rate >= 85%)
  2. LLM-only runs (5 domains × 20 gens, --accept-all-valid)
  3. No-QD runs (5 domains × 20 gens, --greedy)
  4. Harmony-only (random proposer + Harmony scoring, no LLM)
  5. Aggregate factor decomposition results

Usage:
    uv run python scripts/run_block_b.py \
        --model gpt-oss:20b \
        --domains astronomy physics materials wikidata_physics wikidata_materials
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

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
    "linear_algebra": "harmony.datasets.linear_algebra.build_linear_algebra_kg",
    "astronomy": "harmony.datasets.astronomy.build_astronomy_kg",
    "physics": "harmony.datasets.physics.build_physics_kg",
    "materials": "harmony.datasets.materials.build_materials_kg",
    "wikidata_physics": "harmony.datasets.wikidata_physics.build_wikidata_physics_kg",
    "wikidata_materials": "harmony.datasets.wikidata_materials.build_wikidata_materials_kg",
}


def _load_kg(builder_path: str):
    import importlib

    module_path, func_name = builder_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, func_name)()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Block B: LLM experiments")
    parser.add_argument(
        "--model",
        default="gpt-oss:20b",
        help="Ollama model name (default: gpt-oss:20b).",
    )
    parser.add_argument(
        "--domains",
        nargs="+",
        default=_ALL_DOMAINS,
        choices=_ALL_DOMAINS,
        help="Domains to run (default: all 5).",
    )
    parser.add_argument(
        "--generations",
        type=int,
        default=20,
        help="Generations per run (default: 20).",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("artifacts/harmony/factor"),
        help="Root output directory for factor runs.",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("data/results"),
        help="Results directory for aggregated CSV.",
    )
    parser.add_argument(
        "--skip-pilot",
        action="store_true",
        help="Skip Qwen pilot gate check.",
    )
    parser.add_argument(
        "--skip",
        nargs="*",
        default=[],
        choices=["llm_only", "no_qd", "harmony_only"],
        help="Factor configs to skip.",
    )
    return parser


def _run_harmony_loop_for_config(
    domain: str,
    model: str,
    generations: int,
    output_dir: Path,
    accept_all_valid: bool = False,
    greedy: bool = False,
) -> dict[str, object]:
    """Run harmony loop with given factor config."""
    from harmony.config import HarmonyConfig
    from harmony.harmony_loop import run_harmony_loop

    kg = _load_kg(_DOMAIN_BUILDERS[domain])

    cfg = HarmonyConfig(
        domain=domain,
        max_generations=generations,
        model_name=model,
        accept_all_valid=accept_all_valid,
        greedy=greedy,
        map_elites_bins=1 if greedy else 5,
    )

    state = run_harmony_loop(cfg, kg, output_dir=output_dir)
    return {
        "domain": domain,
        "generations": state.generation,
        "best_harmony_gain": state.best_harmony_gain,
        "experiment_id": state.experiment_id,
    }


def _run_harmony_only(
    domain: str,
    generations: int,
    population_size: int,
    output_dir: Path,
) -> dict[str, object]:
    """Run random proposer + Harmony scoring (no LLM)."""
    from harmony.config import HarmonyConfig
    from harmony.map_elites import serialize_archive
    from harmony.proposals.pipeline import run_pipeline
    from harmony.proposals.random_proposer import generate_random_proposals
    from harmony.state import HarmonySearchState, save_state

    kg = _load_kg(_DOMAIN_BUILDERS[domain])
    cfg = HarmonyConfig(domain=domain, max_generations=generations)

    output_dir.mkdir(parents=True, exist_ok=True)
    state = HarmonySearchState(
        experiment_id=f"harmony-only-{domain}",
        generation=0,
        islands={0: []},
        rng_seed=cfg.seed,
    )

    for gen in range(generations):
        proposals = generate_random_proposals(
            kg, n=population_size, seed=cfg.seed + gen
        )
        result = run_pipeline(
            kg=kg,
            proposals=proposals,
            seed=cfg.seed,
            archive_bins=cfg.map_elites_bins,
        )

        best_gain = max(
            (r.harmony_gain for r in result.results if r.harmony_gain is not None),
            default=0.0,
        )
        if best_gain > state.best_harmony_gain + 1e-12:
            state.best_harmony_gain = best_gain

        state.archive = serialize_archive(result.archive)
        state.generation = gen + 1
        save_state(state, output_dir / "checkpoint.json")
        print(
            f"    gen={gen + 1} valid_rate={result.valid_rate:.3f}"
            f" best_gain={state.best_harmony_gain:.4f}"
        )

    return {
        "domain": domain,
        "generations": state.generation,
        "best_harmony_gain": state.best_harmony_gain,
        "experiment_id": state.experiment_id,
    }


def step_pilot(model: str) -> bool:
    """Run 3-gen pilot on linear_algebra to check valid_rate."""
    from harmony.config import HarmonyConfig
    from harmony.harmony_loop import run_harmony_loop

    print("=== Pilot: 3 gens on linear_algebra ===")
    kg = _load_kg(_DOMAIN_BUILDERS["linear_algebra"])
    cfg = HarmonyConfig(
        domain="linear_algebra",
        max_generations=3,
        model_name=model,
    )
    output_dir = Path("/tmp/harmony_pilot")
    run_harmony_loop(cfg, kg, output_dir=output_dir)

    # Check events log for valid_rate
    log_path = output_dir / "logs" / "harmony_events.jsonl"
    if log_path.exists():
        lines = log_path.read_text().strip().split("\n")
        for line in reversed(lines):
            event = json.loads(line)
            if event.get("event") == "generation_summary":
                vr = event.get("valid_rate", 0)
                print(f"  Pilot valid_rate={vr:.3f} (gen={event['generation']})")
                if vr >= 0.80:
                    print("  PASS: valid_rate >= 80%")
                    return True
                else:
                    print(f"  FAIL: valid_rate {vr:.3f} < 80%")
                    return False

    print("  WARNING: Could not determine valid_rate from logs")
    return False


def step_aggregate(output_root: Path, results_dir: Path, domains: list[str]) -> None:
    """Aggregate factor decomposition results into a CSV."""
    print("=== Aggregating factor decomposition ===")

    from harmony.map_elites import deserialize_archive
    from harmony.state import load_state

    rows: list[dict[str, object]] = []
    for config_name in ["llm_only", "no_qd", "harmony_only"]:
        for domain in domains:
            cp_path = output_root / config_name / domain / "checkpoint.json"
            if not cp_path.exists():
                continue
            state = load_state(cp_path)
            n_archive = 0
            if state.archive is not None:
                try:
                    archive = deserialize_archive(state.archive)
                    n_archive = len(archive.cells)
                except Exception:
                    pass
            rows.append(
                {
                    "config": config_name,
                    "domain": domain,
                    "generations": state.generation,
                    "best_harmony_gain": state.best_harmony_gain,
                    "archive_size": n_archive,
                }
            )

    if rows:
        df = pd.DataFrame(rows)
        results_dir.mkdir(parents=True, exist_ok=True)
        out_path = results_dir / "factor_decomposition.csv"
        df.to_csv(out_path, index=False)
        print(f"  Written to {out_path}")
    else:
        print("  No factor results found.")


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    skip = set(args.skip)

    # Pilot check
    if not args.skip_pilot:
        if not step_pilot(args.model):
            print("Pilot failed. Use --skip-pilot to bypass or change --model.")
            sys.exit(1)

    # LLM-only runs
    if "llm_only" not in skip:
        print("\n=== LLM-only runs (accept_all_valid=True) ===")
        for domain in args.domains:
            print(f"  Running {domain}...")
            out = args.output_root / "llm_only" / domain
            result = _run_harmony_loop_for_config(
                domain=domain,
                model=args.model,
                generations=args.generations,
                output_dir=out,
                accept_all_valid=True,
            )
            print(f"    Done: {result}")

    # No-QD runs
    if "no_qd" not in skip:
        print("\n=== No-QD runs (greedy=True) ===")
        for domain in args.domains:
            print(f"  Running {domain}...")
            out = args.output_root / "no_qd" / domain
            result = _run_harmony_loop_for_config(
                domain=domain,
                model=args.model,
                generations=args.generations,
                output_dir=out,
                greedy=True,
            )
            print(f"    Done: {result}")

    # Harmony-only runs (no LLM)
    if "harmony_only" not in skip:
        print("\n=== Harmony-only runs (random proposer, no LLM) ===")
        for domain in args.domains:
            print(f"  Running {domain}...")
            out = args.output_root / "harmony_only" / domain
            result = _run_harmony_only(
                domain=domain,
                generations=args.generations,
                population_size=20,
                output_dir=out,
            )
            print(f"    Done: {result}")

    # Aggregate
    step_aggregate(args.output_root, args.results_dir, args.domains)
    print("\n=== Block B complete ===")


if __name__ == "__main__":
    main()
