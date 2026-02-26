"""Harmony CLI — thin entry point: argparse → run_harmony_loop().

Usage:
    harmony --domain linear_algebra --generations 10 --output-dir /tmp/harmony
    python -m harmony.cli --domain periodic_table --model mistral
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from harmony.config import HarmonyConfig
from harmony.datasets.astronomy import build_astronomy_kg
from harmony.datasets.linear_algebra import build_linear_algebra_kg
from harmony.datasets.materials import build_materials_kg
from harmony.datasets.periodic_table import build_periodic_table_kg
from harmony.datasets.physics import build_physics_kg
from harmony.harmony_loop import run_harmony_loop

_DOMAIN_BUILDERS = {
    "linear_algebra": build_linear_algebra_kg,
    "periodic_table": build_periodic_table_kg,
    "astronomy": build_astronomy_kg,
    "physics": build_physics_kg,
    "materials": build_materials_kg,
}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="harmony",
        description="Harmony-driven theory discovery via island search on a knowledge graph.",
    )
    parser.add_argument(
        "--domain",
        choices=list(_DOMAIN_BUILDERS),
        default="linear_algebra",
        help="KG domain to run discovery on (default: linear_algebra).",
    )
    parser.add_argument(
        "--generations",
        type=int,
        default=50,
        metavar="N",
        help="Maximum number of generations (default: 50).",
    )
    parser.add_argument(
        "--model",
        default="mistral",
        metavar="NAME",
        help="Ollama model name (default: mistral).",
    )
    parser.add_argument(
        "--ollama-url",
        default="http://localhost:11434/api/generate",
        metavar="URL",
        help="Ollama generate endpoint (default: http://localhost:11434/api/generate).",
    )
    parser.add_argument(
        "--output-dir",
        default="harmony_output",
        metavar="DIR",
        help="Output directory for checkpoints and logs (default: harmony_output).",
    )
    parser.add_argument(
        "--resume",
        default=None,
        metavar="PATH",
        help="Path to a checkpoint.json to resume a previous run.",
    )
    parser.add_argument(
        "--allow-remote",
        action="store_true",
        default=False,
        help="Allow Ollama on a remote host (default: localhost only).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42).",
    )
    parser.add_argument(
        "--population-size",
        type=int,
        default=5,
        metavar="N",
        help="Number of LLM proposals per island per generation (default: 5).",
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    kg_builder = _DOMAIN_BUILDERS[args.domain]
    kg = kg_builder()

    cfg = HarmonyConfig(
        domain=args.domain,
        max_generations=args.generations,
        model_name=args.model,
        ollama_url=args.ollama_url,
        allow_remote_ollama=args.allow_remote,
        seed=args.seed,
        population_size=args.population_size,
    )

    output_dir = Path(args.output_dir)
    state = run_harmony_loop(cfg, kg, output_dir=output_dir, resume=args.resume)

    print(
        f"Done. generation={state.generation} "
        f"best_gain={state.best_harmony_gain:.4f} "
        f"experiment_id={state.experiment_id}"
    )
    sys.exit(0)


if __name__ == "__main__":
    main()
