"""Scalability profiler: measure per-component wall time as KG size scales.

Measures Harmony scoring, DistMult training, and component computation
across KG sizes from small (~20 entities) to medium (~250 entities).
Reports timing data suitable for inclusion in the paper appendix.

Usage:
    uv run python analysis/scalability_profile.py [--repeats N] [--output path.json]
"""

from __future__ import annotations

import json
import platform
import statistics
import sys
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class TimingResult:
    """Wall-clock timing for a single measurement."""

    component: str
    n_entities: int
    n_edges: int
    median_seconds: float
    std_seconds: float
    repeats: int
    all_seconds: list[float] = field(default_factory=list)


@dataclass
class ScalabilityReport:
    """Collection of timing results across KG scales."""

    results: list[TimingResult] = field(default_factory=list)

    def add(
        self,
        component: str,
        n_entities: int,
        n_edges: int,
        timings: list[float],
    ) -> None:
        median = statistics.median(timings)
        std = statistics.stdev(timings) if len(timings) > 1 else 0.0
        self.results.append(
            TimingResult(component, n_entities, n_edges, median, std, len(timings), timings)
        )

    def summary(self) -> str:
        lines = [
            f"{'Component':<20} {'Entities':>8} {'Edges':>8} "
            f"{'Median(s)':>10} {'Std(s)':>8} {'N':>3}",
            "-" * 62,
        ]
        for r in sorted(self.results, key=lambda x: (x.component, x.n_entities)):
            lines.append(
                f"{r.component:<20} {r.n_entities:>8} {r.n_edges:>8} "
                f"{r.median_seconds:>10.4f} {r.std_seconds:>8.4f} {r.repeats:>3}"
            )
        return "\n".join(lines)

    def to_json(self) -> str:
        env = {
            "python_version": sys.version,
            "platform": platform.platform(),
            "processor": platform.processor(),
        }
        return json.dumps(
            {"environment": env, "results": [asdict(r) for r in self.results]},
            indent=2,
        )


def _time_component(fn: Callable[..., Any], *args: Any, **kwargs: Any) -> float:
    """Time a callable, returning wall-clock seconds."""
    start = time.perf_counter()
    fn(*args, **kwargs)
    return time.perf_counter() - start


def profile_harmony_components(
    seed: int = 42,
    repeats: int = 3,
) -> ScalabilityReport:
    """Profile Harmony component computation across KG scales.

    Tests small KGs (existing domains) and medium KGs (Wikidata domains).
    First run is a warmup (discarded).
    """
    from harmony.datasets.linear_algebra import build_linear_algebra_kg
    from harmony.datasets.physics import build_physics_kg
    from harmony.datasets.wikidata_materials import build_wikidata_materials_kg
    from harmony.datasets.wikidata_physics import build_wikidata_physics_kg
    from harmony.metric.coherence import coherence
    from harmony.metric.compressibility import compressibility
    from harmony.metric.generativity import generativity
    from harmony.metric.symmetry import symmetry

    report = ScalabilityReport()

    datasets = [
        ("linear_algebra", build_linear_algebra_kg),
        ("physics_small", build_physics_kg),
        ("wikidata_physics", build_wikidata_physics_kg),
        ("wikidata_materials", build_wikidata_materials_kg),
    ]

    components = [
        ("compressibility", compressibility),
        ("coherence", coherence),
        ("symmetry", symmetry),
    ]

    for _domain_name, builder in datasets:
        kg = builder()
        n_ent = kg.num_entities
        n_edg = kg.num_edges

        for comp_name, comp_fn in components:
            # Warmup run (discarded)
            _time_component(comp_fn, kg)
            timings = [_time_component(comp_fn, kg) for _ in range(repeats)]
            report.add(comp_name, n_ent, n_edg, timings)

        # Generativity needs enough edges for train/test split
        if kg.num_edges >= 10:
            _time_component(generativity, kg, seed=seed, dim=50, n_epochs=50)
            timings = [
                _time_component(generativity, kg, seed=seed, dim=50, n_epochs=50)
                for _ in range(repeats)
            ]
            report.add("generativity", n_ent, n_edg, timings)

    return report


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Scalability profiler")
    parser.add_argument("--repeats", type=int, default=3, help="Number of timed runs (default 3)")
    parser.add_argument("--output", type=str, default=None, help="JSON output path")
    args = parser.parse_args()

    print("Scalability Profile: Harmony Components")
    print("=" * 62)
    report = profile_harmony_components(repeats=args.repeats)
    print(report.summary())
    print()

    # Identify bottleneck
    if report.results:
        slowest = max(report.results, key=lambda r: r.median_seconds)
        print(
            f"Bottleneck: {slowest.component} at {slowest.n_entities} entities "
            f"({slowest.median_seconds:.3f}s median)"
        )

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(report.to_json())
        print(f"\nJSON output written to {out_path}")


if __name__ == "__main__":
    main()
