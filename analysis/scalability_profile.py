"""Scalability profiler: measure per-component wall time as KG size scales.

Measures Harmony scoring, DistMult training, and component computation
across KG sizes from small (~20 entities) to medium (~250 entities).
Reports timing data suitable for inclusion in the paper appendix.

Usage:
    uv run python analysis/scalability_profile.py
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field


@dataclass
class TimingResult:
    """Wall-clock timing for a single measurement."""

    component: str
    n_entities: int
    n_edges: int
    seconds: float


@dataclass
class ScalabilityReport:
    """Collection of timing results across KG scales."""

    results: list[TimingResult] = field(default_factory=list)

    def add(self, component: str, n_entities: int, n_edges: int, seconds: float) -> None:
        self.results.append(TimingResult(component, n_entities, n_edges, seconds))

    def summary(self) -> str:
        lines = [
            f"{'Component':<20} {'Entities':>8} {'Edges':>8} {'Time (s)':>10}",
            "-" * 50,
        ]
        for r in sorted(self.results, key=lambda x: (x.component, x.n_entities)):
            lines.append(
                f"{r.component:<20} {r.n_entities:>8} {r.n_edges:>8} {r.seconds:>10.3f}"
            )
        return "\n".join(lines)


def _time_component(fn: object, *args: object, **kwargs: object) -> float:
    """Time a callable, returning wall-clock seconds."""
    start = time.perf_counter()
    fn(*args, **kwargs)  # type: ignore[operator]
    return time.perf_counter() - start


def profile_harmony_components(seed: int = 42) -> ScalabilityReport:
    """Profile Harmony component computation across KG scales.

    Tests small KGs (existing domains) and medium KGs (Wikidata domains).
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
            elapsed = _time_component(comp_fn, kg)
            report.add(comp_name, n_ent, n_edg, elapsed)

        # Generativity needs enough edges for train/test split
        if kg.num_edges >= 10:
            elapsed = _time_component(
                generativity, kg, seed=seed, dim=50, n_epochs=50
            )
            report.add("generativity", n_ent, n_edg, elapsed)

    return report


def main() -> None:
    print("Scalability Profile: Harmony Components")
    print("=" * 50)
    report = profile_harmony_components()
    print(report.summary())
    print()

    # Identify bottleneck
    if report.results:
        slowest = max(report.results, key=lambda r: r.seconds)
        print(f"Bottleneck: {slowest.component} at {slowest.n_entities} entities "
              f"({slowest.seconds:.3f}s)")


if __name__ == "__main__":
    main()
