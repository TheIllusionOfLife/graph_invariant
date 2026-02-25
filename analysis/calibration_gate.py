#!/usr/bin/env python
"""Calibration gate for the Harmony composite metric (PR 2b).

Pre-registered gate (must pass before running expensive discovery experiments):
  - Full metric outperforms frequency baseline by ≥10% on linear_algebra AND periodic_table
  - Bootstrap 95% CI lower bound > frequency mean on both domains
  - All 6 pre-registered weight combinations show consistent direction (harmony > frequency)

Usage:
    uv run python analysis/calibration_gate.py

Writes results to analysis/calibration_gate.md and prints GATE PASSED ✓ or GATE FAILED ✗.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure src/ is on the path when run from the repo root
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from harmony.datasets.linear_algebra import build_linear_algebra_kg
from harmony.datasets.periodic_table import build_periodic_table_kg
from harmony.metric.ablation import run_ablation
from harmony.metric.baselines import (
    baseline_distmult_alone,
    baseline_frequency,
    baseline_random,
)
from harmony.metric.harmony import harmony_score
from harmony.types import KnowledgeGraph

# ── Configuration ──────────────────────────────────────────────────────
N_BOOTSTRAP = 200
SEED = 42
IMPROVEMENT_THRESHOLD = 0.10  # 10% improvement required
MIN_PASSING_DOMAINS = 2  # gate requires all 2 domains to pass

# Pre-registered weight grid: α ∈ {0.3, 0.5, 0.7} × β ∈ {0.1, 0.3}; γ=δ=0.25
WEIGHT_GRID: list[tuple[float, float, float, float]] = [
    (alpha, beta, 0.25, 0.25) for alpha in (0.3, 0.5, 0.7) for beta in (0.1, 0.3)
]

OUTPUT_MD = Path(__file__).parent / "calibration_gate.md"


def _improvement(harmony_mean: float, freq_mean: float) -> float:
    """Relative improvement of harmony over frequency baseline.

    Returns float("inf") when freq_mean <= 0 and harmony_mean > 0 (any positive
    score is an infinite improvement over a zero baseline), 0.0 when both are
    zero or negative, rather than silently inflating via a near-zero clamp.
    """
    if freq_mean <= 0.0:
        return float("inf") if harmony_mean > 0.0 else 0.0
    return (harmony_mean - freq_mean) / freq_mean


def _gate_passes_domain(
    harmony_mean: float,
    ci95_half_width: float,
    freq_mean: float,
) -> bool:
    """Gate logic (pre-registered):
    improvement ≥ 10% AND ci_lower > freq_mean.
    """
    imp = _improvement(harmony_mean, freq_mean)
    ci_lower = harmony_mean - ci95_half_width
    return imp >= IMPROVEMENT_THRESHOLD and ci_lower > freq_mean


def main() -> None:
    print("=" * 60)
    print("Harmony Metric Calibration Gate")
    print("=" * 60)

    # ── Load KGs ──────────────────────────────────────────────────
    print("\n[1/4] Loading knowledge graphs...")
    kgs: dict[str, KnowledgeGraph] = {
        "linear_algebra": build_linear_algebra_kg(),
        "periodic_table": build_periodic_table_kg(),
    }
    for name, kg in kgs.items():
        print(f"  {name}: {kg.num_entities} entities, {kg.num_edges} edges")

    # ── Compute baselines ─────────────────────────────────────────
    print("\n[2/4] Computing baselines...")
    baselines: dict[str, dict[str, float]] = {}
    for name, kg in kgs.items():
        rand = baseline_random(kg, seed=SEED)
        freq = baseline_frequency(kg, seed=SEED)
        dmult = baseline_distmult_alone(kg, seed=SEED)
        baselines[name] = {"random": rand, "frequency": freq, "distmult": dmult}
        print(f"  {name}: random={rand:.4f}  frequency={freq:.4f}  distmult={dmult:.4f}")

    # ── Run ablation (bootstrap CIs for gate check) ────────────────
    print(f"\n[3/4] Running ablation (n_bootstrap={N_BOOTSTRAP}, this may take a few minutes)...")
    ablations: dict[str, list] = {}
    for name, kg in kgs.items():
        print(f"  Ablating {name}...", end="", flush=True)
        rows = run_ablation(kg, n_bootstrap=N_BOOTSTRAP, seed=SEED)
        ablations[name] = rows
        full = next(r for r in rows if r.component == "full")
        print(f" done. full mean={full.mean:.4f} ±{full.ci95_half_width:.4f}")

    # ── Weight grid consistency check ─────────────────────────────
    print("\n[4/4] Weight grid consistency check (6 configurations)...")
    grid_results: dict[str, list[dict]] = {name: [] for name in kgs}
    for kg_name, kg in kgs.items():
        for alpha, beta, gamma, delta in WEIGHT_GRID:
            h = harmony_score(kg, alpha=alpha, beta=beta, gamma=gamma, delta=delta, seed=SEED)
            freq = baselines[kg_name]["frequency"]
            imp = _improvement(h, freq)
            grid_results[kg_name].append(
                {
                    "alpha": alpha,
                    "beta": beta,
                    "gamma": gamma,
                    "delta": delta,
                    "harmony": h,
                    "freq": freq,
                    "improvement": imp,
                    "beats_freq": h > freq,
                }
            )
        all_beat = all(r["beats_freq"] for r in grid_results[kg_name])
        print(f"  {kg_name}: all 6 configs beat frequency? {all_beat}")

    # ── Gate evaluation ───────────────────────────────────────────
    domain_pass: dict[str, bool] = {}
    for name in kgs:
        full_row = next(r for r in ablations[name] if r.component == "full")
        freq_mean = baselines[name]["frequency"]
        passes = _gate_passes_domain(full_row.mean, full_row.ci95_half_width, freq_mean)
        domain_pass[name] = passes
        imp = _improvement(full_row.mean, freq_mean)
        ci_lower = full_row.mean - full_row.ci95_half_width
        print(
            f"\n  {name}:"
            f"\n    harmony_mean={full_row.mean:.4f}  ci95_hw={full_row.ci95_half_width:.4f}"
            f"\n    freq_mean={freq_mean:.4f}  improvement={imp:.1%}"
            f"\n    ci_lower={ci_lower:.4f}  ci_lower>freq? {ci_lower > freq_mean}"
            f"\n    → {'PASS' if passes else 'FAIL'}"
        )

    n_passing = sum(domain_pass.values())
    gate_overall = n_passing >= MIN_PASSING_DOMAINS

    # Weight grid consistency
    grid_consistent = all(all(r["beats_freq"] for r in grid_results[name]) for name in kgs)

    final_pass = gate_overall and grid_consistent

    # ── Write markdown ────────────────────────────────────────────
    _write_markdown(
        kgs, baselines, ablations, grid_results, domain_pass, grid_consistent, final_pass
    )
    print(f"\nResults written to {OUTPUT_MD}")

    # ── Final verdict ─────────────────────────────────────────────
    print("\n" + "=" * 60)
    if final_pass:
        print("GATE PASSED ✓")
        print("Full metric outperforms frequency baseline by ≥10% on both domains")
        print(
            "with bootstrap 95% CI excluding zero gap, and consistent across all 6 weight configs."
        )
    else:
        print("GATE FAILED ✗")
        if not gate_overall:
            print(
                f"Only {n_passing}/{MIN_PASSING_DOMAINS} domains passed the improvement+CI check."
            )
        if not grid_consistent:
            print("Not all 6 pre-registered weight configurations beat the frequency baseline.")
    print("=" * 60)

    sys.exit(0 if final_pass else 1)


def _write_markdown(
    kgs: dict,
    baselines: dict,
    ablations: dict,
    grid_results: dict,
    domain_pass: dict,
    grid_consistent: bool,
    final_pass: bool,
) -> None:
    lines: list[str] = []

    status = "✅ PASSED" if final_pass else "❌ FAILED"
    lines.append(f"# Harmony Metric Calibration Gate — {status}")
    lines.append("")
    lines.append("Pre-registered gate: ≥10% improvement over frequency baseline on ≥2 domains,")
    lines.append("bootstrap 95% CI lower bound > frequency mean on both domains,")
    lines.append("and all 6 weight configurations show consistent direction.")
    lines.append("")

    # Dataset summary
    lines.append("## Datasets")
    lines.append("")
    lines.append("| Domain | Entities | Edges |")
    lines.append("|--------|----------|-------|")
    for name, kg in kgs.items():
        lines.append(f"| {name} | {kg.num_entities} | {kg.num_edges} |")
    lines.append("")

    # Baseline table
    lines.append("## Baselines")
    lines.append("")
    lines.append("| Domain | Random | Frequency | DistMult alone |")
    lines.append("|--------|--------|-----------|---------------|")
    for name, b in baselines.items():
        lines.append(f"| {name} | {b['random']:.4f} | {b['frequency']:.4f} | {b['distmult']:.4f} |")
    lines.append("")

    # Gate check table
    lines.append(f"## Gate Check (n_bootstrap={N_BOOTSTRAP}, seed={SEED})")
    lines.append("")
    hdr = "| Domain | H.mean | CI95 hw | CI lower | Freq mean | Improvement | CI>Freq | Pass? |"
    sep = "|--------|--------|---------|----------|-----------|-------------|---------|-------|"
    lines.append(hdr)
    lines.append(sep)
    for name in kgs:
        full_row = next(r for r in ablations[name] if r.component == "full")
        freq_mean = baselines[name]["frequency"]
        imp = _improvement(full_row.mean, freq_mean)
        ci_lower = full_row.mean - full_row.ci95_half_width
        ci_ok = "✓" if ci_lower > freq_mean else "✗"
        imp_ok = "✓" if imp >= IMPROVEMENT_THRESHOLD else "✗"
        pass_str = "✅" if domain_pass[name] else "❌"
        lines.append(
            f"| {name} | {full_row.mean:.4f} | {full_row.ci95_half_width:.4f} | "
            f"{ci_lower:.4f} | {freq_mean:.4f} | {imp:.1%} {imp_ok} | {ci_ok} | {pass_str} |"
        )
    lines.append("")

    # Ablation tables
    lines.append("## Ablation Results")
    lines.append("")
    for name in kgs:
        lines.append(f"### {name}")
        lines.append("")
        lines.append("| Component | Mean | Std | CI95 half-width | Δ vs full |")
        lines.append("|-----------|------|-----|-----------------|-----------|")
        for row in ablations[name]:
            lines.append(
                f"| {row.component} | {row.mean:.4f} | {row.std:.4f} | "
                f"{row.ci95_half_width:.4f} | {row.delta_vs_full:+.4f} |"
            )
        lines.append("")

    # Weight grid
    lines.append("## Weight Grid Consistency")
    lines.append("")
    beat_str = "✅" if grid_consistent else "❌"
    lines.append(f"All 6 configurations beat frequency baseline: {beat_str}")
    lines.append("")
    for name in kgs:
        lines.append(f"### {name}")
        lines.append("")
        lines.append("| α | β | γ | δ | Harmony | Frequency | Improvement | Beats freq? |")
        lines.append("|---|---|---|---|---------|-----------|-------------|-------------|")
        for r in grid_results[name]:
            beat = "✓" if r["beats_freq"] else "✗"
            lines.append(
                f"| {r['alpha']} | {r['beta']} | {r['gamma']} | {r['delta']} | "
                f"{r['harmony']:.4f} | {r['freq']:.4f} | {r['improvement']:.1%} | {beat} |"
            )
        lines.append("")

    # Final verdict
    lines.append("## Final Verdict")
    lines.append("")
    lines.append(f"**{status}**")
    lines.append("")

    OUTPUT_MD.write_text("\n".join(lines))


if __name__ == "__main__":
    main()
