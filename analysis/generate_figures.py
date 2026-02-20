"""Generate publication-quality figures from analysis results.

Usage:
    uv run python analysis/generate_figures.py \
        --data analysis/results/ --output paper/figures/

Reads figure_data.json produced by analyze_experiments.py and generates
PDF figures suitable for a NeurIPS-format paper.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # Non-interactive backend for CI/headless (must precede pyplot import)

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# ── Style setup ──────────────────────────────────────────────────────

sns.set_theme(style="whitegrid", font_scale=1.1)
plt.rcParams.update(
    {
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "font.family": "serif",
        "font.serif": ["DejaVu Serif", "Times New Roman", "Computer Modern Roman"],
        "text.usetex": False,
    }
)


# ── Helpers ──────────────────────────────────────────────────────────


def _annotate_bars(bars, ax) -> None:
    """Add value labels above each bar in a bar chart."""
    for bar in bars:
        height = bar.get_height()
        if height > 0:
            ax.annotate(
                f"{height:.2f}",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=7,
            )


def _experiment_items(data: dict):
    """Iterate only true experiment entries (exclude metadata buckets)."""
    for name, info in data.items():
        if name.startswith("__"):
            continue
        if not isinstance(info, dict):
            continue
        yield name, info


# ── Figure generators ────────────────────────────────────────────────


def plot_convergence(data: dict, output_path: Path) -> None:
    """Plot score by generation, faceted by experiment."""
    experiments_with_convergence = {
        name: info
        for name, info in _experiment_items(data)
        if info.get("convergence", {}).get("generations")
    }

    if not experiments_with_convergence:
        print("  No convergence data available, skipping convergence.pdf")
        return

    n = len(experiments_with_convergence)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4), squeeze=False)

    for idx, (name, info) in enumerate(experiments_with_convergence.items()):
        ax = axes[0, idx]
        conv = info["convergence"]
        gens = conv["generations"]
        scores = conv["best_scores"]

        ax.plot(gens, scores, "o-", markersize=3, linewidth=1.5, color="tab:blue")
        ax.set_xlabel("Generation")
        ax.set_ylabel("Best Score")
        ax.set_title(name.replace("_", " ").title(), fontsize=10)

        coverage = conv.get("map_elites_coverage", [])
        if coverage and len(coverage) == len(gens):
            ax2 = ax.twinx()
            ax2.plot(
                gens,
                coverage,
                "s--",
                markersize=2,
                linewidth=1,
                color="tab:orange",
                alpha=0.7,
            )
            ax2.set_ylabel("Archive Coverage", color="tab:orange", fontsize=8)
            ax2.tick_params(axis="y", labelcolor="tab:orange", labelsize=7)

    fig.suptitle("Convergence by Experiment", fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    print(f"  Saved {output_path}")


def plot_map_elites_coverage(data: dict, output_path: Path) -> None:
    """Plot MAP-Elites archive coverage summary."""
    me_data = data.get("experiment_map_elites_aspl", {})
    conv = me_data.get("convergence", {}) if isinstance(me_data, dict) else {}
    coverage = conv.get("map_elites_coverage", [])

    if not coverage:
        print("  No MAP-Elites data available, skipping map_elites_heatmap.pdf")
        return

    fig, ax = plt.subplots(figsize=(6, 4))

    gens = conv.get("generations", list(range(len(coverage))))
    ax.bar(gens, coverage, color="tab:green", alpha=0.8, edgecolor="white")
    ax.set_xlabel("Generation")
    ax.set_ylabel("Archive Coverage (cells)")
    ax.set_title("MAP-Elites Archive Growth")

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    print(f"  Saved {output_path}")


def plot_baseline_comparison(data: dict, output_path: Path) -> None:
    """Grouped bar chart: LLM candidates vs baselines."""
    methods: list[str] = []
    val_scores: list[float] = []
    test_scores: list[float] = []
    baselines_collected = False

    for name, info in _experiment_items(data):
        val_s = info.get("val_spearman")
        test_s = info.get("test_spearman")
        if val_s is None:
            continue

        methods.append(f"LLM ({name.replace('_', ' ')})")
        val_scores.append(val_s)
        test_scores.append(test_s if test_s is not None else 0.0)

        if not baselines_collected:
            added_any_baseline = False
            baselines = info.get("baselines", {})
            stat = baselines.get("stat_baselines", {}) if isinstance(baselines, dict) else {}
            if isinstance(stat, dict):
                for bl_name, bl_data in stat.items():
                    if not isinstance(bl_data, dict) or bl_data.get("status") != "ok":
                        continue
                    bl_val = bl_data.get("val_metrics", {})
                    bl_test = bl_data.get("test_metrics", {})
                    bl_val_s = bl_val.get("spearman") if isinstance(bl_val, dict) else None
                    bl_test_s = bl_test.get("spearman") if isinstance(bl_test, dict) else None
                    label = bl_name.replace("_", " ").title()
                    if label not in methods:
                        methods.append(label)
                        val_scores.append(bl_val_s if bl_val_s is not None else 0.0)
                        test_scores.append(bl_test_s if bl_test_s is not None else 0.0)
                        added_any_baseline = True

            pysr = baselines.get("pysr_baseline", {}) if isinstance(baselines, dict) else {}
            if isinstance(pysr, dict) and pysr.get("status") == "ok":
                pysr_val = pysr.get("val_metrics", {})
                pysr_test = pysr.get("test_metrics", {})
                if "PySR" not in methods:
                    methods.append("PySR")
                    val_scores.append(
                        pysr_val.get("spearman", 0.0) if isinstance(pysr_val, dict) else 0.0
                    )
                    test_scores.append(
                        pysr_test.get("spearman", 0.0) if isinstance(pysr_test, dict) else 0.0
                    )
                    added_any_baseline = True
            if added_any_baseline:
                baselines_collected = True

    if not methods:
        print("  No baseline data available, skipping baseline_comparison.pdf")
        return

    x = np.arange(len(methods))
    width = 0.35

    fig, ax = plt.subplots(figsize=(max(8, len(methods) * 1.5), 5))
    bars1 = ax.bar(x - width / 2, val_scores, width, label="Validation", color="tab:blue")
    bars2 = ax.bar(x + width / 2, test_scores, width, label="Test", color="tab:orange")

    ax.set_xlabel("Method")
    ax.set_ylabel("Spearman Correlation")
    ax.set_title("Method Comparison")
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=30, ha="right", fontsize=8)
    ax.legend()
    ax.set_ylim(0, 1.05)

    _annotate_bars(bars1, ax)
    _annotate_bars(bars2, ax)

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    print(f"  Saved {output_path}")


def plot_ood_generalization(data: dict, output_path: Path) -> None:
    """Grouped bar chart of OOD Spearman by category."""
    categories = ["large_random", "extreme_params", "special_topology"]
    experiments_with_ood = {
        name: info
        for name, info in _experiment_items(data)
        if any(info.get("ood", {}).get(c) for c in categories)
    }

    if not experiments_with_ood:
        print("  No OOD data available, skipping ood_generalization.pdf")
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(categories))
    width = 0.8 / max(len(experiments_with_ood), 1)

    for idx, (name, info) in enumerate(experiments_with_ood.items()):
        ood = info.get("ood", {})
        scores = []
        for cat in categories:
            cat_data = ood.get(cat, {}) if isinstance(ood, dict) else {}
            s = cat_data.get("spearman") if isinstance(cat_data, dict) else None
            scores.append(s if s is not None else 0.0)
        offset = (idx - len(experiments_with_ood) / 2 + 0.5) * width
        label = name.replace("_", " ").title()
        ax.bar(x + offset, scores, width, label=label, alpha=0.85)

    ax.set_xlabel("OOD Category")
    ax.set_ylabel("Spearman Correlation")
    ax.set_title("Out-of-Distribution Generalization")
    ax.set_xticks(x)
    ax.set_xticklabels([c.replace("_", " ").title() for c in categories])
    ax.legend(fontsize=8)
    ax.set_ylim(0, 1.05)

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    print(f"  Saved {output_path}")


def plot_benchmark_boxplot(data: dict, output_path: Path) -> None:
    """Box plot of Val/Test Spearman across benchmark seeds."""
    benchmark_data = {
        name: info for name, info in _experiment_items(data) if name.startswith("benchmark/")
    }

    if not benchmark_data:
        for _name, info in _experiment_items(data):
            summary = info.get("summary", info)
            runs = summary.get("runs", []) if isinstance(summary, dict) else []
            if runs:
                val_scores = [r["val_spearman"] for r in runs if r.get("val_spearman") is not None]
                test_scores = [
                    r["test_spearman"] for r in runs if r.get("test_spearman") is not None
                ]
                if val_scores or test_scores:
                    fig, ax = plt.subplots(figsize=(6, 4))
                    plot_data = []
                    labels = []
                    if val_scores:
                        plot_data.append(val_scores)
                        labels.append("Validation")
                    if test_scores:
                        plot_data.append(test_scores)
                        labels.append("Test")
                    ax.boxplot(plot_data, tick_labels=labels)
                    ax.set_ylabel("Spearman Correlation")
                    ax.set_title("Multi-Seed Benchmark Consistency")
                    fig.tight_layout()
                    fig.savefig(output_path)
                    plt.close(fig)
                    print(f"  Saved {output_path}")
                    return

        print("  No benchmark data available, skipping benchmark_boxplot.pdf")
        return

    val_scores = []
    test_scores = []
    for info in benchmark_data.values():
        runs = info.get("runs", [])
        if runs:
            for run in runs:
                if run.get("val_spearman") is not None:
                    val_scores.append(run["val_spearman"])
                if run.get("test_spearman") is not None:
                    test_scores.append(run["test_spearman"])

    if not val_scores and not test_scores:
        print("  No benchmark scores available, skipping benchmark_boxplot.pdf")
        return

    fig, ax = plt.subplots(figsize=(6, 4))
    plot_data = []
    labels = []
    if val_scores:
        plot_data.append(val_scores)
        labels.append("Validation")
    if test_scores:
        plot_data.append(test_scores)
        labels.append("Test")
    ax.boxplot(plot_data, tick_labels=labels)
    ax.set_ylabel("Spearman Correlation")
    ax.set_title("Multi-Seed Benchmark Consistency")

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    print(f"  Saved {output_path}")


def plot_seed_ci_comparison(data: dict, output_path: Path) -> None:
    """Plot seed-aggregated mean test Spearman with 95% CI."""
    aggregates = data.get("__aggregates__", {})
    if not isinstance(aggregates, dict) or not aggregates:
        print("  No aggregate data available, skipping seed_ci_comparison.pdf")
        return

    labels = []
    means = []
    errors = []
    for name, payload in sorted(aggregates.items()):
        test = payload.get("test_spearman", {}) if isinstance(payload, dict) else {}
        mean = test.get("mean") if isinstance(test, dict) else None
        ci = test.get("ci95_half_width") if isinstance(test, dict) else None
        if mean is None:
            continue
        labels.append(name.split("/")[-1])
        means.append(mean)
        errors.append(ci if ci is not None else 0.0)

    if not labels:
        print("  No aggregate test Spearman means available, skipping seed_ci_comparison.pdf")
        return

    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 1.2), 4))
    ax.bar(x, means, yerr=errors, capsize=4, color="tab:blue", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylabel("Test Spearman")
    ax.set_title("Seed-Aggregated Test Spearman (95% CI)")
    ax.set_ylim(0, 1.05)

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    print(f"  Saved {output_path}")


def plot_small_data_tradeoff(data: dict, output_path: Path) -> None:
    """Plot small-data regimes (train20/train35) against full-data runs."""
    aggregates = data.get("__aggregates__", {})
    if not isinstance(aggregates, dict):
        print("  No aggregate data available, skipping small_data_tradeoff.pdf")
        return

    selected = []
    for name, payload in sorted(aggregates.items()):
        label = name.split("/")[-1]
        if "small_data" in label or "full" in label:
            test = payload.get("test_spearman", {}) if isinstance(payload, dict) else {}
            mean = test.get("mean") if isinstance(test, dict) else None
            ci = test.get("ci95_half_width") if isinstance(test, dict) else None
            if mean is not None:
                selected.append((label, mean, ci if ci is not None else 0.0))

    if not selected:
        print("  No small-data aggregate runs available, skipping small_data_tradeoff.pdf")
        return

    labels = [x[0] for x in selected]
    means = [x[1] for x in selected]
    errors = [x[2] for x in selected]

    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 1.2), 4))
    ax.errorbar(x, means, yerr=errors, fmt="o", color="tab:green", capsize=4)
    ax.plot(x, means, "-", color="tab:green", alpha=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylabel("Test Spearman")
    ax.set_title("Small-Data Regime Performance")
    ax.set_ylim(0, 1.05)

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    print(f"  Saved {output_path}")


def plot_bounds_diagnostics(data: dict, output_path: Path) -> None:
    """Plot val/test satisfaction rates for bounds-mode experiments."""
    labels = []
    val_rates = []
    test_rates = []

    for name, info in _experiment_items(data):
        bounds = info.get("bounds_diagnostics", {})
        if not isinstance(bounds, dict):
            continue
        val = bounds.get("val_satisfaction_rate")
        test = bounds.get("test_satisfaction_rate")
        if val is None and test is None:
            continue
        labels.append(name.split("/")[-1])
        val_rates.append(val if val is not None else 0.0)
        test_rates.append(test if test is not None else 0.0)

    if not labels:
        print("  No bounds diagnostics available, skipping bounds_diagnostics.pdf")
        return

    x = np.arange(len(labels))
    width = 0.35
    fig, ax = plt.subplots(figsize=(max(7, len(labels) * 1.2), 4))
    ax.bar(x - width / 2, val_rates, width, label="Validation", color="tab:purple")
    ax.bar(x + width / 2, test_rates, width, label="Test", color="tab:pink")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("Satisfaction Rate")
    ax.set_title("Bounds Satisfaction Rates")
    ax.set_ylim(0, 1.05)
    ax.legend()

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    print(f"  Saved {output_path}")


# ── CLI ──────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate publication figures")
    parser.add_argument(
        "--data",
        type=str,
        default="analysis/results",
        help="Directory containing figure_data.json",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="paper/figures",
        help="Output directory for PDF figures",
    )
    args = parser.parse_args()

    data_path = Path(args.data) / "figure_data.json"
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not data_path.exists():
        print(f"Error: {data_path} not found. Run analyze_experiments.py first.")
        return

    try:
        data = json.loads(data_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        print(f"Error: failed to read {data_path}: {exc}")
        return
    print(f"Loaded data for {len(data)} experiments")

    print("\nGenerating figures...")
    plot_convergence(data, output_dir / "convergence.pdf")
    plot_map_elites_coverage(data, output_dir / "map_elites_heatmap.pdf")
    plot_baseline_comparison(data, output_dir / "baseline_comparison.pdf")
    plot_ood_generalization(data, output_dir / "ood_generalization.pdf")
    plot_benchmark_boxplot(data, output_dir / "benchmark_boxplot.pdf")
    plot_seed_ci_comparison(data, output_dir / "seed_ci_comparison.pdf")
    plot_small_data_tradeoff(data, output_dir / "small_data_tradeoff.pdf")
    plot_bounds_diagnostics(data, output_dir / "bounds_diagnostics.pdf")
    print("\nDone!")


if __name__ == "__main__":
    main()
