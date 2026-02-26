#!/usr/bin/env python
"""Generate NeurIPS-quality figures for the Harmony-Driven Theory Discovery paper.

Reads pipeline outputs (JSONL events, checkpoint archives, metrics CSV) and
produces four PDF figures:
  1. convergence.pdf   — valid_rate + best_harmony_gain by generation
  2. archive_heatmap.pdf — MAP-Elites archive fitness grid
  3. baseline_comparison.pdf — grouped bar chart of Hits@10 across methods
  4. ablation_weights.pdf — Harmony score across 6 weight configurations

Usage:
    python analysis/generate_harmony_figures.py \
        --astronomy artifacts/harmony/astronomy \
        --physics   artifacts/harmony/physics \
        --materials artifacts/harmony/materials \
        --figures-dir paper/figures/
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np

# Ensure src/ is on the path when run from the repo root
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# NeurIPS style constants
# ---------------------------------------------------------------------------
COLUMN_WIDTH = 3.25  # inches — single NeurIPS column
FULL_WIDTH = 6.75  # inches — full NeurIPS text width
COLORS = {
    "random": "#bdbdbd",
    "frequency": "#fdae6b",
    "distmult": "#6baed6",
    "harmony": "#31a354",
}
VALID_RATE_COLOR = "#e6550d"
HARMONY_GAIN_COLOR = "#3182bd"


# ===========================================================================
# Pure data-parsing functions (unit-testable, no matplotlib)
# ===========================================================================


def parse_convergence_data(output_dir: Path) -> dict[str, list[float]]:
    """Parse harmony_events.jsonl → convergence curve arrays.

    Returns dict with keys: generation, valid_rate, best_harmony_gain.
    Each value is a list of floats (same length). Returns empty lists if
    the events file is missing or empty.
    """
    events_path = Path(output_dir) / "logs" / "harmony_events.jsonl"
    result: dict[str, list[float]] = {
        "generation": [],
        "valid_rate": [],
        "best_harmony_gain": [],
    }
    if not events_path.exists():
        return result

    with events_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if event.get("event") != "generation_summary":
                continue
            result["generation"].append(event.get("generation", 0))
            result["valid_rate"].append(event.get("valid_rate", 0.0))
            result["best_harmony_gain"].append(event.get("best_harmony_gain", 0.0))

    return result


def build_heatmap_matrix(output_dir: Path) -> tuple[np.ndarray, int]:
    """Load checkpoint.json → 2D numpy matrix of archive fitness values.

    Returns (matrix, num_bins). Unoccupied cells are NaN.
    Returns (empty array, 0) if checkpoint is missing or has no archive.
    """
    checkpoint_path = Path(output_dir) / "checkpoint.json"
    if not checkpoint_path.exists():
        return np.array([]), 0

    with checkpoint_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    archive_data = data.get("archive")
    if archive_data is None:
        return np.array([]), 0

    num_bins = int(archive_data.get("num_bins", 0))
    if num_bins <= 0:
        return np.array([]), 0

    matrix = np.full((num_bins, num_bins), np.nan)
    for key, cell_data in archive_data.get("cells", {}).items():
        try:
            row_str, col_str = key.split(",")
            row, col = int(row_str), int(col_str)
            if 0 <= row < num_bins and 0 <= col < num_bins:
                matrix[row, col] = float(cell_data["fitness_signal"])
        except (ValueError, KeyError, TypeError):
            continue

    return matrix, num_bins


def parse_metrics_csv(csv_path: Path) -> dict[str, Any]:
    """Parse metrics_table.csv → structured dict for bar chart.

    Returns dict with keys: domains, random, frequency, distmult, harmony.
    Each value is a list (same length). Returns empty lists if file missing.
    """
    csv_path = Path(csv_path)
    result: dict[str, Any] = {
        "domains": [],
        "random": [],
        "frequency": [],
        "distmult": [],
        "harmony": [],
    }
    if not csv_path.exists():
        return result

    import pandas as pd

    df = pd.read_csv(csv_path, index_col=0)
    result["domains"] = list(df.index)
    result["random"] = list(df["random_hits10"])
    result["frequency"] = list(df["freq_hits10"])
    result["distmult"] = list(df["distmult_hits10"])
    result["harmony"] = list(df["harmony_hits10"])
    return result


def build_ablation_data() -> dict[str, Any]:
    """Compute Harmony scores across the 6 pre-registered weight configs.

    Uses the linear_algebra calibration KG (fast, always available).
    Returns dict with keys: labels (list[str]), scores (list[float]).
    """
    from harmony.datasets.linear_algebra import build_linear_algebra_kg
    from harmony.metric.harmony import harmony_score

    weight_grid = [(alpha, beta, 0.25, 0.25) for alpha in (0.3, 0.5, 0.7) for beta in (0.1, 0.3)]

    kg = build_linear_algebra_kg()
    labels: list[str] = []
    scores: list[float] = []
    for alpha, beta, gamma, delta in weight_grid:
        h = harmony_score(kg, alpha=alpha, beta=beta, gamma=gamma, delta=delta, seed=42)
        labels.append(f"α={alpha} β={beta}")
        scores.append(float(h))

    return {"labels": labels, "scores": scores}


# ===========================================================================
# Plotting functions (matplotlib, produce PDF output)
# ===========================================================================


def _apply_neurips_style() -> None:
    """Apply NeurIPS-appropriate matplotlib style settings."""
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.05,
            "axes.grid": True,
            "grid.alpha": 0.3,
            "grid.linewidth": 0.5,
        }
    )


def plot_convergence(
    domain_data: dict[str, dict[str, list[float]]],
    output_path: Path,
) -> None:
    """Plot convergence curves: valid_rate + best_harmony_gain per domain.

    domain_data maps domain_name → parse_convergence_data() result.
    Domains with empty data are skipped.
    """
    import matplotlib.pyplot as plt

    _apply_neurips_style()

    # Filter to domains with data
    active = {k: v for k, v in domain_data.items() if v["generation"]}
    if not active:
        # Create a minimal figure even when empty
        fig, ax = plt.subplots(figsize=(COLUMN_WIDTH, 2))
        ax.text(0.5, 0.5, "No convergence data", ha="center", va="center", transform=ax.transAxes)
        fig.savefig(output_path)
        plt.close(fig)
        return

    n = len(active)
    fig, axes = plt.subplots(1, n, figsize=(FULL_WIDTH, 2.2), squeeze=False)

    for idx, (domain, data) in enumerate(active.items()):
        ax = axes[0, idx]
        gens = data["generation"]

        ax.plot(gens, data["valid_rate"], color=VALID_RATE_COLOR, linewidth=1.5, label="Valid rate")
        ax.set_ylabel("Valid rate", color=VALID_RATE_COLOR)
        ax.set_ylim(0, 1)

        ax2 = ax.twinx()
        ax2.plot(
            gens,
            data["best_harmony_gain"],
            color=HARMONY_GAIN_COLOR,
            linewidth=1.5,
            linestyle="--",
            label="Best gain",
        )
        ax2.set_ylabel("Best harmony gain", color=HARMONY_GAIN_COLOR)

        ax.set_xlabel("Generation")
        ax.set_title(domain.replace("_", " ").title())

        if idx == 0:
            lines1, labels1 = ax.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax.legend(lines1 + lines2, labels1 + labels2, loc="lower right")

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def plot_archive_heatmap(
    domain_data: dict[str, tuple[np.ndarray, int]],
    output_path: Path,
) -> None:
    """Plot MAP-Elites archive heatmaps per domain.

    domain_data maps domain_name → (matrix, num_bins) from build_heatmap_matrix().
    """
    import matplotlib.pyplot as plt

    _apply_neurips_style()

    # Filter to domains with data
    active = {k: (m, n) for k, (m, n) in domain_data.items() if m.size > 0}
    if not active:
        fig, ax = plt.subplots(figsize=(COLUMN_WIDTH, 2))
        ax.text(0.5, 0.5, "No archive data", ha="center", va="center", transform=ax.transAxes)
        fig.savefig(output_path)
        plt.close(fig)
        return

    n = len(active)
    fig, axes = plt.subplots(1, n, figsize=(FULL_WIDTH, 2.5), squeeze=False)

    for idx, (domain, (matrix, num_bins)) in enumerate(active.items()):
        ax = axes[0, idx]
        im = ax.imshow(
            matrix,
            cmap="YlGn",
            aspect="equal",
            origin="lower",
            vmin=0,
        )
        ax.set_xlabel("Gain bin")
        ax.set_ylabel("Simplicity bin")
        ax.set_title(domain.replace("_", " ").title())
        ax.set_xticks(range(num_bins))
        ax.set_yticks(range(num_bins))

        # Annotate occupied cells
        for r in range(num_bins):
            for c in range(num_bins):
                if not np.isnan(matrix[r, c]):
                    ax.text(
                        c,
                        r,
                        f"{matrix[r, c]:.2f}",
                        ha="center",
                        va="center",
                        fontsize=6,
                    )

        fig.colorbar(im, ax=ax, shrink=0.8, label="Fitness")

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def plot_baseline_comparison(
    data: dict[str, Any],
    output_path: Path,
) -> None:
    """Plot grouped bar chart: Hits@10 by method across domains."""
    import matplotlib.pyplot as plt

    _apply_neurips_style()

    domains = data["domains"]
    if not domains:
        fig, ax = plt.subplots(figsize=(COLUMN_WIDTH, 2))
        ax.text(0.5, 0.5, "No metrics data", ha="center", va="center", transform=ax.transAxes)
        fig.savefig(output_path)
        plt.close(fig)
        return

    methods = ["random", "frequency", "distmult", "harmony"]
    method_labels = ["Random", "Frequency", "DistMult", "Harmony (ours)"]

    x = np.arange(len(domains))
    width = 0.18
    fig, ax = plt.subplots(figsize=(COLUMN_WIDTH, 2.5))

    for i, (method, label) in enumerate(zip(methods, method_labels, strict=True)):
        values = data[method]
        offset = (i - 1.5) * width
        bars = ax.bar(
            x + offset,
            values,
            width,
            label=label,
            color=COLORS[method],
            edgecolor="white",
            linewidth=0.5,
        )
        # Add value labels on top
        for bar in bars:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{bar.get_height():.2f}",
                ha="center",
                va="bottom",
                fontsize=5,
            )

    ax.set_ylabel("Hits@10")
    ax.set_xticks(x)
    ax.set_xticklabels([d.replace("_", " ").title() for d in domains])
    ax.set_ylim(0, 1.0)
    ax.legend(loc="upper left", framealpha=0.9)
    ax.set_title("Link Prediction: Hits@10 Comparison")

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def plot_ablation_weights(
    data: dict[str, Any],
    output_path: Path,
) -> None:
    """Plot bar chart: Harmony score by weight configuration."""
    import matplotlib.pyplot as plt

    _apply_neurips_style()

    labels = data["labels"]
    scores = data["scores"]

    fig, ax = plt.subplots(figsize=(COLUMN_WIDTH, 2.5))
    x = np.arange(len(labels))
    bars = ax.bar(x, scores, color=COLORS["harmony"], edgecolor="white", linewidth=0.5)

    for bar in bars:
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.005,
            f"{bar.get_height():.3f}",
            ha="center",
            va="bottom",
            fontsize=6,
        )

    ax.set_ylabel("Harmony score")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_title("Harmony Score by Weight Configuration\n(Linear Algebra KG)")

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


# ===========================================================================
# CLI
# ===========================================================================


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Generate NeurIPS figures for Harmony paper")
    discovery_domains = ["astronomy", "physics", "materials"]
    for domain in discovery_domains:
        parser.add_argument(f"--{domain}", type=Path, metavar="DIR", dest=domain)
    parser.add_argument(
        "--figures-dir",
        type=Path,
        default=Path("paper/figures"),
        help="Output directory for PDF figures",
    )
    parser.add_argument(
        "--metrics-csv",
        type=Path,
        default=None,
        help="Path to metrics_table.csv (default: auto-detect from domain dirs)",
    )
    args = parser.parse_args()

    figures_dir = args.figures_dir
    figures_dir.mkdir(parents=True, exist_ok=True)

    # --- Convergence ---
    print("[1/4] Generating convergence plot...")
    conv_data: dict[str, dict[str, list[float]]] = {}
    for domain in discovery_domains:
        d = getattr(args, domain, None)
        if d is not None:
            conv_data[domain] = parse_convergence_data(d)
    if conv_data:
        plot_convergence(conv_data, figures_dir / "convergence.pdf")
        print(f"  → {figures_dir / 'convergence.pdf'}")
    else:
        print("  (skipped — no domain directories provided)")

    # --- Heatmap ---
    print("[2/4] Generating archive heatmap...")
    heat_data: dict[str, tuple[np.ndarray, int]] = {}
    for domain in discovery_domains:
        d = getattr(args, domain, None)
        if d is not None:
            heat_data[domain] = build_heatmap_matrix(d)
    if heat_data:
        plot_archive_heatmap(heat_data, figures_dir / "archive_heatmap.pdf")
        print(f"  → {figures_dir / 'archive_heatmap.pdf'}")
    else:
        print("  (skipped — no domain directories provided)")

    # --- Baseline comparison ---
    print("[3/4] Generating baseline comparison...")
    csv_path = args.metrics_csv
    if csv_path is None:
        # Try auto-detect from first domain dir
        for domain in discovery_domains:
            d = getattr(args, domain, None)
            if d is not None:
                candidate = d.parent / "metrics_table.csv"
                if candidate.exists():
                    csv_path = candidate
                    break
    if csv_path is not None and csv_path.exists():
        metrics_data = parse_metrics_csv(csv_path)
        plot_baseline_comparison(metrics_data, figures_dir / "baseline_comparison.pdf")
        print(f"  → {figures_dir / 'baseline_comparison.pdf'}")
    else:
        print("  (skipped — no metrics_table.csv found)")

    # --- Ablation weights ---
    print("[4/4] Generating ablation weights plot...")
    ablation_data = build_ablation_data()
    plot_ablation_weights(ablation_data, figures_dir / "ablation_weights.pdf")
    print(f"  → {figures_dir / 'ablation_weights.pdf'}")

    print("\nDone. All figures saved to", figures_dir)


if __name__ == "__main__":
    main()
