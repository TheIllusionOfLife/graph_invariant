#!/usr/bin/env python
"""Populate paper tables with live experiment results.

Reads experiment output CSVs/JSONs and patches LaTeX placeholders
(--- values) in paper/sections/results.tex and paper/sections/appendix.tex.

Usage:
    uv run python scripts/populate_paper_tables.py \
        --results-dir data/results \
        --paper-dir paper/sections
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent))

# Domain display names for LaTeX tables
_DOMAIN_DISPLAY = {
    "astronomy": "Astronomy",
    "physics": "Physics",
    "materials": "Materials",
    "wikidata_physics": "Wikidata Phys.",
    "wikidata_materials": "Wikidata Mat.",
}

# Mapping from domain display name patterns in LaTeX to domain keys
_LATEX_DOMAIN_MAP = {
    "Astronomy": "astronomy",
    "Physics}": "physics",  # matches \multirow{8}{*}{Physics}
    "Materials}": "materials",  # matches \multirow{8}{*}{Materials}
    "Wikidata\\\\Physics": "wikidata_physics",
    "Wikidata\\\\Materials": "wikidata_materials",
}


def _fmt(value: float, precision: int = 2) -> str:
    """Format a float to fixed precision."""
    return f"{value:.{precision}f}"


def _fmt_pm(mean: float, std: float, precision: int = 2) -> str:
    """Format mean ± std for LaTeX."""
    return f"${_fmt(mean, precision)} \\pm {_fmt(std, precision)}$"


def _fmt_p(p_value: float) -> str:
    """Format p-value for LaTeX."""
    if p_value < 0.001:
        return "$< 0.001$"
    if p_value < 0.01:
        return f"${p_value:.3f}$"
    return f"${p_value:.2f}$"


def _fmt_delta(delta: float) -> str:
    """Format Cliff's delta for LaTeX."""
    return f"${delta:+.2f}$"


# ---------------------------------------------------------------------------
# Table 1: Link prediction metrics
# ---------------------------------------------------------------------------


def populate_table1(
    summary_path: Path,
    stat_tests_path: Path,
    results_tex: Path,
) -> str:
    """Replace --- placeholders in Table 1 with live numbers.

    Returns the updated LaTeX content.
    """
    summary = pd.read_csv(summary_path, index_col=0)
    stat_tests = {}
    if stat_tests_path.exists():
        stat_tests = json.loads(stat_tests_path.read_text())

    content = results_tex.read_text()

    for domain in summary.index:
        # Method → (hits10_mean_col, hits10_std_col, mrr_mean_col, mrr_std_col)
        method_map = {
            "TransE": ("transe_hits10", "mrr_transe"),
            "RotatE": ("rotate_hits10", "mrr_rotate"),
            "ComplEx": ("complex_hits10", "mrr_complex"),
            "Harmony-3": ("harmony3_hits10", "mrr_harmony3"),
        }

        for method, (hits_col, mrr_col) in method_map.items():
            hits_mean = summary.loc[domain, f"{hits_col}_mean"]
            hits_std = summary.loc[domain, f"{hits_col}_std"]
            mrr_mean = summary.loc[domain, f"{mrr_col}_mean"]
            mrr_std = summary.loc[domain, f"{mrr_col}_std"]

            hits_str = _fmt_pm(hits_mean, hits_std)
            mrr_str = _fmt_pm(mrr_mean, mrr_std)

            # Replace --- for this method in the domain block
            # Pattern: & {method} & --- & --- & &
            old_pattern = f"& {method}" + r"\s+& --- & ---"
            new_text = f"& {method:<15s} & {hits_str} & {mrr_str}"
            # re.sub treats \p as backreference; bind via default arg
            content = re.sub(
                old_pattern, lambda _, t=new_text: t, content, count=1
            )

        # Populate p-value and Cliff's delta for Harmony row
        if domain in stat_tests:
            st = stat_tests[domain]
            p_val = st.get("hits10_bootstrap", {}).get("p_value")
            cliff_d = st.get("hits10_cliffs_delta")
            if p_val is not None and cliff_d is not None:
                # Replace --- & --- at end of Harmony row
                old_harmony = r"(Harmony \(ours\)\s+& \$[^$]+\$ & \$[^$]+\$) & --- & ---"
                p_str = _fmt_p(p_val)
                d_str = _fmt_delta(cliff_d)
                content = re.sub(
                    old_harmony,
                    lambda m, p=p_str, d=d_str: f"{m.group(1)} & {p} & {d}",
                    content,
                    count=1,
                )

    return content


# ---------------------------------------------------------------------------
# Backtesting table
# ---------------------------------------------------------------------------


def populate_backtest_table(
    backtest_path: Path,
    content: str,
) -> str:
    """Replace --- placeholders in backtesting table."""
    if not backtest_path.exists():
        return content

    backtest = json.loads(backtest_path.read_text())

    for domain, display in _DOMAIN_DISPLAY.items():
        if domain not in backtest:
            continue
        bt = backtest[domain]

        vals = " & ".join(
            _fmt(bt[k])
            for k in [
                "precision_at_5",
                "precision_at_10",
                "precision_at_20",
                "recall_at_5",
                "recall_at_10",
                "recall_at_20",
            ]
        )

        # Replace the --- row for this domain
        old_row = re.escape(display) + r"\s+& --- & --- & --- & --- & --- & ---"
        new_row = f"{display:<17s} & {vals}"
        content = re.sub(old_row, lambda _, r=new_row: r, content, count=1)

    return content


# ---------------------------------------------------------------------------
# Reproducibility table (appendix)
# ---------------------------------------------------------------------------


def populate_reproducibility_table(
    repro_path: Path,
    appendix_tex: Path,
) -> str:
    """Replace --- placeholders in reproducibility table."""
    content = appendix_tex.read_text()
    if not repro_path.exists():
        return content

    repro = pd.read_csv(repro_path)

    for _, row in repro.iterrows():
        domain = row["domain"]
        entities = int(row["entities"])
        edges = int(row["edges"])
        entity_types = int(row["entity_types"])
        edge_types = int(row["edge_types"])
        source = row["source"]

        # Map domain names to LaTeX display
        display_map = {
            "linear_algebra": "Linear algebra",
            "periodic_table": "Periodic table",
            "astronomy": "Astronomy",
            "physics": "Physics",
            "materials": "Materials science",
            "wikidata_physics": "Wikidata Physics",
            "wikidata_materials": "Wikidata Materials",
        }
        display = display_map.get(domain, domain)

        # Replace rows with --- placeholders
        # Pattern: Display & N & M & X & --- & Source
        src_display = source.capitalize() if source[0].islower() else source
        old_pattern = (
            re.escape(display)
            + r"\s+& \d+\s+& \d+\s+& \d+\s+& ---\s+& "
            + re.escape(src_display)
        )
        new_row = (
            f"{display:<18s} & {entities:<3d} & {edges:<3d}"
            f" & {entity_types} & {edge_types} & {src_display}"
        )
        content = re.sub(old_pattern, lambda _, r=new_row: r, content, count=1)

        # Handle Wikidata rows with all ---
        old_wiki = (
            re.escape(display)
            + r"\s+& ---\s+& ---\s+& ---\s+& ---\s+& "
            + re.escape("Wikidata")
        )
        new_wiki = (
            f"{display:<18s} & {entities:<3d} & {edges:<3d}"
            f" & {entity_types} & {edge_types} & Wikidata"
        )
        content = re.sub(old_wiki, lambda _, w=new_wiki: w, content, count=1)

    return content


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Populate paper tables")
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("data/results"),
        help="Root results directory.",
    )
    parser.add_argument(
        "--paper-dir",
        type=Path,
        default=Path("paper/sections"),
        help="Paper sections directory.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print changes without writing files.",
    )
    args = parser.parse_args()

    results_dir = args.results_dir
    paper_dir = args.paper_dir

    # Table 1: Link prediction
    results_tex = paper_dir / "results.tex"
    summary_path = results_dir / "multi_seed" / "summary.csv"
    stat_tests_path = results_dir / "statistical_tests.json"

    if summary_path.exists() and results_tex.exists():
        print("Populating Table 1 (link prediction)...")
        new_results = populate_table1(summary_path, stat_tests_path, results_tex)

        # Backtesting table (also in results.tex)
        backtest_path = results_dir / "backtesting.json"
        new_results = populate_backtest_table(backtest_path, new_results)

        if args.dry_run:
            print(new_results)
        else:
            results_tex.write_text(new_results)
            print(f"  Updated {results_tex}")
    else:
        print(f"  Skipping Table 1: summary={summary_path.exists()}, tex={results_tex.exists()}")

    # Reproducibility table (appendix)
    appendix_tex = paper_dir / "appendix.tex"
    repro_path = results_dir / "reproducibility_table.csv"
    if repro_path.exists() and appendix_tex.exists():
        print("Populating reproducibility table (appendix)...")
        new_appendix = populate_reproducibility_table(repro_path, appendix_tex)
        if args.dry_run:
            print(new_appendix)
        else:
            appendix_tex.write_text(new_appendix)
            print(f"  Updated {appendix_tex}")
    else:
        print(
            f"  Skipping reproducibility: repro={repro_path.exists()},"
            f" tex={appendix_tex.exists()}"
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
