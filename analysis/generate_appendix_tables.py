#!/usr/bin/env python3
"""Generate Harmony-specific appendix tables for the NeurIPS paper.

Produces paper/sections/appendix_tables_generated.tex with three tables:
  1. MLX batch runtime summary per domain
  2. Factor decomposition detail (full + ablated configs)
  3. Statistical tests summary (Harmony vs DistMult-alone)

Usage:
    uv run python analysis/generate_appendix_tables.py \
        --factor-csv data/results/factor_decomposition.csv \
        --stat-tests data/results/statistical_tests.json \
        --mlx-base artifacts/harmony_mlx_full/2026-03-05_mlx_no_think_full \
        --output paper/sections/appendix_tables_generated.tex
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
from pathlib import Path


# ---------------------------------------------------------------------------
# Data extraction functions (unit-testable, no LaTeX)
# ---------------------------------------------------------------------------


def extract_runtime_stats(domain_dir: Path | str) -> dict | None:
    """Parse run.log files under domain_dir/seed_*/run.log to extract runtimes.

    Returns dict with keys: n, mean_sec, std_sec. Returns None if the
    directory does not exist or no valid 'real' timing lines are found.
    """
    domain_dir = Path(domain_dir)
    if not domain_dir.exists():
        return None

    times: list[float] = []
    for seed_dir in sorted(domain_dir.iterdir()):
        log_path = seed_dir / "run.log"
        if not log_path.exists():
            continue
        try:
            text = log_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        match = re.search(r"^real\s+([\d.]+)", text, re.MULTILINE)
        if match:
            times.append(float(match.group(1)))

    if not times:
        return None

    return {
        "n": len(times),
        "mean_sec": statistics.mean(times),
        "std_sec": statistics.pstdev(times),  # population stdev (all seeds observed)
    }


def build_factor_decomp_table(csv_path: Path | str) -> str:
    """Read factor_decomposition.csv → LaTeX tabular for factor decomp detail.

    Returns a complete LaTeX table environment string.
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)

    CONFIG_LABELS = {
        "llm_only": "LLM-only",
        "no_qd": "No-QD",
        "harmony_only": "Harmony-only",
        "full": "Full (LLM+Harmony+MAP-Elites)",
    }
    CONFIG_ORDER = ["llm_only", "no_qd", "harmony_only", "full"]
    DOMAIN_LABELS = {
        "astronomy": "Astronomy",
        "physics": "Physics",
        "materials": "Materials",
        "wikidata_physics": "Wikidata Physics",
        "wikidata_materials": "Wikidata Materials",
    }
    DOMAIN_ORDER = list(DOMAIN_LABELS.keys())

    # Read all rows
    rows: list[dict] = []
    with csv_path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    # Organise by domain → config
    data: dict[str, dict[str, dict]] = {}
    for row in rows:
        domain = row["config"].strip()  # wait — CSV has config, domain columns
        # Actually: config=config_name, domain=domain_name
        config = row["config"].strip()
        dom = row["domain"].strip()
        if dom not in data:
            data[dom] = {}
        data[dom][config] = row

    lines: list[str] = []
    lines.append(r"\subsection{Factor Decomposition Detail}")
    lines.append(r"\begin{table}[h]")
    lines.append(r"  \centering")
    lines.append(r"  \small")
    lines.append(
        r"  \caption{Factor decomposition: best harmony gain per configuration and domain. "
        r"LLM-only bypasses Harmony scoring (gain $=$ 1.0 by construction, not comparable). "
        r"Full = LLM proposer + Harmony scoring + MAP-Elites archive.}"
    )
    lines.append(r"  \label{tab:appendix_factor_decomp}")
    lines.append(r"  \begin{tabular}{llccc}")
    lines.append(r"    \toprule")
    lines.append(r"    Domain & Configuration & Best Gain & Gens & Archive \\")
    lines.append(r"    \midrule")

    for dom in DOMAIN_ORDER:
        if dom not in data:
            continue
        dom_label = DOMAIN_LABELS.get(dom, dom.replace("_", " ").title())
        configs = data[dom]
        n_configs = sum(1 for c in CONFIG_ORDER if c in configs)
        first = True
        for config in CONFIG_ORDER:
            if config not in configs:
                continue
            row = configs[config]
            label = CONFIG_LABELS.get(config, config)
            gain = row.get("best_harmony_gain", "")
            try:
                gain_val = float(gain)
                if config == "llm_only":
                    gain_str = "---"
                else:
                    gain_str = f"{gain_val:.4f}"
            except ValueError:
                gain_str = gain

            gens = row.get("generations", "")
            archive = row.get("archive_size", "")

            domain_col = f"\\multirow{{{n_configs}}}{{*}}{{{dom_label}}}" if first else ""
            lines.append(
                f"    {domain_col} & {label} & {gain_str} & {gens} & {archive} \\\\"
            )
            first = False
        lines.append(r"    \midrule")

    # Remove trailing \midrule and add \bottomrule
    if lines and lines[-1] == r"    \midrule":
        lines[-1] = r"    \bottomrule"

    lines.append(r"  \end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines) + "\n"


def build_statistical_tests_table(json_path: Path | str) -> str:
    """Read statistical_tests.json → LaTeX tabular for significance summary.

    Returns a complete LaTeX table environment string.
    """
    json_path = Path(json_path)
    if not json_path.exists():
        raise FileNotFoundError(json_path)

    DOMAIN_LABELS = {
        "astronomy": "Astronomy",
        "physics": "Physics",
        "materials": "Materials",
        "wikidata_physics": "Wikidata Physics",
        "wikidata_materials": "Wikidata Materials",
    }
    DOMAIN_ORDER = list(DOMAIN_LABELS.keys())

    data = json.loads(json_path.read_text())

    lines: list[str] = []
    lines.append(r"\subsection{Statistical Tests Summary}")
    lines.append(r"\begin{table}[h]")
    lines.append(r"  \centering")
    lines.append(r"  \small")
    lines.append(
        r"  \caption{Paired bootstrap CI (2000 resamples) and Cliff's~$\delta$ effect size "
        r"for Harmony vs.\ DistMult-alone Hits@10. None reach $p < 0.05$. "
        r"$|\delta_C| < 0.147$ = negligible; all observed $|\delta_C| \leq 0.12$.}"
    )
    lines.append(r"  \label{tab:appendix_stat_tests}")
    lines.append(r"  \begin{tabular}{lcccc}")
    lines.append(r"    \toprule")
    lines.append(r"    Domain & $\Delta$Hits@10 & 95\% CI & $p$ & $\delta_C$ \\")
    lines.append(r"    \midrule")

    for dom in DOMAIN_ORDER:
        if dom not in data:
            continue
        entry = data[dom]
        dom_label = DOMAIN_LABELS.get(dom, dom)
        bs = entry.get("hits10_bootstrap", {})
        mean_diff = bs.get("mean_diff", 0.0)
        ci_low = bs.get("ci_low", 0.0)
        ci_high = bs.get("ci_high", 0.0)
        p_value = bs.get("p_value", 1.0)
        delta = entry.get("hits10_cliffs_delta", 0.0)

        sign = "+" if mean_diff >= 0 else ""
        delta_sign = "+" if delta >= 0 else ""
        lines.append(
            f"    {dom_label} & ${sign}{mean_diff:.3f}$ & "
            f"$[{ci_low:.3f}, {ci_high:.3f}]$ & "
            f"${p_value:.3f}$ & "
            f"${delta_sign}{delta:.2f}$ \\\\"
        )

    lines.append(r"    \bottomrule")
    lines.append(r"  \end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines) + "\n"


def build_runtime_table(
    mlx_base_dir: Path | str,
    run_label: str = r"harmony\_mlx\_full (2026-03-05)",
) -> str | None:
    """Build LaTeX runtime summary table from MLX batch run.log files.

    Returns None if mlx_base_dir does not exist or has no data.
    """
    if mlx_base_dir is None:
        return None

    mlx_base_dir = Path(mlx_base_dir)
    DOMAIN_LABELS = {
        "astronomy": "Astronomy",
        "physics": "Physics",
        "materials": "Materials",
        "wikidata_physics": "Wikidata Physics",
        "wikidata_materials": "Wikidata Materials",
    }
    DOMAIN_ORDER = list(DOMAIN_LABELS.keys())

    rows: list[tuple[str, dict]] = []
    for dom in DOMAIN_ORDER:
        stats = extract_runtime_stats(mlx_base_dir / dom)
        if stats:
            rows.append((dom, stats))

    if not rows:
        return None

    lines: list[str] = []
    lines.append(r"\subsection{MLX Batch Runtime Summary}")
    lines.append(r"\begin{table}[h]")
    lines.append(r"  \centering")
    lines.append(r"  \small")
    lines.append(
        r"  \caption{Wall-clock runtime (seconds) for MLX batch runs "
        r"(\texttt{mlx-community/Qwen3.5-35B-A3B-4bit} via \texttt{mlx\_lm}, "
        r"Apple M2 Pro, 10 seeds per domain). Times include LLM inference, "
        r"Harmony scoring, and checkpointing.}"
    )
    lines.append(r"  \label{tab:appendix_mlx_runtime}")
    lines.append(r"  \begin{tabular}{lcccc}")
    lines.append(r"    \toprule")
    lines.append(r"    Domain & Mean (s) & Std (s) & Seeds & Mean (min) \\")
    lines.append(r"    \midrule")
    for dom, stats in rows:
        label = DOMAIN_LABELS.get(dom, dom)
        mean_s = stats["mean_sec"]
        std_s = stats["std_sec"]
        n = stats["n"]
        mean_min = mean_s / 60.0
        lines.append(
            f"    {label} & {mean_s:.1f} & {std_s:.1f} & {n}/{n} & {mean_min:.1f} \\\\"
        )
    lines.append(r"    \bottomrule")
    lines.append(r"  \end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Main generation function
# ---------------------------------------------------------------------------


def generate_appendix_tables(
    factor_csv: Path | str,
    stat_tests_json: Path | str,
    mlx_base_dir: Path | str | None,
    output_path: Path | str,
) -> None:
    """Generate appendix_tables_generated.tex with Harmony-specific tables.

    Parameters
    ----------
    factor_csv: Path to data/results/factor_decomposition.csv
    stat_tests_json: Path to data/results/statistical_tests.json
    mlx_base_dir: Base directory for MLX batch runs (seed_* subdirs per domain)
    output_path: Destination .tex file
    """
    output_path = Path(output_path)

    sections: list[str] = []
    sections.append("% Auto-generated by analysis/generate_appendix_tables.py. Do not edit manually.\n")

    # Table 1: MLX runtime (optional)
    runtime_tex = build_runtime_table(mlx_base_dir) if mlx_base_dir else None
    if runtime_tex:
        sections.append(runtime_tex)

    # Table 2: Factor decomposition
    sections.append(build_factor_decomp_table(factor_csv))

    # Table 3: Statistical tests
    sections.append(build_statistical_tests_table(stat_tests_json))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(sections), encoding="utf-8")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Harmony appendix tables")
    parser.add_argument(
        "--factor-csv",
        type=Path,
        default=Path("data/results/factor_decomposition.csv"),
    )
    parser.add_argument(
        "--stat-tests",
        type=Path,
        default=Path("data/results/statistical_tests.json"),
    )
    parser.add_argument(
        "--mlx-base",
        type=Path,
        default=Path("artifacts/harmony_mlx_full/2026-03-05_mlx_no_think_full"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("paper/sections/appendix_tables_generated.tex"),
    )
    args = parser.parse_args()

    generate_appendix_tables(
        factor_csv=args.factor_csv,
        stat_tests_json=args.stat_tests,
        mlx_base_dir=args.mlx_base,
        output_path=args.output,
    )
    print(f"Written: {args.output}")


if __name__ == "__main__":
    main()
