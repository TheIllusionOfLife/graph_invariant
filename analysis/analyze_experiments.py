"""Cross-experiment analysis entry point.

Usage:
    uv run python analysis/analyze_experiments.py \
        --artifacts-root artifacts/ --output analysis/results/
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure the analysis/ directory is on sys.path so submodule imports resolve.
_analysis_dir = Path(__file__).resolve().parent
if str(_analysis_dir) not in sys.path:
    sys.path.insert(0, str(_analysis_dir))

from experiment_analysis import *  # noqa: F401,F403,E402
from experiment_loader import *  # noqa: F401,F403,E402
from report_writers import *  # noqa: F401,F403,E402

from experiment_analysis import (  # noqa: E402
    _normalize_candidate_code_for_report,
    build_comparison_table,
    build_seed_aggregates,
    extract_acceptance_funnel,
    extract_bounds_diagnostics,
    extract_convergence_data,
    extract_repair_breakdown,
)
from experiment_loader import discover_experiments, discover_matrix_summaries  # noqa: E402
from report_writers import (  # noqa: E402
    write_analysis_report,
    write_appendix_tables_tex,
    write_figure_data_json,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Cross-experiment analysis")
    parser.add_argument("--artifacts-root", type=str, default="artifacts")
    parser.add_argument("--output", type=str, default="analysis/results")
    parser.add_argument(
        "--appendix-tex-output",
        type=str,
        default="paper/sections/appendix_tables_generated.tex",
    )
    args = parser.parse_args()

    artifacts_root = Path(args.artifacts_root)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Discovering experiments in {artifacts_root}...")
    experiments = discover_experiments(artifacts_root)
    matrix_summaries = discover_matrix_summaries(artifacts_root)
    print(f"Found {len(experiments)} experiments: {list(experiments.keys())}")

    if not experiments:
        print("No experiments found. Nothing to analyze.")
        return

    report_path = output_dir / "analysis_report.md"
    write_analysis_report(experiments, report_path)
    print(f"Report written to {report_path}")

    figure_data_path = output_dir / "figure_data.json"
    figure_data = write_figure_data_json(experiments, figure_data_path, matrix_summaries)
    print(f"Figure data written to {figure_data_path}")

    appendix_tex_path = Path(args.appendix_tex_output)
    write_appendix_tables_tex(figure_data, appendix_tex_path)
    print(f"Appendix tables written to {appendix_tex_path}")

    table = build_comparison_table(experiments)
    print(f"\n{'Experiment':<35} {'Mode':<15} {'Success':<8} {'Val ρ':<10} {'Test ρ':<10}")
    print("-" * 78)
    for row in table:
        val_s = f"{row['val_spearman']:.4f}" if row.get("val_spearman") is not None else "N/A"
        test_s = f"{row['test_spearman']:.4f}" if row.get("test_spearman") is not None else "N/A"
        print(
            f"{row['experiment']:<35} {row.get('fitness_mode', ''):<15} "
            f"{str(row.get('success', '')):<8} {val_s:<10} {test_s:<10}"
        )


if __name__ == "__main__":
    main()
