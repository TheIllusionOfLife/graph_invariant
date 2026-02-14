"""Cross-experiment analysis for graph invariant discovery results.

Usage:
    uv run python analysis/analyze_experiments.py \\
        --artifacts-root artifacts/ --output analysis/results/

Reads phase1_summary.json, baselines_summary.json, ood_validation.json,
and events.jsonl from each experiment directory under artifacts-root.
Produces a markdown report and JSON figure data for generate_figures.py.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

# ── Data loading ─────────────────────────────────────────────────────


def _load_json_safe(path: Path) -> dict:
    """Load a JSON file, returning {} on missing or corrupt files."""
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def load_experiment_summary(experiment_dir: Path) -> dict:
    """Load phase1_summary.json from an experiment directory."""
    return _load_json_safe(Path(experiment_dir) / "phase1_summary.json")


def load_baselines_summary(experiment_dir: Path) -> dict:
    """Load baselines_summary.json from an experiment directory."""
    return _load_json_safe(Path(experiment_dir) / "baselines_summary.json")


def load_ood_results(experiment_dir: Path) -> dict:
    """Load OOD validation results from experiment_dir/ood/ood_validation.json."""
    return _load_json_safe(Path(experiment_dir) / "ood" / "ood_validation.json")


def load_event_log(experiment_dir: Path) -> list[dict]:
    """Load events.jsonl from experiment_dir/logs/events.jsonl.

    Returns a list of parsed event dicts. Returns [] if file is missing.
    """
    events_path = Path(experiment_dir) / "logs" / "events.jsonl"
    if not events_path.exists():
        return []
    events: list[dict] = []
    try:
        with events_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    events.append(json.loads(line))
    except (json.JSONDecodeError, OSError):
        pass
    return events


# ── Analysis functions ───────────────────────────────────────────────


def extract_convergence_data(events: list[dict]) -> dict[str, list]:
    """Extract generation-by-generation convergence data from event log.

    Returns dict with keys:
        - generations: list of generation numbers
        - best_scores: list of best scores per generation
        - map_elites_coverage: list of archive coverage counts (if present)
    """
    generations: list[int] = []
    best_scores: list[float] = []
    coverages: list[int] = []

    for event in events:
        if event.get("event") != "generation_summary":
            continue
        payload = event.get("payload", {})
        gen = payload.get("generation")
        score = payload.get("best_score")
        if gen is not None and score is not None:
            generations.append(gen)
            best_scores.append(score)
            me_stats = payload.get("map_elites_stats", {})
            if "coverage" in me_stats:
                coverages.append(me_stats["coverage"])

    result: dict[str, list] = {
        "generations": generations,
        "best_scores": best_scores,
    }
    if coverages:
        result["map_elites_coverage"] = coverages
    return result


def _get_spearman(metrics: dict | None) -> float | None:
    """Safely extract spearman from a metrics dict."""
    if not isinstance(metrics, dict):
        return None
    val = metrics.get("spearman")
    if isinstance(val, (int, float)):
        return float(val)
    return None


def build_comparison_table(experiments: dict[str, dict]) -> list[dict[str, Any]]:
    """Build a cross-experiment comparison table.

    Args:
        experiments: dict mapping experiment name to {summary, baselines, ood}.

    Returns:
        List of row dicts with standardized columns for comparison.
    """
    rows: list[dict[str, Any]] = []
    for name, data in experiments.items():
        summary = data.get("summary", {})
        baselines = data.get("baselines", {})
        ood = data.get("ood", {})

        row: dict[str, Any] = {
            "experiment": name,
            "fitness_mode": summary.get("fitness_mode", "unknown"),
            "success": summary.get("success", False),
            "final_generation": summary.get("final_generation"),
            "val_spearman": _get_spearman(summary.get("val_metrics")),
            "test_spearman": _get_spearman(summary.get("test_metrics")),
            "stop_reason": summary.get("stop_reason"),
        }

        # Bounds metrics (for upper/lower bound experiments)
        bounds = summary.get("bounds_metrics", {})
        if isinstance(bounds, dict) and bounds:
            val_bounds = bounds.get("val", {})
            test_bounds = bounds.get("test", {})
            row["val_bound_score"] = val_bounds.get("bound_score") if val_bounds else None
            row["test_bound_score"] = test_bounds.get("bound_score") if test_bounds else None
            row["val_satisfaction_rate"] = (
                val_bounds.get("satisfaction_rate") if val_bounds else None
            )

        # Baseline comparison
        bc = summary.get("baseline_comparison", {})
        if isinstance(bc, dict) and bc:
            pysr = bc.get("pysr", {})
            row["pysr_val_spearman"] = _get_spearman(pysr) if isinstance(pysr, dict) else None

        # Statistical baselines
        stat = baselines.get("stat_baselines", {})
        if isinstance(stat, dict):
            for bl_name, bl_data in stat.items():
                if isinstance(bl_data, dict):
                    row[f"bl_{bl_name}_val_spearman"] = _get_spearman(bl_data.get("val_metrics"))

        # PySR baseline from baselines_summary
        pysr_bl = baselines.get("pysr_baseline", {})
        if isinstance(pysr_bl, dict) and pysr_bl.get("status") == "ok":
            row["bl_pysr_val_spearman"] = _get_spearman(pysr_bl.get("val_metrics"))

        # OOD results
        for category in ("large_random", "extreme_params", "special_topology"):
            cat_data = ood.get(category, {})
            if isinstance(cat_data, dict):
                row[f"ood_{category}_spearman"] = _get_spearman(cat_data)

        # Self-correction stats
        sc = summary.get("self_correction_stats", {})
        if isinstance(sc, dict):
            row["self_correction_attempts"] = sc.get("attempted_repairs", 0)
            row["self_correction_successes"] = sc.get("successful_repairs", 0)

        rows.append(row)

    return rows


# ── Output functions ─────────────────────────────────────────────────


def write_analysis_report(experiments: dict[str, dict], output_path: Path) -> None:
    """Write a markdown analysis report summarizing all experiments.

    Args:
        experiments: dict mapping experiment name to
            {summary, baselines, ood, convergence}.
        output_path: Path to write the markdown report.
    """
    lines: list[str] = ["# Cross-Experiment Analysis Report", ""]

    # Summary table
    table_data = build_comparison_table(experiments)
    if table_data:
        lines.extend(["## Experiment Comparison", ""])
        headers = ["Experiment", "Mode", "Success", "Val Spearman", "Test Spearman", "Generations"]
        lines.append("| " + " | ".join(headers) + " |")
        lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
        for row in table_data:
            val_s = f"{row['val_spearman']:.4f}" if row.get("val_spearman") is not None else "N/A"
            test_s = (
                f"{row['test_spearman']:.4f}" if row.get("test_spearman") is not None else "N/A"
            )
            cells = [
                row["experiment"],
                row.get("fitness_mode", ""),
                str(row.get("success", "")),
                val_s,
                test_s,
                str(row.get("final_generation", "")),
            ]
            lines.append("| " + " | ".join(cells) + " |")
        lines.append("")

    # Per-experiment details
    for name, data in experiments.items():
        summary = data.get("summary", {})
        convergence = data.get("convergence", {})
        baselines = data.get("baselines", {})
        ood = data.get("ood", {})

        lines.extend([f"## {name}", ""])

        # Basic info
        lines.append(f"- Fitness mode: {summary.get('fitness_mode', 'unknown')}")
        lines.append(f"- Success: {summary.get('success', False)}")
        lines.append(f"- Stop reason: {summary.get('stop_reason', 'N/A')}")
        lines.append(f"- Final generation: {summary.get('final_generation', 'N/A')}")

        # Metrics
        val_s = _get_spearman(summary.get("val_metrics"))
        test_s = _get_spearman(summary.get("test_metrics"))
        if val_s is not None:
            lines.append(f"- Validation Spearman: {val_s:.4f}")
        if test_s is not None:
            lines.append(f"- Test Spearman: {test_s:.4f}")

        # Bounds metrics
        bounds = summary.get("bounds_metrics", {})
        if isinstance(bounds, dict) and bounds:
            val_b = bounds.get("val", {})
            test_b = bounds.get("test", {})
            if val_b:
                lines.append(f"- Val bound score: {val_b.get('bound_score', 'N/A')}")
                lines.append(f"- Val satisfaction rate: {val_b.get('satisfaction_rate', 'N/A')}")
            if test_b:
                lines.append(f"- Test bound score: {test_b.get('bound_score', 'N/A')}")

        # Convergence
        if convergence:
            scores = convergence.get("best_scores", [])
            if scores:
                lines.append(f"- Convergence: {scores[0]:.3f} → {scores[-1]:.3f}")
            coverage = convergence.get("map_elites_coverage", [])
            if coverage:
                lines.append(f"- MAP-Elites coverage: {coverage[0]} → {coverage[-1]} cells")

        # Baselines
        stat = baselines.get("stat_baselines", {})
        if isinstance(stat, dict) and stat:
            lines.extend(["", "### Baselines", ""])
            for bl_name, bl_data in stat.items():
                if isinstance(bl_data, dict):
                    bl_val = _get_spearman(bl_data.get("val_metrics"))
                    bl_test = _get_spearman(bl_data.get("test_metrics"))
                    val_str = f"{bl_val:.4f}" if bl_val is not None else "N/A"
                    test_str = f"{bl_test:.4f}" if bl_test is not None else "N/A"
                    lines.append(f"- {bl_name}: val={val_str}, test={test_str}")

        pysr_bl = baselines.get("pysr_baseline", {})
        if isinstance(pysr_bl, dict) and pysr_bl.get("status") == "ok":
            pysr_val = _get_spearman(pysr_bl.get("val_metrics"))
            pysr_test = _get_spearman(pysr_bl.get("test_metrics"))
            val_str = f"{pysr_val:.4f}" if pysr_val is not None else "N/A"
            test_str = f"{pysr_test:.4f}" if pysr_test is not None else "N/A"
            lines.append(f"- PySR: val={val_str}, test={test_str}")

        # OOD
        if ood:
            lines.extend(["", "### OOD Generalization", ""])
            for category, cat_data in ood.items():
                if isinstance(cat_data, dict):
                    ood_s = _get_spearman(cat_data)
                    ood_str = f"{ood_s:.4f}" if ood_s is not None else "N/A"
                    valid = cat_data.get("valid_count", "?")
                    total = cat_data.get("total_count", "?")
                    lines.append(f"- {category}: spearman={ood_str} ({valid}/{total} valid)")

        # Best candidate code
        code = summary.get("best_candidate_code")
        if code:
            lines.extend(["", "### Best Candidate Code", "", "```python", code, "```"])

        # Self-correction stats
        sc = summary.get("self_correction_stats", {})
        if isinstance(sc, dict) and sc.get("enabled"):
            lines.extend(["", "### Self-Correction", ""])
            lines.append(f"- Attempted repairs: {sc.get('attempted_repairs', 0)}")
            lines.append(f"- Successful repairs: {sc.get('successful_repairs', 0)}")

        lines.append("")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_figure_data_json(experiments: dict[str, dict], output_path: Path) -> None:
    """Write JSON data for figure generation scripts.

    Args:
        experiments: dict mapping experiment name to
            {summary, baselines, ood, convergence}.
        output_path: Path to write the JSON file.
    """
    figure_data: dict[str, Any] = {}
    for name, data in experiments.items():
        summary = data.get("summary", {})
        entry: dict[str, Any] = {
            "fitness_mode": summary.get("fitness_mode"),
            "success": summary.get("success"),
            "val_spearman": _get_spearman(summary.get("val_metrics")),
            "test_spearman": _get_spearman(summary.get("test_metrics")),
            "convergence": data.get("convergence", {}),
            "baselines": data.get("baselines", {}),
            "ood": data.get("ood", {}),
        }

        # Include bounds metrics if present
        bounds = summary.get("bounds_metrics")
        if bounds:
            entry["bounds_metrics"] = bounds

        # Include self-correction stats
        sc = summary.get("self_correction_stats")
        if sc:
            entry["self_correction_stats"] = sc

        figure_data[name] = entry

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(figure_data, indent=2) + "\n", encoding="utf-8")


# ── Artifact discovery ───────────────────────────────────────────────


EXPERIMENT_DIRS = [
    "experiment_map_elites_aspl",
    "experiment_algebraic_connectivity",
    "experiment_upper_bound_aspl",
    "experiment_v2",
]


def discover_experiments(artifacts_root: Path) -> dict[str, dict]:
    """Discover and load all experiment data from the artifacts root.

    Returns dict mapping experiment name to {summary, baselines, ood, convergence}.
    """
    experiments: dict[str, dict] = {}
    root = Path(artifacts_root)

    for exp_name in EXPERIMENT_DIRS:
        exp_dir = root / exp_name
        if not exp_dir.exists():
            continue
        summary = load_experiment_summary(exp_dir)
        if not summary:
            continue

        baselines = load_baselines_summary(exp_dir)
        ood = load_ood_results(exp_dir)
        events = load_event_log(exp_dir)
        convergence = extract_convergence_data(events)

        experiments[exp_name] = {
            "summary": summary,
            "baselines": baselines,
            "ood": ood,
            "convergence": convergence,
        }

    # Discover benchmark results
    for bench_dir in sorted(root.glob("benchmark_aspl/benchmark_*")):
        if not bench_dir.is_dir():
            continue
        bench_summary_path = bench_dir / "benchmark_summary.json"
        if bench_summary_path.exists():
            bench_data = _load_json_safe(bench_summary_path)
            experiments[f"benchmark/{bench_dir.name}"] = {
                "summary": bench_data,
                "baselines": {},
                "ood": {},
                "convergence": {},
            }

    return experiments


# ── CLI ──────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="Cross-experiment analysis")
    parser.add_argument(
        "--artifacts-root",
        type=str,
        default="artifacts",
        help="Root directory containing experiment artifacts",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="analysis/results",
        help="Output directory for analysis results",
    )
    args = parser.parse_args()

    artifacts_root = Path(args.artifacts_root)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Discovering experiments in {artifacts_root}...")
    experiments = discover_experiments(artifacts_root)
    print(f"Found {len(experiments)} experiments: {list(experiments.keys())}")

    if not experiments:
        print("No experiments found. Nothing to analyze.")
        return

    # Write report
    report_path = output_dir / "analysis_report.md"
    write_analysis_report(experiments, report_path)
    print(f"Report written to {report_path}")

    # Write figure data
    figure_data_path = output_dir / "figure_data.json"
    write_figure_data_json(experiments, figure_data_path)
    print(f"Figure data written to {figure_data_path}")

    # Print summary table
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
