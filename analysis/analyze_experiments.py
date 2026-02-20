"""Cross-experiment analysis for graph invariant discovery results.

Usage:
    uv run python analysis/analyze_experiments.py \
        --artifacts-root artifacts/ --output analysis/results/

Reads phase1_summary.json, baselines_summary.json, ood_validation.json,
and events.jsonl from each experiment directory under artifacts-root.
Produces a markdown report and JSON figure data for generate_figures.py.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from graph_invariant.stats_utils import mean_std_ci95, safe_float

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
                    try:
                        events.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
    except OSError:
        pass
    return events


# ── Helpers ──────────────────────────────────────────────────────────


def _get_spearman(metrics: dict | None) -> float | None:
    """Safely extract spearman from a metrics dict."""
    if not isinstance(metrics, dict):
        return None
    val = metrics.get("spearman")
    if isinstance(val, (int, float)):
        return float(val)
    return None


def _ast_node_count(code: str | None) -> int | None:
    if not isinstance(code, str) or not code.strip():
        return None
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None
    return sum(1 for _ in ast.walk(tree))


def _group_seed_base(name: str) -> str | None:
    """Return a shared base id for seeded runs.

    Examples:
    - neurips_matrix/map_elites_aspl_full/seed_11 -> neurips_matrix/map_elites_aspl_full
    - benchmark/benchmark_... -> handled separately from benchmark runs payload.
    """
    parts = name.split("/")
    if parts and re.fullmatch(r"seed_\d+", parts[-1]):
        return "/".join(parts[:-1])
    return None


# ── Analysis functions ───────────────────────────────────────────────


def extract_convergence_data(events: list[dict]) -> dict[str, list]:
    """Extract generation-by-generation convergence data from event log."""
    generations: list[int] = []
    best_scores: list[float] = []
    coverages: list[int] = []

    for event in events:
        if event.get("event_type") != "generation_summary":
            continue
        payload = event.get("payload", {})
        gen = payload.get("generation")
        score = payload.get("best_val_score")
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


def extract_convergence_data_from_log_file(events_path: Path) -> dict[str, list]:
    """Extract convergence data from a JSONL log file without loading all events."""
    if not events_path.exists():
        return {"generations": [], "best_scores": []}

    generations: list[int] = []
    best_scores: list[float] = []
    coverages: list[int] = []

    try:
        with events_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if event.get("event_type") != "generation_summary":
                    continue
                payload = event.get("payload", {})
                gen = payload.get("generation")
                score = payload.get("best_val_score")
                if gen is None or score is None:
                    continue
                generations.append(gen)
                best_scores.append(score)
                me_stats = payload.get("map_elites_stats", {})
                if "coverage" in me_stats:
                    coverages.append(me_stats["coverage"])
    except OSError:
        return {"generations": [], "best_scores": []}

    result: dict[str, list] = {
        "generations": generations,
        "best_scores": best_scores,
    }
    if coverages:
        result["map_elites_coverage"] = coverages
    return result


def extract_acceptance_funnel(events: list[dict]) -> dict[str, list[float] | list[int]]:
    """Compute per-generation attempted/evaluated/rejected candidate counts."""
    per_gen: dict[int, dict[str, int]] = defaultdict(
        lambda: {"attempted": 0, "evaluated": 0, "rejected": 0}
    )

    for event in events:
        event_type = event.get("event_type")
        payload = event.get("payload", {})
        if not isinstance(payload, dict):
            continue
        generation = payload.get("generation")
        if not isinstance(generation, int):
            continue
        if event_type == "candidate_evaluated":
            per_gen[generation]["evaluated"] += 1
            per_gen[generation]["attempted"] += 1
        elif event_type == "candidate_rejected":
            per_gen[generation]["rejected"] += 1
            per_gen[generation]["attempted"] += 1

    generations = sorted(per_gen.keys())
    attempted = [per_gen[g]["attempted"] for g in generations]
    evaluated = [per_gen[g]["evaluated"] for g in generations]
    rejected = [per_gen[g]["rejected"] for g in generations]
    rates = [
        (evaluated[idx] / attempted[idx]) if attempted[idx] > 0 else 0.0
        for idx in range(len(generations))
    ]

    return {
        "generations": generations,
        "attempted": attempted,
        "evaluated": evaluated,
        "rejected": rejected,
        "acceptance_rate": rates,
    }


def extract_repair_breakdown(events: list[dict], summary: dict[str, Any]) -> dict[str, Any]:
    """Summarize repair outcomes and rejection reasons."""
    result = {
        "repair_attempts": 0,
        "repair_successes": 0,
        "repair_failures": 0,
        "rejection_reasons": {},
    }

    for event in events:
        event_type = event.get("event_type")
        payload = event.get("payload", {})
        if event_type == "candidate_repair_attempted":
            result["repair_attempts"] += 1
        elif event_type == "candidate_repair_result":
            status = payload.get("status") if isinstance(payload, dict) else None
            if status == "success":
                result["repair_successes"] += 1
            elif status == "failed":
                result["repair_failures"] += 1
        elif event_type == "candidate_rejected" and isinstance(payload, dict):
            reason = payload.get("reason")
            if isinstance(reason, str):
                current = result["rejection_reasons"].get(reason, 0)
                result["rejection_reasons"][reason] = int(current) + 1

    summary_sc = summary.get("self_correction_stats", {})
    if isinstance(summary_sc, dict):
        result["summary_attempted_repairs"] = int(summary_sc.get("attempted_repairs", 0))
        result["summary_successful_repairs"] = int(summary_sc.get("successful_repairs", 0))
        result["summary_failed_repairs"] = int(summary_sc.get("failed_repairs", 0))
        failure_categories = summary_sc.get("failure_categories", {})
        if isinstance(failure_categories, dict):
            result["failure_categories"] = failure_categories

    return result


def extract_bounds_diagnostics(events: list[dict], summary: dict[str, Any]) -> dict[str, Any]:
    """Collect bounds-specific diagnostics from summary and logs."""
    diagnostics: dict[str, Any] = {}

    bounds_metrics = summary.get("bounds_metrics", {})
    if isinstance(bounds_metrics, dict):
        val = bounds_metrics.get("val", {})
        test = bounds_metrics.get("test", {})
        if isinstance(val, dict):
            diagnostics["val_bound_score"] = safe_float(val.get("bound_score"))
            diagnostics["val_satisfaction_rate"] = safe_float(val.get("satisfaction_rate"))
            diagnostics["val_mean_gap"] = safe_float(val.get("mean_gap"))
        if isinstance(test, dict):
            diagnostics["test_bound_score"] = safe_float(test.get("bound_score"))
            diagnostics["test_satisfaction_rate"] = safe_float(test.get("satisfaction_rate"))
            diagnostics["test_mean_gap"] = safe_float(test.get("mean_gap"))

    eval_gaps: list[float] = []
    eval_satisfaction: list[float] = []
    for event in events:
        if event.get("event_type") != "candidate_evaluated":
            continue
        payload = event.get("payload", {})
        if not isinstance(payload, dict):
            continue
        gap = safe_float(payload.get("mean_gap"))
        sat = safe_float(payload.get("satisfaction_rate"))
        if gap is not None:
            eval_gaps.append(gap)
        if sat is not None:
            eval_satisfaction.append(sat)

    if eval_gaps:
        diagnostics["candidate_mean_gap_min"] = min(eval_gaps)
        diagnostics["candidate_mean_gap_max"] = max(eval_gaps)
    if eval_satisfaction:
        diagnostics["candidate_satisfaction_min"] = min(eval_satisfaction)
        diagnostics["candidate_satisfaction_max"] = max(eval_satisfaction)

    return diagnostics


def build_comparison_table(experiments: dict[str, dict]) -> list[dict[str, Any]]:
    """Build a cross-experiment comparison table."""
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

        bounds = summary.get("bounds_metrics", {})
        if isinstance(bounds, dict) and bounds:
            val_bounds = bounds.get("val", {})
            test_bounds = bounds.get("test", {})
            if isinstance(val_bounds, dict):
                row["val_bound_score"] = safe_float(val_bounds.get("bound_score"))
                row["val_satisfaction_rate"] = safe_float(val_bounds.get("satisfaction_rate"))
            if isinstance(test_bounds, dict):
                row["test_bound_score"] = safe_float(test_bounds.get("bound_score"))

        bc = summary.get("baseline_comparison", {})
        if isinstance(bc, dict) and bc:
            pysr = bc.get("pysr", {})
            if isinstance(pysr, dict):
                row["pysr_val_spearman"] = safe_float(pysr.get("val_spearman"))

        stat = baselines.get("stat_baselines", {})
        if isinstance(stat, dict):
            for bl_name, bl_data in stat.items():
                if isinstance(bl_data, dict):
                    row[f"bl_{bl_name}_val_spearman"] = _get_spearman(bl_data.get("val_metrics"))

        pysr_bl = baselines.get("pysr_baseline", {})
        if isinstance(pysr_bl, dict) and pysr_bl.get("status") == "ok":
            row["bl_pysr_val_spearman"] = _get_spearman(pysr_bl.get("val_metrics"))

        for category in ("large_random", "extreme_params", "special_topology"):
            cat_data = ood.get(category, {})
            if isinstance(cat_data, dict):
                row[f"ood_{category}_spearman"] = _get_spearman(cat_data)

        sc = summary.get("self_correction_stats", {})
        if isinstance(sc, dict):
            row["self_correction_attempts"] = sc.get("attempted_repairs", 0)
            row["self_correction_successes"] = sc.get("successful_repairs", 0)

        rows.append(row)

    return rows


def build_seed_aggregates(experiments: dict[str, dict]) -> dict[str, dict[str, Any]]:
    """Aggregate seeded runs into mean/std/CI summaries."""
    grouped: dict[str, list[tuple[str, dict[str, Any]]]] = defaultdict(list)

    # Seeded phase1 directories discovered individually.
    for name, data in experiments.items():
        base = _group_seed_base(name)
        if base is not None:
            grouped[base].append((name, data))

    # Benchmark summaries include a runs list in a single file.
    for name, data in experiments.items():
        if not name.startswith("benchmark/"):
            continue
        summary = data.get("summary", {})
        runs = summary.get("runs", []) if isinstance(summary, dict) else []
        if not isinstance(runs, list) or not runs:
            continue
        grouped[name] = []
        for run in runs:
            if not isinstance(run, dict):
                continue
            pseudo_summary = {
                "val_metrics": {"spearman": run.get("val_spearman")},
                "test_metrics": {"spearman": run.get("test_spearman")},
                "success": bool(run.get("success", False)),
            }
            grouped[name].append((f"{name}/seed_{run.get('seed')}", {"summary": pseudo_summary}))

    aggregates: dict[str, dict[str, Any]] = {}
    for base, entries in grouped.items():
        val_scores: list[float] = []
        test_scores: list[float] = []
        complexity: list[float] = []
        successes = 0
        for _, data in entries:
            summary = data.get("summary", {})
            val_s = _get_spearman(summary.get("val_metrics"))
            test_s = _get_spearman(summary.get("test_metrics"))
            if val_s is not None:
                val_scores.append(val_s)
            if test_s is not None:
                test_scores.append(test_s)
            code_nodes = _ast_node_count(summary.get("best_candidate_code"))
            if code_nodes is not None:
                complexity.append(float(code_nodes))
            if bool(summary.get("success", False)):
                successes += 1

        aggregates[base] = {
            "seed_count": len(entries),
            "success_count": successes,
            "success_rate": (successes / len(entries)) if entries else 0.0,
            "val_spearman": mean_std_ci95(val_scores),
            "test_spearman": mean_std_ci95(test_scores),
            "complexity_ast_nodes": mean_std_ci95(complexity),
            "seed_keys": [name for name, _ in entries],
        }

    return aggregates


# ── Output functions ─────────────────────────────────────────────────


def write_analysis_report(experiments: dict[str, dict], output_path: Path) -> None:
    """Write a markdown analysis report summarizing all experiments."""
    lines: list[str] = ["# Cross-Experiment Analysis Report", ""]

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

    aggregates = build_seed_aggregates(experiments)
    if aggregates:
        lines.extend(["## Multi-Seed Aggregates", ""])
        lines.append(
            "| Experiment Group | Seeds | Val mean±std | Val CI95 | Test mean±std | Test CI95 |"
        )
        lines.append("| --- | --- | --- | --- | --- | --- |")
        for group, payload in sorted(aggregates.items()):
            val = payload["val_spearman"]
            test = payload["test_spearman"]
            val_mean_std = (
                f"{val['mean']:.4f} ± {val['std']:.4f}" if val.get("mean") is not None else "N/A"
            )
            val_ci = (
                f"±{val['ci95_half_width']:.4f}"
                if val.get("ci95_half_width") is not None
                else "N/A"
            )
            test_mean_std = (
                f"{test['mean']:.4f} ± {test['std']:.4f}" if test.get("mean") is not None else "N/A"
            )
            test_ci = (
                f"±{test['ci95_half_width']:.4f}"
                if test.get("ci95_half_width") is not None
                else "N/A"
            )
            lines.append(
                "| {group} | {seed_count} | {val_mean_std} | {val_ci} | "
                "{test_mean_std} | {test_ci} |".format(
                    group=group,
                    seed_count=payload["seed_count"],
                    val_mean_std=val_mean_std,
                    val_ci=val_ci,
                    test_mean_std=test_mean_std,
                    test_ci=test_ci,
                )
            )
        lines.append("")

    for name, data in experiments.items():
        summary = data.get("summary", {})
        convergence = data.get("convergence", {})
        baselines = data.get("baselines", {})
        ood = data.get("ood", {})

        lines.extend([f"## {name}", ""])
        lines.append(f"- Fitness mode: {summary.get('fitness_mode', 'unknown')}")
        lines.append(f"- Success: {summary.get('success', False)}")
        lines.append(f"- Stop reason: {summary.get('stop_reason', 'N/A')}")
        lines.append(f"- Final generation: {summary.get('final_generation', 'N/A')}")

        val_s = _get_spearman(summary.get("val_metrics"))
        test_s = _get_spearman(summary.get("test_metrics"))
        if val_s is not None:
            lines.append(f"- Validation Spearman: {val_s:.4f}")
        if test_s is not None:
            lines.append(f"- Test Spearman: {test_s:.4f}")

        code_nodes = _ast_node_count(summary.get("best_candidate_code"))
        if code_nodes is not None:
            lines.append(f"- Best formula AST nodes: {code_nodes}")

        bounds = data.get("bounds_diagnostics", {})
        if bounds:
            lines.extend(["", "### Bounds Diagnostics", ""])
            for key in sorted(bounds.keys()):
                lines.append(f"- {key}: {bounds[key]}")

        if convergence:
            scores = convergence.get("best_scores", [])
            if scores:
                lines.append(f"- Convergence: {scores[0]:.3f} -> {scores[-1]:.3f}")
            coverage = convergence.get("map_elites_coverage", [])
            if coverage:
                lines.append(f"- MAP-Elites coverage: {coverage[0]} -> {coverage[-1]} cells")

        funnel = data.get("acceptance_funnel", {})
        if funnel and funnel.get("generations"):
            final_idx = len(funnel["generations"]) - 1
            lines.extend(["", "### Acceptance Funnel", ""])
            lines.append(f"- Final generation attempted: {funnel['attempted'][final_idx]}")
            lines.append(f"- Final generation accepted: {funnel['evaluated'][final_idx]}")
            lines.append(
                f"- Final generation acceptance rate: {funnel['acceptance_rate'][final_idx]:.3f}"
            )

        repair = data.get("repair_breakdown", {})
        if repair:
            lines.extend(["", "### Repair Breakdown", ""])
            lines.append(f"- Repair attempts: {repair.get('repair_attempts', 0)}")
            lines.append(f"- Repair successes: {repair.get('repair_successes', 0)}")
            lines.append(f"- Repair failures: {repair.get('repair_failures', 0)}")

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

        if ood:
            lines.extend(["", "### OOD Generalization", ""])
            for category, cat_data in ood.items():
                if isinstance(cat_data, dict):
                    ood_s = _get_spearman(cat_data)
                    ood_str = f"{ood_s:.4f}" if ood_s is not None else "N/A"
                    valid = cat_data.get("valid_count", "?")
                    total = cat_data.get("total_count", "?")
                    lines.append(f"- {category}: spearman={ood_str} ({valid}/{total} valid)")

        code = summary.get("best_candidate_code")
        if code:
            lines.extend(["", "### Best Candidate Code", "", "```python", code, "```"])

        lines.append("")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_figure_data_json(experiments: dict[str, dict], output_path: Path) -> None:
    """Write JSON data for figure generation scripts."""
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
            "acceptance_funnel": data.get("acceptance_funnel", {}),
            "repair_breakdown": data.get("repair_breakdown", {}),
            "bounds_diagnostics": data.get("bounds_diagnostics", {}),
        }

        bounds = summary.get("bounds_metrics")
        if bounds:
            entry["bounds_metrics"] = bounds

        sc = summary.get("self_correction_stats")
        if sc:
            entry["self_correction_stats"] = sc

        runs = summary.get("runs")
        if runs:
            entry["runs"] = runs

        figure_data[name] = entry

    figure_data["__aggregates__"] = build_seed_aggregates(experiments)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(figure_data, indent=2) + "\n", encoding="utf-8")


# ── Artifact discovery ───────────────────────────────────────────────


EXPERIMENT_DIRS = [
    "experiment_map_elites_aspl",
    "experiment_algebraic_connectivity",
    "experiment_upper_bound_aspl",
    "experiment_v2",
]


def _load_experiment_entry(exp_dir: Path, name: str) -> tuple[str, dict] | None:
    summary = load_experiment_summary(exp_dir)
    if not summary:
        return None
    baselines = load_baselines_summary(exp_dir)
    ood = load_ood_results(exp_dir)
    events = load_event_log(exp_dir)
    convergence = extract_convergence_data(events)
    acceptance_funnel = extract_acceptance_funnel(events)
    repair_breakdown = extract_repair_breakdown(events, summary)
    bounds_diagnostics = extract_bounds_diagnostics(events, summary)
    return (
        name,
        {
            "summary": summary,
            "baselines": baselines,
            "ood": ood,
            "convergence": convergence,
            "acceptance_funnel": acceptance_funnel,
            "repair_breakdown": repair_breakdown,
            "bounds_diagnostics": bounds_diagnostics,
        },
    )


def discover_experiments(artifacts_root: Path) -> dict[str, dict]:
    """Discover and load all experiment data from the artifacts root."""
    experiments: dict[str, dict] = {}
    root = Path(artifacts_root)

    for exp_name in EXPERIMENT_DIRS:
        exp_dir = root / exp_name
        if not exp_dir.exists():
            continue
        loaded = _load_experiment_entry(exp_dir, exp_name)
        if loaded is not None:
            key, value = loaded
            experiments[key] = value

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
                "acceptance_funnel": {},
                "repair_breakdown": {},
                "bounds_diagnostics": {},
            }

    # Seeded matrix artifacts: artifacts_root/neurips_matrix/<exp>/seed_<seed>/
    matrix_root = root / "neurips_matrix"
    if matrix_root.exists():
        for seed_dir in sorted(matrix_root.glob("*/seed_*")):
            if not seed_dir.is_dir():
                continue
            rel_name = str(seed_dir.relative_to(root))
            loaded = _load_experiment_entry(seed_dir, rel_name)
            if loaded is not None:
                key, value = loaded
                experiments[key] = value

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

    report_path = output_dir / "analysis_report.md"
    write_analysis_report(experiments, report_path)
    print(f"Report written to {report_path}")

    figure_data_path = output_dir / "figure_data.json"
    write_figure_data_json(experiments, figure_data_path)
    print(f"Figure data written to {figure_data_path}")

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
