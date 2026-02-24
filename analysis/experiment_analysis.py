"""Analysis functions for graph invariant experiment data."""

from __future__ import annotations

import ast
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from graph_invariant.stats_utils import mean_std_ci95, safe_float

# ── Helpers ──────────────────────────────────────────────────────────


def get_spearman(metrics: dict | None) -> float | None:
    """Safely extract spearman from a metrics dict."""
    if not isinstance(metrics, dict):
        return None
    val = metrics.get("spearman")
    if isinstance(val, (int, float)):
        return float(val)
    return None


def ast_node_count(code: str | None) -> int | None:
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


def _clamp_ci95_to_metric_bounds(
    stats: dict[str, Any], lower: float = -1.0, upper: float = 1.0
) -> dict[str, Any]:
    """Clamp CI95 half-width so implied interval stays within [lower, upper]."""
    mean = safe_float(stats.get("mean"))
    ci = safe_float(stats.get("ci95_half_width"))
    if mean is None or ci is None:
        return stats
    max_half_width = max(0.0, min(upper - mean, mean - lower))
    if ci > max_half_width:
        clamped = dict(stats)
        clamped["ci95_half_width"] = max_half_width
        clamped["ci95_clamped_to_bounds"] = True
        return clamped
    return stats


def normalize_candidate_code_for_report(code: str) -> str:
    """Ensure emitted candidate snippets are self-contained for readers."""
    if "np." in code and not re.search(
        r"(^|\n)\s*(import\s+numpy\s+as\s+np|from\s+numpy\s+import\s+)", code
    ):
        return f"import numpy as np\n\n{code}"
    return code


def _status_code(value: Any, default: int = 1) -> int:
    """Parse matrix run status into a numeric return-code style integer."""
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        stripped = value.strip()
        if re.fullmatch(r"[+-]?\d+", stripped):
            return int(stripped)
    return default


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
            "val_spearman": get_spearman(summary.get("val_metrics")),
            "test_spearman": get_spearman(summary.get("test_metrics")),
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
                    row[f"bl_{bl_name}_val_spearman"] = get_spearman(bl_data.get("val_metrics"))

        pysr_bl = baselines.get("pysr_baseline", {})
        if isinstance(pysr_bl, dict) and pysr_bl.get("status") == "ok":
            row["bl_pysr_val_spearman"] = get_spearman(pysr_bl.get("val_metrics"))

        for category in ("large_random", "extreme_params", "special_topology"):
            cat_data = ood.get(category, {})
            if isinstance(cat_data, dict):
                row[f"ood_{category}_spearman"] = get_spearman(cat_data)

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
            val_s = get_spearman(summary.get("val_metrics"))
            test_s = get_spearman(summary.get("test_metrics"))
            if val_s is not None:
                val_scores.append(val_s)
            if test_s is not None:
                test_scores.append(test_s)
            code_nodes = ast_node_count(summary.get("best_candidate_code"))
            if code_nodes is not None:
                complexity.append(float(code_nodes))
            if bool(summary.get("success", False)):
                successes += 1

        val_stats = _clamp_ci95_to_metric_bounds(mean_std_ci95(val_scores))
        test_stats = _clamp_ci95_to_metric_bounds(mean_std_ci95(test_scores))

        aggregates[base] = {
            "seed_count": len(entries),
            "success_count": successes,
            "success_rate": (successes / len(entries)) if entries else 0.0,
            "val_spearman": val_stats,
            "test_spearman": test_stats,
            "val_ci95_clamped_to_bounds": bool(val_stats.get("ci95_clamped_to_bounds", False)),
            "test_ci95_clamped_to_bounds": bool(test_stats.get("ci95_clamped_to_bounds", False)),
            "complexity_ast_nodes": mean_std_ci95(complexity),
            "seed_keys": [name for name, _ in entries],
        }

    return aggregates


def build_appendix_payload(
    experiments: dict[str, dict], matrix_summaries: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    """Build structured appendix payloads for rebuttal-ready tables."""
    aggregates = build_seed_aggregates(experiments)

    small_data_tradeoff: dict[str, dict[str, Any]] = {}
    for group, payload in sorted(aggregates.items()):
        label = group.split("/")[-1]
        if "small_data" in label or "aspl_full" in label:
            small_data_tradeoff[group] = payload

    bounds_diagnostics: dict[str, dict[str, Any]] = {}
    for name, data in sorted(experiments.items()):
        bounds = data.get("bounds_diagnostics", {})
        if not isinstance(bounds, dict):
            continue
        val_sat = bounds.get("val_satisfaction_rate")
        test_sat = bounds.get("test_satisfaction_rate")
        if val_sat is None and test_sat is None:
            continue
        bounds_diagnostics[name] = {
            "val_bound_score": bounds.get("val_bound_score"),
            "val_satisfaction_rate": val_sat,
            "val_mean_gap": bounds.get("val_mean_gap"),
            "test_bound_score": bounds.get("test_bound_score"),
            "test_satisfaction_rate": test_sat,
            "test_mean_gap": bounds.get("test_mean_gap"),
        }

    runtime_by_experiment: dict[str, list[float]] = defaultdict(list)
    completion_by_experiment: dict[str, dict[str, int]] = defaultdict(
        lambda: {"total": 0, "completed": 0, "criteria_success": 0}
    )
    seed_notes: dict[str, str] = {}
    for matrix_root, summary in matrix_summaries.items():
        runs = summary.get("runs", []) if isinstance(summary, dict) else []
        if not isinstance(runs, list):
            continue
        grouped_runs: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for run in runs:
            if not isinstance(run, dict):
                continue
            experiment = run.get("experiment")
            if not isinstance(experiment, str) or not experiment:
                continue
            grouped_runs[experiment].append(run)
            group_key = f"{matrix_root}/{experiment}"
            status = _status_code(run.get("status", 1), default=1)
            completion_by_experiment[group_key]["total"] += 1
            if status == 0:
                completion_by_experiment[group_key]["completed"] += 1
            if bool(run.get("success", False)):
                completion_by_experiment[group_key]["criteria_success"] += 1
            duration = safe_float(run.get("duration_sec"))
            if duration is not None:
                runtime_by_experiment[group_key].append(duration)

        for experiment, exp_runs in grouped_runs.items():
            total = len(exp_runs)
            completed = sum(1 for run in exp_runs if _status_code(run.get("status", 1)) == 0)
            if total == 0 or completed == total:
                continue
            missing = [run for run in exp_runs if _status_code(run.get("status", 1)) != 0]
            missing_seeds = ", ".join(
                f"seed_{run.get('seed')}"
                for run in missing
                if isinstance(run.get("seed"), int | str)
            )
            reason = next(
                (
                    str(run.get("note"))
                    for run in missing
                    if isinstance(run.get("note"), str) and run.get("note")
                ),
                "incomplete run",
            )
            group = f"{matrix_root}/{experiment}"
            seed_notes[group] = (
                f"{experiment} uses {completed}/{total} completed seeds; "
                f"missing {missing_seeds} ({reason})."
            )

    runtime_summary: dict[str, dict[str, Any]] = {}
    for group_key, durations in sorted(runtime_by_experiment.items()):
        counts = completion_by_experiment[group_key]
        runtime_summary[group_key] = {
            "duration_sec": mean_std_ci95(durations),
            "total_runs": counts["total"],
            "completed_runs": counts["completed"],
            "criteria_success_runs": counts["criteria_success"],
        }

    # --- Self-correction failure breakdown ---
    sc_failure_breakdown: dict[str, dict] = {}
    for name, data in sorted(experiments.items()):
        summary = data.get("summary", {})
        sc_stats = summary.get("self_correction_stats", {})
        if not isinstance(sc_stats, dict) or not sc_stats.get("enabled", False):
            continue
        failure_cats = sc_stats.get("failure_categories", {})
        if not isinstance(failure_cats, dict):
            continue
        sc_failure_breakdown[name] = {
            "attempted": sc_stats.get("attempted_repairs", "N/A"),
            "no_valid_train_predictions": failure_cats.get("no_valid_train_predictions", "N/A"),
            "below_train_threshold": failure_cats.get("below_train_threshold", "N/A"),
            "below_novelty_threshold": failure_cats.get("below_novelty_threshold", "N/A"),
        }

    # --- Compute profile ---
    compute_profile: dict[str, dict] = {}
    for name, data in sorted(experiments.items()):
        summary = data.get("summary", {})
        gens = summary.get("final_generation")
        if not isinstance(gens, int):
            continue
        cfg = summary.get("config", {}) if isinstance(summary.get("config"), dict) else {}
        # Read islands and population from persisted config; fall back to system defaults
        island_temps = cfg.get("island_temperatures", [])
        islands = len(island_temps) if island_temps else 4
        pop = int(cfg.get("population_size", 5))
        pysr_sec = cfg.get("pysr_timeout_in_seconds")
        pysr_budget = f"{int(pysr_sec)} s" if isinstance(pysr_sec, (int, float)) else "60 s"
        sc_stats = summary.get("self_correction_stats", {})
        repair_calls = sc_stats.get("attempted_repairs", 0) if isinstance(sc_stats, dict) else 0
        gen_calls = gens * islands * pop
        compute_profile[name] = {
            "generations": gens,
            "llm_gen_calls": gen_calls,
            "repair_calls": repair_calls,
            "total_calls": gen_calls + repair_calls,
            "pysr_budget": pysr_budget,
        }

    return {
        "appendix_seed_aggregates": aggregates,
        "appendix_seed_notes": seed_notes,
        "appendix_small_data_tradeoff": small_data_tradeoff,
        "appendix_bounds_diagnostics": bounds_diagnostics,
        "appendix_runtime_summary": runtime_summary,
        "sc_failure_breakdown": sc_failure_breakdown,
        "compute_profile": compute_profile,
    }
