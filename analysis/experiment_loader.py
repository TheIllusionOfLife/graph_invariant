"""Data loading and experiment discovery for graph invariant analysis."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from experiment_analysis import (
    extract_acceptance_funnel,
    extract_bounds_diagnostics,
    extract_convergence_data,
    extract_repair_breakdown,
)

from graph_invariant.stats_utils import mean_std_ci95, safe_float  # noqa: F401

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

    # Seeded matrix artifacts: artifacts_root/neurips_matrix*/<exp>/seed_<seed>/
    for matrix_root in sorted(root.glob("neurips_matrix*")):
        if not matrix_root.is_dir():
            continue
        for seed_dir in sorted(matrix_root.glob("*/seed_*")):
            if not seed_dir.is_dir():
                continue
            rel_name = str(seed_dir.relative_to(root))
            loaded = _load_experiment_entry(seed_dir, rel_name)
            if loaded is not None:
                key, value = loaded
                experiments[key] = value

    # Ablation seeds: artifacts_root/ablation_*/<seed>/
    for ablation_root in sorted(root.glob("ablation_*")):
        if not ablation_root.is_dir():
            continue
        for seed_dir in sorted(ablation_root.glob("seed_*")):
            if not seed_dir.is_dir():
                continue
            rel_name = str(seed_dir.relative_to(root))
            loaded = _load_experiment_entry(seed_dir, rel_name)
            if loaded is not None:
                key, value = loaded
                experiments[key] = value

    return experiments


def discover_matrix_summaries(artifacts_root: Path) -> dict[str, dict[str, Any]]:
    """Load matrix_summary.json files from artifacts_root/neurips_matrix*."""
    root = Path(artifacts_root)
    discovered: dict[str, dict[str, Any]] = {}
    for matrix_root in sorted(root.glob("neurips_matrix*")):
        if not matrix_root.is_dir():
            continue
        summary_path = matrix_root / "matrix_summary.json"
        if not summary_path.exists():
            continue
        payload = _load_json_safe(summary_path)
        if not payload:
            continue
        discovered[str(matrix_root.relative_to(root))] = payload
    return discovered
