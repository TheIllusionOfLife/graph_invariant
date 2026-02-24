import argparse
import json
from pathlib import Path


def write_report(artifacts_dir: str | Path) -> Path:
    def _load_json_or_default(path: Path) -> dict:
        if not path.exists():
            return {}
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return {}

    root = Path(artifacts_dir)
    phase1_summary_path = root / "phase1_summary.json"
    baseline_summary_path = root / "baselines_summary.json"
    phase1 = _load_json_or_default(phase1_summary_path)
    baselines = _load_json_or_default(baseline_summary_path)

    lines = ["# Phase 1 Report", "", "## Run Summary", ""]
    lines.append(f"- Success: {phase1.get('success', False)}")
    lines.append(f"- Best candidate: {phase1.get('best_candidate_id', 'n/a')}")
    lines.append(f"- Stop reason: {phase1.get('stop_reason', 'n/a')}")

    val_metrics = phase1.get("val_metrics", {})
    test_metrics = phase1.get("test_metrics", {})
    if val_metrics:
        lines.extend(
            [
                "",
                "## Candidate Metrics",
                "",
                f"- Validation Spearman: {val_metrics.get('spearman', 0.0)}",
                f"- Test Spearman: {test_metrics.get('spearman', 0.0)}",
            ]
        )

    stat = baselines.get("stat_baselines", {})
    if stat:
        lines.extend(["", "## Baselines", ""])
        for name, payload in stat.items():
            lines.append(f"- {name}: {payload.get('status', 'unknown')}")

    self_correction = phase1.get("self_correction_stats", {})
    if isinstance(self_correction, dict) and self_correction:
        lines.extend(["", "## Self-Correction", ""])
        lines.append(f"- Enabled: {self_correction.get('enabled', False)}")
        lines.append(f"- Attempted repairs: {self_correction.get('attempted_repairs', 0)}")
        lines.append(f"- Successful repairs: {self_correction.get('successful_repairs', 0)}")
        lines.append(f"- Failed repairs: {self_correction.get('failed_repairs', 0)}")

    ood_path = root / "ood" / "ood_validation.json"
    ood = _load_json_or_default(ood_path)
    if ood:
        lines.extend(["", "## OOD Validation", ""])
        for category, metrics in ood.items():
            if not isinstance(metrics, dict):
                continue
            valid = metrics.get("valid_count", 0)
            total = metrics.get("total_count", 0)
            spearman = metrics.get("spearman")
            bound_score = metrics.get("bound_score")
            if spearman is not None:
                score_str = f"spearman={spearman:.4f}"
            elif bound_score is not None:
                score_str = f"bound_score={bound_score:.4f}"
            else:
                score_str = "no valid predictions"
            lines.append(f"- {category}: {score_str} ({valid}/{total} valid)")

    # MAP-Elites archive coverage from generation logs
    events_path = root / "logs" / "events.jsonl"
    if events_path.exists():
        try:
            primary_coverages: list[int] = []
            topology_coverages: list[int] = []
            with events_path.open("r", encoding="utf-8") as f:
                for line in f:
                    if '"generation_summary"' not in line:
                        continue
                    event = json.loads(line)
                    payload = event.get("payload", {})
                    primary_stats = payload.get("map_elites_stats_primary") or payload.get(
                        "map_elites_stats"
                    )
                    topology_stats = payload.get("map_elites_stats_topology")
                    if isinstance(primary_stats, dict) and "coverage" in primary_stats:
                        primary_coverages.append(int(primary_stats["coverage"]))
                    if isinstance(topology_stats, dict) and "coverage" in topology_stats:
                        topology_coverages.append(int(topology_stats["coverage"]))
            if primary_coverages:
                lines.extend(["", "## MAP-Elites Archive", ""])
                lines.append(f"- Final coverage: {primary_coverages[-1]} cells")
                lines.append(
                    f"- Coverage progression: {' -> '.join(str(c) for c in primary_coverages)}"
                )
                if topology_coverages:
                    lines.append(f"- Topology final coverage: {topology_coverages[-1]} cells")
                    lines.append(
                        f"- Topology progression: {' -> '.join(str(c) for c in topology_coverages)}"
                    )
        except (json.JSONDecodeError, OSError, KeyError):
            pass

    report_path = root / "report.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Graph invariant discovery CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    phase1 = sub.add_parser("phase1")
    phase1.add_argument("--config", type=str, default=None)
    phase1.add_argument("--resume", type=str, default=None)

    report = sub.add_parser("report")
    report.add_argument("--artifacts", type=str, required=True)

    benchmark = sub.add_parser("benchmark")
    benchmark.add_argument("--config", type=str, default=None)

    ood = sub.add_parser("ood-validate")
    ood.add_argument("--summary", type=str, required=True)
    ood.add_argument("--output", type=str, required=True)
    ood.add_argument("--seed", type=int, default=42)
    ood.add_argument("--num-large", type=int, default=100)
    ood.add_argument("--num-extreme", type=int, default=50)

    args = parser.parse_args()

    if args.command == "phase1":
        from .config import Phase1Config
        from .phase1_loop import run_phase1

        cfg = Phase1Config.from_json(args.config) if args.config else Phase1Config()
        return run_phase1(cfg, resume=args.resume)
    if args.command == "report":
        report_path = write_report(args.artifacts)
        print(f"Report written to {report_path}")
        return 0
    if args.command == "benchmark":
        from .benchmark import run_benchmark
        from .config import Phase1Config

        cfg = Phase1Config.from_json(args.config) if args.config else Phase1Config()
        return run_benchmark(cfg)
    if args.command == "ood-validate":
        from .ood_validation import run_ood_validation

        return run_ood_validation(
            summary_path=args.summary,
            output_dir=args.output,
            seed=args.seed,
            num_large=args.num_large,
            num_extreme=args.num_extreme,
        )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
