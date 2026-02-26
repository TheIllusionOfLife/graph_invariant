#!/usr/bin/env python
"""Harmony report generator — loads checkpoint + events, writes JSON + Markdown summary.

Usage:
    python analysis/harmony_report.py --output-dir artifacts/harmony/astronomy --domain astronomy
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

# Ensure src/ is on the path when run from the repo root
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from harmony.map_elites import archive_stats, deserialize_archive
from harmony.state import load_state


def generate_report(output_dir: Path, domain: str) -> dict[str, Any]:
    """Load checkpoint.json + harmony_events.jsonl → structured report dict.

    Writes {output_dir}/report.json and {output_dir}/report.md.

    Parameters
    ----------
    output_dir:
        Directory containing checkpoint.json and logs/harmony_events.jsonl.
    domain:
        Domain label (e.g. "astronomy") written into the report.

    Returns
    -------
    Full structured report as a plain dict (JSON-serialisable).
    """
    output_dir = Path(output_dir)
    checkpoint_path = output_dir / "checkpoint.json"
    events_path = output_dir / "logs" / "harmony_events.jsonl"

    state = load_state(checkpoint_path)

    # Parse events JSONL for valid_rate curve
    valid_rate_curve: list[dict[str, Any]] = []
    if events_path.exists():
        with events_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if event.get("event") == "generation_summary":
                    valid_rate_curve.append(
                        {
                            "generation": event.get("generation"),
                            "valid_rate": event.get("valid_rate"),
                            "best_harmony_gain": event.get("best_harmony_gain"),
                        }
                    )

    # Archive stats + top proposals
    best_proposals: list[dict[str, Any]] = []
    archive_stats_data: dict[str, Any] = {}
    heatmap_data: list[dict[str, Any]] = []

    if state.archive is not None:
        archive = deserialize_archive(state.archive)
        archive_stats_data = dict(archive_stats(archive))

        # Top-5 cells sorted by fitness descending
        sorted_cells = sorted(
            archive.cells.items(),
            key=lambda item: item[1].fitness_signal,
            reverse=True,
        )
        for (row, col), cell in sorted_cells[:5]:
            best_proposals.append(
                {
                    "row": row,
                    "col": col,
                    "fitness_signal": cell.fitness_signal,
                    "proposal": cell.proposal.to_dict(),
                }
            )

        # Heatmap: all occupied cells
        for (row, col), cell in archive.cells.items():
            heatmap_data.append(
                {
                    "row": row,
                    "col": col,
                    "fitness": cell.fitness_signal,
                }
            )

    report: dict[str, Any] = {
        "domain": domain,
        "experiment_id": state.experiment_id,
        "total_generations": state.generation,
        "best_harmony_gain": state.best_harmony_gain,
        "no_improve_count": state.no_improve_count,
        "valid_rate_curve": valid_rate_curve,
        "archive_stats": archive_stats_data,
        "best_proposals": best_proposals,
        "heatmap_data": heatmap_data,
    }

    # Write JSON
    report_json_path = output_dir / "report.json"
    with report_json_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    # Write Markdown
    _write_markdown(report, output_dir / "report.md")

    return report


def _write_markdown(report: dict[str, Any], path: Path) -> None:
    lines: list[str] = []
    lines.append(f"# Harmony Discovery Report — {report['domain']}")
    lines.append("")
    lines.append(f"**Experiment ID:** {report['experiment_id']}")
    lines.append(f"**Total Generations:** {report['total_generations']}")
    lines.append(f"**Best Harmony Gain:** {report['best_harmony_gain']:.4f}")
    lines.append(f"**No-Improve Count:** {report['no_improve_count']}")
    lines.append("")

    stats = report.get("archive_stats", {})
    if stats:
        lines.append("## Archive Statistics")
        lines.append("")
        lines.append(
            f"- Coverage: {stats.get('coverage', 0)} / {stats.get('total_cells', 0)} cells"
        )
        lines.append(f"- Best Fitness: {stats.get('best_fitness', 0.0):.4f}")
        lines.append(f"- Mean Fitness: {stats.get('mean_fitness', 0.0):.4f}")
        lines.append("")

    proposals = report.get("best_proposals", [])
    if proposals:
        lines.append("## Top Proposals")
        lines.append("")
        for i, entry in enumerate(proposals, 1):
            p = entry["proposal"]
            lines.append(f"### {i}. {p.get('claim', '(no claim)')}")
            lines.append(f"- **Type:** {p.get('proposal_type', 'unknown')}")
            lines.append(f"- **Fitness:** {entry['fitness_signal']:.4f}")
            lines.append(f"- **Justification:** {p.get('justification', '')}")
            lines.append(f"- **Falsification:** {p.get('falsification_condition', '')}")
            lines.append("")

    curve = report.get("valid_rate_curve", [])
    if curve:
        lines.append("## Valid Rate Curve")
        lines.append("")
        lines.append("| Generation | Valid Rate | Best Gain |")
        lines.append("|-----------|------------|-----------|")
        for entry in curve:
            vr = entry.get("valid_rate") or 0.0
            bg = entry.get("best_harmony_gain") or 0.0
            lines.append(f"| {entry.get('generation', '?')} | {vr:.3f} | {bg:.4f} |")
        lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Generate Harmony discovery report")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--domain", type=str, required=True)
    args = parser.parse_args()

    report = generate_report(args.output_dir, args.domain)
    print(f"Report written to {args.output_dir / 'report.json'}")
    print(f"Best harmony gain: {report['best_harmony_gain']:.4f}")
    print(f"Total generations: {report['total_generations']}")
    proposals = report.get("best_proposals", [])
    print(f"Top proposals in archive: {len(proposals)}")


if __name__ == "__main__":
    main()
