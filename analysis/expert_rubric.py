#!/usr/bin/env python
"""Expert rubric for manual scoring of top proposals from discovery runs.

After a live Harmony run, extract top-5 proposals from report.json,
fill scores (1–5) for each dimension, then run rubric_summary() to
compute the mean scores used in the NeurIPS evaluation section.

Usage:
    # Generate template from report
    python analysis/expert_rubric.py --report artifacts/harmony/astronomy/report.json \
        --output artifacts/harmony/astronomy/rubric_template.json

    # Score rubric_template.json manually (1–5 per dimension), then:
    python analysis/expert_rubric.py --load artifacts/harmony/astronomy/rubric.json --summarize
"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

_SCORE_MIN = 1
_SCORE_MAX = 5
_SCORE_FIELDS = ("domain_plausibility", "novelty", "falsifiability", "clarity")


@dataclass
class RubricEntry:
    """Scores for a single proposal across four evaluation dimensions."""

    proposal_id: str
    claim: str
    domain_plausibility: int  # 1–5: is the proposed relation real?
    novelty: int  # 1–5: not in textbooks?
    falsifiability: int  # 1–5: testable prediction?
    clarity: int  # 1–5: unambiguous claim?
    notes: str = ""


@dataclass
class ExpertRubric:
    """Collection of scored entries for one domain's top proposals."""

    domain: str
    entries: list[RubricEntry]
    scorer: str  # "self" | "domain-expert"
    scored_at: str  # ISO date string


def save_rubric(rubric: ExpertRubric, path: Path) -> None:
    """Serialize rubric to JSON at path."""
    data = {
        "domain": rubric.domain,
        "scorer": rubric.scorer,
        "scored_at": rubric.scored_at,
        "entries": [asdict(e) for e in rubric.entries],
    }
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def load_rubric(path: Path) -> ExpertRubric:
    """Deserialize rubric from JSON at path."""
    data = json.loads(path.read_text(encoding="utf-8"))
    entries = [RubricEntry(**e) for e in data["entries"]]
    return ExpertRubric(
        domain=data["domain"],
        entries=entries,
        scorer=data["scorer"],
        scored_at=data["scored_at"],
    )


def rubric_summary(rubric: ExpertRubric) -> dict[str, float]:
    """Compute mean scores per dimension and overall mean.

    Returns
    -------
    dict with keys: mean_plausibility, mean_novelty, mean_falsifiability,
    mean_clarity, overall.

    Raises
    ------
    ValueError
        If any score is outside [1, 5].
    """
    if not rubric.entries:
        return {
            "mean_plausibility": 0.0,
            "mean_novelty": 0.0,
            "mean_falsifiability": 0.0,
            "mean_clarity": 0.0,
            "overall": 0.0,
        }

    for entry in rubric.entries:
        for field_name in _SCORE_FIELDS:
            val = getattr(entry, field_name)
            if not (_SCORE_MIN <= val <= _SCORE_MAX):
                raise ValueError(
                    f"Score {field_name}={val} for proposal '{entry.proposal_id}' "
                    f"is out of range [{_SCORE_MIN}, {_SCORE_MAX}]"
                )

    n = len(rubric.entries)
    mean_plausibility = sum(e.domain_plausibility for e in rubric.entries) / n
    mean_novelty = sum(e.novelty for e in rubric.entries) / n
    mean_falsifiability = sum(e.falsifiability for e in rubric.entries) / n
    mean_clarity = sum(e.clarity for e in rubric.entries) / n
    overall = (mean_plausibility + mean_novelty + mean_falsifiability + mean_clarity) / 4

    return {
        "mean_plausibility": mean_plausibility,
        "mean_novelty": mean_novelty,
        "mean_falsifiability": mean_falsifiability,
        "mean_clarity": mean_clarity,
        "overall": overall,
    }


def _generate_template(report_path: Path, output_path: Path) -> None:
    """Generate a blank rubric template from top proposals in report.json."""
    report = json.loads(report_path.read_text(encoding="utf-8"))
    proposals = report.get("best_proposals", [])[:5]

    entries = []
    for entry in proposals:
        p = entry["proposal"]
        entries.append(
            {
                "proposal_id": p.get("id", "unknown"),
                "claim": p.get("claim", ""),
                "domain_plausibility": 0,
                "novelty": 0,
                "falsifiability": 0,
                "clarity": 0,
                "notes": "",
            }
        )

    template = {
        "domain": report.get("domain", "unknown"),
        "scorer": "self",
        "scored_at": "YYYY-MM-DD",
        "entries": entries,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(template, indent=2), encoding="utf-8")
    print(f"Template written to {output_path}")
    print("Fill in scores (1–5) for each proposal, then run with --load --summarize")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Expert rubric tool for Harmony proposals")
    parser.add_argument("--report", type=Path, help="Path to report.json (for template generation)")
    parser.add_argument("--output", type=Path, help="Output path for rubric template JSON")
    parser.add_argument("--load", type=Path, help="Path to a filled rubric JSON to summarize")
    parser.add_argument("--summarize", action="store_true", help="Print rubric summary")
    args = parser.parse_args()

    if args.report and args.output:
        _generate_template(args.report, args.output)
    elif args.load and args.summarize:
        rubric = load_rubric(args.load)
        summary = rubric_summary(rubric)
        print(f"\nRubric Summary — {rubric.domain} (scorer: {rubric.scorer})")
        print(f"  Entries scored: {len(rubric.entries)}")
        for key, val in summary.items():
            print(f"  {key}: {val:.2f}")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
