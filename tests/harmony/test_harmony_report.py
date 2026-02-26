"""Tests for analysis.harmony_report — report generation from checkpoint + events.

TDD: written BEFORE implementation. Verifies:
  - generate_report() returns dict with expected keys
  - Top-5 proposals extracted from archive by fitness
  - report.md written after call
  - Empty archive handled gracefully
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

# Ensure analysis/ is importable
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "analysis"))

from harmony.map_elites import HarmonyMapElites, serialize_archive, try_insert
from harmony.proposals.types import Proposal, ProposalType
from harmony.state import HarmonySearchState, save_state


def _make_proposal(pid: str, claim_suffix: str = "") -> Proposal:
    return Proposal(
        id=pid,
        proposal_type=ProposalType.ADD_EDGE,
        claim=f"Claim about domain structure {claim_suffix}",
        justification="This follows from observed patterns in the data.",
        falsification_condition="If the relation cannot be reproduced experimentally.",
        kg_domain="astronomy",
        source_entity="hot_jupiter",
        target_entity="disk_instability",
        edge_type="EXPLAINS",
    )


def _write_events_jsonl(events_path: Path, events: list[dict]) -> None:
    events_path.parent.mkdir(parents=True, exist_ok=True)
    with events_path.open("w", encoding="utf-8") as f:
        for event in events:
            f.write(json.dumps(event) + "\n")


def _make_synthetic_checkpoint(
    tmp_path: Path,
    archive: HarmonyMapElites | None = None,
    generation: int = 5,
    best_gain: float = 0.12,
) -> Path:
    """Create a minimal checkpoint.json in tmp_path and return the path."""
    state = HarmonySearchState(
        experiment_id="test-exp-abc",
        generation=generation,
        islands={0: [], 1: []},
        best_harmony_gain=best_gain,
        no_improve_count=2,
        archive=serialize_archive(archive) if archive is not None else None,
    )
    checkpoint_path = tmp_path / "checkpoint.json"
    save_state(state, checkpoint_path)
    return checkpoint_path


class TestGenerateReport:
    def test_generate_report_from_synthetic_checkpoint(self, tmp_path: Path):
        """generate_report returns dict with all required top-level keys."""
        from harmony_report import generate_report

        _make_synthetic_checkpoint(tmp_path)

        events_path = tmp_path / "logs" / "harmony_events.jsonl"
        _write_events_jsonl(
            events_path,
            [
                {
                    "event": "generation_summary",
                    "generation": 1,
                    "valid_rate": 0.6,
                    "best_harmony_gain": 0.08,
                },
                {
                    "event": "generation_summary",
                    "generation": 2,
                    "valid_rate": 0.7,
                    "best_harmony_gain": 0.12,
                },
            ],
        )

        report = generate_report(tmp_path, "astronomy")

        required_keys = {
            "domain",
            "experiment_id",
            "total_generations",
            "best_harmony_gain",
            "no_improve_count",
            "valid_rate_curve",
            "archive_stats",
            "best_proposals",
            "heatmap_data",
        }
        assert required_keys.issubset(report.keys())
        assert report["domain"] == "astronomy"
        assert report["experiment_id"] == "test-exp-abc"
        assert report["total_generations"] == 5
        assert report["best_harmony_gain"] == pytest.approx(0.12)
        assert len(report["valid_rate_curve"]) == 2
        assert report["valid_rate_curve"][0]["valid_rate"] == pytest.approx(0.6)

    def test_report_extracts_top5_proposals(self, tmp_path: Path):
        """Top-5 proposals by fitness_signal are extracted from archive."""
        from harmony_report import generate_report

        archive = HarmonyMapElites(num_bins=10)
        # Insert 10 proposals with different fitness signals
        for i in range(10):
            p = _make_proposal(f"p{i}", str(i))
            try_insert(
                archive,
                p,
                fitness_signal=float(i) / 10.0,
                descriptor=(i * 0.09, i * 0.09),
            )
        _make_synthetic_checkpoint(tmp_path, archive=archive)

        report = generate_report(tmp_path, "astronomy")

        proposals = report["best_proposals"]
        assert len(proposals) == 5
        # Should be sorted by fitness descending
        fitnesses = [p["fitness_signal"] for p in proposals]
        assert fitnesses == sorted(fitnesses, reverse=True)
        # Top proposal should be the one with fitness 0.9 (index 9)
        assert proposals[0]["fitness_signal"] == pytest.approx(0.9)

    def test_report_writes_json(self, tmp_path: Path):
        """report.json is written to output_dir after generate_report call."""
        from harmony_report import generate_report

        _make_synthetic_checkpoint(tmp_path)
        generate_report(tmp_path, "physics")

        json_path = tmp_path / "report.json"
        assert json_path.exists()
        data = json.loads(json_path.read_text())
        assert data["domain"] == "physics"

    def test_report_writes_markdown(self, tmp_path: Path):
        """report.md is written to output_dir after generate_report call."""
        from harmony_report import generate_report

        _make_synthetic_checkpoint(tmp_path)
        generate_report(tmp_path, "materials")

        md_path = tmp_path / "report.md"
        assert md_path.exists()
        content = md_path.read_text()
        assert "materials" in content.lower()

    def test_report_handles_empty_archive(self, tmp_path: Path):
        """No archive → best_proposals is empty list, report still generated."""
        from harmony_report import generate_report

        _make_synthetic_checkpoint(tmp_path, archive=None)
        report = generate_report(tmp_path, "astronomy")

        assert report["best_proposals"] == []
        assert report["heatmap_data"] == []
        assert report["archive_stats"] == {}

    def test_report_handles_missing_events_file(self, tmp_path: Path):
        """Missing events JSONL → valid_rate_curve is empty, no crash."""
        from harmony_report import generate_report

        _make_synthetic_checkpoint(tmp_path)
        # Do NOT create events file
        report = generate_report(tmp_path, "astronomy")

        assert report["valid_rate_curve"] == []

    def test_report_handles_malformed_jsonl(self, tmp_path: Path):
        """Malformed JSONL lines are skipped without crashing report generation."""
        from harmony_report import generate_report

        _make_synthetic_checkpoint(tmp_path)

        events_path = tmp_path / "logs" / "harmony_events.jsonl"
        events_path.parent.mkdir(parents=True, exist_ok=True)
        with events_path.open("w", encoding="utf-8") as f:
            f.write('{"event":"generation_summary","generation":1,"valid_rate":0.5}\n')
            f.write("TRUNCATED LINE {{not json\n")
            f.write('{"event":"generation_summary","generation":2,"valid_rate":0.7}\n')

        report = generate_report(tmp_path, "astronomy")

        # Both valid lines parsed; the corrupted line is skipped
        assert len(report["valid_rate_curve"]) == 2

    def test_report_heatmap_data_has_row_col_fitness(self, tmp_path: Path):
        """Each heatmap entry has row, col, fitness keys."""
        from harmony_report import generate_report

        archive = HarmonyMapElites(num_bins=5)
        try_insert(archive, _make_proposal("p1"), fitness_signal=0.5, descriptor=(0.2, 0.8))
        try_insert(archive, _make_proposal("p2"), fitness_signal=0.3, descriptor=(0.6, 0.4))
        _make_synthetic_checkpoint(tmp_path, archive=archive)

        report = generate_report(tmp_path, "astronomy")

        assert len(report["heatmap_data"]) == 2
        for entry in report["heatmap_data"]:
            assert "row" in entry
            assert "col" in entry
            assert "fitness" in entry
