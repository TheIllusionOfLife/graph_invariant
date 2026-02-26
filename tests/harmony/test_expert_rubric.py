"""Tests for analysis.expert_rubric — ExpertRubric dataclass + save/load/score.

TDD: written BEFORE implementation. Verifies:
  - RubricEntry and ExpertRubric instantiate correctly
  - save_rubric / load_rubric round-trip preserves all fields
  - rubric_summary computes correct averages
  - Out-of-range scores raise ValueError
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "analysis"))


class TestRubricEntry:
    def test_rubric_entry_instantiates(self):
        from expert_rubric import RubricEntry

        entry = RubricEntry(
            proposal_id="p001",
            claim="Hot Jupiters form via disk instability in early protoplanetary disks.",
            domain_plausibility=4,
            novelty=3,
            falsifiability=5,
            clarity=4,
        )
        assert entry.proposal_id == "p001"
        assert entry.domain_plausibility == 4
        assert entry.notes == ""

    def test_rubric_entry_accepts_notes(self):
        from expert_rubric import RubricEntry

        entry = RubricEntry(
            proposal_id="p002",
            claim="A claim about cosmic structure and its formation.",
            domain_plausibility=3,
            novelty=4,
            falsifiability=3,
            clarity=5,
            notes="Borderline — needs literature check.",
        )
        assert entry.notes == "Borderline — needs literature check."


class TestExpertRubric:
    def test_expert_rubric_instantiates(self):
        from expert_rubric import ExpertRubric, RubricEntry

        entry = RubricEntry(
            proposal_id="p001",
            claim="Disk instability explains hot Jupiter formation.",
            domain_plausibility=4,
            novelty=3,
            falsifiability=5,
            clarity=4,
        )
        rubric = ExpertRubric(
            domain="astronomy",
            entries=[entry],
            scorer="self",
            scored_at="2026-02-26",
        )
        assert rubric.domain == "astronomy"
        assert len(rubric.entries) == 1

    def test_rubric_save_load_round_trip(self, tmp_path: Path):
        from expert_rubric import ExpertRubric, RubricEntry, load_rubric, save_rubric

        entries = [
            RubricEntry(
                proposal_id="p001",
                claim="Disk instability explains hot Jupiter formation periods.",
                domain_plausibility=4,
                novelty=3,
                falsifiability=5,
                clarity=4,
                notes="Strong claim",
            ),
            RubricEntry(
                proposal_id="p002",
                claim="Stellar metallicity generalizes planet occurrence rates broadly.",
                domain_plausibility=5,
                novelty=4,
                falsifiability=4,
                clarity=3,
            ),
        ]
        rubric = ExpertRubric(
            domain="astronomy",
            entries=entries,
            scorer="domain-expert",
            scored_at="2026-02-26",
        )
        path = tmp_path / "rubric.json"
        save_rubric(rubric, path)

        loaded = load_rubric(path)

        assert loaded.domain == rubric.domain
        assert loaded.scorer == rubric.scorer
        assert loaded.scored_at == rubric.scored_at
        assert len(loaded.entries) == 2
        assert loaded.entries[0].proposal_id == "p001"
        assert loaded.entries[0].domain_plausibility == 4
        assert loaded.entries[0].notes == "Strong claim"
        assert loaded.entries[1].proposal_id == "p002"

    def test_save_rubric_writes_valid_json(self, tmp_path: Path):
        import json

        from expert_rubric import ExpertRubric, RubricEntry, save_rubric

        rubric = ExpertRubric(
            domain="physics",
            entries=[
                RubricEntry(
                    proposal_id="p001",
                    claim="Symmetry breaking explains mass generation in standard model.",
                    domain_plausibility=5,
                    novelty=2,
                    falsifiability=5,
                    clarity=5,
                )
            ],
            scorer="self",
            scored_at="2026-02-26",
        )
        path = tmp_path / "rubric.json"
        save_rubric(rubric, path)

        data = json.loads(path.read_text())
        assert data["domain"] == "physics"
        assert data["scorer"] == "self"
        assert len(data["entries"]) == 1
        assert data["entries"][0]["proposal_id"] == "p001"


class TestRubricSummary:
    def _make_rubric(self, scores: list[tuple[int, int, int, int]]) -> object:
        from expert_rubric import ExpertRubric, RubricEntry

        entries = [
            RubricEntry(
                proposal_id=f"p{i:03d}",
                claim=f"Claim {i} about the domain structure and observations.",
                domain_plausibility=d,
                novelty=n,
                falsifiability=f,
                clarity=c,
            )
            for i, (d, n, f, c) in enumerate(scores)
        ]
        return ExpertRubric(domain="test", entries=entries, scorer="self", scored_at="2026-02-26")

    def test_rubric_summary_averages_correctly(self):
        from expert_rubric import rubric_summary

        # 2 entries: scores are symmetric so means equal the scores
        rubric = self._make_rubric([(4, 3, 5, 4), (2, 5, 3, 4)])
        summary = rubric_summary(rubric)

        assert summary["mean_plausibility"] == pytest.approx(3.0)
        assert summary["mean_novelty"] == pytest.approx(4.0)
        assert summary["mean_falsifiability"] == pytest.approx(4.0)
        assert summary["mean_clarity"] == pytest.approx(4.0)
        assert summary["overall"] == pytest.approx((3.0 + 4.0 + 4.0 + 4.0) / 4)

    def test_rubric_summary_single_entry(self):
        from expert_rubric import rubric_summary

        rubric = self._make_rubric([(5, 5, 5, 5)])
        summary = rubric_summary(rubric)

        assert summary["mean_plausibility"] == pytest.approx(5.0)
        assert summary["overall"] == pytest.approx(5.0)

    def test_rubric_summary_rejects_out_of_range_high(self):
        from expert_rubric import rubric_summary

        rubric = self._make_rubric([(6, 3, 3, 3)])  # 6 is out of [1, 5]
        with pytest.raises(ValueError, match="out of range"):
            rubric_summary(rubric)

    def test_rubric_summary_rejects_out_of_range_low(self):
        from expert_rubric import rubric_summary

        rubric = self._make_rubric([(0, 3, 3, 3)])  # 0 is out of [1, 5]
        with pytest.raises(ValueError, match="out of range"):
            rubric_summary(rubric)

    def test_rubric_summary_empty_entries_returns_zeros(self):
        from expert_rubric import ExpertRubric, rubric_summary

        rubric = ExpertRubric(
            domain="test", entries=[], scorer="self", scored_at="2026-02-26"
        )
        summary = rubric_summary(rubric)
        assert summary["overall"] == pytest.approx(0.0)
        assert summary["mean_plausibility"] == pytest.approx(0.0)

    def test_rubric_summary_has_expected_keys(self):
        from expert_rubric import rubric_summary

        rubric = self._make_rubric([(3, 3, 3, 3)])
        summary = rubric_summary(rubric)

        assert set(summary.keys()) == {
            "mean_plausibility",
            "mean_novelty",
            "mean_falsifiability",
            "mean_clarity",
            "overall",
        }
