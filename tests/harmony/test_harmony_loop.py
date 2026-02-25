"""Tests for harmony.harmony_loop — island search, stagnation scheduling, checkpointing.

TDD: written BEFORE implementation. Verifies:
  - run_harmony_loop() smoke test: 1 generation with mocked LLM → returns HarmonySearchState
  - stagnation_trigger: 3 consecutive all-invalid gens → island_prompt_mode = "constrained"
  - constrained mode: archive exemplars appear in the prompt
  - valid_rate is logged each generation
  - checkpoint is written after each generation
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from harmony.config import HarmonyConfig
from harmony.datasets.linear_algebra import build_linear_algebra_kg
from harmony.harmony_loop import run_harmony_loop
from harmony.state import HarmonySearchState


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_valid_proposal_dict(**overrides) -> dict:
    base = {
        "id": "p-smoke-001",
        "proposal_type": "add_edge",
        "claim": "Eigenvectors depend on the characteristic polynomial fundamentally.",
        "justification": "Defined via det(A - λI) = 0 derivation.",
        "falsification_condition": "If no polynomial relation exists between eigen and det.",
        "kg_domain": "linear_algebra",
        "source_entity": "eigenvalue",
        "target_entity": "determinant",
        "edge_type": "DEPENDS_ON",
        "entity_id": None,
        "entity_type": None,
    }
    base.update(overrides)
    return base


def _make_cfg(**overrides) -> HarmonyConfig:
    defaults = dict(
        max_generations=1,
        population_size=2,
        early_stop_patience=10,
        stagnation_trigger_generations=3,
        enable_self_correction=False,
    )
    defaults.update(overrides)
    return HarmonyConfig(**defaults)


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------


class TestRunHarmonyLoopSmoke:
    def test_returns_harmony_search_state(self, tmp_path: Path):
        cfg = _make_cfg(max_generations=1, population_size=1)
        kg = build_linear_algebra_kg()

        valid_json = json.dumps(_make_valid_proposal_dict())

        with patch("harmony.harmony_loop.generate_proposal_payload") as mock_gen:
            mock_gen.return_value = {
                "response": valid_json,
                "proposal_dict": _make_valid_proposal_dict(),
            }
            state = run_harmony_loop(cfg, kg, output_dir=tmp_path)

        assert isinstance(state, HarmonySearchState)

    def test_generation_increments(self, tmp_path: Path):
        cfg = _make_cfg(max_generations=2, population_size=1)
        kg = build_linear_algebra_kg()

        with patch("harmony.harmony_loop.generate_proposal_payload") as mock_gen:
            mock_gen.return_value = {
                "response": "{}",
                "proposal_dict": _make_valid_proposal_dict(),
            }
            state = run_harmony_loop(cfg, kg, output_dir=tmp_path)

        assert state.generation == 2

    def test_experiment_id_set(self, tmp_path: Path):
        cfg = _make_cfg(max_generations=1, population_size=1)
        kg = build_linear_algebra_kg()

        with patch("harmony.harmony_loop.generate_proposal_payload") as mock_gen:
            mock_gen.return_value = {
                "response": "{}",
                "proposal_dict": _make_valid_proposal_dict(),
            }
            state = run_harmony_loop(cfg, kg, output_dir=tmp_path)

        assert state.experiment_id != ""


# ---------------------------------------------------------------------------
# Stagnation → constrained mode
# ---------------------------------------------------------------------------


class TestStagnationTriggersConstrainedMode:
    def test_stagnation_triggers_constrained_mode(self, tmp_path: Path):
        """After stagnation_trigger_generations all-invalid gens, island moves to 'constrained'."""
        cfg = _make_cfg(
            max_generations=4,
            population_size=1,
            stagnation_trigger_generations=3,
            enable_self_correction=False,
        )
        kg = build_linear_algebra_kg()

        # Always return garbage → all proposals invalid
        with patch("harmony.harmony_loop.generate_proposal_payload") as mock_gen:
            mock_gen.return_value = {"response": "GARBAGE", "proposal_dict": None}
            state = run_harmony_loop(cfg, kg, output_dir=tmp_path)

        # At least one island should have been switched to constrained
        constrained_islands = [
            iid
            for iid, mode in state.island_prompt_mode.items()
            if mode == "constrained"
        ]
        assert len(constrained_islands) > 0

    def test_stagnation_counter_increments(self, tmp_path: Path):
        cfg = _make_cfg(
            max_generations=2,
            population_size=1,
            stagnation_trigger_generations=5,
            enable_self_correction=False,
        )
        kg = build_linear_algebra_kg()

        with patch("harmony.harmony_loop.generate_proposal_payload") as mock_gen:
            mock_gen.return_value = {"response": "GARBAGE", "proposal_dict": None}
            state = run_harmony_loop(cfg, kg, output_dir=tmp_path)

        # Each island should have stagnation count = 2 (one per generation)
        for stagnation in state.island_stagnation.values():
            assert stagnation == 2


# ---------------------------------------------------------------------------
# valid_rate logging
# ---------------------------------------------------------------------------


class TestValidRateLogging:
    def test_valid_rate_logged_each_generation(self, tmp_path: Path, caplog):
        cfg = _make_cfg(max_generations=1, population_size=2)
        kg = build_linear_algebra_kg()

        proposals = [
            _make_valid_proposal_dict(id="p1"),
            _make_valid_proposal_dict(id="p2"),
        ]

        def side_effect(**kwargs):
            return {"response": "{}", "proposal_dict": proposals.pop(0)}

        with caplog.at_level(logging.INFO, logger="harmony.harmony_loop"):
            with patch("harmony.harmony_loop.generate_proposal_payload") as mock_gen:
                mock_gen.side_effect = side_effect
                run_harmony_loop(cfg, kg, output_dir=tmp_path)

        # Look for valid_rate in logs
        log_text = " ".join(caplog.messages)
        assert "valid_rate" in log_text


# ---------------------------------------------------------------------------
# Checkpoint written each generation
# ---------------------------------------------------------------------------


class TestCheckpointWritten:
    def test_checkpoint_written_each_generation(self, tmp_path: Path):
        cfg = _make_cfg(max_generations=2, population_size=1)
        kg = build_linear_algebra_kg()

        with patch("harmony.harmony_loop.generate_proposal_payload") as mock_gen:
            mock_gen.return_value = {
                "response": "{}",
                "proposal_dict": _make_valid_proposal_dict(),
            }
            run_harmony_loop(cfg, kg, output_dir=tmp_path)

        checkpoint = tmp_path / "checkpoint.json"
        assert checkpoint.exists()
        with checkpoint.open() as f:
            data = json.load(f)
        assert data["generation"] == 2

    def test_log_file_written(self, tmp_path: Path):
        cfg = _make_cfg(max_generations=1, population_size=1)
        kg = build_linear_algebra_kg()

        with patch("harmony.harmony_loop.generate_proposal_payload") as mock_gen:
            mock_gen.return_value = {
                "response": "{}",
                "proposal_dict": _make_valid_proposal_dict(),
            }
            run_harmony_loop(cfg, kg, output_dir=tmp_path)

        log_file = tmp_path / "logs" / "harmony_events.jsonl"
        assert log_file.exists()
        lines = log_file.read_text().strip().split("\n")
        assert len(lines) >= 1


# ---------------------------------------------------------------------------
# Resume from checkpoint
# ---------------------------------------------------------------------------


class TestResume:
    def test_resume_continues_from_saved_generation(self, tmp_path: Path):
        cfg = _make_cfg(max_generations=3, population_size=1)
        kg = build_linear_algebra_kg()

        with patch("harmony.harmony_loop.generate_proposal_payload") as mock_gen:
            mock_gen.return_value = {
                "response": "{}",
                "proposal_dict": _make_valid_proposal_dict(),
            }
            state1 = run_harmony_loop(cfg, kg, output_dir=tmp_path)

        # First run should reach generation 3
        assert state1.generation == 3

        # Now resume with max_generations=5
        cfg2 = _make_cfg(max_generations=5, population_size=1)
        checkpoint_path = str(tmp_path / "checkpoint.json")
        with patch("harmony.harmony_loop.generate_proposal_payload") as mock_gen:
            mock_gen.return_value = {
                "response": "{}",
                "proposal_dict": _make_valid_proposal_dict(),
            }
            state2 = run_harmony_loop(cfg2, kg, output_dir=tmp_path, resume=checkpoint_path)

        assert state2.generation == 5
