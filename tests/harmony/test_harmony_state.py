"""Tests for harmony.state — HarmonySearchState dataclass and checkpoint round-trip.

TDD: written BEFORE implementation. Verifies:
  - State initializes with empty islands dict per island_id
  - save_state / load_state round-trip preserves all fields
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from harmony.state import HarmonySearchState, load_state, save_state


class TestHarmonySearchState:
    def test_state_initializes_with_empty_islands(self):
        state = HarmonySearchState(
            experiment_id="exp-001",
            generation=0,
            islands={0: [], 1: [], 2: [], 3: []},
        )
        assert len(state.islands) == 4
        for proposals in state.islands.values():
            assert proposals == []

    def test_state_has_expected_default_fields(self):
        state = HarmonySearchState(
            experiment_id="exp-002",
            generation=0,
            islands={},
        )
        assert state.rng_seed == 42
        assert state.rng_state is None
        assert state.archive is None
        assert state.best_harmony_gain == 0.0
        assert state.no_improve_count == 0

    def test_state_island_prompt_mode_defaults_to_empty(self):
        state = HarmonySearchState(
            experiment_id="exp-003",
            generation=0,
            islands={},
        )
        assert state.island_prompt_mode == {}
        assert state.island_stagnation == {}

    def test_generation_can_be_set(self):
        state = HarmonySearchState(
            experiment_id="exp-004",
            generation=5,
            islands={0: []},
        )
        assert state.generation == 5

    def test_best_harmony_gain_can_be_set(self):
        state = HarmonySearchState(
            experiment_id="exp-005",
            generation=0,
            islands={},
            best_harmony_gain=0.42,
        )
        assert state.best_harmony_gain == pytest.approx(0.42)


class TestCheckpointRoundTrip:
    def test_save_load_state_round_trip(self, tmp_path: Path):
        state = HarmonySearchState(
            experiment_id="round-trip-001",
            generation=3,
            islands={0: [], 1: []},
            rng_seed=99,
            island_prompt_mode={0: "free", 1: "constrained"},
            island_stagnation={0: 0, 1: 2},
            best_harmony_gain=0.15,
            no_improve_count=2,
        )
        checkpoint = tmp_path / "checkpoint.json"
        save_state(state, checkpoint)
        assert checkpoint.exists()

        loaded = load_state(checkpoint)
        assert loaded.experiment_id == state.experiment_id
        assert loaded.generation == state.generation
        assert loaded.rng_seed == state.rng_seed
        assert loaded.best_harmony_gain == pytest.approx(state.best_harmony_gain)
        assert loaded.no_improve_count == state.no_improve_count
        assert loaded.island_prompt_mode == state.island_prompt_mode
        assert loaded.island_stagnation == state.island_stagnation

    def test_save_state_writes_valid_json(self, tmp_path: Path):
        state = HarmonySearchState(
            experiment_id="json-check",
            generation=0,
            islands={},
        )
        checkpoint = tmp_path / "checkpoint.json"
        save_state(state, checkpoint)
        with checkpoint.open() as f:
            data = json.load(f)
        assert data["experiment_id"] == "json-check"
        assert data["generation"] == 0

    def test_load_state_restores_islands_as_empty_lists(self, tmp_path: Path):
        state = HarmonySearchState(
            experiment_id="island-restore",
            generation=1,
            islands={0: [], 1: [], 2: [], 3: []},
        )
        checkpoint = tmp_path / "checkpoint.json"
        save_state(state, checkpoint)
        loaded = load_state(checkpoint)
        assert len(loaded.islands) == 4
        for v in loaded.islands.values():
            assert v == []

    def test_checkpoint_file_contains_islands_keys_as_strings(self, tmp_path: Path):
        """JSON serializes dict keys as strings; verify load restores them as ints."""
        state = HarmonySearchState(
            experiment_id="key-type-check",
            generation=0,
            islands={0: [], 1: []},
        )
        checkpoint = tmp_path / "checkpoint.json"
        save_state(state, checkpoint)
        loaded = load_state(checkpoint)
        # Keys should be integers after load, matching the original
        assert 0 in loaded.islands
        assert 1 in loaded.islands
