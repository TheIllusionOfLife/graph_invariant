"""HarmonySearchState — island search state with checkpoint persistence.

Mirrors the CheckpointState / save_checkpoint / load_checkpoint pattern from
graph_invariant/logging_io.py, adapted for the Harmony proposal-engine loop.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class HarmonySearchState:
    """Full mutable state of one Harmony island-search run.

    Persisted as JSON after every generation so runs can be resumed.
    """

    experiment_id: str
    generation: int
    islands: dict[int, list[dict[str, Any]]]  # island_id → list of Proposal.to_dict()

    rng_seed: int = 42
    rng_state: dict[str, Any] | None = None

    # Prompt-mode scheduler state
    island_prompt_mode: dict[int, str] = field(default_factory=dict)  # "free" | "constrained"
    island_stagnation: dict[int, int] = field(default_factory=dict)
    island_recent_failures: dict[int, list[str]] = field(default_factory=dict)
    island_constrained_generations: dict[int, int] = field(default_factory=dict)

    # Archive — serialized HarmonyMapElites (None until first generation completes)
    archive: dict[str, Any] | None = None

    # Fitness tracking
    best_harmony_gain: float = 0.0
    no_improve_count: int = 0


def save_state(state: HarmonySearchState, path: Path) -> None:
    """Persist *state* as a JSON file at *path* using an atomic write.

    Writes to a ``.tmp`` sibling first, then renames to *path* so that a
    crash mid-write never leaves a truncated/corrupt checkpoint.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    data: dict[str, Any] = {
        "experiment_id": state.experiment_id,
        "generation": state.generation,
        # JSON requires string keys; store island_id as str, restore as int on load
        "islands": {str(k): v for k, v in state.islands.items()},
        "rng_seed": state.rng_seed,
        "rng_state": state.rng_state,
        "island_prompt_mode": {str(k): v for k, v in state.island_prompt_mode.items()},
        "island_stagnation": {str(k): v for k, v in state.island_stagnation.items()},
        "island_recent_failures": {str(k): v for k, v in state.island_recent_failures.items()},
        "island_constrained_generations": {
            str(k): v for k, v in state.island_constrained_generations.items()
        },
        "archive": state.archive,
        "best_harmony_gain": state.best_harmony_gain,
        "no_improve_count": state.no_improve_count,
    }
    tmp_path = path.with_suffix(".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    tmp_path.replace(path)


def load_state(path: Path) -> HarmonySearchState:
    """Restore a HarmonySearchState from a JSON checkpoint file."""
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    return HarmonySearchState(
        experiment_id=data["experiment_id"],
        generation=data["generation"],
        islands={int(k): v for k, v in data.get("islands", {}).items()},
        rng_seed=data.get("rng_seed", 42),
        rng_state=data.get("rng_state"),
        island_prompt_mode={int(k): v for k, v in data.get("island_prompt_mode", {}).items()},
        island_stagnation={int(k): v for k, v in data.get("island_stagnation", {}).items()},
        island_recent_failures={
            int(k): v for k, v in data.get("island_recent_failures", {}).items()
        },
        island_constrained_generations={
            int(k): v for k, v in data.get("island_constrained_generations", {}).items()
        },
        archive=data.get("archive"),
        best_harmony_gain=data.get("best_harmony_gain", 0.0),
        no_improve_count=data.get("no_improve_count", 0),
    )
