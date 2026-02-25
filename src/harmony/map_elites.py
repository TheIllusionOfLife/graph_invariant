"""MAP-Elites diversity archive for Harmony proposal quality-diversity search.

Maintains a 2D behavioral grid where cells are selected by a (simplicity, gain)
descriptor. Each cell keeps only the proposal with the highest harmony_gain
(fitness_signal). Adapted from graph_invariant/map_elites.py with Candidate
replaced by Proposal.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from harmony.proposals.types import Proposal

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class ArchiveCell:
    """A single cell in the MAP-Elites archive holding one proposal."""

    proposal: Proposal
    fitness_signal: float


@dataclass(slots=True)
class HarmonyMapElites:
    """2D behavioral grid mapping (simplicity, gain) bins to elite proposals."""

    num_bins: int
    archive_id: str = "primary"
    cells: dict[tuple[int, int], ArchiveCell] = field(default_factory=dict)


def _bin_index(value: float, num_bins: int) -> int:
    """Map a [0,1] value to a bin index in [0, num_bins-1]."""
    if num_bins <= 0:
        raise ValueError("num_bins must be > 0")
    clamped = max(0.0, min(1.0, value))
    idx = int(clamped * num_bins)
    return min(idx, num_bins - 1)


def try_insert(
    archive: HarmonyMapElites,
    proposal: Proposal,
    fitness_signal: float,
    descriptor: tuple[float, float],
) -> bool:
    """Insert proposal if cell is empty or fitness_signal beats the incumbent."""
    row_value, col_value = descriptor
    row = _bin_index(row_value, archive.num_bins)
    col = _bin_index(col_value, archive.num_bins)
    key = (row, col)
    existing = archive.cells.get(key)
    if existing is not None and existing.fitness_signal >= fitness_signal:
        return False
    archive.cells[key] = ArchiveCell(proposal=proposal, fitness_signal=fitness_signal)
    return True


def sample_diverse_exemplars(
    archive: HarmonyMapElites,
    rng: np.random.Generator,
    count: int,
) -> list[Proposal]:
    """Uniform sample from occupied cells for prompt diversity."""
    eligible = [cell.proposal for cell in archive.cells.values()]
    if not eligible:
        return []
    if len(eligible) <= count:
        return list(eligible)
    indices = rng.choice(len(eligible), size=count, replace=False)
    return [eligible[i] for i in indices]


def archive_stats(archive: HarmonyMapElites) -> dict[str, int | float]:
    """Return summary statistics: coverage, total_cells, best/mean fitness."""
    total_cells = archive.num_bins * archive.num_bins
    coverage = len(archive.cells)
    if coverage == 0:
        return {
            "archive_id": archive.archive_id,
            "coverage": 0,
            "total_cells": total_cells,
            "best_fitness": 0.0,
            "mean_fitness": 0.0,
        }
    fitnesses = [cell.fitness_signal for cell in archive.cells.values()]
    return {
        "archive_id": archive.archive_id,
        "coverage": coverage,
        "total_cells": total_cells,
        "best_fitness": max(fitnesses),
        "mean_fitness": sum(fitnesses) / len(fitnesses),
    }


def serialize_archive(archive: HarmonyMapElites) -> dict[str, Any]:
    """Convert archive to a JSON-safe dict for checkpoint persistence."""
    cells_data = {}
    for (row, col), cell in archive.cells.items():
        key = f"{row},{col}"
        cells_data[key] = {
            "proposal": cell.proposal.to_dict(),
            "fitness_signal": cell.fitness_signal,
        }
    return {
        "num_bins": archive.num_bins,
        "archive_id": archive.archive_id,
        "cells": cells_data,
    }


def deserialize_archive(data: dict[str, Any]) -> HarmonyMapElites:
    """Restore archive from a JSON-safe dict.

    Gracefully skips malformed cell entries to tolerate corrupted checkpoints.
    """
    num_bins = int(data["num_bins"])
    if num_bins <= 0:
        raise ValueError(f"Cannot deserialize archive with num_bins={num_bins}; must be > 0")
    archive_id = str(data.get("archive_id", "primary"))
    cells: dict[tuple[int, int], ArchiveCell] = {}
    for key, cell_data in data.get("cells", {}).items():
        try:
            row_str, col_str = key.split(",")
            row, col = int(row_str), int(col_str)
            proposal = Proposal.from_dict(cell_data["proposal"])
            cells[(row, col)] = ArchiveCell(
                proposal=proposal,
                fitness_signal=float(cell_data["fitness_signal"]),
            )
        except (ValueError, KeyError, TypeError):
            logger.warning("Skipping malformed archive cell: %s", key)
            continue
    return HarmonyMapElites(num_bins=num_bins, archive_id=archive_id, cells=cells)
