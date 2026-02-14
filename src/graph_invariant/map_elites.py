"""MAP-Elites diversity archive for quality-diversity search.

Maintains a 2D behavioral grid indexed by (simplicity_score, novelty_bonus).
Each cell keeps only the candidate with the highest raw fitness_signal.
Runs alongside the existing island model to provide diverse prompt exemplars.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from .types import Candidate

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class ArchiveCell:
    """A single cell in the MAP-Elites archive holding one candidate."""

    candidate: Candidate
    fitness_signal: float


@dataclass(slots=True)
class MapElitesArchive:
    """2D behavioral grid mapping (simplicity, novelty) bins to elite candidates."""

    num_bins: int
    cells: dict[tuple[int, int], ArchiveCell] = field(default_factory=dict)


def _bin_index(value: float, num_bins: int) -> int:
    """Map a [0,1] value to a bin index in [0, num_bins-1]."""
    if num_bins <= 0:
        raise ValueError("num_bins must be > 0")
    clamped = max(0.0, min(1.0, value))
    idx = int(clamped * num_bins)
    return min(idx, num_bins - 1)


def try_insert(archive: MapElitesArchive, candidate: Candidate, fitness_signal: float) -> bool:
    """Insert candidate if cell is empty or fitness_signal beats the incumbent."""
    row = _bin_index(candidate.simplicity_score, archive.num_bins)
    col = _bin_index(candidate.novelty_bonus, archive.num_bins)
    key = (row, col)
    existing = archive.cells.get(key)
    if existing is not None and existing.fitness_signal >= fitness_signal:
        return False
    archive.cells[key] = ArchiveCell(candidate=candidate, fitness_signal=fitness_signal)
    return True


def sample_diverse_exemplars(
    archive: MapElitesArchive,
    rng: np.random.Generator,
    count: int,
    exclude_island: int | None = None,
) -> list[Candidate]:
    """Uniform sample from occupied cells for prompt diversity."""
    eligible = [
        cell.candidate
        for cell in archive.cells.values()
        if exclude_island is None or cell.candidate.island_id != exclude_island
    ]
    if not eligible:
        return []
    if len(eligible) <= count:
        return list(eligible)
    indices = rng.choice(len(eligible), size=count, replace=False)
    return [eligible[i] for i in indices]


def archive_stats(archive: MapElitesArchive) -> dict[str, int | float]:
    """Return summary statistics: coverage, total_cells, best/mean fitness."""
    total_cells = archive.num_bins * archive.num_bins
    coverage = len(archive.cells)
    if coverage == 0:
        return {
            "coverage": 0,
            "total_cells": total_cells,
            "best_fitness": 0.0,
            "mean_fitness": 0.0,
        }
    fitnesses = [cell.fitness_signal for cell in archive.cells.values()]
    return {
        "coverage": coverage,
        "total_cells": total_cells,
        "best_fitness": max(fitnesses),
        "mean_fitness": sum(fitnesses) / len(fitnesses),
    }


def serialize_archive(archive: MapElitesArchive) -> dict[str, Any]:
    """Convert archive to a JSON-safe dict for checkpoint persistence."""
    cells_data = {}
    for (row, col), cell in archive.cells.items():
        key = f"{row},{col}"
        cells_data[key] = {
            "candidate": asdict(cell.candidate),
            "fitness_signal": cell.fitness_signal,
        }
    return {"num_bins": archive.num_bins, "cells": cells_data}


def deserialize_archive(data: dict[str, Any]) -> MapElitesArchive:
    """Restore archive from a JSON-safe dict.

    Gracefully skips malformed cell entries to tolerate corrupted checkpoints.
    """
    num_bins = int(data["num_bins"])
    if num_bins <= 0:
        raise ValueError(f"Cannot deserialize archive with num_bins={num_bins}; must be > 0")
    cells: dict[tuple[int, int], ArchiveCell] = {}
    for key, cell_data in data.get("cells", {}).items():
        try:
            row_str, col_str = key.split(",")
            row, col = int(row_str), int(col_str)
            candidate = Candidate(**cell_data["candidate"])
            cells[(row, col)] = ArchiveCell(
                candidate=candidate,
                fitness_signal=float(cell_data["fitness_signal"]),
            )
        except (ValueError, KeyError, TypeError):
            logger.warning("Skipping malformed archive cell: %s", key)
            continue
    return MapElitesArchive(num_bins=num_bins, cells=cells)
