import numpy as np
import pytest

from graph_invariant.types import Candidate

# ── bin_index ────────────────────────────────────────────────────────


def test_bin_index_zero_maps_to_first_bin():
    from graph_invariant.map_elites import _bin_index

    assert _bin_index(0.0, num_bins=5) == 0


def test_bin_index_one_maps_to_last_bin():
    from graph_invariant.map_elites import _bin_index

    assert _bin_index(1.0, num_bins=5) == 4


def test_bin_index_midpoint_maps_to_middle():
    from graph_invariant.map_elites import _bin_index

    assert _bin_index(0.5, num_bins=4) == 2


def test_bin_index_clamps_negative():
    from graph_invariant.map_elites import _bin_index

    assert _bin_index(-0.1, num_bins=5) == 0


def test_bin_index_clamps_above_one():
    from graph_invariant.map_elites import _bin_index

    assert _bin_index(1.5, num_bins=5) == 4


def test_bin_index_two_bins():
    from graph_invariant.map_elites import _bin_index

    assert _bin_index(0.0, num_bins=2) == 0
    assert _bin_index(0.49, num_bins=2) == 0
    assert _bin_index(0.5, num_bins=2) == 1
    assert _bin_index(1.0, num_bins=2) == 1


# ── try_insert ───────────────────────────────────────────────────────


def _make_candidate(
    simplicity: float = 0.5,
    novelty: float = 0.5,
    code: str = "def new_invariant(s): return 1.0",
    cid: str = "c1",
) -> Candidate:
    return Candidate(
        id=cid,
        code=code,
        simplicity_score=simplicity,
        novelty_bonus=novelty,
    )


def test_try_insert_into_empty_archive():
    from graph_invariant.map_elites import MapElitesArchive, try_insert

    archive = MapElitesArchive(num_bins=5, cells={})
    candidate = _make_candidate()
    assert try_insert(archive, candidate, fitness_signal=0.8) is True
    assert len(archive.cells) == 1


def test_try_insert_replaces_worse_incumbent():
    from graph_invariant.map_elites import MapElitesArchive, try_insert

    archive = MapElitesArchive(num_bins=5, cells={})
    c1 = _make_candidate(cid="c1")
    c2 = _make_candidate(cid="c2")
    try_insert(archive, c1, fitness_signal=0.5)
    assert try_insert(archive, c2, fitness_signal=0.9) is True
    cell = next(iter(archive.cells.values()))
    assert cell.candidate.id == "c2"
    assert cell.fitness_signal == 0.9


def test_try_insert_keeps_better_incumbent():
    from graph_invariant.map_elites import MapElitesArchive, try_insert

    archive = MapElitesArchive(num_bins=5, cells={})
    c1 = _make_candidate(cid="c1")
    c2 = _make_candidate(cid="c2")
    try_insert(archive, c1, fitness_signal=0.9)
    assert try_insert(archive, c2, fitness_signal=0.5) is False
    cell = next(iter(archive.cells.values()))
    assert cell.candidate.id == "c1"


def test_try_insert_different_cells():
    from graph_invariant.map_elites import MapElitesArchive, try_insert

    archive = MapElitesArchive(num_bins=5, cells={})
    c_low = _make_candidate(simplicity=0.1, novelty=0.1, cid="low")
    c_high = _make_candidate(simplicity=0.9, novelty=0.9, cid="high")
    try_insert(archive, c_low, fitness_signal=0.5)
    try_insert(archive, c_high, fitness_signal=0.5)
    assert len(archive.cells) == 2


# ── sample_diverse_exemplars ─────────────────────────────────────────


def test_sample_from_empty_archive_returns_empty():
    from graph_invariant.map_elites import MapElitesArchive, sample_diverse_exemplars

    archive = MapElitesArchive(num_bins=5, cells={})
    rng = np.random.default_rng(42)
    result = sample_diverse_exemplars(archive, rng, count=3)
    assert result == []


def test_sample_returns_up_to_count():
    from graph_invariant.map_elites import MapElitesArchive, sample_diverse_exemplars, try_insert

    archive = MapElitesArchive(num_bins=5, cells={})
    for i in range(10):
        c = _make_candidate(
            simplicity=i / 10,
            novelty=i / 10,
            cid=f"c{i}",
            code=f"def new_invariant(s): return {i}",
        )
        try_insert(archive, c, fitness_signal=0.5)
    rng = np.random.default_rng(42)
    result = sample_diverse_exemplars(archive, rng, count=3)
    assert len(result) <= 3
    assert all(isinstance(c, Candidate) for c in result)


def test_sample_excludes_island():
    from graph_invariant.map_elites import MapElitesArchive, sample_diverse_exemplars, try_insert

    archive = MapElitesArchive(num_bins=5, cells={})
    c1 = _make_candidate(simplicity=0.1, novelty=0.1, cid="c1")
    c1.island_id = 0
    c2 = _make_candidate(simplicity=0.9, novelty=0.9, cid="c2")
    c2.island_id = 1
    try_insert(archive, c1, fitness_signal=0.5)
    try_insert(archive, c2, fitness_signal=0.5)
    rng = np.random.default_rng(42)
    result = sample_diverse_exemplars(archive, rng, count=10, exclude_island=0)
    assert all(c.island_id != 0 for c in result)


def test_sample_fewer_available_than_requested():
    from graph_invariant.map_elites import MapElitesArchive, sample_diverse_exemplars, try_insert

    archive = MapElitesArchive(num_bins=5, cells={})
    c = _make_candidate(cid="only")
    try_insert(archive, c, fitness_signal=0.5)
    rng = np.random.default_rng(42)
    result = sample_diverse_exemplars(archive, rng, count=5)
    assert len(result) == 1


# ── archive_stats ────────────────────────────────────────────────────


def test_archive_stats_empty():
    from graph_invariant.map_elites import MapElitesArchive, archive_stats

    archive = MapElitesArchive(num_bins=5, cells={})
    stats = archive_stats(archive)
    assert stats["archive_id"] == "primary"
    assert stats["coverage"] == 0
    assert stats["total_cells"] == 25
    assert stats["best_fitness"] == 0.0
    assert stats["mean_fitness"] == 0.0


def test_archive_stats_populated():
    from graph_invariant.map_elites import MapElitesArchive, archive_stats, try_insert

    archive = MapElitesArchive(num_bins=5, cells={})
    c1 = _make_candidate(simplicity=0.1, novelty=0.1, cid="c1")
    c2 = _make_candidate(simplicity=0.9, novelty=0.9, cid="c2")
    try_insert(archive, c1, fitness_signal=0.4)
    try_insert(archive, c2, fitness_signal=0.8)
    stats = archive_stats(archive)
    assert stats["archive_id"] == "primary"
    assert stats["coverage"] == 2
    assert stats["total_cells"] == 25
    assert stats["best_fitness"] == pytest.approx(0.8)
    assert stats["mean_fitness"] == pytest.approx(0.6)


# ── serialize / deserialize ──────────────────────────────────────────


def test_serialize_deserialize_roundtrip():
    from graph_invariant.map_elites import (
        MapElitesArchive,
        deserialize_archive,
        serialize_archive,
        try_insert,
    )

    archive = MapElitesArchive(num_bins=5, archive_id="topology", cells={})
    c1 = _make_candidate(simplicity=0.3, novelty=0.7, cid="c1")
    c2 = _make_candidate(simplicity=0.8, novelty=0.2, cid="c2")
    try_insert(archive, c1, fitness_signal=0.6)
    try_insert(archive, c2, fitness_signal=0.9)

    data = serialize_archive(archive)
    restored = deserialize_archive(data)

    assert restored.num_bins == archive.num_bins
    assert restored.archive_id == "topology"
    assert len(restored.cells) == len(archive.cells)
    for key, cell in archive.cells.items():
        assert key in restored.cells
        assert restored.cells[key].candidate.id == cell.candidate.id
        assert restored.cells[key].fitness_signal == cell.fitness_signal


def test_serialize_empty_archive():
    from graph_invariant.map_elites import (
        MapElitesArchive,
        deserialize_archive,
        serialize_archive,
    )

    archive = MapElitesArchive(num_bins=3, cells={})
    data = serialize_archive(archive)
    restored = deserialize_archive(data)
    assert restored.num_bins == 3
    assert restored.cells == {}


def test_deserialize_archive_skips_malformed_cells():
    from graph_invariant.map_elites import deserialize_archive

    data = {
        "num_bins": 5,
        "cells": {
            "0,0": {
                "candidate": {
                    "id": "good",
                    "code": "def new_invariant(s): return 1.0",
                    "island_id": 0,
                    "generation": 0,
                    "train_score": 0.0,
                    "val_score": 0.0,
                    "simplicity_score": 0.5,
                    "novelty_bonus": 0.5,
                },
                "fitness_signal": 0.8,
            },
            "bad_key": {"candidate": {}, "fitness_signal": "not_a_number"},
            "1,1": {"missing_candidate_key": True},
        },
    }
    archive = deserialize_archive(data)
    assert len(archive.cells) == 1
    assert archive.cells[(0, 0)].candidate.id == "good"


def test_try_insert_uses_descriptor_when_provided():
    from graph_invariant.map_elites import MapElitesArchive, try_insert

    archive = MapElitesArchive(num_bins=5, archive_id="topology", cells={})
    candidate = _make_candidate(simplicity=0.1, novelty=0.9, cid="c1")
    inserted = try_insert(archive, candidate, fitness_signal=0.7, descriptor=(0.8, 0.2))
    assert inserted is True
    assert (4, 1) in archive.cells
    assert archive.cells[(4, 1)].candidate.id == "c1"


def test_serialize_is_json_safe():
    import json

    from graph_invariant.map_elites import (
        MapElitesArchive,
        serialize_archive,
        try_insert,
    )

    archive = MapElitesArchive(num_bins=5, cells={})
    c = _make_candidate(cid="c1")
    try_insert(archive, c, fitness_signal=0.5)
    data = serialize_archive(archive)
    # Should not raise
    json.dumps(data)
