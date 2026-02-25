"""Tests for harmony.map_elites.HarmonyMapElites — insert, replace, stats, serialize.

TDD: these tests are written BEFORE implementation. They verify:
  - try_insert populates empty cells
  - Better fitness replaces incumbent; worse fitness does not
  - archive_stats coverage matches occupied cell count
  - sample_diverse_exemplars returns unique proposals
  - serialize / deserialize produces equivalent archive
"""

from __future__ import annotations

import numpy as np
import pytest

from harmony.map_elites import (
    HarmonyMapElites,
    archive_stats,
    deserialize_archive,
    sample_diverse_exemplars,
    serialize_archive,
    try_insert,
)
from harmony.proposals.types import Proposal, ProposalType


def _make_proposal(pid: str = "p1") -> Proposal:
    return Proposal(
        id=pid,
        proposal_type=ProposalType.ADD_ENTITY,
        claim="A new mathematical concept can be formalized in this domain.",
        justification="It is observed in multiple proofs and theorems consistently.",
        falsification_condition="If the concept reduces to an existing one without loss.",
        kg_domain="test_domain",
        entity_id=f"entity_{pid}",
        entity_type="concept",
    )


class TestTryInsert:
    def test_insert_into_empty_archive(self):
        archive = HarmonyMapElites(num_bins=5)
        p = _make_proposal("p1")
        inserted = try_insert(archive, p, fitness_signal=0.5, descriptor=(0.5, 0.5))
        assert inserted is True
        assert len(archive.cells) == 1

    def test_better_fitness_replaces_incumbent(self):
        archive = HarmonyMapElites(num_bins=5)
        p1 = _make_proposal("p1")
        p2 = _make_proposal("p2")
        try_insert(archive, p1, fitness_signal=0.3, descriptor=(0.5, 0.5))
        inserted = try_insert(archive, p2, fitness_signal=0.8, descriptor=(0.5, 0.5))
        assert inserted is True
        assert list(archive.cells.values())[0].proposal.id == "p2"

    def test_worse_fitness_does_not_replace(self):
        archive = HarmonyMapElites(num_bins=5)
        p1 = _make_proposal("p1")
        p2 = _make_proposal("p2")
        try_insert(archive, p1, fitness_signal=0.8, descriptor=(0.5, 0.5))
        inserted = try_insert(archive, p2, fitness_signal=0.3, descriptor=(0.5, 0.5))
        assert inserted is False
        assert list(archive.cells.values())[0].proposal.id == "p1"

    def test_equal_fitness_does_not_replace(self):
        archive = HarmonyMapElites(num_bins=5)
        p1 = _make_proposal("p1")
        p2 = _make_proposal("p2")
        try_insert(archive, p1, fitness_signal=0.5, descriptor=(0.5, 0.5))
        inserted = try_insert(archive, p2, fitness_signal=0.5, descriptor=(0.5, 0.5))
        assert inserted is False
        assert list(archive.cells.values())[0].proposal.id == "p1"

    def test_different_descriptors_populate_different_cells(self):
        archive = HarmonyMapElites(num_bins=5)
        try_insert(archive, _make_proposal("p1"), fitness_signal=0.5, descriptor=(0.1, 0.1))
        try_insert(archive, _make_proposal("p2"), fitness_signal=0.5, descriptor=(0.9, 0.9))
        assert len(archive.cells) == 2

    def test_descriptor_clamped_to_bins(self):
        archive = HarmonyMapElites(num_bins=5)
        # Values beyond [0,1] should clamp gracefully, not raise
        try_insert(archive, _make_proposal("p1"), fitness_signal=0.5, descriptor=(1.5, -0.1))
        assert len(archive.cells) == 1


class TestArchiveStats:
    def test_archive_coverage_matches_occupied_cells(self):
        archive = HarmonyMapElites(num_bins=5)
        descriptors = [(0.1, 0.1), (0.5, 0.5), (0.9, 0.9)]
        for i, desc in enumerate(descriptors):
            try_insert(archive, _make_proposal(f"p{i}"), fitness_signal=0.5, descriptor=desc)
        stats = archive_stats(archive)
        assert stats["coverage"] == len(archive.cells)
        assert stats["total_cells"] == 25  # 5×5

    def test_empty_archive_stats(self):
        archive = HarmonyMapElites(num_bins=3)
        stats = archive_stats(archive)
        assert stats["coverage"] == 0
        assert stats["total_cells"] == 9
        assert stats["best_fitness"] == 0.0
        assert stats["mean_fitness"] == 0.0

    def test_best_fitness_is_maximum(self):
        archive = HarmonyMapElites(num_bins=5)
        try_insert(archive, _make_proposal("p1"), fitness_signal=0.3, descriptor=(0.1, 0.1))
        try_insert(archive, _make_proposal("p2"), fitness_signal=0.9, descriptor=(0.5, 0.5))
        stats = archive_stats(archive)
        assert stats["best_fitness"] == pytest.approx(0.9)


class TestSampleDiverse:
    def test_sample_diverse_exemplars_returns_unique(self):
        archive = HarmonyMapElites(num_bins=5)
        for i in range(5):
            descriptor = (i * 0.2, i * 0.2)
            try_insert(archive, _make_proposal(f"p{i}"), fitness_signal=0.5, descriptor=descriptor)
        rng = np.random.default_rng(42)
        samples = sample_diverse_exemplars(archive, rng, count=3)
        ids = [s.id for s in samples]
        assert len(ids) == len(set(ids))

    def test_sample_all_when_count_exceeds_occupied(self):
        archive = HarmonyMapElites(num_bins=5)
        for i in range(2):
            try_insert(archive, _make_proposal(f"p{i}"), fitness_signal=0.5, descriptor=(i * 0.5, i * 0.5))
        rng = np.random.default_rng(42)
        samples = sample_diverse_exemplars(archive, rng, count=10)
        assert len(samples) == 2

    def test_sample_empty_archive_returns_empty(self):
        archive = HarmonyMapElites(num_bins=5)
        rng = np.random.default_rng(42)
        samples = sample_diverse_exemplars(archive, rng, count=5)
        assert samples == []


class TestSerializeDeserialize:
    def test_serialize_deserialize_round_trip(self):
        archive = HarmonyMapElites(num_bins=5, archive_id="test")
        p = _make_proposal("p1")
        try_insert(archive, p, fitness_signal=0.7, descriptor=(0.4, 0.6))
        data = serialize_archive(archive)
        restored = deserialize_archive(data)
        assert restored.num_bins == archive.num_bins
        assert restored.archive_id == archive.archive_id
        assert len(restored.cells) == len(archive.cells)
        restored_cell = list(restored.cells.values())[0]
        assert restored_cell.proposal.id == "p1"
        assert restored_cell.fitness_signal == pytest.approx(0.7)

    def test_serialize_produces_json_safe_dict(self):
        archive = HarmonyMapElites(num_bins=3, archive_id="serial_test")
        try_insert(archive, _make_proposal("p1"), fitness_signal=0.5, descriptor=(0.5, 0.5))
        data = serialize_archive(archive)
        assert isinstance(data["num_bins"], int)
        assert isinstance(data["archive_id"], str)
        assert isinstance(data["cells"], dict)
        # Cell keys must be strings like "row,col"
        for key in data["cells"]:
            assert "," in key

    def test_deserialize_empty_cells(self):
        data = {"num_bins": 4, "archive_id": "empty", "cells": {}}
        archive = deserialize_archive(data)
        assert archive.num_bins == 4
        assert len(archive.cells) == 0

    def test_round_trip_preserves_all_proposal_fields(self):
        archive = HarmonyMapElites(num_bins=5)
        p = Proposal(
            id="full-proposal",
            proposal_type=ProposalType.ADD_EDGE,
            claim="Eigenvectors depend on the determinant in a fundamental way.",
            justification="This follows from the characteristic polynomial definition.",
            falsification_condition="If a matrix has zero determinant but non-trivial eigenvectors.",
            kg_domain="linear_algebra",
            source_entity="eigenvector",
            target_entity="determinant",
            edge_type="DEPENDS_ON",
        )
        try_insert(archive, p, fitness_signal=0.42, descriptor=(0.3, 0.7))
        restored = deserialize_archive(serialize_archive(archive))
        cell = list(restored.cells.values())[0]
        assert cell.proposal.claim == p.claim
        assert cell.proposal.proposal_type == ProposalType.ADD_EDGE
        assert cell.proposal.edge_type == "DEPENDS_ON"
