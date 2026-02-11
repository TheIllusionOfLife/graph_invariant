import json
from pathlib import Path

from graph_invariant.logging_io import (
    append_jsonl,
    load_checkpoint,
    rotate_generation_checkpoints,
    save_checkpoint,
)
from graph_invariant.types import Candidate, CheckpointState


def test_append_jsonl_writes_one_record(tmp_path):
    path = tmp_path / "events.jsonl"
    append_jsonl("candidate_evaluated", {"score": 0.5}, path)
    lines = path.read_text().splitlines()
    assert len(lines) == 1
    payload = json.loads(lines[0])
    assert payload["event_type"] == "candidate_evaluated"
    assert payload["payload"]["score"] == 0.5


def test_checkpoint_roundtrip(tmp_path):
    ckpt_path = tmp_path / "state.json"
    state = CheckpointState(
        experiment_id="exp",
        generation=3,
        islands={0: [Candidate(id="c1", code="def new_invariant(G):\n    return 1")]},
        rng_seed=42,
        island_recent_failures={0: ["static_invalid: bad syntax"]},
    )
    save_checkpoint(state, ckpt_path)
    loaded = load_checkpoint(ckpt_path)
    assert loaded.generation == 3
    assert loaded.rng_seed == 42
    assert loaded.islands[0][0].id == "c1"
    assert loaded.island_recent_failures[0] == ["static_invalid: bad syntax"]


def test_load_checkpoint_defaults_missing_recent_failures(tmp_path):
    ckpt_path = tmp_path / "state.json"
    payload = {
        "experiment_id": "exp",
        "generation": 1,
        "rng_seed": 42,
        "islands": {"0": []},
    }
    ckpt_path.write_text(json.dumps(payload), encoding="utf-8")
    loaded = load_checkpoint(ckpt_path)
    assert loaded.island_recent_failures == {}


def test_rotate_generation_checkpoints_uses_numeric_generation_order(tmp_path):
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    for generation in [1, 2, 3, 9, 10, 11]:
        path = checkpoint_dir / f"gen_{generation}.json"
        path.write_text("{}", encoding="utf-8")

    rotate_generation_checkpoints(checkpoint_dir, keep_last=3)
    remaining = sorted(
        checkpoint_dir.glob("gen_*.json"),
        key=lambda p: int(Path(p).stem.split("_")[-1]),
    )
    assert [p.name for p in remaining] == ["gen_9.json", "gen_10.json", "gen_11.json"]
