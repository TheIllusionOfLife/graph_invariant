import json

from graph_invariant.logging_io import append_jsonl, load_checkpoint, save_checkpoint
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
        generation=3,
        islands={0: [Candidate(id="c1", code="def new_invariant(G):\n    return 1")]},
        rng_seed=42,
    )
    save_checkpoint(state, ckpt_path)
    loaded = load_checkpoint(ckpt_path)
    assert loaded.generation == 3
    assert loaded.rng_seed == 42
    assert loaded.islands[0][0].id == "c1"
