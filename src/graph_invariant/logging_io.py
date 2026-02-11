import json
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

from .types import Candidate, CheckpointState


def append_jsonl(event_type: str, payload: dict, path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "timestamp": datetime.now(UTC).isoformat(),
        "event_type": event_type,
        "payload": payload,
    }
    with target.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def save_checkpoint(state: CheckpointState, path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "generation": state.generation,
        "rng_seed": state.rng_seed,
        "islands": {
            str(island): [asdict(candidate) for candidate in candidates]
            for island, candidates in state.islands.items()
        },
    }
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load_checkpoint(path: str | Path) -> CheckpointState:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    islands: dict[int, list[Candidate]] = {}
    for island, candidates in payload["islands"].items():
        islands[int(island)] = [Candidate(**candidate) for candidate in candidates]
    return CheckpointState(
        generation=int(payload["generation"]),
        islands=islands,
        rng_seed=int(payload["rng_seed"]),
    )
