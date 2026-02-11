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
        "experiment_id": state.experiment_id,
        "generation": state.generation,
        "rng_seed": state.rng_seed,
        "rng_state": state.rng_state,
        "best_val_score": state.best_val_score,
        "no_improve_count": state.no_improve_count,
        "island_stagnation": {str(k): v for k, v in state.island_stagnation.items()},
        "island_prompt_mode": {str(k): v for k, v in state.island_prompt_mode.items()},
        "island_constrained_generations": {
            str(k): v for k, v in state.island_constrained_generations.items()
        },
        "island_recent_failures": {
            str(k): list(v) for k, v in state.island_recent_failures.items()
        },
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
    island_stagnation = {
        int(island): int(value) for island, value in payload.get("island_stagnation", {}).items()
    }
    island_prompt_mode = {
        int(island): str(value) for island, value in payload.get("island_prompt_mode", {}).items()
    }
    island_constrained_generations = {
        int(island): int(value)
        for island, value in payload.get("island_constrained_generations", {}).items()
    }
    island_recent_failures = {
        int(island): [str(item) for item in values]
        for island, values in payload.get("island_recent_failures", {}).items()
        if isinstance(values, list)
    }
    return CheckpointState(
        experiment_id=str(payload.get("experiment_id", "phase1")),
        generation=int(payload["generation"]),
        islands=islands,
        rng_seed=int(payload["rng_seed"]),
        rng_state=payload.get("rng_state"),
        best_val_score=float(payload.get("best_val_score", 0.0)),
        no_improve_count=int(payload.get("no_improve_count", 0)),
        island_stagnation=island_stagnation,
        island_prompt_mode=island_prompt_mode,
        island_constrained_generations=island_constrained_generations,
        island_recent_failures=island_recent_failures,
    )


def rotate_generation_checkpoints(checkpoint_dir: str | Path, keep_last: int) -> None:
    root = Path(checkpoint_dir)
    if keep_last <= 0 or not root.exists():
        return
    checkpoints = sorted(root.glob("gen_*.json"), key=lambda p: int(p.stem.split("_")[-1]))
    for old_path in checkpoints[:-keep_last]:
        old_path.unlink(missing_ok=True)


def write_json(payload: dict, path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
