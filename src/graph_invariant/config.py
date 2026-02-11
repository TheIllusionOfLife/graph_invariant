import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class Phase1Config:
    seed: int = 42
    target_name: str = "average_shortest_path_length"
    num_train_graphs: int = 50
    num_val_graphs: int = 200
    num_test_graphs: int = 200
    max_generations: int = 20
    population_size: int = 5
    migration_interval: int = 10
    timeout_sec: float = 2.0
    memory_mb: int = 256
    alpha: float = 0.6
    beta: float = 0.2
    gamma: float = 0.2
    artifacts_dir: str = "artifacts"
    model_name: str = "llama3"
    ollama_url: str = "http://localhost:11434/api/generate"

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "Phase1Config":
        return cls(**values)

    @classmethod
    def from_json(cls, path: str | Path) -> "Phase1Config":
        with Path(path).open("r", encoding="utf-8") as f:
            return cls.from_dict(json.load(f))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
