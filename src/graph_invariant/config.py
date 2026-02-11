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
    island_temperatures: tuple[float, float, float, float] = (0.3, 0.3, 0.8, 1.2)
    timeout_sec: float = 2.0
    memory_mb: int = 256
    alpha: float = 0.6
    beta: float = 0.2
    gamma: float = 0.2
    train_score_threshold: float = 0.3
    early_stop_patience: int = 10
    artifacts_dir: str = "artifacts"
    model_name: str = "gpt-oss:20b"
    ollama_url: str = "http://localhost:11434/api/generate"
    allow_remote_ollama: bool = False
    checkpoint_keep_last: int = 3
    experiment_id: str | None = None
    enable_constrained_fallback: bool = True
    stagnation_trigger_generations: int = 5
    constrained_recovery_generations: int = 3
    allow_late_constrained_recovery: bool = True
    run_baselines: bool = False
    persist_candidate_code_in_summary: bool = False
    success_spearman_threshold: float = 0.85

    def __post_init__(self) -> None:
        self.island_temperatures = tuple(float(x) for x in self.island_temperatures)
        if len(self.island_temperatures) != 4:
            raise ValueError("island_temperatures must contain exactly 4 values")
        if self.stagnation_trigger_generations < 1:
            raise ValueError("stagnation_trigger_generations must be >= 1")
        if self.constrained_recovery_generations < 1:
            raise ValueError("constrained_recovery_generations must be >= 1")

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "Phase1Config":
        return cls(**values)

    @classmethod
    def from_json(cls, path: str | Path) -> "Phase1Config":
        with Path(path).open("r", encoding="utf-8") as f:
            return cls.from_dict(json.load(f))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
