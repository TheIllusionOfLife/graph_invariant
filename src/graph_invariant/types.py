from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class Candidate:
    id: str
    code: str
    island_id: int = 0
    generation: int = 0
    train_score: float = 0.0
    val_score: float = 0.0
    simplicity_score: float = 0.0
    novelty_bonus: float = 0.0


@dataclass(slots=True)
class EvaluationResult:
    rho_spearman: float
    r_pearson: float
    rmse: float
    mae: float
    valid_count: int
    error_count: int


@dataclass(slots=True)
class CheckpointState:
    experiment_id: str
    generation: int
    islands: dict[int, list[Candidate]] = field(default_factory=dict)
    rng_seed: int = 42
    rng_state: dict[str, Any] | None = None
    best_val_score: float = 0.0
    no_improve_count: int = 0
    island_stagnation: dict[int, int] = field(default_factory=dict)
    island_prompt_mode: dict[int, str] = field(default_factory=dict)
    island_constrained_generations: dict[int, int] = field(default_factory=dict)
    island_recent_failures: dict[int, list[str]] = field(default_factory=dict)
