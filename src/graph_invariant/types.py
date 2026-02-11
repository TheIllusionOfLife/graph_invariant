from dataclasses import dataclass, field


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
    generation: int
    islands: dict[int, list[Candidate]] = field(default_factory=dict)
    rng_seed: int = 42
