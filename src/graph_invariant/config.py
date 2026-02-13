import json
import warnings
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
    sandbox_max_workers: int | None = None
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
    run_baselines: bool = True
    persist_candidate_code_in_summary: bool = False
    success_spearman_threshold: float = 0.85
    pysr_niterations: int = 30
    pysr_populations: int = 8
    pysr_procs: int = 0
    pysr_timeout_in_seconds: float | None = 60.0
    benchmark_seeds: tuple[int, ...] = (11, 22, 33, 44, 55)
    novelty_bootstrap_samples: int = 1000
    novelty_threshold: float = 0.7
    enforce_pysr_parity_for_success: bool = True
    pysr_parity_epsilon: float = 0.0
    require_baselines_for_success: bool = True
    persist_prompt_and_response_logs: bool = False
    llm_timeout_sec: float = 60.0
    enable_self_correction: bool = True
    self_correction_max_retries: int = 1
    self_correction_feedback_window: int = 3
    novelty_gate_threshold: float = 0.15
    fitness_mode: str = "correlation"
    bound_tolerance: float = 1e-9
    success_bound_score_threshold: float = 0.7
    success_satisfaction_threshold: float = 0.95
    enable_map_elites: bool = False
    map_elites_bins: int = 5

    def __post_init__(self) -> None:
        self.island_temperatures = tuple(float(x) for x in self.island_temperatures)
        if len(self.island_temperatures) != 4:
            raise ValueError("island_temperatures must contain exactly 4 values")
        if self.stagnation_trigger_generations < 1:
            raise ValueError("stagnation_trigger_generations must be >= 1")
        if self.constrained_recovery_generations < 1:
            raise ValueError("constrained_recovery_generations must be >= 1")
        if self.sandbox_max_workers is not None and self.sandbox_max_workers < 1:
            raise ValueError("sandbox_max_workers must be >= 1")
        if not (0.0 <= self.success_spearman_threshold <= 1.0):
            raise ValueError("success_spearman_threshold must be between 0.0 and 1.0")
        if self.alpha < 0.0 or self.beta < 0.0 or self.gamma < 0.0:
            raise ValueError("alpha, beta, gamma must be >= 0.0")
        total = self.alpha + self.beta + self.gamma
        if total <= 0.0:
            raise ValueError("alpha, beta, gamma must sum to > 0.0")
        if abs(total - 1.0) > 1e-9:
            warnings.warn(
                "alpha, beta, gamma did not sum to 1.0; normalizing weights",
                stacklevel=2,
            )
            self.alpha /= total
            self.beta /= total
            self.gamma /= total
        if self.pysr_niterations < 1:
            raise ValueError("pysr_niterations must be >= 1")
        if self.pysr_populations < 1:
            raise ValueError("pysr_populations must be >= 1")
        if self.pysr_procs < 0:
            raise ValueError("pysr_procs must be >= 0")
        if self.pysr_timeout_in_seconds is not None and self.pysr_timeout_in_seconds <= 0.0:
            raise ValueError("pysr_timeout_in_seconds must be > 0.0")
        self.benchmark_seeds = tuple(int(seed) for seed in self.benchmark_seeds)
        if not self.benchmark_seeds:
            raise ValueError("benchmark_seeds must contain at least one seed")
        if self.novelty_bootstrap_samples < 1:
            raise ValueError("novelty_bootstrap_samples must be >= 1")
        if not (0.0 <= self.novelty_threshold <= 1.0):
            raise ValueError("novelty_threshold must be between 0.0 and 1.0")
        if self.pysr_parity_epsilon < 0.0:
            raise ValueError("pysr_parity_epsilon must be >= 0.0")
        if self.llm_timeout_sec <= 0.0:
            raise ValueError("llm_timeout_sec must be > 0.0")
        if self.self_correction_max_retries < 0:
            raise ValueError("self_correction_max_retries must be >= 0")
        if self.self_correction_feedback_window < 1:
            raise ValueError("self_correction_feedback_window must be >= 1")
        if not (0.0 <= self.novelty_gate_threshold <= 1.0):
            raise ValueError("novelty_gate_threshold must be between 0.0 and 1.0")
        if self.fitness_mode not in {"correlation", "upper_bound", "lower_bound"}:
            raise ValueError("fitness_mode must be 'correlation', 'upper_bound', or 'lower_bound'")
        if self.bound_tolerance < 0.0:
            raise ValueError("bound_tolerance must be >= 0.0")
        if not (0.0 <= self.success_bound_score_threshold <= 1.0):
            raise ValueError("success_bound_score_threshold must be between 0.0 and 1.0")
        if not (0.0 <= self.success_satisfaction_threshold <= 1.0):
            raise ValueError("success_satisfaction_threshold must be between 0.0 and 1.0")
        if self.map_elites_bins < 2:
            raise ValueError("map_elites_bins must be >= 2")

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "Phase1Config":
        return cls(**values)

    @classmethod
    def from_json(cls, path: str | Path) -> "Phase1Config":
        with Path(path).open("r", encoding="utf-8") as f:
            return cls.from_dict(json.load(f))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
