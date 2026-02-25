"""HarmonyConfig: validated dataclass configuration for Harmony experiments."""

from __future__ import annotations

import json
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class HarmonyConfig:
    """Configuration for Harmony framework experiments.

    Intentionally standalone from ``Phase1Config``: the two configs serve
    different subsystems (KG theory discovery vs graph-formula discovery) and
    share only a small set of LLM/infra defaults by convention, not inheritance.
    """

    seed: int = 42
    domain: str = "linear_algebra"
    max_generations: int = 20
    population_size: int = 5
    migration_interval: int = 10
    island_temperatures: tuple[float, ...] = (0.3, 0.3, 0.8, 1.2)
    # Harmony metric weights (must sum to 1.0)
    alpha: float = 0.25  # compressibility
    beta: float = 0.25  # coherence
    gamma: float = 0.25  # symmetry
    delta: float = 0.25  # generativity
    # Dataset split
    hidden_ratio: float = 0.1
    # Proposal engine
    model_name: str = "gpt-oss:20b"
    ollama_url: str = "http://localhost:11434/api/generate"
    llm_timeout_sec: float = 60.0
    enable_self_correction: bool = True
    self_correction_max_retries: int = 1
    # MAP-Elites
    enable_map_elites: bool = True
    map_elites_bins: int = 5
    # Stagnation recovery
    stagnation_trigger_generations: int = 5
    constrained_recovery_generations: int = 3
    # Early stopping
    early_stop_patience: int = 10
    # Artifacts
    artifacts_dir: str = "artifacts/harmony"
    experiment_id: str | None = None
    # Weight sensitivity grid
    alpha_grid: tuple[float, ...] = (0.3, 0.5, 0.7)
    beta_grid: tuple[float, ...] = (0.1, 0.3)

    def __post_init__(self) -> None:
        self.island_temperatures = tuple(float(x) for x in self.island_temperatures)
        if len(self.island_temperatures) < 1:
            raise ValueError("island_temperatures must contain at least 1 value")
        if self.stagnation_trigger_generations < 1:
            raise ValueError("stagnation_trigger_generations must be >= 1")
        if self.constrained_recovery_generations < 1:
            raise ValueError("constrained_recovery_generations must be >= 1")
        if self.llm_timeout_sec <= 0.0:
            raise ValueError("llm_timeout_sec must be > 0.0")
        if self.self_correction_max_retries < 0:
            raise ValueError("self_correction_max_retries must be >= 0")
        if self.map_elites_bins < 2:
            raise ValueError("map_elites_bins must be >= 2")
        if not (0.0 < self.hidden_ratio < 1.0):
            raise ValueError("hidden_ratio must be in (0, 1)")
        self._validate_weights()
        self.alpha_grid = tuple(float(x) for x in self.alpha_grid)
        self.beta_grid = tuple(float(x) for x in self.beta_grid)

    def _validate_weights(self) -> None:
        for name, val in [
            ("alpha", self.alpha),
            ("beta", self.beta),
            ("gamma", self.gamma),
            ("delta", self.delta),
        ]:
            if val < 0.0:
                raise ValueError(f"{name} must be >= 0.0")
        total = self.alpha + self.beta + self.gamma + self.delta
        if total <= 0.0:
            raise ValueError("alpha, beta, gamma, delta must sum to > 0.0")
        if abs(total - 1.0) > 1e-9:
            warnings.warn(
                "alpha, beta, gamma, delta did not sum to 1.0; normalizing weights",
                stacklevel=3,
            )
            self.alpha /= total
            self.beta /= total
            self.gamma /= total
            self.delta /= total

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> HarmonyConfig:
        return cls(**values)

    @classmethod
    def from_json(cls, path: str | Path) -> HarmonyConfig:
        with Path(path).open("r", encoding="utf-8") as f:
            return cls.from_dict(json.load(f))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
