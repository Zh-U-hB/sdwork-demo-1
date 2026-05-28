"""Individual representation for the genetic algorithm."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Individual:
    """One candidate solution with genes, decoded params, and objectives."""

    id: int
    generation: int
    genes: dict[str, float]
    params: dict[str, Any] = field(default_factory=dict)
    objectives: dict[str, float] = field(default_factory=dict)
    fitness: float | None = None
    feasible: bool = True
    error: str | None = None
    sim_dir: str | None = None

    def copy_with(self, **kwargs: Any) -> Individual:
        data = {
            "id": self.id,
            "generation": self.generation,
            "genes": dict(self.genes),
            "params": dict(self.params),
            "objectives": dict(self.objectives),
            "fitness": self.fitness,
            "feasible": self.feasible,
            "error": self.error,
            "sim_dir": self.sim_dir,
        }
        data.update(kwargs)
        return Individual(**data)
