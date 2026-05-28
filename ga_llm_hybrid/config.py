"""Load and validate hybrid optimizer configuration (YAML/JSON)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, field_validator


class ProjectConfig(BaseModel):
    name: str = "hybrid_optimization"
    seed: int = 42
    max_rounds: int = 10
    debug_mode: bool = False
    min_rounds: int = 2
    morris_enabled: bool = True
    morris_trajectories: int = 6


class GAConfig(BaseModel):
    population_size: int = 50
    max_generations: int = 100
    crossover_rate: float = 0.9
    mutation_rate: float = 0.1
    tournament_size: int = 3
    elite_fraction: float = 0.05
    elite_count: int | None = None
    sbx_eta: float = 15.0
    poly_mutation_eta: float = 20.0
    convergence_generations: int = 15
    convergence_threshold: float = 0.01
    init_mode: Literal["random", "llm_guided", "warm_start"] = "random"
    llm_seed_fraction: float = 0.2
    global_explore_fraction: float = 0.2
    warm_start_path: str | None = None


class EnergyPlusConfig(BaseModel):
    backend: Literal["arch_model", "template_idf"] = "arch_model"
    executable_path: str | None = None
    epw_path: str | None = None
    template_idf: str | None = None
    parallel_jobs: int = 4
    simulation_timeout: int = 300
    partition_enabled: bool = True
    perimeter_depth: float = 4.0
    fixed_geometry: dict[str, Any] = Field(default_factory=dict)


class LLMConfigSection(BaseModel):
    enabled: bool = True
    analysis_frequency: int = 2
    temperature: float = 0.3
    max_tokens: int = 4000
    provider: str | None = None
    model: str | None = None


class ParamDef(BaseModel):
    name: str
    type: Literal["continuous", "categorical", "ordinal", "boolean"] = "continuous"
    range: list[float] | None = None
    values: list[str] | None = None
    unit: str = "-"
    llm_adjustable: bool = True
    fixed: bool = False


class ObjectiveDef(BaseModel):
    name: str
    weight: float = 1.0
    direction: Literal["minimize", "maximize"] = "minimize"
    output_key: str = "eui_mj_m2"


class HybridConfig(BaseModel):
    model_config = {"extra": "ignore"}

    project: ProjectConfig = Field(default_factory=ProjectConfig)
    ga: GAConfig = Field(default_factory=GAConfig)
    energyplus: EnergyPlusConfig = Field(default_factory=EnergyPlusConfig)
    llm: LLMConfigSection = Field(default_factory=LLMConfigSection)
    parameters: list[ParamDef] = Field(default_factory=list)
    objectives: list[ObjectiveDef] = Field(default_factory=list)
    mappings: list[dict[str, Any]] = Field(default_factory=list)
    building_context: dict[str, str] = Field(default_factory=dict)

    @field_validator("objectives")
    @classmethod
    def _at_least_one_objective(cls, v: list[ObjectiveDef]) -> list[ObjectiveDef]:
        if not v:
            return [ObjectiveDef(name="eui_mj_m2", weight=1.0, direction="minimize")]
        return v


def load_config(path: str | Path) -> HybridConfig:
    """Load configuration from YAML or JSON file."""
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    if p.suffix.lower() in {".yaml", ".yml"}:
        raw = yaml.safe_load(text)
    else:
        raw = json.loads(text)
    if "parameters" in raw and isinstance(raw["parameters"], dict):
        flat: list[dict] = []
        type_map = {
            "continuous": "continuous",
            "categorical": "categorical",
            "ordinal": "ordinal",
            "boolean": "boolean",
            "booleans": "boolean",
        }
        for ptype, items in raw["parameters"].items():
            for item in items:
                item = dict(item)
                item.setdefault("type", type_map.get(ptype, ptype))
                flat.append(item)
        raw["parameters"] = flat
    return HybridConfig.model_validate(raw)
