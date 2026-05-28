"""Genetic algorithm optimizer for generate_20260528().

This GA targets the newest parametric model in `scripts/generate_20260528.py`.
Fitness is EUI (MJ/m²) computed from EnergyPlus results via the direct simulation
path (JSON -> IDF -> EnergyPlus).
"""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Generator

from scripts.ep_sim_utils import read_eplustbl, run_ep_simulation_direct as run_ep_simulation
from scripts.generate_20260528 import generate_20260528


@dataclass(frozen=True)
class GeneSpec:
    name: str
    low: float
    high: float
    step: float
    is_int: bool = False


DEFAULT_GENES_20260528: list[GeneSpec] = [
    GeneSpec("total_area", 6000.0, 16000.0, 100.0),
    GeneSpec("lobby_height", 3.0, 9.0, 0.1),
    GeneSpec("floor_height", 3.0, 5.0, 0.1),
    GeneSpec("setback_south", 0.0, 30.0, 0.5),
    GeneSpec("setback_west", 0.0, 30.0, 0.5),
    GeneSpec("setback_north", 0.0, 30.0, 0.5),
    GeneSpec("setback_east", 0.0, 30.0, 0.5),
    GeneSpec("low_aspect_ratio", 0.5, 2.0, 0.05),
    GeneSpec("mid_aspect_ratio", 0.5, 2.0, 0.05),
    GeneSpec("high_aspect_ratio", 0.5, 2.0, 0.05),
    GeneSpec("boundary_shift", 0.0, 200.0, 1.0),
    GeneSpec("group_size", 1, 4, 1, is_int=True),
    GeneSpec("low_offset_angle", 0.0, 360.0, 1.0),
    GeneSpec("mid_offset_angle", 0.0, 360.0, 1.0),
    GeneSpec("high_offset_angle", 0.0, 360.0, 1.0),
    GeneSpec("low_offset_distance", 0.0, 10.0, 0.1),
    GeneSpec("mid_offset_distance", 0.0, 10.0, 0.1),
    GeneSpec("high_offset_distance", 0.0, 10.0, 0.1),
    GeneSpec("min_support_overlap_ratio", 0.1, 1.0, 0.05),
    GeneSpec("platform_edge_walk_distance", 1.0, 12.0, 0.5),
    GeneSpec("add_aerial_platforms", 0, 1, 1, is_int=True),  # bool encoded as 0/1
]


PENALTY = 1e6


@dataclass
class GAConfig:
    pop_size: int = 10
    n_gen: int = 10
    mutation_rate: float = 0.15
    mutation_sigma: float = 0.2
    crossover_alpha: float = 0.5
    elite_count: int = 1
    tournament_size: int = 3
    cache_path: str = "output/ga20260528_cache.json"
    checkpoint_path: str = "output/ga20260528_checkpoint.json"


def _clamp(value: float, spec: GeneSpec) -> float:
    steps = round((value - spec.low) / spec.step)
    value = spec.low + steps * spec.step
    value = max(spec.low, min(spec.high, value))
    if spec.is_int:
        value = int(round(value))
    return value


def _params_hash(params: dict) -> str:
    payload = json.dumps(params, sort_keys=True)
    return hashlib.md5(payload.encode()).hexdigest()


def _load_cache(path: str) -> dict[str, float]:
    p = Path(path)
    if p.exists():
        try:
            return json.loads(p.read_text())
        except (json.JSONDecodeError, OSError):
            return {}
    return {}


def _save_cache(cache: dict[str, float], path: str) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(cache, indent=2))


def random_individual(genes: list[GeneSpec]) -> dict:
    ind = {}
    for g in genes:
        n_steps = int(round((g.high - g.low) / g.step))
        step_idx = random.randint(0, n_steps)
        val = g.low + step_idx * g.step
        if g.is_int:
            val = int(val)
        ind[g.name] = val
    return ind


def crossover(p1: dict, p2: dict, genes: list[GeneSpec], alpha: float) -> tuple[dict, dict]:
    c1, c2 = {}, {}
    for g in genes:
        v1, v2 = p1[g.name], p2[g.name]
        if g.is_int:
            c1[g.name] = v1 if random.random() < 0.5 else v2
            c2[g.name] = v2 if random.random() < 0.5 else v1
        else:
            lo = min(v1, v2)
            hi = max(v1, v2)
            span = hi - lo
            c1[g.name] = _clamp(random.uniform(lo - alpha * span, hi + alpha * span), g)
            c2[g.name] = _clamp(random.uniform(lo - alpha * span, hi + alpha * span), g)
    return c1, c2


def mutate(ind: dict, genes: list[GeneSpec], rate: float, sigma: float) -> dict:
    result = dict(ind)
    for g in genes:
        if random.random() < rate:
            if g.is_int:
                delta = random.choice([-1, 1])
                result[g.name] = _clamp(result[g.name] + delta * g.step, g)
            else:
                noise = random.gauss(0, sigma * (g.high - g.low))
                result[g.name] = _clamp(result[g.name] + noise, g)
    return result


def tournament_select(population: list[dict], fitness: list[float], k: int) -> dict:
    indices = random.sample(range(len(population)), k)
    best = min(indices, key=lambda i: fitness[i])
    return dict(population[best])


def evaluate_fitness(
    individual: dict,
    cache: dict[str, float],
    *,
    fixed_params: dict,
) -> tuple[float, dict | None]:
    """Return (EUI, model_dict). model_dict is None if evaluation failed."""
    full_params = {**fixed_params, **individual}
    # decode bool
    if "add_aerial_platforms" in full_params:
        full_params["add_aerial_platforms"] = bool(int(full_params["add_aerial_platforms"]))

    key = _params_hash(full_params)
    if key in cache:
        return cache[key], None

    try:
        model = generate_20260528(**full_params)
    except Exception:
        cache[key] = PENALTY
        return PENALTY, None

    try:
        result_dir = run_ep_simulation(model, full_params.get("building_name", "GA_20260528"))
    except Exception:
        cache[key] = PENALTY
        return PENALTY, model

    if not result_dir:
        cache[key] = PENALTY
        return PENALTY, model

    sim = read_eplustbl(result_dir)
    if not sim.get("exists"):
        cache[key] = PENALTY
        return PENALTY, model

    total_gj = sim["site_energy"].get("Total Site Energy", 0.0)
    area = sim["building_area"].get("Net Conditioned Building Area", 0.0)
    if area <= 0:
        cache[key] = PENALTY
        return PENALTY, model

    eui = total_gj * 1000 / area
    cache[key] = eui
    return eui, model


@dataclass
class GenerationResult:
    gen: int
    best_fitness: float
    best_params: dict
    best_model: dict | None
    pop_fitness: list[float]
    avg_fitness: float
    worst_fitness: float


def run_ga(
    config: GAConfig,
    *,
    fixed_params: dict,
    genes: list[GeneSpec] | None = None,
    seed: int | None = None,
) -> Generator[GenerationResult, None, None]:
    if seed is not None:
        random.seed(seed)
    genes = genes or DEFAULT_GENES_20260528
    cache = _load_cache(config.cache_path)

    population = [random_individual(genes) for _ in range(config.pop_size)]
    fitness = [PENALTY] * config.pop_size
    best_model = None

    for i, ind in enumerate(population):
        fit, model = evaluate_fitness(ind, cache, fixed_params=fixed_params)
        fitness[i] = fit
        if fit < PENALTY and model is not None:
            best_model = model

    _save_cache(cache, config.cache_path)

    elite_count = min(config.elite_count, config.pop_size)

    for gen in range(config.n_gen):
        ranked = sorted(range(config.pop_size), key=lambda i: fitness[i])
        best_idx = ranked[0]
        yield GenerationResult(
            gen=gen,
            best_fitness=fitness[best_idx],
            best_params=dict(population[best_idx]),
            best_model=best_model,
            pop_fitness=list(fitness),
            avg_fitness=sum(fitness) / len(fitness),
            worst_fitness=fitness[ranked[-1]],
        )

        new_pop = [dict(population[ranked[i]]) for i in range(elite_count)]
        while len(new_pop) < config.pop_size:
            p1 = tournament_select(population, fitness, config.tournament_size)
            p2 = tournament_select(population, fitness, config.tournament_size)
            c1, c2 = crossover(p1, p2, genes, config.crossover_alpha)
            c1 = mutate(c1, genes, config.mutation_rate, config.mutation_sigma)
            c2 = mutate(c2, genes, config.mutation_rate, config.mutation_sigma)
            new_pop.append(c1)
            if len(new_pop) < config.pop_size:
                new_pop.append(c2)

        population = new_pop
        fitness = [PENALTY] * config.pop_size
        best_model = None
        best_gen_fitness = PENALTY
        for i, ind in enumerate(population):
            fit, model = evaluate_fitness(ind, cache, fixed_params=fixed_params)
            fitness[i] = fit
            if fit < best_gen_fitness and model is not None:
                best_gen_fitness = fit
                best_model = model

        _save_cache(cache, config.cache_path)

    ranked = sorted(range(config.pop_size), key=lambda i: fitness[i])
    best_idx = ranked[0]
    yield GenerationResult(
        gen=config.n_gen,
        best_fitness=fitness[best_idx],
        best_params=dict(population[best_idx]),
        best_model=best_model,
        pop_fitness=list(fitness),
        avg_fitness=sum(fitness) / len(fitness),
        worst_fitness=fitness[ranked[-1]],
    )


@dataclass
class CheckpointState:
    generation: int
    population: list[dict]
    fitness: list[float]
    config: dict = field(default_factory=dict)
    history: list[dict] = field(default_factory=list)


def save_checkpoint(state: CheckpointState, path: str) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "generation": state.generation,
        "population": state.population,
        "fitness": state.fitness,
        "config": state.config,
        "history": state.history,
    }
    p.write_text(json.dumps(data, indent=2, ensure_ascii=False))


def load_checkpoint(path: str) -> CheckpointState | None:
    p = Path(path)
    if not p.exists():
        return None
    data = json.loads(p.read_text())
    return CheckpointState(
        generation=data["generation"],
        population=data["population"],
        fitness=data["fitness"],
        config=data.get("config", {}),
        history=data.get("history", []),
    )

