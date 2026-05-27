"""Genetic algorithm optimizer for parametric L-shape building energy performance.

Minimizes EUI (MJ/m²) by searching the parameter space of generate_l_gradient().
"""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Generator

from scripts.ep_sim_utils import read_eplustbl, run_ep_simulation_direct as run_ep_simulation
from scripts.generate_l_gradient import generate_l_gradient

# ---------------------------------------------------------------------------
# Gene definition
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class GeneSpec:
    name: str
    low: float
    high: float
    step: float
    is_int: bool = False


# 12 optimizable parameters (excludes building_name, site_size, add_courtyard_marker)
DEFAULT_GENES: list[GeneSpec] = [
    GeneSpec("floors",              2,  14,   1,    is_int=True),
    GeneSpec("lobby_height",      3.0, 9.0,  0.1),
    GeneSpec("floor_height",      3.0, 5.0,  0.1),
    GeneSpec("base_x",            0.0, 30.0, 0.5),
    GeneSpec("base_y",            0.0, 30.0, 0.5),
    GeneSpec("arm_width",         6.0, 30.0, 0.5),
    GeneSpec("horizontal_length",20.0, 80.0, 0.5),
    GeneSpec("vertical_length",  20.0, 80.0, 0.5),
    GeneSpec("scatter_gap",       0.0, 18.0, 0.5),
    GeneSpec("min_fragment_scale",0.3,  1.0, 0.01),
    GeneSpec("merge_power",       0.4,  3.0, 0.05),
    GeneSpec("top_solid_floors",    1,   5,   1,   is_int=True),
]

# Fixed parameters
FIXED_PARAMS = {
    "building_name": "GA_Candidate",
    "site_size": 100.0,
    "add_courtyard_marker": False,
}

# Large penalty for invalid / failed individuals
PENALTY = 1e6

# ---------------------------------------------------------------------------
# GA config
# ---------------------------------------------------------------------------

@dataclass
class GAConfig:
    pop_size: int = 10
    n_gen: int = 10
    mutation_rate: float = 0.15
    mutation_sigma: float = 0.2      # relative std for Gaussian mutation
    crossover_alpha: float = 0.5     # BLX-α expansion factor
    elite_count: int = 1
    tournament_size: int = 3
    cache_path: str = "output/ga_cache.json"
    checkpoint_path: str = "output/ga_checkpoint.json"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _clamp(value: float, spec: GeneSpec) -> float:
    """Snap value to nearest valid step and clamp to range."""
    steps = round((value - spec.low) / spec.step)
    value = spec.low + steps * spec.step
    value = max(spec.low, min(spec.high, value))
    if spec.is_int:
        value = int(round(value))
    return value


def _params_hash(params: dict) -> str:
    """Deterministic hash for caching."""
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


# ---------------------------------------------------------------------------
# Individual representation: dict of {gene_name: value}
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Operators
# ---------------------------------------------------------------------------

def crossover(p1: dict, p2: dict, genes: list[GeneSpec], alpha: float) -> tuple[dict, dict]:
    """BLX-α crossover for floats, uniform crossover for ints."""
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


# ---------------------------------------------------------------------------
# Fitness evaluation with caching
# ---------------------------------------------------------------------------

def evaluate_fitness(
    individual: dict,
    cache: dict[str, float],
) -> tuple[float, dict | None]:
    """Return (EUI, model_dict). model_dict is None if evaluation failed."""
    full_params = {**FIXED_PARAMS, **individual}

    # Check cache
    key = _params_hash(full_params)
    if key in cache:
        return cache[key], None

    # Try to generate model
    try:
        model = generate_l_gradient(**full_params)
    except Exception:
        cache[key] = PENALTY
        return PENALTY, None

    # Check constraint: height < 50m
    mass_zones = [z for z in model["zones"] if z["dimensions"]["height"] > 1.0]
    if not mass_zones:
        cache[key] = PENALTY
        return PENALTY, None
    max_h = max(z["origin"]["z"] + z["dimensions"]["height"] for z in mass_zones)
    if max_h >= 50:
        cache[key] = PENALTY
        return PENALTY, model

    # Run simulation
    try:
        result_dir = run_ep_simulation(model, full_params["building_name"])
    except Exception:
        cache[key] = PENALTY
        return PENALTY, model

    if not result_dir:
        cache[key] = PENALTY
        return PENALTY, model

    # Parse results
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


# ---------------------------------------------------------------------------
# Per-generation result
# ---------------------------------------------------------------------------

@dataclass
class GenerationResult:
    gen: int
    best_fitness: float
    best_params: dict
    best_model: dict | None
    pop_fitness: list[float]
    avg_fitness: float
    worst_fitness: float


# ---------------------------------------------------------------------------
# Main GA loop (generator)
# ---------------------------------------------------------------------------

def run_ga(
    config: GAConfig,
    genes: list[GeneSpec] | None = None,
    seed: int | None = None,
) -> Generator[GenerationResult, None, None]:
    """Yield GenerationResult after each generation."""
    if seed is not None:
        random.seed(seed)
    genes = genes or DEFAULT_GENES

    cache = _load_cache(config.cache_path)

    # Initialize population
    population = [random_individual(genes) for _ in range(config.pop_size)]
    fitness = [PENALTY] * config.pop_size
    best_model = None

    # Evaluate initial population
    for i, ind in enumerate(population):
        fit, model = evaluate_fitness(ind, cache)
        fitness[i] = fit
        if fit < PENALTY and model is not None:
            best_model = model

    _save_cache(cache, config.cache_path)

    elite_count = min(config.elite_count, config.pop_size)

    for gen in range(config.n_gen):
        # Sort by fitness (lower is better)
        ranked = sorted(range(config.pop_size), key=lambda i: fitness[i])

        # Yield result for this generation
        best_idx = ranked[0]
        gen_result = GenerationResult(
            gen=gen,
            best_fitness=fitness[best_idx],
            best_params=dict(population[best_idx]),
            best_model=best_model,
            pop_fitness=list(fitness),
            avg_fitness=sum(fitness) / len(fitness),
            worst_fitness=fitness[ranked[-1]],
        )
        yield gen_result

        # Elitism
        new_pop = [dict(population[ranked[i]]) for i in range(elite_count)]

        # Fill rest via selection + crossover + mutation
        while len(new_pop) < config.pop_size:
            p1 = tournament_select(population, fitness, config.tournament_size)
            p2 = tournament_select(population, fitness, config.tournament_size)
            c1, c2 = crossover(p1, p2, genes, config.crossover_alpha)
            c1 = mutate(c1, genes, config.mutation_rate, config.mutation_sigma)
            c2 = mutate(c2, genes, config.mutation_rate, config.mutation_sigma)
            new_pop.append(c1)
            if len(new_pop) < config.pop_size:
                new_pop.append(c2)

        # Evaluate new individuals
        population = new_pop
        fitness = [PENALTY] * config.pop_size
        best_model = None
        best_gen_fitness = PENALTY
        for i, ind in enumerate(population):
            fit, model = evaluate_fitness(ind, cache)
            fitness[i] = fit
            if fit < best_gen_fitness and model is not None:
                best_gen_fitness = fit
                best_model = model

        _save_cache(cache, config.cache_path)

    # Final generation result
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


# ---------------------------------------------------------------------------
# Checkpoint save / load
# ---------------------------------------------------------------------------

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
