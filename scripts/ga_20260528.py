"""GA optimizer for the 2026-05-28 boundary-offset parametric model.

The optimizer keeps total floor area fixed by default and searches geometric
parameters that affect orientation, compactness, offsets, and aerial-platform
shape. EnergyPlus simulation uses the existing rectangular-zone converter, so
only mass_block zones are sent to IDF; aerial platforms remain in the saved
geometry model but are excluded from the energy model until the converter
supports arbitrary prism footprints.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Generator

from scripts.ep_sim_utils import read_eplustbl, run_ep_simulation_direct
from scripts.generate_20260528 import generate_20260528
from scripts.idf_converter import convert_and_run


PENALTY = 1e6


@dataclass(frozen=True)
class GeneSpec:
    name: str
    low: float
    high: float
    step: float
    is_int: bool = False


DEFAULT_GENES: list[GeneSpec] = [
    GeneSpec("boundary_shift", 0.0, 300.0, 5.0),
    GeneSpec("low_aspect_ratio", 0.65, 1.75, 0.05),
    GeneSpec("mid_aspect_ratio", 0.65, 1.75, 0.05),
    GeneSpec("high_aspect_ratio", 0.65, 1.75, 0.05),
    GeneSpec("low_offset_angle", 0.0, 355.0, 5.0),
    GeneSpec("mid_offset_angle", 0.0, 355.0, 5.0),
    GeneSpec("high_offset_angle", 0.0, 355.0, 5.0),
    GeneSpec("low_offset_distance", 0.0, 5.0, 0.25),
    GeneSpec("mid_offset_distance", 0.0, 5.0, 0.25),
    GeneSpec("high_offset_distance", 0.0, 5.0, 0.25),
    GeneSpec("platform_edge_walk_distance", 2.0, 10.0, 0.5),
]


BASELINE_INDIVIDUAL = {
    "boundary_shift": 40.0,
    "low_aspect_ratio": 1.0,
    "mid_aspect_ratio": 1.0,
    "high_aspect_ratio": 1.0,
    "low_offset_angle": 45.0,
    "mid_offset_angle": 180.0,
    "high_offset_angle": 315.0,
    "low_offset_distance": 2.0,
    "mid_offset_distance": 2.0,
    "high_offset_distance": 2.0,
    "platform_edge_walk_distance": 5.0,
}


FIXED_PARAMS = {
    "building_name": "GA20260528_Candidate",
    "site_size": 100.0,
    "total_area": 10000.0,
    "lobby_height": 6.0,
    "floor_height": 4.0,
    "setback_south": 15.0,
    "setback_west": 15.0,
    "setback_north": 10.0,
    "setback_east": 10.0,
    "group_size": 2,
    "min_support_overlap_ratio": 0.5,
    "add_aerial_platforms": True,
    "add_open_space_markers": False,
}


@dataclass
class GAConfig:
    pop_size: int = 6
    n_gen: int = 4
    mutation_rate: float = 0.18
    mutation_sigma: float = 0.15
    crossover_alpha: float = 0.5
    elite_count: int = 1
    tournament_size: int = 3
    cache_path: str = "output/ga20260528_cache.json"
    history_path: str = "output/ga20260528_history.json"
    best_model_path: str = "output/ga20260528_best_model.json"
    output_base: str = "output/ga20260528_runs"


@dataclass
class GenerationResult:
    gen: int
    best_fitness: float
    best_params: dict
    best_model: dict | None
    pop_fitness: list[float]
    avg_fitness: float
    worst_fitness: float


def _clamp(value: float, spec: GeneSpec) -> float:
    steps = round((value - spec.low) / spec.step)
    value = spec.low + steps * spec.step
    value = max(spec.low, min(spec.high, value))
    if spec.is_int:
        return int(round(value))
    return round(value, 6)


def _params_hash(params: dict) -> str:
    payload = json.dumps(params, sort_keys=True)
    return hashlib.md5(payload.encode()).hexdigest()


def _load_cache(path: str) -> dict[str, float]:
    p = Path(path)
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text())
    except (json.JSONDecodeError, OSError):
        return {}


def _save_json(payload: dict | list, path: str) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def random_individual(genes: list[GeneSpec]) -> dict:
    ind = {}
    for gene in genes:
        n_steps = int(round((gene.high - gene.low) / gene.step))
        value = gene.low + random.randint(0, n_steps) * gene.step
        ind[gene.name] = int(value) if gene.is_int else round(value, 6)
    return ind


def crossover(p1: dict, p2: dict, genes: list[GeneSpec], alpha: float) -> tuple[dict, dict]:
    c1, c2 = {}, {}
    for gene in genes:
        v1, v2 = p1[gene.name], p2[gene.name]
        if gene.is_int:
            c1[gene.name] = v1 if random.random() < 0.5 else v2
            c2[gene.name] = v2 if random.random() < 0.5 else v1
            continue
        lo, hi = min(v1, v2), max(v1, v2)
        span = hi - lo
        c1[gene.name] = _clamp(random.uniform(lo - alpha * span, hi + alpha * span), gene)
        c2[gene.name] = _clamp(random.uniform(lo - alpha * span, hi + alpha * span), gene)
    return c1, c2


def mutate(individual: dict, genes: list[GeneSpec], rate: float, sigma: float) -> dict:
    result = dict(individual)
    for gene in genes:
        if random.random() >= rate:
            continue
        if gene.is_int:
            result[gene.name] = _clamp(result[gene.name] + random.choice([-1, 1]) * gene.step, gene)
        else:
            noise = random.gauss(0, sigma * (gene.high - gene.low))
            result[gene.name] = _clamp(result[gene.name] + noise, gene)
    return result


def tournament_select(population: list[dict], fitness: list[float], k: int) -> dict:
    indices = random.sample(range(len(population)), min(k, len(population)))
    return dict(population[min(indices, key=lambda idx: fitness[idx])])


def simulation_model(model: dict) -> dict:
    return {
        **model,
        "zones": [zone for zone in model["zones"] if zone.get("category") == "mass_block"],
        "metadata": {
            **model.get("metadata", {}),
            "energy_model_note": "aerial_platform zones excluded because the current IDF converter supports rectangular boxes only",
        },
    }


def compute_eui(result_dir: str) -> float:
    sim = read_eplustbl(result_dir)
    if not sim.get("exists"):
        return PENALTY
    total_gj = sim["site_energy"].get("Total Site Energy", 0.0)
    area = sim["building_area"].get("Net Conditioned Building Area", 0.0)
    if area <= 0:
        return PENALTY
    return total_gj * 1000 / area


def evaluate_fitness(individual: dict, cache: dict[str, float], config: GAConfig) -> tuple[float, dict | None]:
    params = {**FIXED_PARAMS, **individual}
    key = _params_hash(params)
    if key in cache:
        return cache[key], None

    try:
        model = generate_20260528(**params)
    except Exception:
        cache[key] = PENALTY
        return PENALTY, None

    result_dir = run_ep_simulation_direct(
        simulation_model(model),
        params["building_name"],
        output_base=Path(config.output_base) / key[:8],
    )
    if not result_dir:
        cache[key] = PENALTY
        return PENALTY, model

    eui = compute_eui(result_dir)
    cache[key] = eui
    return eui, model


def run_ga(config: GAConfig, genes: list[GeneSpec] | None = None, seed: int | None = None) -> Generator[GenerationResult, None, None]:
    if seed is not None:
        random.seed(seed)
    genes = genes or DEFAULT_GENES
    cache = _load_cache(config.cache_path)

    population = [dict(BASELINE_INDIVIDUAL)]
    population.extend(random_individual(genes) for _ in range(max(0, config.pop_size - 1)))
    fitness = []
    best_model = None
    for individual in population:
        fit, model = evaluate_fitness(individual, cache, config)
        fitness.append(fit)
        if fit < PENALTY and model is not None:
            best_model = model
    _save_json(cache, config.cache_path)

    for gen in range(config.n_gen + 1):
        ranked = sorted(range(len(population)), key=lambda idx: fitness[idx])
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
        if gen == config.n_gen:
            break

        elite_count = min(config.elite_count, config.pop_size)
        new_population = [dict(population[ranked[i]]) for i in range(elite_count)]
        while len(new_population) < config.pop_size:
            p1 = tournament_select(population, fitness, config.tournament_size)
            p2 = tournament_select(population, fitness, config.tournament_size)
            c1, c2 = crossover(p1, p2, genes, config.crossover_alpha)
            new_population.append(mutate(c1, genes, config.mutation_rate, config.mutation_sigma))
            if len(new_population) < config.pop_size:
                new_population.append(mutate(c2, genes, config.mutation_rate, config.mutation_sigma))

        population = new_population
        fitness = []
        best_model = None
        best_gen_fitness = PENALTY
        for individual in population:
            fit, model = evaluate_fitness(individual, cache, config)
            fitness.append(fit)
            if fit < best_gen_fitness and model is not None:
                best_gen_fitness = fit
                best_model = model
        _save_json(cache, config.cache_path)


def idf_only_check(output_dir: str) -> str | None:
    model = generate_20260528(**FIXED_PARAMS)
    return convert_and_run(
        simulation_model(model),
        output_dir=output_dir,
        run_simulation=False,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run GA optimization for generate_20260528().")
    parser.add_argument("--pop-size", type=int, default=6)
    parser.add_argument("--generations", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cache-path", default="output/ga20260528_cache.json")
    parser.add_argument("--history-path", default="output/ga20260528_history.json")
    parser.add_argument("--best-model-path", default="output/ga20260528_best_model.json")
    parser.add_argument("--output-base", default="output/ga20260528_runs")
    parser.add_argument("--idf-only-check", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.idf_only_check:
        out = idf_only_check("output/ga20260528_idf_check")
        print(f"IDF-only check output: {out}")
        return

    config = GAConfig(
        pop_size=args.pop_size,
        n_gen=args.generations,
        cache_path=args.cache_path,
        history_path=args.history_path,
        best_model_path=args.best_model_path,
        output_base=args.output_base,
    )
    history = []
    best_model = None
    for result in run_ga(config, seed=args.seed):
        row = {
            "generation": result.gen,
            "best_fitness": result.best_fitness,
            "best_params": result.best_params,
            "avg_fitness": result.avg_fitness,
            "worst_fitness": result.worst_fitness,
        }
        history.append(row)
        print(
            f"gen={result.gen} best_eui={result.best_fitness:.2f} "
            f"avg={result.avg_fitness:.2f} worst={result.worst_fitness:.2f}"
        )
        if result.best_model is not None:
            best_model = result.best_model
        _save_json(history, config.history_path)
        if best_model is not None:
            _save_json(best_model, config.best_model_path)

    print(f"History saved to {config.history_path}")
    if best_model is not None:
        print(f"Best model saved to {config.best_model_path}")


if __name__ == "__main__":
    main()
