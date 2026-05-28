"""Genetic algorithm optimizer for generate_20260528().

This GA targets the newest parametric model in `scripts/generate_20260528.py`.
Fitness is EUI (MJ/m²) computed from EnergyPlus results via the direct simulation
path (JSON -> IDF -> EnergyPlus).
"""

from __future__ import annotations

import hashlib
import json
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Generator

from scripts.ep_sim_utils import read_eplustbl, run_ep_simulation_direct as run_ep_simulation
from scripts.generate_20260528 import generate_20260528
from scripts.zone_partition import partition_model_by_floor


@dataclass(frozen=True)
class GeneSpec:
    name: str
    low: float
    high: float
    step: float
    is_int: bool = False


# Five genes only; shared_* uniformly controls all three blocks (see generate_20260528).
DEFAULT_GENES_20260528: list[GeneSpec] = [
    GeneSpec("shared_aspect_ratio", 0.5, 2.0, 0.05),
    GeneSpec("shared_offset_angle", 0.0, 360.0, 1.0),
    GeneSpec("shared_offset_distance", 0.0, 10.0, 0.1),
    GeneSpec("boundary_shift", 0.0, 200.0, 1.0),
    GeneSpec("window_wwr", 0.0, 0.8, 0.05),
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
    # GA run output organization:
    # - each GA run gets a directory under run_root (unless run_dir is provided)
    # - each simulation evaluation is written under <run_dir>/sims/<eval_id>/
    run_dir: str | None = None
    run_root: str = "output/ga_runs"
    use_cache: bool = True  # if False: run every evaluation (no dedupe)
    # Partition settings (applied before JSON->IDF conversion)
    partition_enabled: bool = True
    perimeter_depth: float = 4.0


def _ensure_run_dir(config: GAConfig) -> Path:
    if config.run_dir:
        p = Path(config.run_dir)
        p.mkdir(parents=True, exist_ok=True)
        return p
    ts = time.strftime("%Y%m%d_%H%M%S")
    ns = time.time_ns() % 1_000_000_000
    p = Path(config.run_root) / f"ga_{ts}_{ns:09d}"
    p.mkdir(parents=True, exist_ok=True)
    return p


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


def baseline_individual_from_fixed(fixed_params: dict, genes: list[GeneSpec]) -> dict:
    """Build a valid-ish individual from current fixed params (UI baseline)."""
    ind: dict = {}
    for g in genes:
        if g.name in fixed_params:
            val = fixed_params[g.name]
            if g.is_int:
                try:
                    val = int(val)
                except Exception:
                    val = int(round(float(val)))
            else:
                val = float(val)
            ind[g.name] = _clamp(val, g)
        else:
            # fallback to midpoint
            mid = (g.low + g.high) / 2
            ind[g.name] = _clamp(mid, g)
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
    k = max(1, min(int(k), len(population)))
    indices = random.sample(range(len(population)), k)
    best = min(indices, key=lambda i: fitness[i])
    return dict(population[best])


def _prepare_geo_params(merged: dict) -> dict:
    """Merge fixed + individual and enforce unified shared building controls."""
    from scripts.facade_params import decode_generator_bools

    geo = {k: v for k, v in merged.items() if not str(k).startswith("_")}
    if geo.get("shared_aspect_ratio") is not None:
        for key in ("low_aspect_ratio", "mid_aspect_ratio", "high_aspect_ratio"):
            geo.pop(key, None)
    if geo.get("shared_offset_angle") is not None:
        for key in ("low_offset_angle", "mid_offset_angle", "high_offset_angle"):
            geo.pop(key, None)
    if geo.get("shared_offset_distance") is not None:
        for key in ("low_offset_distance", "mid_offset_distance", "high_offset_distance"):
            geo.pop(key, None)
    return decode_generator_bools(geo)


def evaluate_fitness(
    individual: dict,
    cache: dict[str, float],
    *,
    fixed_params: dict,
    sims_dir: Path,
    eval_subdir: str,
    use_cache: bool,
) -> tuple[float, dict | None, str | None]:
    """Return (EUI, model_dict, result_dir). model_dict is None if evaluation failed."""
    geo_params = _prepare_geo_params({**fixed_params, **individual})

    key = _params_hash(geo_params)
    if use_cache and key in cache:
        return cache[key], None, None

    try:
        raw_model = generate_20260528(**geo_params)
    except Exception:
        cache[key] = PENALTY
        return PENALTY, None, None

    model = raw_model
    if fixed_params.get("_partition_enabled", True):
        try:
            model = partition_model_by_floor(
                raw_model,
                perimeter_depth=float(fixed_params.get("_perimeter_depth", 4.0)),
                lobby_height=float(geo_params.get("lobby_height", 6.0)),
                floor_height=float(geo_params.get("floor_height", 4.0)),
            )
        except Exception:
            cache[key] = PENALTY
            return PENALTY, raw_model, None

    try:
        from scripts.facade_params import make_ep_defaults_for_geometry

        ep_defaults = make_ep_defaults_for_geometry(geo_params)
        eval_id = eval_subdir
        result_dir = run_ep_simulation(
            model,
            geo_params.get("building_name", "GA_20260528"),
            defaults=ep_defaults,
            output_base=sims_dir,
            run_id=eval_id,
            weather_file=fixed_params.get("_weather_file"),
        )
    except Exception:
        cache[key] = PENALTY
        return PENALTY, model, None

    if not result_dir:
        cache[key] = PENALTY
        return PENALTY, model, None

    sim = read_eplustbl(result_dir)
    if not sim.get("exists"):
        cache[key] = PENALTY
        return PENALTY, model, result_dir

    total_gj = sim["site_energy"].get("Total Site Energy", 0.0)
    area = sim["building_area"].get("Net Conditioned Building Area", 0.0)
    if area <= 0:
        cache[key] = PENALTY
        return PENALTY, model, result_dir

    eui = total_gj * 1000 / area
    cache[key] = eui
    return eui, model, result_dir


@dataclass
class GenerationResult:
    gen: int
    best_fitness: float
    best_params: dict
    best_model: dict | None
    best_result_dir: str | None
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

    run_dir = _ensure_run_dir(config)
    sims_dir = run_dir / "sims"
    sims_dir.mkdir(parents=True, exist_ok=True)

    # Write GA run metadata once per execution
    meta = {
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "pop_size": int(config.pop_size),
        "n_gen": int(config.n_gen),
        "mutation_rate": float(config.mutation_rate),
        "mutation_sigma": float(config.mutation_sigma),
        "crossover_alpha": float(config.crossover_alpha),
        "elite_count": int(config.elite_count),
        "tournament_size": int(config.tournament_size),
        "seed": int(seed) if seed is not None else None,
        "batch_size": int(config.pop_size),
        "gene_count": len(genes),
        "gene_names": [g.name for g in genes],
        "use_cache": bool(config.use_cache),
        "expected_evaluations": int(config.pop_size) * (int(config.n_gen) + 1),
        "partition_enabled": bool(config.partition_enabled),
        "perimeter_depth": float(config.perimeter_depth),
    }
    (run_dir / "run_meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))

    # If caller didn't override, keep cache/checkpoint inside this GA run dir
    if config.cache_path == "output/ga20260528_cache.json":
        config.cache_path = str(run_dir / "ga_cache.json")
    if config.checkpoint_path == "output/ga20260528_checkpoint.json":
        config.checkpoint_path = str(run_dir / "checkpoint.json")

    cache = _load_cache(config.cache_path) if config.use_cache else {}

    # Inject partition settings into the fixed params passed into evaluation.
    fixed_params = dict(fixed_params)
    fixed_params["_partition_enabled"] = bool(config.partition_enabled)
    fixed_params["_perimeter_depth"] = float(config.perimeter_depth)

    population = [random_individual(genes) for _ in range(config.pop_size)]
    # Force-populate a baseline individual derived from the current UI params to
    # avoid the common case where all-random individuals violate constraints.
    if population:
        population[0] = baseline_individual_from_fixed(fixed_params, genes)
    fitness = [PENALTY] * config.pop_size
    best_model = None
    best_result_dir = None

    # Always keep a baseline model for UI display/debugging, even if all individuals
    # are penalized. This mirrors the user's current sidebar parameters.
    try:
        baseline_ind = baseline_individual_from_fixed(fixed_params, genes)
        baseline_geo = _prepare_geo_params({**fixed_params, **baseline_ind})
        raw = generate_20260528(**baseline_geo)
        if fixed_params.get("_partition_enabled", True):
            raw = partition_model_by_floor(
                raw,
                perimeter_depth=float(fixed_params.get("_perimeter_depth", 4.0)),
                lobby_height=float(baseline_geo.get("lobby_height", 6.0)),
                floor_height=float(baseline_geo.get("floor_height", 4.0)),
            )
        best_model = raw
    except Exception:
        # If even baseline fails, we fall back to None and let the UI instruct the user.
        best_model = None

    for i, ind in enumerate(population):
        eval_subdir = f"gen_init/ind_{i:02d}_{time.time_ns() % 1_000_000_000:09d}"
        fit, model, rdir = evaluate_fitness(
            ind,
            cache,
            fixed_params=fixed_params,
            sims_dir=sims_dir,
            eval_subdir=eval_subdir,
            use_cache=config.use_cache,
        )
        fitness[i] = fit
        if (fit < PENALTY and model is not None) or (best_model is None and model is not None):
            best_model = model
            best_result_dir = rdir

    if config.use_cache:
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
            best_result_dir=best_result_dir,
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
        best_result_dir = None
        for i, ind in enumerate(population):
            eval_subdir = f"gen_{gen:02d}/ind_{i:02d}_{time.time_ns() % 1_000_000_000:09d}"
            fit, model, rdir = evaluate_fitness(
                ind,
                cache,
                fixed_params=fixed_params,
                sims_dir=sims_dir,
                eval_subdir=eval_subdir,
                use_cache=config.use_cache,
            )
            fitness[i] = fit
            if fit < best_gen_fitness and model is not None:
                best_gen_fitness = fit
                best_model = model
                best_result_dir = rdir

        if config.use_cache:
            _save_cache(cache, config.cache_path)

    ranked = sorted(range(config.pop_size), key=lambda i: fitness[i])
    best_idx = ranked[0]
    yield GenerationResult(
        gen=config.n_gen,
        best_fitness=fitness[best_idx],
        best_params=dict(population[best_idx]),
        best_model=best_model,
        best_result_dir=best_result_dir,
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
