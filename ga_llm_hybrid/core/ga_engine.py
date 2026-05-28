"""Genetic algorithm engine with SBX crossover and polynomial mutation."""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path
from typing import Any, Callable

from ga_llm_hybrid.config import GAConfig, ObjectiveDef
from ga_llm_hybrid.core.individual import Individual
from ga_llm_hybrid.core.objective import pareto_front, scalar_fitness
from ga_llm_hybrid.core.parameter_space import ParameterSpace

logger = logging.getLogger(__name__)

EvaluateFn = Callable[[list[Individual], int], list[Individual]]


class GAEngine:
    """Run generational GA over encoded parameter genes."""

    def __init__(
        self,
        space: ParameterSpace,
        config: GAConfig,
        objectives: list[ObjectiveDef],
        evaluate_fn: EvaluateFn,
        seed: int = 42,
    ) -> None:
        self.space = space
        self.config = config
        self.objectives = objectives
        self.evaluate_fn = evaluate_fn
        self.rng = random.Random(seed)
        self._history_best: list[float] = []
        self._next_id = 0

    def _new_id(self) -> int:
        self._next_id += 1
        return self._next_id

    def init_population(
        self,
        seeds: list[dict[str, Any]] | None = None,
        warm_start: list[dict[str, Any]] | None = None,
    ) -> list[Individual]:
        """Build initial population (LHS + optional seeds / warm start)."""
        n = self.config.population_size
        cfg = self.config
        pop_params: list[dict[str, Any]] = []

        if warm_start:
            pop_params.extend(warm_start[:n])
        elif cfg.init_mode == "warm_start" and cfg.warm_start_path:
            loaded = self._load_warm_start(cfg.warm_start_path)
            pop_params.extend(loaded[:n])

        if seeds:
            n_seeds = min(len(seeds), max(1, int(n * cfg.llm_seed_fraction)))
            pop_params.extend(seeds[:n_seeds])

        n_explore = int(n * cfg.global_explore_fraction)
        n_lhs = n - len(pop_params) - n_explore
        if n_lhs > 0:
            pop_params.extend(self.space.latin_hypercube_population(n_lhs, use_original=False))
        if n_explore > 0:
            pop_params.extend(self.space.latin_hypercube_population(n_explore, use_original=True))

        while len(pop_params) < n:
            pop_params.append(self.space.sample_random(use_original=False))
        pop_params = pop_params[:n]

        pop: list[Individual] = []
        for p in pop_params:
            full_p = self.space.complete_params(p)
            genes = self.space.encode(full_p)
            pop.append(
                Individual(
                    id=self._new_id(),
                    generation=0,
                    genes=genes,
                    params=full_p,
                )
            )
        return pop

    @staticmethod
    def _load_warm_start(path: str) -> list[dict[str, Any]]:
        p = Path(path)
        if not p.exists():
            logger.warning("warm_start_path not found: %s", path)
            return []
        data = json.loads(p.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return [d for d in data if isinstance(d, dict)]
        if isinstance(data, dict) and "population" in data:
            return list(data["population"])
        return []

    def run(
        self,
        max_generations: int | None = None,
        seeds: list[dict[str, Any]] | None = None,
        warm_start: list[dict[str, Any]] | None = None,
    ) -> tuple[list[Individual], list[list[Individual]]]:
        """Run GA; return final population and per-generation snapshots."""
        max_gen = max_generations or self.config.max_generations
        pop = self.init_population(seeds=seeds, warm_start=warm_start)
        pop = self.evaluate_fn(pop, 0)
        snapshots: list[list[Individual]] = [list(pop)]
        best_fit = min((p.fitness or 1e9) for p in pop)
        self._history_best = [best_fit]
        stall = 0

        for gen in range(1, max_gen + 1):
            offspring = self._breed(pop)
            for ind in offspring:
                ind.generation = gen
            offspring = self.evaluate_fn(offspring, gen)
            pop = self._environmental_selection(pop, offspring)
            snapshots.append(list(pop))
            best_fit = min((p.fitness or 1e9) for p in pop)
            self._history_best.append(best_fit)

            if len(self._history_best) >= 2:
                prev = self._history_best[-2]
                rel = abs(best_fit - prev) / max(abs(prev), 1e-9)
                if rel < self.config.convergence_threshold:
                    stall += 1
                else:
                    stall = 0
            if stall >= self.config.convergence_generations:
                logger.info("GA converged at generation %d", gen)
                break

        return pop, snapshots

    def _elite_count(self, n: int) -> int:
        by_fraction = max(1, int(n * self.config.elite_fraction))
        by_count = self.config.elite_count
        if by_count is not None:
            return max(1, min(by_count, n))
        return by_fraction

    def _tournament_select(self, pop: list[Individual], k: int | None = None) -> Individual:
        k = min(k or self.config.tournament_size, len(pop))
        contenders = self.rng.sample(pop, k)
        return min(contenders, key=lambda x: x.fitness or 1e9)

    def _breed(self, pop: list[Individual]) -> list[Individual]:
        n = len(pop)
        elite_n = self._elite_count(n)
        sorted_pop = sorted(pop, key=lambda x: x.fitness or 1e9)
        offspring: list[Individual] = [
            sorted_pop[i].copy_with(id=self._new_id()) for i in range(elite_n)
        ]
        while len(offspring) < n:
            p1 = self._tournament_select(pop)
            p2 = self._tournament_select(pop)
            if self.rng.random() < self.config.crossover_rate:
                c1_genes, c2_genes = self._crossover(p1.genes, p2.genes)
            else:
                c1_genes, c2_genes = dict(p1.genes), dict(p2.genes)
            c1_genes = self._mutate(c1_genes)
            c2_genes = self._mutate(c2_genes)
            for genes in (c1_genes, c2_genes):
                if len(offspring) >= n:
                    break
                params = self.space.decode(genes)
                offspring.append(
                    Individual(
                        id=self._new_id(),
                        generation=p1.generation + 1,
                        genes=genes,
                        params=params,
                    )
                )
        return offspring[:n]

    def _environmental_selection(
        self, parents: list[Individual], offspring: list[Individual]
    ) -> list[Individual]:
        combined = parents + offspring
        combined.sort(key=lambda x: x.fitness or 1e9)
        return combined[: len(parents)]

    def _crossover(
        self, g1: dict[str, float], g2: dict[str, float]
    ) -> tuple[dict[str, float], dict[str, float]]:
        c1, c2 = {}, {}
        for key in g1:
            if self.space.param_type(key) == "continuous":
                c1[key], c2[key] = self._sbx_pair(g1[key], g2[key])
            else:
                c1[key], c2[key] = self._uniform_discrete_pair(g1[key], g2[key])
        return c1, c2

    def _sbx_pair(self, a: float, b: float) -> tuple[float, float]:
        eta = self.config.sbx_eta
        u = self.rng.random()
        if u <= 0.5:
            beta = (2 * u) ** (1 / (eta + 1))
        else:
            beta = (2 - 2 * u) ** (-1 / (eta + 1))
        v1 = 0.5 * ((1 + beta) * a + (1 - beta) * b)
        v2 = 0.5 * ((1 - beta) * a + (1 + beta) * b)
        return max(0.0, min(1.0, v1)), max(0.0, min(1.0, v2))

    def _uniform_discrete_pair(self, a: float, b: float) -> tuple[float, float]:
        if self.rng.random() < 0.5:
            return a, b
        return b, a

    def _mutate(self, genes: dict[str, float]) -> dict[str, float]:
        out = dict(genes)
        for key, val in out.items():
            if self.rng.random() > self.config.mutation_rate:
                continue
            if self.space.param_type(key) == "continuous":
                out[key] = self._polynomial_mutate(val)
            else:
                out[key] = self._discrete_reset_mutate(key, val)
        return out

    def _polynomial_mutate(self, val: float) -> float:
        eta = self.config.poly_mutation_eta
        u = self.rng.random()
        if u < 0.5:
            delta = (2 * u) ** (1 / (eta + 1)) - 1
        else:
            delta = 1 - (2 * (1 - u)) ** (1 / (eta + 1))
        return max(0.0, min(1.0, val + delta * 0.25))

    def _discrete_reset_mutate(self, key: str, val: float) -> float:
        spec = self.space.current_spec(key)
        if spec["type"] == "boolean":
            return 1.0 - val if self.rng.random() < 0.5 else val
        vals = spec.get("values", [])
        if len(vals) <= 1:
            return val
        # Random reset: pick a different index fraction.
        current_idx = int(round(val * (len(vals) - 1)))
        choices = [i for i in range(len(vals)) if i != current_idx]
        new_idx = self.rng.choice(choices)
        return new_idx / (len(vals) - 1)

    @staticmethod
    def assign_fitness(pop: list[Individual], objectives: list[ObjectiveDef]) -> None:
        for ind in pop:
            if ind.feasible:
                ind.fitness = scalar_fitness(ind, objectives)

    @staticmethod
    def get_pareto(pop: list[Individual], objectives: list[ObjectiveDef]) -> list[Individual]:
        return pareto_front(pop, objectives)
