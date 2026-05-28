"""Multi-objective aggregation and Pareto utilities."""

from __future__ import annotations

from ga_llm_hybrid.config import ObjectiveDef
from ga_llm_hybrid.core.individual import Individual


def scalar_fitness(ind: Individual, objectives: list[ObjectiveDef]) -> float:
    """Weighted sum of normalized objectives (lower is better)."""
    total = 0.0
    w_sum = sum(o.weight for o in objectives) or 1.0
    for obj in objectives:
        val = ind.objectives.get(obj.name)
        if val is None:
            return 1e9
        score = val if obj.direction == "minimize" else -val
        total += obj.weight * score / w_sum
    return total


def dominates(a: Individual, b: Individual, objectives: list[ObjectiveDef]) -> bool:
    """True if a Pareto-dominates b."""
    better_any = False
    for obj in objectives:
        va = a.objectives.get(obj.name)
        vb = b.objectives.get(obj.name)
        if va is None or vb is None:
            return False
        if obj.direction == "minimize":
            if va > vb:
                return False
            if va < vb:
                better_any = True
        else:
            if va < vb:
                return False
            if va > vb:
                better_any = True
    return better_any


def pareto_front(population: list[Individual], objectives: list[ObjectiveDef]) -> list[Individual]:
    """Return non-dominated feasible individuals."""
    feasible = [p for p in population if p.feasible and p.fitness is not None]
    front: list[Individual] = []
    for p in feasible:
        if not any(dominates(q, p, objectives) for q in feasible if q is not p):
            front.append(p)
    return sorted(front, key=lambda x: x.fitness or 1e9)
