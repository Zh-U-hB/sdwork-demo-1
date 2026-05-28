"""Tests for GA engine operators and convergence."""

from __future__ import annotations

from ga_llm_hybrid.config import GAConfig, ObjectiveDef, ParamDef
from ga_llm_hybrid.core.ga_engine import GAEngine
from ga_llm_hybrid.core.individual import Individual
from ga_llm_hybrid.core.parameter_space import ParameterSpace


def _dummy_evaluate(pop: list[Individual], generation: int) -> list[Individual]:
    for ind in pop:
        ind.objectives = {"eui_mj_m2": sum(ind.genes.values())}
        ind.feasible = True
    GAEngine.assign_fitness(pop, [ObjectiveDef(name="eui_mj_m2", weight=1.0)])
    return pop


def test_sbx_crossover_stays_in_unit_interval():
    defs = [ParamDef(name="x", type="continuous", range=[0.0, 10.0])]
    space = ParameterSpace(defs, seed=1)
    engine = GAEngine(space, GAConfig(population_size=10, max_generations=1), [], _dummy_evaluate, seed=1)
    g1 = {"x": 0.2}
    g2 = {"x": 0.8}
    c1, c2 = engine._crossover(g1, g2)
    assert 0.0 <= c1["x"] <= 1.0
    assert 0.0 <= c2["x"] <= 1.0


def test_mutation_respects_bounds():
    defs = [ParamDef(name="x", type="continuous", range=[0.0, 10.0])]
    space = ParameterSpace(defs, seed=2)
    engine = GAEngine(space, GAConfig(mutation_rate=1.0), [], _dummy_evaluate, seed=2)
    out = engine._mutate({"x": 0.5})
    assert 0.0 <= out["x"] <= 1.0


def test_tournament_selects_better():
    defs = [ParamDef(name="x", type="continuous", range=[0.0, 1.0])]
    space = ParameterSpace(defs, seed=3)
    engine = GAEngine(space, GAConfig(), [], _dummy_evaluate, seed=3)
    good = Individual(id=1, generation=0, genes={"x": 0.1}, fitness=0.1, feasible=True)
    bad = Individual(id=2, generation=0, genes={"x": 0.9}, fitness=0.9, feasible=True)
    pop = [good, bad] * 5
    picked = [engine._tournament_select(pop) for _ in range(20)]
    assert sum(1 for p in picked if p.id == 1) > sum(1 for p in picked if p.id == 2)


def test_discrete_crossover_is_uniform_swap():
    defs = [
        ParamDef(name="x", type="continuous", range=[0.0, 1.0]),
        ParamDef(name="cat", type="categorical", values=["a", "b", "c"]),
    ]
    space = ParameterSpace(defs, seed=10)
    engine = GAEngine(space, GAConfig(crossover_rate=1.0), [], _dummy_evaluate, seed=10)
    for _ in range(30):
        c1, c2 = engine._crossover({"x": 0.1, "cat": 0.0}, {"x": 0.9, "cat": 1.0})
        assert c1["cat"] in (0.0, 1.0)
        assert c2["cat"] in (0.0, 1.0)


def test_ga_run_converges_in_few_generations():
    defs = [
        ParamDef(name="a", type="continuous", range=[0.0, 1.0]),
        ParamDef(name="b", type="categorical", values=["x", "y"]),
    ]
    space = ParameterSpace(defs, seed=4)

    def evaluate(pop: list[Individual], generation: int) -> list[Individual]:
        for ind in pop:
            ind.objectives = {"eui_mj_m2": ind.genes.get("a", 0.5)}
            ind.feasible = True
        GAEngine.assign_fitness(pop, [ObjectiveDef(name="eui_mj_m2")])
        return pop

    cfg = GAConfig(
        population_size=12,
        max_generations=20,
        convergence_generations=3,
        convergence_threshold= 0.05,
    )
    engine = GAEngine(space, cfg, [ObjectiveDef(name="eui_mj_m2")], evaluate, seed=4)
    pop, snaps = engine.run()
    assert len(snaps) >= 2
    assert min(p.fitness or 1e9 for p in pop) < 0.3
