"""Main orchestrator: GA rounds + LLM guidance loop."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

from ga_llm_hybrid.config import HybridConfig, load_config
from ga_llm_hybrid.core.ga_engine import GAEngine
from ga_llm_hybrid.core.individual import Individual
from ga_llm_hybrid.core.parameter_space import ParameterSpace
from ga_llm_hybrid.energyplus.backend import create_backend
from ga_llm_hybrid.io.data_recorder import DataRecorder
from ga_llm_hybrid.io.report_generator import build_final_report
from ga_llm_hybrid.llm.analyzer import LLMAnalyzer
from ga_llm_hybrid.llm.validator import apply_range_updates, validate_llm_analysis
from ga_llm_hybrid.sensitivity.morris import morris_screening

logger = logging.getLogger(__name__)


class HybridOrchestrator:
    """Coordinate GA search, simulation, Morris, and LLM analysis rounds."""

    def __init__(self, config: HybridConfig, output_dir: str | Path, config_path: Path | None = None) -> None:
        self.config = config
        self.config_path = config_path
        if config.project.debug_mode:
            config.ga.population_size = min(config.ga.population_size, 5)
            config.ga.max_generations = min(config.ga.max_generations, 3)
            config.project.max_rounds = min(config.project.max_rounds, 1)

        self.space = ParameterSpace(config.parameters, seed=config.project.seed)
        self.recorder = DataRecorder(Path(output_dir))
        self.backend = create_backend(config)
        self._pareto_history: list[list[Individual]] = []
        self._llm_assessments: list[dict[str, Any]] = []
        self._seed_params: list[dict[str, Any]] = []

    def run(self) -> dict[str, Any]:
        """Execute full hybrid optimization loop."""
        if self.config_path:
            self.recorder.save_config_copy(self.config_path)

        max_rounds = self.config.project.max_rounds
        llm_freq = self.config.llm.analysis_frequency

        for rnd in range(1, max_rounds + 1):
            logger.info("=== Hybrid round %d / %d ===", rnd, max_rounds)
            round_dir = self.recorder.round_dir(rnd)
            t0 = time.time()

            final_pop, snapshots = self._run_ga_round(round_dir, rnd)
            for gen_idx, snap in enumerate(snapshots):
                self.recorder.write_population_csv(round_dir, snap, gen_idx)

            pareto = GAEngine.get_pareto(final_pop, self.config.objectives)
            self.recorder.write_pareto_csv(round_dir, pareto)
            self._pareto_history.append(pareto)

            morris: dict[str, dict[str, float]] = {}
            run_morris = (
                self.config.project.morris_enabled
                and not self.config.project.debug_mode
                and (rnd == 1 or (self.config.llm.enabled and rnd % llm_freq == 0))
            )
            if run_morris:
                morris = self._run_morris(round_dir)

            llm_result: dict[str, Any] = {}
            if self.config.llm.enabled and rnd % llm_freq == 0:
                ctx = {**self.config.building_context, "objectives": self._objectives_text()}
                analyzer = LLMAnalyzer(self.space, building_context=ctx)
                llm_raw = analyzer.analyze(final_pop, pareto, morris, round_dir)
                llm_result = validate_llm_analysis(llm_raw, self.space, morris)
                apply_range_updates(self.space, llm_result)
                self._llm_assessments.append(llm_result)
                self._seed_params = self._extract_seeds(llm_result, pareto)

            self.recorder.write_summary(
                round_dir,
                rnd,
                pareto,
                extra={"elapsed_s": time.time() - t0, "llm_applied": bool(llm_result)},
            )
            self.recorder.write_parameter_space(self.space.all_current_specs())

            if self._check_convergence(rnd):
                logger.info("Convergence reached at round %d", rnd)
                break

        report = build_final_report(
            len(self._pareto_history),
            self._pareto_history[-1] if self._pareto_history else [],
            self._llm_assessments,
        )
        self.recorder.write_final_report(report)
        return report

    def _run_ga_round(self, round_dir: Path, round_num: int) -> tuple[list[Individual], list[list[Individual]]]:
        objectives = self.config.objectives

        def evaluate_fn(pop: list[Individual], generation: int) -> list[Individual]:
            evaluated = self.backend.evaluate_batch(pop, generation, round_dir)
            GAEngine.assign_fitness(evaluated, objectives)
            return evaluated

        seeds = list(self._seed_params) if self._seed_params else None
        engine = GAEngine(
            self.space,
            self.config.ga,
            objectives,
            evaluate_fn,
            seed=self.config.project.seed + round_num,
        )
        pop, snapshots = engine.run(seeds=seeds)
        return pop, snapshots

    def _run_morris(self, round_dir: Path) -> dict[str, dict[str, float]]:
        primary = self.config.objectives[0].name if self.config.objectives else "eui_mj_m2"
        traj_id = 0

        def eval_one(params: dict[str, Any]) -> dict[str, float]:
            nonlocal traj_id
            traj_id += 1
            ind = Individual(
                id=traj_id,
                generation=0,
                genes=self.space.encode(params),
                params=params,
            )
            res = self.backend.evaluate_batch([ind], -1, round_dir)
            return res[0].objectives

        morris = morris_screening(
            self.space,
            eval_one,
            n_trajectories=self.config.project.morris_trajectories,
            seed=self.config.project.seed,
            primary_objective=primary,
        )
        self.recorder.write_morris(round_dir, morris)
        return morris

    def _check_convergence(self, round_num: int) -> bool:
        if round_num < self.config.project.min_rounds:
            return False
        if len(self._pareto_history) < 3:
            return False
        recent = self._pareto_history[-3:]
        bests = [min(p.fitness or 1e9 for p in front) for front in recent if front]
        if len(bests) < 3:
            return False
        changes = [abs(bests[i] - bests[i - 1]) / max(abs(bests[i - 1]), 1e-9) for i in range(1, 3)]
        if all(c < 0.02 for c in changes):
            return True
        if len(self._llm_assessments) >= 2:
            last_two = self._llm_assessments[-2:]
            low_potential = all(
                a.get("convergence_assessment", {}).get("remaining_potential") == "low"
                for a in last_two
            )
            no_dirs = all(
                not a.get("exploration_guidance", {}).get("new_directions")
                for a in last_two
            )
            if low_potential and no_dirs:
                return True
        return False

    def _extract_seeds(
        self, llm_result: dict[str, Any], pareto: list[Individual]
    ) -> list[dict[str, Any]]:
        seeds: list[dict[str, Any]] = []
        for item in llm_result.get("seed_solutions", []):
            partial = {k: v for k, v in item.items() if k != "reason" and k in self.space.names}
            if partial:
                seeds.append(self.space.complete_params(partial))
        for p in pareto[:5]:
            seeds.append(self.space.complete_params(dict(p.params)))
        return seeds

    def _objectives_text(self) -> str:
        parts = []
        for o in self.config.objectives:
            parts.append(f"{o.name} ({o.direction})")
        return ", ".join(parts)


def run_optimization(config_path: str | Path, output_dir: str | Path = "output/ga_llm_hybrid") -> dict[str, Any]:
    """Entry point: load config and run hybrid optimization."""
    path = Path(config_path)
    config = load_config(path)
    orch = HybridOrchestrator(config, output_dir, config_path=path)
    return orch.run()
