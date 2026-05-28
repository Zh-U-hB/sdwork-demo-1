"""Template IDF backend with parallel EnergyPlus runs."""

from __future__ import annotations

import logging
import os
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from ga_llm_hybrid.config import HybridConfig
from ga_llm_hybrid.core.individual import Individual
from ga_llm_hybrid.energyplus.backend import PENALTY, EvaluationBackend
from ga_llm_hybrid.energyplus.idf_generator import apply_mappings
from ga_llm_hybrid.energyplus.result_parser import parse_simulation_outputs, pick_objectives

logger = logging.getLogger(__name__)


def _run_one_sim(
    idf_path: str,
    epw_path: str,
    out_dir: str,
    executable: str,
    timeout: int,
) -> dict[str, float]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cmd = [executable, "-w", epw_path, "-d", str(out), idf_path]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=timeout)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        raise RuntimeError(str(exc)) from exc
    return parse_simulation_outputs(out)


class TemplateIdfBackend(EvaluationBackend):
    """Evaluate individuals via template IDF + parallel EnergyPlus."""

    def __init__(self, config: HybridConfig) -> None:
        self.cfg = config
        self.objectives = config.objectives
        ep = config.energyplus
        self.template = Path(ep.template_idf or "")
        self.epw = Path(ep.epw_path or "")
        self.executable = ep.executable_path or os.environ.get(
            "ENERGYPLUS_EXE", "/usr/local/EnergyPlus-25-1-0/energyplus"
        )
        self.parallel = ep.parallel_jobs
        self.timeout = ep.simulation_timeout
        self.mappings = config.mappings

    def evaluate_batch(
        self,
        individuals: list[Individual],
        generation: int,
        run_dir: Path,
    ) -> list[Individual]:
        if not self.template.exists():
            raise FileNotFoundError(f"Template IDF not found: {self.template}")

        sub = "morris" if generation < 0 else f"gen_{generation:02d}"
        sim_root = run_dir / "sims" / sub
        idf_dir = run_dir / "idf" / sub
        sim_root.mkdir(parents=True, exist_ok=True)
        idf_dir.mkdir(parents=True, exist_ok=True)

        jobs: list[tuple[Individual, Path, Path]] = []
        for ind in individuals:
            idf_path = idf_dir / f"ind_{ind.id:04d}.idf"
            apply_mappings(self.template, ind.params, self.mappings, idf_path)
            sim_dir = sim_root / f"ind_{ind.id:04d}"
            ind.sim_dir = str(sim_dir)
            jobs.append((ind, idf_path, sim_dir))

        with ProcessPoolExecutor(max_workers=self.parallel) as pool:
            futures = {}
            for ind, idf_path, sim_dir in jobs:
                fut = pool.submit(
                    _run_with_retry,
                    str(idf_path),
                    str(self.epw),
                    str(sim_dir),
                    self.executable,
                    self.timeout,
                )
                futures[fut] = ind

            for fut in as_completed(futures):
                ind = futures[fut]
                try:
                    ind.objectives = pick_objectives(fut.result(), self.objectives)
                    ind.feasible = True
                    ind.error = None
                except Exception as exc:
                    logger.warning("Sim failed ind=%s: %s", ind.id, exc)
                    ind.feasible = False
                    ind.fitness = PENALTY
                    ind.error = str(exc)
                    ind.objectives = {"annual_energy_consumption": PENALTY}
                    if ind.sim_dir:
                        Path(ind.sim_dir, "error.txt").write_text(str(exc), encoding="utf-8")

        return individuals


def _run_with_retry(
    idf_path: str,
    epw_path: str,
    out_dir: str,
    executable: str,
    timeout: int,
) -> dict[str, float]:
    last_err: Exception | None = None
    for _ in range(2):
        try:
            return _run_one_sim(idf_path, epw_path, out_dir, executable, timeout)
        except Exception as exc:
            last_err = exc
    raise RuntimeError(str(last_err))
