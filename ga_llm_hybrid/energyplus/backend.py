"""Evaluation backends: arch_model (generate_20260528) and template IDF."""

from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from ga_llm_hybrid.config import HybridConfig, ObjectiveDef
from ga_llm_hybrid.core.individual import Individual
from ga_llm_hybrid.energyplus.result_parser import PENALTY, objectives_from_eplustbl, pick_objectives

logger = logging.getLogger(__name__)


class EvaluationBackend(ABC):
    """Run EnergyPlus for a batch of individuals."""

    @abstractmethod
    def evaluate_batch(
        self,
        individuals: list[Individual],
        generation: int,
        run_dir: Path,
    ) -> list[Individual]:
        """Evaluate objectives for each individual; mutate in place."""


class ArchModelBackend(EvaluationBackend):
    """Use generate_20260528 + partition + direct IDF converter."""

    _EP_KEYS = frozenset(
        {
            "lights_watts_per_floor_area",
            "people_per_floor_area",
            "heating_setpoint",
            "cooling_setpoint",
        }
    )

    def __init__(self, config: HybridConfig) -> None:
        self.config = config
        self.ep_cfg = config.energyplus
        self.objectives: list[ObjectiveDef] = config.objectives
        self.fixed_geometry = dict(self.ep_cfg.fixed_geometry)

    def evaluate_batch(
        self,
        individuals: list[Individual],
        generation: int,
        run_dir: Path,
    ) -> list[Individual]:
        from scripts.ep_sim_utils import read_eplustbl, run_ep_simulation_direct
        from scripts.facade_params import (
            FACADE_GEOMETRY_KEYS,
            decode_generator_bools,
            make_ep_defaults_for_geometry,
        )
        from scripts.generate_20260528 import generate_20260528
        from scripts.zone_partition import partition_model_by_floor

        sim_root = run_dir / "sims" / self._sim_subdir(generation)
        sim_root.mkdir(parents=True, exist_ok=True)

        for ind in individuals:
            eval_id = f"ind_{ind.id:04d}"
            sim_dir = sim_root / eval_id
            sim_dir.mkdir(parents=True, exist_ok=True)
            ind.sim_dir = str(sim_dir)
            try:
                geo = {**self.fixed_geometry}
                ep_overrides: dict[str, float] = {}
                for k, v in ind.params.items():
                    if k in FACADE_GEOMETRY_KEYS:
                        geo[k] = v
                    elif k in self._EP_KEYS:
                        ep_overrides[k] = float(v)
                    elif not k.startswith("_"):
                        geo[k] = v

                geo = decode_generator_bools(geo)
                model = generate_20260528(**geo)
                if self.ep_cfg.partition_enabled:
                    model = partition_model_by_floor(
                        model,
                        perimeter_depth=self.ep_cfg.perimeter_depth,
                        lobby_height=float(geo.get("lobby_height", 6.0)),
                        floor_height=float(geo.get("floor_height", 4.0)),
                    )
                (sim_dir / "model.json").write_text(
                    json.dumps(model, indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )

                defaults = make_ep_defaults_for_geometry(geo, ep_overrides)
                self._apply_ep_overrides(defaults, ep_overrides)

                result_dir = run_ep_simulation_direct(
                    model,
                    geo.get("building_name", model.get("building_name")),
                    defaults=defaults,
                    output_base=sim_root,
                    run_id=eval_id,
                )
                if not result_dir:
                    raise RuntimeError("EnergyPlus simulation returned no result directory")

                sim = read_eplustbl(result_dir)
                full_obj = objectives_from_eplustbl(sim)
                ind.objectives = pick_objectives(full_obj, self.objectives)
                ind.feasible = (
                    bool(ind.objectives)
                    and full_obj.get("eui_mj_m2", PENALTY) < PENALTY / 2
                )
                ind.error = None
            except Exception as exc:
                logger.exception("Simulation failed ind=%s", ind.id)
                ind.feasible = False
                ind.fitness = PENALTY
                ind.error = str(exc)
                (sim_dir / "error.txt").write_text(str(exc), encoding="utf-8")
                ind.objectives = pick_objectives(
                    {
                        "eui_mj_m2": PENALTY,
                        "annual_energy_consumption": PENALTY,
                        "peak_cooling_load": PENALTY,
                        "peak_heating_load": PENALTY,
                        "thermal_discomfort_hours": PENALTY,
                    },
                    self.objectives,
                )

        return individuals

    @staticmethod
    def _sim_subdir(generation: int) -> str:
        if generation < 0:
            return "morris"
        return f"gen_{generation:02d}"

    @staticmethod
    def _apply_ep_overrides(defaults: Any, overrides: dict[str, float]) -> None:
        paths = {
            "lights_watts_per_floor_area": ("lights", "watts_per_floor_area"),
            "people_per_floor_area": ("people", "people_per_floor_area"),
            "window_wwr": ("window", "wwr"),
            "heating_setpoint": ("hvac", "heating_setpoint"),
            "cooling_setpoint": ("hvac", "cooling_setpoint"),
        }
        for key, val in overrides.items():
            if key not in paths:
                continue
            obj = defaults
            *parents, attr = paths[key]
            for p in parents:
                obj = getattr(obj, p)
            setattr(obj, attr, val)


def create_backend(config: HybridConfig) -> EvaluationBackend:
    if config.energyplus.backend == "template_idf":
        from ga_llm_hybrid.energyplus.template_backend import TemplateIdfBackend

        return TemplateIdfBackend(config)
    return ArchModelBackend(config)
