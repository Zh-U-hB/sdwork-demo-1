"""Persist round outputs for reproducibility."""

from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path
from typing import Any

from ga_llm_hybrid.core.individual import Individual


class DataRecorder:
    """Write population, Pareto front, and summaries per round."""

    def __init__(self, output_root: Path) -> None:
        self.root = output_root
        self.root.mkdir(parents=True, exist_ok=True)

    def round_dir(self, round_num: int) -> Path:
        d = self.root / f"round_{round_num:02d}"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def save_config_copy(self, config_path: Path) -> None:
        dest = self.root / "config.yaml"
        if config_path.exists():
            shutil.copy2(config_path, dest)

    def write_population_csv(self, round_dir: Path, population: list[Individual], generation: int) -> Path:
        path = round_dir / "population.csv"
        if not population:
            return path
        param_names: list[str] = []
        obj_names: list[str] = []
        for ind in population:
            for k in ind.params:
                if k not in param_names:
                    param_names.append(k)
            for k in ind.objectives:
                if k not in obj_names:
                    obj_names.append(k)
        fieldnames = ["generation", "individual_id", *param_names, *obj_names, "fitness", "feasible"]
        write_header = not path.exists()
        with path.open("a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            if write_header:
                w.writeheader()
            for ind in population:
                row: dict[str, Any] = {
                    "generation": generation,
                    "individual_id": ind.id,
                    "fitness": ind.fitness,
                    "feasible": ind.feasible,
                }
                row.update(ind.params)
                row.update(ind.objectives)
                w.writerow(row)
        return path

    def write_pareto_csv(self, round_dir: Path, pareto: list[Individual]) -> Path:
        path = round_dir / "pareto_front.csv"
        if not pareto:
            path.write_text("", encoding="utf-8")
            return path
        param_names = list(pareto[0].params.keys())
        obj_names = list(pareto[0].objectives.keys())
        fieldnames = ["individual_id", *param_names, *obj_names, "fitness"]
        with path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            w.writeheader()
            for ind in pareto:
                row: dict[str, Any] = {"individual_id": ind.id, "fitness": ind.fitness}
                row.update(ind.params)
                row.update(ind.objectives)
                w.writerow(row)
        return path

    def write_summary(
        self,
        round_dir: Path,
        round_num: int,
        pareto: list[Individual],
        extra: dict[str, Any] | None = None,
    ) -> None:
        summary = {
            "round": round_num,
            "pareto_count": len(pareto),
            "best_fitness": min((p.fitness or 1e9) for p in pareto) if pareto else None,
        }
        if extra:
            summary.update(extra)
        (round_dir / "summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def write_morris(self, round_dir: Path, morris: dict[str, Any]) -> None:
        (round_dir / "morris_sensitivity.json").write_text(
            json.dumps(morris, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def write_final_report(self, report: dict[str, Any]) -> None:
        (self.root / "final_report.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def write_parameter_space(self, specs: dict[str, Any]) -> None:
        (self.root / "parameter_space.json").write_text(
            json.dumps(specs, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
