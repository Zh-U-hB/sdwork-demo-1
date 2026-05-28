"""Parallel EnergyPlus simulation runner with timeout and retry."""

from __future__ import annotations

import logging
import os
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from ga_llm_hybrid.core.individual import Individual

logger = logging.getLogger(__name__)


def _run_one(
    idf_path: str,
    epw_path: str | None,
    exe: str,
    out_dir: str,
    timeout: int,
) -> dict[str, Any]:
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    cmd = [exe, "-w", epw_path, "-d", out_dir, idf_path] if epw_path else [exe, "-d", out_dir, idf_path]
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        ok = proc.returncode == 0
        return {
            "success": ok,
            "output_dir": out_dir,
            "stdout": proc.stdout[-2000:] if proc.stdout else "",
            "stderr": proc.stderr[-2000:] if proc.stderr else "",
        }
    except subprocess.TimeoutExpired:
        return {"success": False, "error": "timeout", "output_dir": out_dir}
    except Exception as exc:
        return {"success": False, "error": str(exc), "output_dir": out_dir}


class SimulationRunner:
    """Batch EnergyPlus runs with process pool."""

    def __init__(
        self,
        executable: str | None = None,
        epw_path: str | None = None,
        parallel_jobs: int | None = None,
        timeout: int = 300,
        retries: int = 1,
    ) -> None:
        self.executable = executable or os.environ.get(
            "ENERGYPLUS_EXE", "/usr/local/EnergyPlus-25-1-0/energyplus"
        )
        self.epw_path = epw_path
        self.parallel_jobs = parallel_jobs or max(1, (os.cpu_count() or 2) - 1)
        self.timeout = timeout
        self.retries = retries

    def run_batch(
        self, jobs: list[tuple[Individual, Path]]
    ) -> list[tuple[Individual, dict[str, Any]]]:
        results: list[tuple[Individual, dict[str, Any]]] = []
        with ProcessPoolExecutor(max_workers=self.parallel_jobs) as pool:
            futures = {}
            for ind, idf in jobs:
                out_dir = str(idf.parent / f"out_{ind.id:04d}")
                fut = pool.submit(
                    _run_one,
                    str(idf),
                    self.epw_path,
                    self.executable,
                    out_dir,
                    self.timeout,
                )
                futures[fut] = ind
            for fut in as_completed(futures):
                ind = futures[fut]
                res = fut.result()
                if not res.get("success") and self.retries > 0:
                    idf_for_ind = next((str(idf) for i, idf in jobs if i.id == ind.id), "")
                    res = _run_one(
                        idf_for_ind,
                        self.epw_path,
                        self.executable,
                        res.get("output_dir", ""),
                        self.timeout,
                    )
                results.append((ind, res))
        return results
