"""Parameter space definition, encoding, and range management."""

from __future__ import annotations

import copy
import random
from typing import Any

import numpy as np

from ga_llm_hybrid.config import ParamDef


class ParameterSpace:
    """Manages parameter definitions and current search ranges."""

    def __init__(self, definitions: list[ParamDef], seed: int = 42) -> None:
        self._defs = {d.name: d for d in definitions}
        self._original: dict[str, dict[str, Any]] = {}
        self._current: dict[str, dict[str, Any]] = {}
        self._rng = random.Random(seed)
        self._np_rng = np.random.default_rng(seed)
        for name, d in self._defs.items():
            spec = self._spec_from_def(d)
            self._original[name] = copy.deepcopy(spec)
            self._current[name] = copy.deepcopy(spec)

    @staticmethod
    def _spec_from_def(d: ParamDef) -> dict[str, Any]:
        if d.type == "continuous":
            lo, hi = d.range or [0.0, 1.0]
            return {"type": "continuous", "min": float(lo), "max": float(hi), "unit": d.unit}
        if d.type == "boolean":
            return {"type": "boolean"}
        vals = list(d.values or [])
        return {"type": d.type, "values": vals}

    @property
    def names(self) -> list[str]:
        return list(self._defs.keys())

    def param_type(self, name: str) -> str:
        """Return ``continuous`` or ``discrete`` (categorical / ordinal / boolean)."""
        spec = self._current.get(name, {})
        t = spec.get("type", "continuous")
        return "continuous" if t == "continuous" else "discrete"

    def definitions(self) -> list[ParamDef]:
        return list(self._defs.values())

    def original_spec(self, name: str) -> dict[str, Any]:
        return copy.deepcopy(self._original[name])

    def current_spec(self, name: str) -> dict[str, Any]:
        return copy.deepcopy(self._current[name])

    def all_current_specs(self) -> dict[str, dict[str, Any]]:
        return copy.deepcopy(self._current)

    def update_range(
        self,
        name: str,
        new_min: float | None = None,
        new_max: float | None = None,
        new_values: list[str] | None = None,
    ) -> None:
        if name not in self._current:
            return
        spec = self._current[name]
        if spec["type"] == "continuous":
            if new_min is not None:
                spec["min"] = float(new_min)
            if new_max is not None:
                spec["max"] = float(new_max)
        elif new_values is not None:
            spec["values"] = list(new_values)

    def add_parameter(self, name: str, ptype: str, **kwargs: Any) -> None:
        d = ParamDef(name=name, type=ptype, **kwargs)  # type: ignore[arg-type]
        self._defs[name] = d
        spec = self._spec_from_def(d)
        self._original[name] = copy.deepcopy(spec)
        self._current[name] = copy.deepcopy(spec)

    def sample_random(self, use_original: bool = False) -> dict[str, Any]:
        """Sample one random individual in decoded form."""
        out: dict[str, Any] = {}
        specs = self._original if use_original else self._current
        for name, spec in specs.items():
            out[name] = self._sample_one(name, spec)
        return out

    def _sample_one(self, name: str, spec: dict[str, Any]) -> Any:
        t = spec["type"]
        if t == "continuous":
            return float(self._rng.uniform(spec["min"], spec["max"]))
        if t == "boolean":
            return bool(self._rng.getrandbits(1))
        idx = self._rng.randrange(len(spec["values"]))
        return spec["values"][idx] if t == "categorical" else idx

    def latin_hypercube_population(self, n: int, use_original: bool = False) -> list[dict[str, Any]]:
        """Latin Hypercube Sampling for continuous dims; random for discrete."""
        specs = self._original if use_original else self._current
        cont_names = [n for n, s in specs.items() if s["type"] == "continuous"]
        pop: list[dict[str, Any]] = []
        if cont_names:
            dim = len(cont_names)
            lhs = np.zeros((n, dim))
            for j in range(dim):
                perm = self._np_rng.permutation(n)
                lhs[:, j] = (perm + self._np_rng.random(n)) / n
        else:
            lhs = None
        for i in range(n):
            ind: dict[str, Any] = {}
            ci = 0
            for name, spec in specs.items():
                if spec["type"] == "continuous" and lhs is not None:
                    lo, hi = spec["min"], spec["max"]
                    ind[name] = float(lo + lhs[i, ci] * (hi - lo))
                    ci += 1
                else:
                    ind[name] = self._sample_one(name, spec)
            pop.append(ind)
        return pop

    def complete_params(self, partial: dict[str, Any]) -> dict[str, Any]:
        """Fill missing keys in *partial* with mid-range / default values."""
        out = self.sample_random(use_original=False)
        out.update(partial)
        return out

    def encode(self, decoded: dict[str, Any]) -> dict[str, float]:
        """Encode decoded individual to [0,1] genes (discrete as fraction)."""
        genes: dict[str, float] = {}
        for name, spec in self._current.items():
            val = decoded.get(name)
            if spec["type"] == "continuous":
                lo, hi = spec["min"], spec["max"]
                if val is None:
                    genes[name] = 0.5
                else:
                    genes[name] = 0.5 if hi <= lo else (float(val) - lo) / (hi - lo)
            elif spec["type"] == "boolean":
                genes[name] = 0.0 if val is None else (1.0 if val else 0.0)
            else:
                vals = spec["values"]
                if isinstance(val, str) and val in vals:
                    idx = vals.index(val)
                elif isinstance(val, (int, float)):
                    idx = int(val)
                else:
                    idx = 0
                genes[name] = idx / max(len(vals) - 1, 1)
        return genes

    def decode(self, genes: dict[str, float]) -> dict[str, Any]:
        """Decode genes to parameter values."""
        out: dict[str, Any] = {}
        for name, spec in self._current.items():
            g = float(genes.get(name, 0.5))
            g = max(0.0, min(1.0, g))
            if spec["type"] == "continuous":
                lo, hi = spec["min"], spec["max"]
                out[name] = lo + g * (hi - lo)
            elif spec["type"] == "boolean":
                out[name] = g >= 0.5
            else:
                vals = spec["values"]
                idx = int(round(g * max(len(vals) - 1, 0)))
                idx = max(0, min(len(vals) - 1, idx))
                out[name] = vals[idx]
        return out

    def summary_table(self) -> str:
        lines = ["name | type | current_range | unit"]
        for name, d in self._defs.items():
            spec = self._current[name]
            if spec["type"] == "continuous":
                rng = f"{spec['min']:.4g} ~ {spec['max']:.4g}"
            else:
                rng = str(spec.get("values", ["T", "F"]))
            lines.append(f"{name} | {spec['type']} | {rng} | {d.unit}")
        return "\n".join(lines)
