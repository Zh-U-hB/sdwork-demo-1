"""Morris elementary-effects sensitivity analysis."""

from __future__ import annotations

import logging
from typing import Any, Callable

import numpy as np

from ga_llm_hybrid.core.parameter_space import ParameterSpace

logger = logging.getLogger(__name__)


def morris_screening(
    space: ParameterSpace,
    evaluate_fn: Callable[[dict[str, Any]], dict[str, float]],
    n_trajectories: int = 10,
    n_levels: int = 4,
    seed: int = 42,
    primary_objective: str = "eui_mj_m2",
) -> dict[str, dict[str, float]]:
    """Morris screening with independent elementary effects per parameter.

    Uses random trajectories on a discrete grid (standard Morris-style OAT steps),
    without chaining all dimensions into one long walk (avoids confounded paths).

    Parameters
    ----------
    space
        Parameter space (current ranges).
    evaluate_fn
        ``params -> objectives`` (one EnergyPlus eval per call).
    n_trajectories
        Number of random trajectories.
    n_levels
        Grid levels for continuous parameters.
    seed
        RNG seed.
    primary_objective
        Objective key used to compute elementary effects.

    Returns
    -------
    dict
        ``{param_name: {objective_name: mu_star}}``
    """
    rng = np.random.default_rng(seed)
    names = space.names
    if not names:
        return {}

    delta = n_levels / (2.0 * (n_levels - 1))
    effects: dict[str, list[float]] = {n: [] for n in names}

    for t in range(n_trajectories):
        base_genes = _random_grid_point(space, n_levels, rng)
        x0 = space.decode(base_genes)
        y0 = evaluate_fn(x0)
        y0_val = float(y0.get(primary_objective, y0.get("annual_energy_consumption", 0.0)))

        order = rng.permutation(len(names))
        for idx in order:
            pname = names[idx]
            x1 = _perturb_param(space, dict(x0), pname, delta)
            y1 = evaluate_fn(x1)
            y1_val = float(y1.get(primary_objective, y1.get("annual_energy_consumption", 0.0)))
            effects[pname].append(abs(y1_val - y0_val))
            x0, y0_val = x1, y1_val

        if (t + 1) % max(1, n_trajectories // 3) == 0:
            logger.info("Morris trajectory %d / %d complete", t + 1, n_trajectories)

    result: dict[str, dict[str, float]] = {}
    for pname, ees in effects.items():
        mu_star = float(np.mean(ees)) if ees else 0.0
        result[pname] = {primary_objective: mu_star}
    return result


def _random_grid_point(
    space: ParameterSpace, n_levels: int, rng: np.random.Generator
) -> dict[str, float]:
    """Sample a random point on the Morris grid in gene space [0, 1]."""
    genes: dict[str, float] = {}
    for name in space.names:
        spec = space.current_spec(name)
        if spec["type"] == "continuous":
            level = rng.integers(0, n_levels)
            genes[name] = level / max(n_levels - 1, 1)
        elif spec["type"] == "boolean":
            genes[name] = float(rng.integers(0, 2))
        else:
            n_vals = len(spec.get("values", []))
            if n_vals <= 1:
                genes[name] = 0.0
            else:
                idx = rng.integers(0, n_vals)
                genes[name] = idx / (n_vals - 1)
    return genes


def _perturb_param(
    space: ParameterSpace,
    params: dict[str, Any],
    pname: str,
    delta: float,
) -> dict[str, Any]:
    """One Morris step: increase *pname* by one grid delta."""
    out = dict(params)
    spec = space.current_spec(pname)
    if spec["type"] == "continuous":
        span = spec["max"] - spec["min"]
        out[pname] = min(spec["max"], float(params[pname]) + delta * span)
    elif spec["type"] == "boolean":
        out[pname] = not bool(params[pname])
    else:
        vals = spec["values"]
        cur = params[pname]
        idx = vals.index(cur) if isinstance(cur, str) and cur in vals else 0
        out[pname] = vals[min(idx + 1, len(vals) - 1)]
    return out
