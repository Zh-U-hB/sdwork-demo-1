"""Validate and sanitize LLM analysis JSON."""

from __future__ import annotations

import copy
import logging
from typing import Any

from ga_llm_hybrid.core.parameter_space import ParameterSpace

logger = logging.getLogger(__name__)

HARD_EXPAND = 0.5
MAX_NARROW_FRAC = 0.30
MORRIS_LOW = 0.05


def validate_llm_analysis(
    analysis: dict[str, Any],
    space: ParameterSpace,
    morris: dict[str, dict[str, float]] | None = None,
) -> dict[str, Any]:
    """Apply safety rules to LLM suggestions.

    Parameters
    ----------
    analysis
        Parsed LLM JSON.
    space
        Parameter space with original/current ranges.
    morris
        Morris mu_star per parameter.

    Returns
    -------
    dict
        Sanitized analysis with confidence flags.
    """
    out = copy.deepcopy(analysis)
    morris = morris or {}
    high = out.get("important_parameters", {}).get("high_impact", [])
    narrow = out.get("exploration_guidance", {}).get("narrow_range", [])

    validated_high: list[dict[str, Any]] = []
    for item in high:
        name = item.get("name")
        if not name or name not in space.names:
            continue
        orig = space.original_spec(name)
        rec = item.get("recommended_range")
        conf = item.get("confidence", "medium")
        sens_llm = float(item.get("sensitivity", 0))
        sens_morris = _morris_max(morris.get(name, {}))

        if morris and sens_morris < MORRIS_LOW and sens_llm > 0.2:
            item["confidence"] = "low"
            conf = "low"
        if morris and sens_morris > 0.2 and sens_llm < MORRIS_LOW:
            item["confidence"] = "low"
            conf = "low"

        if conf == "low":
            logger.info("Rejecting low-confidence LLM range for %s", name)
            continue

        if orig["type"] == "continuous" and rec and len(rec) >= 2:
            lo, hi = _clamp_range(
                float(rec[0]),
                float(rec[1]),
                float(orig["min"]),
                float(orig["max"]),
            )
            lo, hi = _limit_narrow(
                lo, hi, float(space.current_spec(name)["min"]), float(space.current_spec(name)["max"])
            )
            lo, hi = _clamp_range(lo, hi, float(orig["min"]), float(orig["max"]))
            item["recommended_range"] = [lo, hi]
        validated_high.append(item)

    out.setdefault("important_parameters", {})["high_impact"] = validated_high

    validated_narrow: list[dict[str, Any]] = []
    for item in narrow:
        name = item.get("param")
        if not name or name not in space.names:
            continue
        orig = space.original_spec(name)
        if orig["type"] != "continuous":
            continue
        conf = item.get("confidence", "medium")
        if conf == "low":
            continue
        lo = float(item.get("new_min", orig["min"]))
        hi = float(item.get("new_max", orig["max"]))
        lo, hi = _clamp_range(lo, hi, float(orig["min"]), float(orig["max"]))
        cur = space.current_spec(name)
        lo, hi = _limit_narrow(lo, hi, float(cur["min"]), float(cur["max"]))
        item["new_min"], item["new_max"] = lo, hi
        validated_narrow.append(item)

    out.setdefault("exploration_guidance", {})["narrow_range"] = validated_narrow
    return out


def apply_range_updates(
    space: ParameterSpace,
    validated: dict[str, Any],
) -> None:
    """Apply validated LLM suggestions to parameter space (mixed update)."""
    for item in validated.get("important_parameters", {}).get("high_impact", []):
        name = item.get("name")
        rec = item.get("recommended_range")
        conf = item.get("confidence", "medium")
        if not name or not rec:
            continue
        cur = space.current_spec(name)
        if cur["type"] != "continuous":
            continue
        w_llm = 0.7 if conf == "high" else 0.4
        w_cur = 1.0 - w_llm
        new_min = w_cur * cur["min"] + w_llm * float(rec[0])
        new_max = w_cur * cur["max"] + w_llm * float(rec[1])
        space.update_range(name, new_min=new_min, new_max=new_max)

    for item in validated.get("exploration_guidance", {}).get("narrow_range", []):
        name = item.get("param")
        if not name:
            continue
        space.update_range(name, new_min=item.get("new_min"), new_max=item.get("new_max"))

    for item in validated.get("important_parameters", {}).get("new_parameters", []):
        pname = item.get("name")
        ptype = item.get("type", "boolean")
        if pname and pname not in space.names:
            if ptype == "boolean":
                space.add_parameter(pname, "boolean")
            elif ptype == "continuous":
                space.add_parameter(
                    pname,
                    "continuous",
                    range=item.get("range", [0.0, 1.0]),
                )


def _morris_max(sens: dict[str, float]) -> float:
    return max(sens.values()) if sens else 0.0


def _clamp_range(
    lo: float, hi: float, orig_min: float, orig_max: float
) -> tuple[float, float]:
    span = orig_max - orig_min
    hard_lo = orig_min - HARD_EXPAND * span
    hard_hi = orig_max + HARD_EXPAND * span
    lo = max(hard_lo, min(lo, hard_hi))
    hi = max(hard_lo, min(hi, hard_hi))
    if lo > hi:
        lo, hi = hi, lo
    return lo, hi


def _limit_narrow(
    new_lo: float, new_hi: float, cur_lo: float, cur_hi: float
) -> tuple[float, float]:
    cur_span = cur_hi - cur_lo
    if cur_span <= 0:
        return new_lo, new_hi
    new_span = new_hi - new_lo
    min_span = cur_span * (1.0 - MAX_NARROW_FRAC)
    if new_span < min_span:
        mid = (new_lo + new_hi) / 2
        half = min_span / 2
        new_lo, new_hi = mid - half, mid + half
    return new_lo, new_hi
