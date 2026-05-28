"""Parse EnergyPlus output files into objective metrics."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

PENALTY = 1e6

# Map config objective names to internal keys produced here.
_OBJECTIVE_ALIASES: dict[str, str] = {
    "eui_mj_m2": "eui_mj_m2",
    "annual_energy_consumption": "eui_mj_m2",
    "peak_cooling_load": "peak_cooling_load",
    "peak_heating_load": "peak_heating_load",
    "thermal_discomfort_hours": "thermal_discomfort_hours",
    "daylighting_hours": "daylighting_hours",
}


def parse_simulation_outputs(sim_dir: Path) -> dict[str, float]:
    """Extract objectives from eplustbl.csv under *sim_dir*."""
    from scripts.ep_sim_utils import read_eplustbl

    tbl = sim_dir / "eplustbl.csv"
    if not tbl.exists():
        for sub in sim_dir.rglob("eplustbl.csv"):
            tbl = sub.parent
            break
        else:
            return _penalty_objectives()
    return objectives_from_eplustbl(read_eplustbl(str(sim_dir)))


def objectives_from_eplustbl(sim: dict[str, Any]) -> dict[str, float]:
    """Build objective dict from :func:`scripts.ep_sim_utils.read_eplustbl` result."""
    if not sim.get("exists"):
        return _penalty_objectives()

    area = sim.get("building_area", {}).get("Net Conditioned Building Area", 0.0)
    total_gj = sim.get("site_energy", {}).get("Total Site Energy", 0.0)
    if area <= 0 or total_gj <= 0:
        return _penalty_objectives()

    eui_mj_m2 = float(total_gj * 1000.0 / area)

    peak_cooling = 0.0
    peak_heating = 0.0
    for item in sim.get("demand_end_uses", []):
        label = item.get("end_use", "")
        demand = float(item.get("demand_w", 0.0))
        if label == "Cooling":
            peak_cooling = max(peak_cooling, demand)
        elif label == "Heating":
            peak_heating = max(peak_heating, demand)

    discomfort = _parse_discomfort_hours(sim.get("path"))
    daylight = _parse_daylight_hours(sim.get("path"))

    return {
        "eui_mj_m2": eui_mj_m2,
        "annual_energy_consumption": eui_mj_m2,
        "peak_cooling_load": peak_cooling,
        "peak_heating_load": peak_heating,
        "thermal_discomfort_hours": discomfort,
        "daylighting_hours": daylight,
    }


def pick_objectives(
    full: dict[str, float],
    objective_defs: list[Any] | None = None,
) -> dict[str, float]:
    """Return only objectives referenced by config (via name or output_key)."""
    if not objective_defs:
        return full
    out: dict[str, float] = {}
    for obj in objective_defs:
        key = getattr(obj, "output_key", None) or obj.name
        alias = _OBJECTIVE_ALIASES.get(obj.name, obj.name)
        val = full.get(alias, full.get(key, full.get(obj.name)))
        if val is not None:
            out[obj.name] = float(val)
    return out


def _penalty_objectives() -> dict[str, float]:
    return {
        "eui_mj_m2": PENALTY,
        "annual_energy_consumption": PENALTY,
        "peak_cooling_load": PENALTY,
        "peak_heating_load": PENALTY,
        "thermal_discomfort_hours": PENALTY,
        "daylighting_hours": PENALTY,
    }


def _parse_discomfort_hours(tbl_path: Path | str | None) -> float:
    """Scan tabular output for thermal comfort hour counts when present."""
    if not tbl_path:
        return 0.0
    path = Path(tbl_path)
    if not path.exists():
        return 0.0
    text = path.read_text(encoding="utf-8", errors="replace")
    for pattern in (
        r"Time Setpoint Not Met.*?During Occupied Heating\s*,\s*([\d.]+)",
        r"Time Setpoint Not Met.*?During Occupied Cooling\s*,\s*([\d.]+)",
        r"Fanger.*?PMV.*?hours\s*,\s*([\d.]+)",
    ):
        m = re.search(pattern, text, re.IGNORECASE)
        if m:
            try:
                return float(m.group(1))
            except ValueError:
                continue
    # Sum heating + cooling setpoint not met as discomfort proxy (hours).
    heat = re.search(
        r"Time Setpoint Not Met During Occupied Heating\s*,\s*([\d.]+)",
        text,
        re.IGNORECASE,
    )
    cool = re.search(
        r"Time Setpoint Not Met During Occupied Cooling\s*,\s*([\d.]+)",
        text,
        re.IGNORECASE,
    )
    total = 0.0
    if heat:
        total += float(heat.group(1))
    if cool:
        total += float(cool.group(1))
    return total


def _parse_daylight_hours(tbl_path: Path | str | None) -> float:
    if not tbl_path:
        return 0.0
    path = Path(tbl_path)
    if not path.exists():
        return 0.0
    text = path.read_text(encoding="utf-8", errors="replace")
    m = re.search(r"Daylighting.*?hours\s*,\s*([\d.]+)", text, re.IGNORECASE)
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            pass
    return 0.0
