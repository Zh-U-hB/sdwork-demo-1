"""Shared EnergyPlus simulation and result-parsing utilities.

Used by parametric_l_app.py and ga_optimizer_app.py.

Two simulation paths are available:

* **Direct path** (:func:`run_ep_simulation_direct`) — programmatic
  JSON → IDF conversion via :mod:`scripts.idf_converter`, then a direct
  ``energyplus`` subprocess call.  No LLM, no MCP.  Fast and deterministic.

* **MCP path** (:func:`run_ep_simulation`) — original path via the
  EnergyPlus-Agent MCP server (LLM-driven IDF construction).  Kept for
  backward compatibility.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import csv
import os
import time
from pathlib import Path

from langchain_core.messages import HumanMessage

from src.agent.llm import create_llm
from src.agent.nodes.energyplus import (
    ENERGYPLUS_SYSTEM_PROMPT as EP_SYSTEM_PROMPT,
    _build_task_prompt as _build_ep_task_prompt,
    _get_mcp_server_config as _get_ep_mcp_config,
    _get_plugin_path as _get_ep_plugin_path,
)
from src.models.zone import BuildingModel, Dimensions, Point3D, Zone

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MASS_HEIGHT_THRESHOLD = 1.0

# ---------------------------------------------------------------------------
# Model conversion
# ---------------------------------------------------------------------------


def convert_model_to_building_model(model_dict: dict, building_name: str) -> BuildingModel:
    zones = []
    for z in model_dict["zones"]:
        if z["dimensions"]["height"] < MASS_HEIGHT_THRESHOLD:
            continue
        zones.append(Zone(
            name=z["name"],
            origin=Point3D(**z["origin"]),
            dimensions=Dimensions(**z["dimensions"]),
        ))
    if not zones:
        raise ValueError("模型中没有有效的建筑体块（高度 < 1m 的被跳过）。")
    return BuildingModel(building_name=building_name, zones=zones)


# ---------------------------------------------------------------------------
# Weather path
# ---------------------------------------------------------------------------


def resolve_weather_path() -> str:
    plugin_path = _get_ep_plugin_path()
    weather = os.getenv("ENERGYPLUS_WEATHER_FILE", "data/weather/Shenzhen.epw")
    if plugin_path and not Path(weather).is_absolute():
        weather = str(plugin_path / weather)
    return weather


# ---------------------------------------------------------------------------
# Simulation runner
# ---------------------------------------------------------------------------


async def async_run_ep_simulation(model_dict: dict, building_name: str) -> str | None:
    """Run EnergyPlus simulation via MCP and return the eplustbl.csv directory."""
    from langchain_mcp_adapters.client import MultiServerMCPClient
    from langchain_mcp_adapters.tools import load_mcp_tools
    from langgraph.prebuilt import create_react_agent

    building_model = convert_model_to_building_model(model_dict, building_name)

    server_config = _get_ep_mcp_config()
    if server_config is None:
        raise RuntimeError(
            "EnergyPlus Agent 插件未找到。\n"
            "请确认 plugins/energyplus_agent/ 存在并已安装依赖。"
        )

    weather_file = resolve_weather_path()

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    abs_output_dir = Path.cwd() / "output" / f"ep_sim_{timestamp}"
    idf_output_path = str(abs_output_dir / "building.idf")

    task = _build_ep_task_prompt(building_model, idf_output_path, weather_file)
    llm = create_llm()

    client = MultiServerMCPClient(server_config)
    async with client.session("energyplus") as session:
        tools = await load_mcp_tools(session)
        agent = create_react_agent(
            model=llm,
            tools=tools,
            prompt=EP_SYSTEM_PROMPT,
        )
        await agent.ainvoke({"messages": [HumanMessage(content=task)]})

    csv_files = list(abs_output_dir.rglob("eplustbl.csv"))
    if csv_files:
        return str(csv_files[0].parent)

    plugin_path = _get_ep_plugin_path()
    if plugin_path:
        plugin_output = plugin_path / "output"
        if plugin_output.exists():
            csv_files = sorted(
                plugin_output.rglob("eplustbl.csv"),
                key=lambda f: f.stat().st_mtime,
                reverse=True,
            )
            if csv_files:
                return str(csv_files[0].parent)

    return None


def run_ep_simulation(model_dict: dict, building_name: str) -> str | None:
    """Synchronous wrapper around the MCP path — safe to call from Streamlit."""
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(
            asyncio.run,
            async_run_ep_simulation(model_dict, building_name),
        )
        return future.result()


# ---------------------------------------------------------------------------
# Direct path (no LLM / no MCP)
# ---------------------------------------------------------------------------


def run_ep_simulation_direct(
    model_dict: dict,
    building_name: str | None = None,
    defaults=None,
    output_base: str | Path = "output/direct_energyplus",
    weather_file: str | Path | None = None,
) -> str | None:
    """Convert *model_dict* to IDF programmatically and run EnergyPlus directly.

    This is the fast, deterministic alternative to :func:`run_ep_simulation`.
    It calls :func:`scripts.idf_converter.convert_and_run` which:

    1. Sanitises zone names to ASCII (required by EnergyPlus / eppy latin-1).
    2. Computes 6-surface box geometry with correct vertex order.
    3. Detects and patches shared interior walls.
    4. Writes materials, constructions, schedules, HVAC and loads.
    5. Saves the IDF and executes ``energyplus`` directly.

    Parameters
    ----------
    model_dict:
        BuildingModel JSON dict (``building_name`` + ``zones``).
    building_name:
        Override the building name.  Falls back to ``model_dict["building_name"]``.
    defaults:
        :class:`scripts.idf_defaults.ConverterDefaults` instance.  Use
        :func:`scripts.idf_defaults.make_default_settings` and mutate to
        override location, materials, schedules etc.  ``None`` → built-in
        defaults (Shenzhen, concrete walls, ideal loads).
    output_base:
        Parent directory for timestamped output sub-folders.
    weather_file:
        Path to ``.epw`` weather file.  Auto-resolved when ``None``.

    Returns
    -------
    str | None
        Path to the directory containing ``eplustbl.csv``, or ``None`` on failure.
    """
    from scripts.idf_converter import convert_and_run  # avoid module-level dep on eppy

    if building_name:
        # Inject override into a shallow copy so we don't mutate the caller's dict
        model_dict = {**model_dict, "building_name": building_name}

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_dir = Path(output_base) / f"run_{timestamp}"

    try:
        return convert_and_run(
            model_dict,
            output_dir=output_dir,
            weather_file=weather_file,
            defaults=defaults,
            run_simulation=True,
        )
    except Exception as exc:
        print(f"[run_ep_simulation_direct] ERROR: {exc}")
        return None


# ---------------------------------------------------------------------------
# CSV helpers
# ---------------------------------------------------------------------------


def parse_float(value: str) -> float:
    try:
        return float(value.strip())
    except (AttributeError, ValueError):
        return 0.0


def row_total_gj(row: list[str], start_index: int = 2) -> float:
    total = 0.0
    for value in row[start_index:]:
        total += parse_float(value)
    return total


def dedupe_metric_rows(rows: list[dict], key_field: str, value_field: str) -> list[dict]:
    seen: dict[str, dict] = {}
    for row in rows:
        key = row[key_field]
        value = row[value_field]
        if key not in seen or value > seen[key][value_field]:
            seen[key] = row
    return [seen[key] for key in seen]


def end_use_total(end_uses: list[dict], name: str) -> float:
    for item in end_uses:
        if item["end_use"] == name:
            return item["total_gj"]
    return 0.0


# ---------------------------------------------------------------------------
# Result parsing
# ---------------------------------------------------------------------------


def read_eplustbl(result_dir: str) -> dict:
    path = Path(result_dir) / "eplustbl.csv"
    if not path.exists():
        return {"path": path, "exists": False}

    rows: list[list[str]] = []
    with path.open(newline="", encoding="utf-8", errors="replace") as file:
        rows = list(csv.reader(file))

    site_energy: dict[str, float] = {}
    building_area: dict[str, float] = {}
    end_uses: list[dict] = []
    demand_end_uses: list[dict] = []
    zone_summary: list[dict] = []
    zone_energy: dict[str, dict[str, float]] = {}

    for i, row in enumerate(rows):
        if len(row) < 2:
            continue

        label = row[1].strip()
        section = row[0].strip()

        if label in {"Total Site Energy", "Net Site Energy", "Total Source Energy", "Net Source Energy"}:
            site_energy[label] = parse_float(row[2]) if len(row) > 2 else 0.0

        if label in {"Total Building Area", "Net Conditioned Building Area", "Unconditioned Building Area"}:
            building_area[label] = parse_float(row[2]) if len(row) > 2 else 0.0

        if label in {"Heating", "Cooling", "Interior Lighting", "Exterior Lighting", "Interior Equipment", "Fans", "Pumps"}:
            nearby_title = ""
            for back in range(max(0, i - 4), i):
                if len(rows[back]) > 0 and rows[back][0].strip() == "REPORT:":
                    nearby_title = rows[back][1].strip() if len(rows[back]) > 1 else ""
            item = {"end_use": label, "total_gj": row_total_gj(row)}
            if nearby_title == "Annual Building Utility Performance Summary" or i < 100:
                end_uses.append(item)
            elif nearby_title == "Demand End Use Components Summary":
                demand_end_uses.append({"end_use": label, "demand_w": row_total_gj(row)})

        if label.startswith("ZONE_") and len(row) > 2:
            zone_summary.append({
                "zone": label,
                "area_m2": parse_float(row[2]),
                "conditioned": row[3].strip() if len(row) > 3 else "",
                "volume_m3": parse_float(row[5]) if len(row) > 5 else 0.0,
            })

        meter_name = label
        if ":Zone:" in meter_name or ":ZONE:" in meter_name:
            parts = meter_name.split(":")
            zone_name = parts[-1].upper()
            bucket = zone_energy.setdefault(zone_name, {"heating_gj": 0.0, "cooling_gj": 0.0, "lighting_gj": 0.0})
            energy_gj = parse_float(row[2]) if len(row) > 2 else 0.0
            lowered = meter_name.lower()
            if lowered.startswith("heating:energytransfer:zone:"):
                bucket["heating_gj"] += energy_gj
            elif lowered.startswith("cooling:energytransfer:zone:"):
                bucket["cooling_gj"] += energy_gj
            elif lowered.startswith("interiorlights:electricity:zone:"):
                bucket["lighting_gj"] += energy_gj

    for values in zone_energy.values():
        values["total_gj"] = values["heating_gj"] + values["cooling_gj"] + values["lighting_gj"]

    end_uses = dedupe_metric_rows(end_uses, "end_use", "total_gj")
    demand_end_uses = dedupe_metric_rows(demand_end_uses, "end_use", "demand_w")

    return {
        "path": path,
        "exists": True,
        "site_energy": site_energy,
        "building_area": building_area,
        "end_uses": end_uses,
        "demand_end_uses": demand_end_uses,
        "zone_summary": zone_summary,
        "zone_energy": zone_energy,
    }


# ---------------------------------------------------------------------------
# Energy mapping
# ---------------------------------------------------------------------------


def model_energy_map(model: dict, sim_data: dict) -> dict[str, dict[str, float]]:
    mass_zones = [z for z in model["zones"] if z["dimensions"]["height"] > 1.0]
    zone_energy = sim_data.get("zone_energy", {})
    mapped: dict[str, dict[str, float]] = {}

    if len(zone_energy) >= len(mass_zones):
        for i, zone in enumerate(mass_zones, start=1):
            candidates = [
                zone["name"].upper(),
                f"ZONE_{i:02d}",
                f"ZONE_{i}",
            ]
            for candidate in candidates:
                if candidate in zone_energy:
                    mapped[zone["name"]] = {**zone_energy[candidate], "source": "meter"}
                    break
        if len(mapped) == len(mass_zones):
            return mapped

    total_area = sum(z["dimensions"]["length"] * z["dimensions"]["width"] for z in mass_zones)
    heating = end_use_total(sim_data.get("end_uses", []), "Heating")
    cooling = end_use_total(sim_data.get("end_uses", []), "Cooling")
    lighting = end_use_total(sim_data.get("end_uses", []), "Interior Lighting")

    for zone in mass_zones:
        area = zone["dimensions"]["length"] * zone["dimensions"]["width"]
        share = area / total_area if total_area else 0.0
        mapped[zone["name"]] = {
            "heating_gj": heating * share,
            "cooling_gj": cooling * share,
            "lighting_gj": lighting * share,
            "total_gj": (heating + cooling + lighting) * share,
            "source": "area_estimate",
        }

    return mapped
