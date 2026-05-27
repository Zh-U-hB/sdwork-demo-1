"""EnergyPlus simulation node.

Connects to the bundled EnergyPlus-Agent plugin (plugins/energyplus_agent/)
via its MCP server, converts the zone-box model produced by the upstream
pipeline into a full EnergyPlus IDF, and runs the simulation.

Plugin path resolution (first match wins):
  1. ENERGYPLUS_AGENT_PATH env var (explicit override)
  2. plugins/energyplus_agent/  relative to this project's root  ← default

MCP transport:
  ENERGYPLUS_AGENT_TRANSPORT = "stdio"           (default — spawns uv subprocess)
                              | "http"
                              | "sse"
                              | "streamable_http"
  ENERGYPLUS_AGENT_URL        = http://localhost:8000   (only for http transports)

Other optional env vars:
  ENERGYPLUS_WEATHER_FILE  – EPW path relative to plugin root
                             (default: data/weather/Shenzhen.epw)
  ENERGYPLUS_LOCATION_NAME / _LATITUDE / _LONGITUDE / _TIMEZONE / _ELEVATION
"""

import os
from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.prebuilt import create_react_agent

from src.agent.llm import create_llm
from src.agent.state import AgentState
from src.models.zone import BuildingModel

# ---------------------------------------------------------------------------
# Plugin root detection
# ---------------------------------------------------------------------------

# Project root = four levels up from this file:
#   src/agent/nodes/energyplus.py  →  src/agent/nodes  →  src/agent  →  src  →  project root
_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_BUNDLED_PLUGIN = _PROJECT_ROOT / "plugins" / "energyplus_agent"


def _get_plugin_path() -> Path | None:
    """Return the resolved EnergyPlus-Agent directory, or None if not available."""
    # Explicit override takes priority
    explicit = os.getenv("ENERGYPLUS_AGENT_PATH", "").strip()
    if explicit:
        p = Path(explicit)
        if p.exists():
            return p

    # Bundled plugin (git submodule)
    if _BUNDLED_PLUGIN.exists() and (_BUNDLED_PLUGIN / "main.py").exists():
        return _BUNDLED_PLUGIN

    return None


# ---------------------------------------------------------------------------
# MCP server config builder
# ---------------------------------------------------------------------------


def _get_mcp_server_config() -> dict | None:
    """Build the langchain-mcp-adapters server config dict.

    Returns None when the plugin is not available or the transport is
    misconfigured, signalling the node to skip gracefully.
    """
    transport = os.getenv("ENERGYPLUS_AGENT_TRANSPORT", "stdio").lower()

    if transport == "stdio":
        plugin_path = _get_plugin_path()
        if plugin_path is None:
            return None
        return {
            "energyplus": {
                "command": "uv",
                "args": [
                    "run",
                    "--directory", str(plugin_path),
                    "python", "-c",
                    # Bypass Typer CLI and start FastMCP server directly
                    (
                        "import sys, os; "
                        f"sys.path.insert(0, {str(plugin_path)!r}); "
                        "os.chdir(" + repr(str(plugin_path)) + "); "
                        "from pathlib import Path; "
                        "from src.validator.data_model import BaseSchema; "
                        "BaseSchema.set_idf(Path('data/dependencies/Energy+.idd')); "
                        "from src.mcp.server import mcp; "
                        "mcp.run()"
                    ),
                ],
                "transport": "stdio",
                "env": {
                    k: v for k, v in os.environ.items()
                    if not k.startswith(("LLM_API_KEY", "OPENAI_", "ANTHROPIC_"))
                },
            }
        }

    # HTTP-based transports
    base_url = os.getenv("ENERGYPLUS_AGENT_URL", "http://localhost:8000").rstrip("/")
    if transport == "sse":
        return {"energyplus": {"url": f"{base_url}/sse", "transport": "sse"}}
    # "http" | "streamable_http"
    return {
        "energyplus": {
            "url": f"{base_url}/mcp",
            "transport": "streamable_http",
        }
    }


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

ENERGYPLUS_SYSTEM_PROMPT = """You are an EnergyPlus building energy simulation expert.
You receive a building model as JSON with rectangular box zones (rooms). Each zone has:
  origin: {x, y, z}  (lower-left-front corner, metres)
  dimensions: {length (X-axis), width (Y-axis), height (Z-axis)}  (metres)

Use the MCP tools to build a complete, valid EnergyPlus model and run the simulation.

═══════════════════════════════════════════════════════════
KEY TOOL: create_zone  (ALL-IN-ONE — creates zone + ALL surfaces automatically)

  create_zone(
      name            = "<zone name>",
      floor_vertices  = [  ← REQUIRED. Counterclockwise when viewed from ABOVE.
          {"X": ox,   "Y": oy,   "Z": oz},   # SW corner
          {"X": ox+L, "Y": oy,   "Z": oz},   # SE corner
          {"X": ox+L, "Y": oy+W, "Z": oz},   # NE corner
          {"X": ox,   "Y": oy+W, "Z": oz},   # NW corner
      ],
      ceiling_height  = H,   ← REQUIRED. Must be a NUMBER (not "autocalculate").
      x_origin=ox, y_origin=oy, z_origin=oz,
  )

  Auto-created surfaces (all use construction "Default_Construction"):
    {name}_Wall_1  = south wall  (y=oy)
    {name}_Wall_2  = east  wall  (x=ox+L)
    {name}_Wall_3  = north wall  (y=oy+W)
    {name}_Wall_4  = west  wall  (x=ox)
    {name}_Floor   = floor       (z=oz,   boundary=Ground)
    {name}_Ceiling = ceiling     (z=oz+H, boundary=Adiabatic)

  ⚠ CRITICAL: You MUST create a construction named "Default_Construction" BEFORE any
    create_zone call, or zone creation will succeed but validate_config will fail.

SHARED WALLS (adjacent zones touching on the same plane)
  Two zones share a wall when one zone's max-X == another zone's min-X (or same for Y).
  After ALL zones are created, update the shared surface pairs:
    update_surface(
        name                             = "{ZoneA}_Wall_2",   # east wall of A
        outside_boundary_condition       = "Surface",
        outside_boundary_condition_object= "{ZoneB}_Wall_4",   # west wall of B
        sun_exposure                     = "NoSun",
        wind_exposure                    = "NoWind",
    )
    update_surface(
        name                             = "{ZoneB}_Wall_4",
        outside_boundary_condition       = "Surface",
        outside_boundary_condition_object= "{ZoneA}_Wall_2",
        sun_exposure                     = "NoSun",
        wind_exposure                    = "NoWind",
    )
  Adjacency on Y-axis: ZoneA Wall_3 (north) ↔ ZoneB Wall_1 (south).

═══════════════════════════════════════════════════════════
MATERIAL & CONSTRUCTION SETUP  (do this BEFORE create_zone)

  create_standard_material(
      name="Concrete200", roughness="MediumRough",
      thickness=0.20, conductivity=1.63, density=2240, specific_heat=900)

  create_construction(name="Default_Construction", layers=["Concrete200"])

═══════════════════════════════════════════════════════════
SCHEDULES

  create_schedule_type_limits(
      name="Fraction", lower_limit_value=0, upper_limit_value=1,
      numeric_type="Continuous", unit_type="Dimensionless")

  create_schedule_type_limits(
      name="Temperature", lower_limit_value=-100, upper_limit_value=100,
      numeric_type="Continuous", unit_type="Temperature")

  create_schedule_type_limits(
      name="ActivityLevel", lower_limit_value=0, upper_limit_value=1000,
      numeric_type="Continuous", unit_type="ActivityLevel")

  # Schedule data format — use EXACTLY this nested dict structure:
  ALWAYS_ON_DATA = [{"Through": "12/31", "Days": [
      {"For": "AllDays", "Times": [{"Until": {"Time": "24:00", "Value": 1.0}}]}]}]

  OCCUPANCY_DATA = [{"Through": "12/31", "Days": [
      {"For": "Weekdays", "Times": [
          {"Until": {"Time": "08:00", "Value": 0.0}},
          {"Until": {"Time": "18:00", "Value": 1.0}},
          {"Until": {"Time": "24:00", "Value": 0.0}}]},
      {"For": "AllOtherDays", "Times": [
          {"Until": {"Time": "24:00", "Value": 0.0}}]}]}]

  ACTIVITY_DATA  = [{"Through": "12/31", "Days": [
      {"For": "AllDays", "Times": [{"Until": {"Time": "24:00", "Value": 120.0}}]}]}]

  HEATING_DATA   = [{"Through": "12/31", "Days": [
      {"For": "AllDays", "Times": [{"Until": {"Time": "24:00", "Value": 21.0}}]}]}]

  COOLING_DATA   = [{"Through": "12/31", "Days": [
      {"For": "AllDays", "Times": [{"Until": {"Time": "24:00", "Value": 26.0}}]}]}]

  create_schedule_compact(name="AlwaysOn",          schedule_type_limits_name="Fraction",     times=ALWAYS_ON_DATA)
  create_schedule_compact(name="OccupancySchedule", schedule_type_limits_name="Fraction",     times=OCCUPANCY_DATA)
  create_schedule_compact(name="ActivityLevel",     schedule_type_limits_name="ActivityLevel",times=ACTIVITY_DATA)
  create_schedule_compact(name="HeatingSetpoint",   schedule_type_limits_name="Temperature",  times=HEATING_DATA)
  create_schedule_compact(name="CoolingSetpoint",   schedule_type_limits_name="Temperature",  times=COOLING_DATA)

═══════════════════════════════════════════════════════════
HVAC, PEOPLE, LIGHTS  (one thermostat shared; one IdealLoads + People + Lights per zone)

  create_hvac_thermostat(
      name="SharedThermostat",
      heating_setpoint_schedule_name="HeatingSetpoint",
      cooling_setpoint_schedule_name="CoolingSetpoint")

  # Per zone:
  create_hvac_ideal_loads_system(zone_name=<zone>, template_thermostat_name="SharedThermostat")

  create_people(
      name="<zone>_People",
      zone_or_zonelist_or_space_or_spacelist_name=<zone>,
      number_of_people_schedule_name="OccupancySchedule",
      activity_level_schedule_name="ActivityLevel",
      number_of_people_calculation_method="People/Area",
      people_per_floor_area=0.05)

  create_light(
      name="<zone>_Lights",
      zone_or_zone_list_or_space_or_space_list_name=<zone>,
      schedule_name="AlwaysOn",
      design_level_calculation_method="Watts/Area",
      watts_per_floor_area=10.0)

═══════════════════════════════════════════════════════════
COMPLETE WORKFLOW (strict order):

  1.  create_building(name, north_axis=0, terrain="Suburbs")
  2.  create_location(name, latitude, longitude, time_zone, elevation)
  3.  create_standard_material(...)       ← "Concrete200"
  4.  create_construction(...)            ← "Default_Construction"
  5.  For each zone → create_zone(name, floor_vertices=[...], ceiling_height=H, ...)
  6.  Detect shared walls → update_surface(...) pairs
  7.  create_schedule_type_limits × 3    (Fraction, Temperature, ActivityLevel)
  8.  create_schedule_compact × 5        (AlwaysOn, OccupancySchedule, ActivityLevel, HeatingSetpoint, CoolingSetpoint)
  9.  create_hvac_thermostat             (SharedThermostat)
 10.  For each zone → create_hvac_ideal_loads_system
 11.  For each zone → create_people + create_light
 12.  validate_config → fix any errors → validate again if needed
 13.  run_simulation(epw_path=<weather_file>, output_dir=<output_directory>)

Do NOT skip steps 3-4. Do NOT skip validate_config before run_simulation.
"""

# ---------------------------------------------------------------------------
# Task prompt builder
# ---------------------------------------------------------------------------


def _build_task_prompt(
    building_model: BuildingModel,
    idf_output_path: str,
    weather_file: str,
) -> str:
    def _env_float(key: str, default: str) -> float:
        try:
            return float(os.getenv(key, default))
        except (ValueError, TypeError):
            return float(default)

    location = {
        "name":      os.getenv("ENERGYPLUS_LOCATION_NAME", "Shenzhen"),
        "latitude":  _env_float("ENERGYPLUS_LATITUDE",  "22.55"),
        "longitude": _env_float("ENERGYPLUS_LONGITUDE", "114.10"),
        "timezone":  _env_float("ENERGYPLUS_TIMEZONE",  "8"),
        "elevation": _env_float("ENERGYPLUS_ELEVATION", "5"),
    }

    # Build a zone name mapping: Chinese/Unicode → ASCII-safe names for EnergyPlus
    zone_name_map = {
        z.name: f"Zone_{i+1:02d}_{z.name.encode('ascii', errors='ignore').decode() or f'Zone{i+1}'}"
        for i, z in enumerate(building_model.zones)
    }
    # Keep names short and purely ASCII (eppy uses latin-1 to save IDF)
    ascii_zone_names = {
        orig: f"Zone_{i+1:02d}"
        for i, orig in enumerate(zone_name_map)
    }
    zone_mapping_text = "\n".join(
        f"  {orig!r:30s} → \"{ascii_name}\""
        for orig, ascii_name in ascii_zone_names.items()
    )

    return f"""Below is the building zone model to convert into a full EnergyPlus simulation.

BUILDING MODEL (JSON):
{building_model.model_dump_json(indent=2)}

ASCII NAME MAPPING  ← USE THESE NAMES for all EnergyPlus objects (building name,
zone names, surface names, etc.). EnergyPlus IDF uses latin-1 encoding and cannot
store non-ASCII characters. The JSON names are only for reference.

  Building name → "Building_01"
{zone_mapping_text}

  Use "Building_01" for create_building(name=...).
  Use the mapped zone names (Zone_01, Zone_02, ...) for create_zone, create_people,
  create_light, create_hvac_ideal_loads_system, and all surface names.

SIMULATION SETTINGS:
  IDF output path : {idf_output_path}
  Weather file    : {weather_file}
  Location        : {location['name']}
                    lat={location['latitude']}°  lon={location['longitude']}°
                    UTC+{location['timezone']}  elevation={location['elevation']} m

Please follow the 13-step workflow from the system prompt.

Key reminders:
• ALL EnergyPlus object names must be ASCII only (letters, digits, underscore).
• Detect shared walls: check if any zone's max-X (origin.x + length) equals another
  zone's min-X (origin.x), OR same for Y axis.
• When calling run_simulation:
    epw_path   = "{weather_file}"
    output_dir = "{str(Path(idf_output_path).parent)}"
"""


# ---------------------------------------------------------------------------
# LangGraph node
# ---------------------------------------------------------------------------


async def energyplus_node(state: AgentState) -> dict:
    """LangGraph node: generate IDF and run EnergyPlus simulation via MCP.

    Gracefully skips when:
    - No zones are present in state, or
    - The EnergyPlus-Agent plugin is not found / not configured.
    """
    zones = state.get("zones", [])
    if not zones:
        return {
            "messages": [AIMessage(content="[EnergyPlus] Skipped: no zones in state.")]
        }

    server_config = _get_mcp_server_config()
    if server_config is None:
        plugin_path = _get_plugin_path()
        if plugin_path is None:
            hint = (
                "[EnergyPlus] Plugin not found.\n"
                "Expected location: plugins/energyplus_agent/\n"
                "Run:  git submodule update --init --recursive\n"
                "Then: cd plugins/energyplus_agent && uv sync"
            )
        else:
            hint = (
                f"[EnergyPlus] Plugin found at {plugin_path} but transport "
                "configuration is invalid. Check ENERGYPLUS_AGENT_TRANSPORT in .env."
            )
        return {"messages": [AIMessage(content=hint)]}

    building_name = state.get("building_name", "Unnamed Building")
    output_path = state.get("output_path", "output/building.json")
    idf_output_path = state.get(
        "idf_output_path",
        str(Path(output_path).with_suffix(".idf")),
    )

    plugin_path = _get_plugin_path()
    weather_file = os.getenv("ENERGYPLUS_WEATHER_FILE", "data/weather/Shenzhen.epw")
    # For stdio transport: make weather path absolute relative to plugin root
    if plugin_path and not Path(weather_file).is_absolute():
        weather_file = str(plugin_path / weather_file)

    building_model = BuildingModel(building_name=building_name, zones=zones)
    task = _build_task_prompt(building_model, idf_output_path, weather_file)

    llm = create_llm()

    try:
        from langchain_mcp_adapters.client import MultiServerMCPClient  # noqa: PLC0415
        from langchain_mcp_adapters.tools import load_mcp_tools  # noqa: PLC0415

        client = MultiServerMCPClient(server_config)
        # Use a persistent session: keeps the MCP subprocess alive for the entire
        # ReAct loop instead of restarting on every tool call.
        async with client.session("energyplus") as session:
            tools = await load_mcp_tools(session)
            agent = create_react_agent(
                model=llm,
                tools=tools,
                prompt=ENERGYPLUS_SYSTEM_PROMPT,
            )
            result = await agent.ainvoke(
                {"messages": [HumanMessage(content=task)]}
            )

        last_msg = result["messages"][-1]
        content = (
            last_msg.content
            if isinstance(last_msg.content, str)
            else str(last_msg.content)
        )
        summary = f"[EnergyPlus] Simulation completed.\n{content}"
        return {
            "messages": [AIMessage(content=summary)],
            "simulation_result": content,
            "idf_output_path": idf_output_path,
        }

    except ImportError:
        msg = (
            "[EnergyPlus] langchain-mcp-adapters is not installed.\n"
            "Run: pip install langchain-mcp-adapters"
        )
        return {"messages": [AIMessage(content=msg)]}
    except Exception as exc:  # noqa: BLE001
        msg = f"[EnergyPlus] Simulation failed: {exc}"
        return {
            "messages": [AIMessage(content=msg)],
            "simulation_result": msg,
        }
