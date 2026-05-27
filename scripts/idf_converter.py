"""Programmatic JSON → IDF converter and EnergyPlus runner.

Uses **idfpy** (https://github.com/ITOTI-Y/idfpy) — type-safe Pydantic models
for all EnergyPlus IDF objects.  No IDD file required at runtime.

Converts a BuildingModel zone-box JSON dict directly to an EnergyPlus IDF file
and optionally runs the simulation — no LLM, no MCP, pure Python.

Public API
----------
::

    from scripts.idf_converter import convert_and_run
    from scripts.idf_defaults import make_default_settings

    # All defaults (Shenzhen, concrete walls, ideal loads)
    result_dir = convert_and_run(model_dict, output_dir="output/sim_001")

    # Override location only
    defaults = make_default_settings()
    defaults.location.latitude  = 39.93
    defaults.location.longitude = 116.28
    defaults.location.name      = "Beijing"
    result_dir = convert_and_run(model_dict, output_dir="output/sim_002",
                                 defaults=defaults)

    # Generate IDF only, no simulation
    result_dir = convert_and_run(model_dict, output_dir="output/sim_003",
                                 run_simulation=False)
"""

from __future__ import annotations

import re
import time
from pathlib import Path
from typing import NamedTuple

from scripts.idf_defaults import (
    ConverterDefaults,
    GlazingDef,
    LightDef,
    MaterialDef,
    PeopleDef,
    ScheduleDef,
    ScheduleTypeDef,
    WindowDef,
    make_default_settings,
)

# ---------------------------------------------------------------------------
# idfpy imports
# ---------------------------------------------------------------------------
from idfpy import IDF
from idfpy.models import (
    Building,
    BuildingSurfaceDetailed,
    Construction,
    FenestrationSurfaceDetailed,
    GlobalGeometryRules,
    HVACTemplateThermostat,
    HVACTemplateZoneIdealLoadsAirSystem,
    Lights,
    Material,
    OutputDiagnostics,
    OutputTableSummaryReports,
    OutputVariable,
    OutputVariableDictionary,
    People,
    RunPeriod,
    ScheduleCompact,
    ScheduleTypeLimits,
    SimulationControl,
    SiteLocation,
    Timestep,
    Version,
    WindowMaterialSimpleGlazingSystem,
    Zone,
)
from idfpy.models.thermal_zones import BuildingSurfaceDetailedVerticesItem
from idfpy.models.schedules import ScheduleCompactDataItem
from idfpy.models.outputs import (
    OutputDiagnosticsDiagnosticsItem,
    OutputTableSummaryReportsReportsItem,
)
from idfpy.sim import simulate

# ---------------------------------------------------------------------------
# ASCII name sanitisation
# ---------------------------------------------------------------------------

_NON_ASCII = re.compile(r"[^\x00-\x7f]")
_UNSAFE    = re.compile(r"[^A-Za-z0-9_\-]")


def _ascii_name(name: str, index: int, prefix: str = "Zone") -> str:
    """Return a short ASCII-safe IDF name for *name*.

    If *name* is already pure ASCII it is returned with unsafe chars replaced.
    Otherwise a positional fallback ``{prefix}_{index:02d}`` is used.
    """
    if _NON_ASCII.search(name):
        return f"{prefix}_{index:02d}"
    safe = _UNSAFE.sub("_", name)[:48]
    return safe or f"{prefix}_{index:02d}"


# ---------------------------------------------------------------------------
# Surface vertex helpers
# ---------------------------------------------------------------------------
# EnergyPlus GlobalGeometryRules: UpperLeftCorner + CounterClockWise + World.
# Vertices are ordered CCW when viewed from OUTSIDE.

Vec3 = tuple[float, float, float]


def _verts(pts: list[Vec3]) -> list[BuildingSurfaceDetailedVerticesItem]:
    return [
        BuildingSurfaceDetailedVerticesItem(
            vertex_x_coordinate=x,
            vertex_y_coordinate=y,
            vertex_z_coordinate=z,
        )
        for x, y, z in pts
    ]


def _floor_pts(ox: float, oy: float, oz: float, L: float, W: float) -> list[Vec3]:
    """Floor at z=oz, outward normal = −Z (CCW viewed from below)."""
    return [(ox, oy+W, oz), (ox+L, oy+W, oz), (ox+L, oy, oz), (ox, oy, oz)]


def _ceiling_pts(ox: float, oy: float, oz: float, L: float, W: float, H: float) -> list[Vec3]:
    """Ceiling at z=oz+H, outward normal = +Z (CCW viewed from above)."""
    return [(ox, oy, oz+H), (ox+L, oy, oz+H), (ox+L, oy+W, oz+H), (ox, oy+W, oz+H)]


def _south_pts(ox: float, oy: float, oz: float, L: float, H: float) -> list[Vec3]:
    """South wall at y=oy, outward normal = −Y."""
    return [(ox, oy, oz), (ox+L, oy, oz), (ox+L, oy, oz+H), (ox, oy, oz+H)]


def _north_pts(ox: float, oy: float, oz: float, L: float, W: float, H: float) -> list[Vec3]:
    """North wall at y=oy+W, outward normal = +Y."""
    return [(ox+L, oy+W, oz), (ox, oy+W, oz), (ox, oy+W, oz+H), (ox+L, oy+W, oz+H)]


def _west_pts(ox: float, oy: float, oz: float, W: float, H: float) -> list[Vec3]:
    """West wall at x=ox, outward normal = −X."""
    return [(ox, oy+W, oz), (ox, oy, oz), (ox, oy, oz+H), (ox, oy+W, oz+H)]


def _east_pts(ox: float, oy: float, oz: float, L: float, W: float, H: float) -> list[Vec3]:
    """East wall at x=ox+L, outward normal = +X."""
    return [(ox+L, oy, oz), (ox+L, oy+W, oz), (ox+L, oy+W, oz+H), (ox+L, oy, oz+H)]


def _window_pts(wall_pts: list[Vec3], wwr: float) -> list[Vec3] | None:
    """Return inset vertices for a centred rectangular window on an exterior wall."""
    if wwr <= 0.0:
        return None
    xs = [v[0] for v in wall_pts]
    ys = [v[1] for v in wall_pts]
    zs = [v[2] for v in wall_pts]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    z_min, z_max = min(zs), max(zs)
    wall_w = max(x_max - x_min, y_max - y_min)
    wall_h = z_max - z_min
    if wall_w < 0.5 or wall_h < 0.5:
        return None
    win_w = wall_w * wwr
    win_h = wall_h * wwr
    mw = (wall_w - win_w) / 2
    mh = (wall_h - win_h) / 2
    z0 = z_min + mh
    z1 = z0 + win_h
    if x_max - x_min > y_max - y_min:  # wall spans X (south or north)
        x0, x1 = x_min + mw, x_min + mw + win_w
        y_val = (y_min + y_max) / 2
        if wall_pts[0][0] < wall_pts[1][0]:
            return [(x0, y_val, z0), (x1, y_val, z0), (x1, y_val, z1), (x0, y_val, z1)]
        else:
            return [(x1, y_val, z0), (x0, y_val, z0), (x0, y_val, z1), (x1, y_val, z1)]
    else:                               # wall spans Y (west or east)
        y0, y1 = y_min + mw, y_min + mw + win_w
        x_val = (x_min + x_max) / 2
        if wall_pts[0][1] > wall_pts[1][1]:
            return [(x_val, y1, z0), (x_val, y0, z0), (x_val, y0, z1), (x_val, y1, z1)]
        else:
            return [(x_val, y0, z0), (x_val, y1, z0), (x_val, y1, z1), (x_val, y0, z1)]


# ---------------------------------------------------------------------------
# Zone box helper
# ---------------------------------------------------------------------------

class _ZoneBox(NamedTuple):
    ascii_name: str
    ox: float; oy: float; oz: float
    L: float;  W: float;  H: float

    @property
    def x_max(self) -> float: return self.ox + self.L
    @property
    def y_max(self) -> float: return self.oy + self.W
    @property
    def z_max(self) -> float: return self.oz + self.H


# ---------------------------------------------------------------------------
# Shared wall detection
# ---------------------------------------------------------------------------

def _shared_walls(boxes: list[_ZoneBox], tol: float = 0.01) -> list[tuple[str, str]]:
    """Return list of (wall_A_name, wall_B_name) surface pairs that share a face."""
    pairs: list[tuple[str, str]] = []
    n = len(boxes)

    def overlap(a0, a1, b0, b1) -> bool:
        return (min(a1, b1) - max(a0, b0)) > tol

    for i in range(n):
        for j in range(i + 1, n):
            a, b = boxes[i], boxes[j]
            # A east ↔ B west
            if abs(a.x_max - b.ox) < tol and overlap(a.oy, a.y_max, b.oy, b.y_max) and overlap(a.oz, a.z_max, b.oz, b.z_max):
                pairs.append((f"{a.ascii_name}_Wall_East", f"{b.ascii_name}_Wall_West"))
            # B east ↔ A west
            elif abs(b.x_max - a.ox) < tol and overlap(a.oy, a.y_max, b.oy, b.y_max) and overlap(a.oz, a.z_max, b.oz, b.z_max):
                pairs.append((f"{b.ascii_name}_Wall_East", f"{a.ascii_name}_Wall_West"))
            # A north ↔ B south
            elif abs(a.y_max - b.oy) < tol and overlap(a.ox, a.x_max, b.ox, b.x_max) and overlap(a.oz, a.z_max, b.oz, b.z_max):
                pairs.append((f"{a.ascii_name}_Wall_North", f"{b.ascii_name}_Wall_South"))
            # B north ↔ A south
            elif abs(b.y_max - a.oy) < tol and overlap(a.ox, a.x_max, b.ox, b.x_max) and overlap(a.oz, a.z_max, b.oz, b.z_max):
                pairs.append((f"{b.ascii_name}_Wall_North", f"{a.ascii_name}_Wall_South"))
    return pairs


# ---------------------------------------------------------------------------
# IDF builder helpers — one function per object type
# ---------------------------------------------------------------------------

def _make_surface(
    name: str, surface_type: str, construction: str, zone: str,
    boundary: str, sun: str, wind: str,
    pts: list[Vec3],
    boundary_object: str = "",
) -> BuildingSurfaceDetailed:
    return BuildingSurfaceDetailed(
        name=name,
        surface_type=surface_type,
        construction_name=construction,
        zone_name=zone,
        outside_boundary_condition=boundary,
        outside_boundary_condition_object=boundary_object or None,
        sun_exposure=sun,
        wind_exposure=wind,
        view_factor_to_ground="autocalculate",
        vertices=_verts(pts),
    )


def _make_window(name: str, construction: str, wall_name: str, pts: list[Vec3]) -> FenestrationSurfaceDetailed:
    assert len(pts) == 4, "Window must have exactly 4 vertices"
    return FenestrationSurfaceDetailed(
        name=name,
        surface_type="Window",
        construction_name=construction,
        building_surface_name=wall_name,
        multiplier=1,
        number_of_vertices=4,
        vertex_1_x_coordinate=pts[0][0], vertex_1_y_coordinate=pts[0][1], vertex_1_z_coordinate=pts[0][2],
        vertex_2_x_coordinate=pts[1][0], vertex_2_y_coordinate=pts[1][1], vertex_2_z_coordinate=pts[1][2],
        vertex_3_x_coordinate=pts[2][0], vertex_3_y_coordinate=pts[2][1], vertex_3_z_coordinate=pts[2][2],
        vertex_4_x_coordinate=pts[3][0], vertex_4_y_coordinate=pts[3][1], vertex_4_z_coordinate=pts[3][2],
    )


def _make_schedule_compact(sched: ScheduleDef) -> ScheduleCompact:
    return ScheduleCompact(
        name=sched.name,
        schedule_type_limits_name=sched.type_limits_name,
        data=[ScheduleCompactDataItem(field=entry) for entry in sched.data],
    )


# ---------------------------------------------------------------------------
# Zone geometry builder
# ---------------------------------------------------------------------------

def _build_zone(
    idf: IDF,
    box: _ZoneBox,
    defaults: ConverterDefaults,
    shared_wall_names: set[str],
) -> None:
    """Add Zone + 6 BuildingSurface:Detailed objects for one rectangular zone."""
    zn = box.ascii_name
    ox, oy, oz = box.ox, box.oy, box.oz
    L,  W,  H  = box.L,  box.W,  box.H

    idf.add(Zone(name=zn))

    ext_wall  = "ExteriorWallConstruction"
    int_wall  = "InteriorWallConstruction"
    ext_roof  = "ExteriorRoofConstruction"
    floor_con = "FloorConstruction"
    win_con   = defaults.window.construction_name
    wwr       = defaults.window.wwr

    # Defined in order: (surface_name, type, construction, boundary, sun, wind, pts_fn)
    wall_specs = [
        (f"{zn}_Floor",      "Floor",   floor_con, "Ground",   "NoSun",       "NoWind",       _floor_pts(ox, oy, oz, L, W)),
        (f"{zn}_Ceiling",    "Ceiling", ext_roof,  "Outdoors", "SunExposed",  "WindExposed",  _ceiling_pts(ox, oy, oz, L, W, H)),
        (f"{zn}_Wall_South", "Wall",    ext_wall,  "Outdoors", "SunExposed",  "WindExposed",  _south_pts(ox, oy, oz, L, H)),
        (f"{zn}_Wall_North", "Wall",    ext_wall,  "Outdoors", "SunExposed",  "WindExposed",  _north_pts(ox, oy, oz, L, W, H)),
        (f"{zn}_Wall_West",  "Wall",    ext_wall,  "Outdoors", "SunExposed",  "WindExposed",  _west_pts(ox, oy, oz, W, H)),
        (f"{zn}_Wall_East",  "Wall",    ext_wall,  "Outdoors", "SunExposed",  "WindExposed",  _east_pts(ox, oy, oz, L, W, H)),
    ]

    for sname, stype, scon, boundary, sun, wind, pts in wall_specs:
        is_shared = sname in shared_wall_names
        idf.add(_make_surface(
            name=sname, surface_type=stype,
            construction=int_wall if is_shared else scon,
            zone=zn,
            boundary=boundary if not is_shared else "Outdoors",  # will be patched
            sun="NoSun"  if is_shared else sun,
            wind="NoWind" if is_shared else wind,
            pts=pts,
        ))

        # Add window on exterior walls only
        if stype == "Wall" and not is_shared and wwr > 0.0:
            win_pts = _window_pts(pts, wwr)
            if win_pts:
                idf.add(_make_window(
                    name=f"{sname}_Window",
                    construction=win_con,
                    wall_name=sname,
                    pts=win_pts,
                ))


def _patch_shared_walls(idf: IDF, pairs: list[tuple[str, str]]) -> None:
    """Set boundary condition on detected interior surface pairs."""
    for wall_a, wall_b in pairs:
        sa = idf.get(BuildingSurfaceDetailed, wall_a)
        sb = idf.get(BuildingSurfaceDetailed, wall_b)
        if sa is None or sb is None:
            continue
        for surf, partner in [(sa, wall_b), (sb, wall_a)]:
            surf.outside_boundary_condition        = "Surface"
            surf.outside_boundary_condition_object = partner
            surf.sun_exposure                      = "NoSun"
            surf.wind_exposure                     = "NoWind"
            surf.construction_name                 = "InteriorWallConstruction"


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

MASS_HEIGHT_THRESHOLD = 1.0  # metres — skip courtyard markers etc.


def convert_and_run(
    model_dict: dict,
    output_dir: str | Path = "output/direct_sim",
    weather_file: str | Path | None = None,
    defaults: ConverterDefaults | None = None,
    run_simulation: bool = True,
    idd_path: str | Path | None = None,  # kept for API compatibility, unused by idfpy
) -> str | None:
    """Convert *model_dict* to IDF using idfpy and optionally run EnergyPlus.

    Parameters
    ----------
    model_dict:
        BuildingModel JSON dict with ``building_name`` and ``zones``.
    output_dir:
        Directory for the IDF file and simulation results.
    weather_file:
        Path to ``.epw`` file.  Auto-resolved when ``None``.
    defaults:
        :class:`scripts.idf_defaults.ConverterDefaults` instance — mutate the
        result of :func:`scripts.idf_defaults.make_default_settings` to override
        any subset of parameters.  ``None`` uses factory defaults.
    run_simulation:
        ``False`` → write IDF only, return the output directory path.
    idd_path:
        Ignored (idfpy does not require an IDD file).  Kept for drop-in
        compatibility with callers that pass it.

    Returns
    -------
    str | None
        Path to the directory containing ``eplustbl.csv`` on success, or
        ``None`` if the simulation failed / was skipped.
    """
    defaults = defaults or make_default_settings()

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if weather_file is None:
        from scripts.ep_sim_utils import resolve_weather_path
        weather_file = resolve_weather_path()
    epw_path = Path(weather_file)

    building_name = _ascii_name(
        model_dict.get("building_name", "Building"), 0, prefix="Building"
    )

    # Filter zones
    raw_zones = [
        z for z in model_dict.get("zones", [])
        if z["dimensions"]["height"] >= MASS_HEIGHT_THRESHOLD
    ]
    if not raw_zones:
        raise ValueError("No valid zones (all below height threshold).")

    # Build ASCII name map and box list
    boxes: list[_ZoneBox] = []
    for idx, z in enumerate(raw_zones, 1):
        aname = _ascii_name(z["name"], idx, prefix="Zone")
        o, d = z["origin"], z["dimensions"]
        boxes.append(_ZoneBox(
            ascii_name=aname,
            ox=float(o["x"]), oy=float(o["y"]), oz=float(o["z"]),
            L=float(d["length"]), W=float(d["width"]), H=float(d["height"]),
        ))

    # Detect shared walls before building IDF
    shared_pairs = _shared_walls(boxes)
    shared_wall_names: set[str] = {w for pair in shared_pairs for w in pair}

    # ── Build IDF ─────────────────────────────────────────────────────────────
    idf = IDF()

    # Settings & envelope (note: idfpy auto-adds a Version header)
    idf.add(SimulationControl(
        do_zone_sizing_calculation="No",
        do_system_sizing_calculation="No",
        do_plant_sizing_calculation="No",
        run_simulation_for_sizing_periods="Yes",
        run_simulation_for_weather_file_run_periods="Yes",
        do_hvac_sizing_simulation_for_sizing_periods="Yes",
        maximum_number_of_hvac_sizing_simulation_passes=1,
    ))
    idf.add(Timestep(number_of_timesteps_per_hour=4))
    idf.add(RunPeriod(
        name="FullYear",
        begin_month=1, begin_day_of_month=1,
        end_month=12, end_day_of_month=31,
    ))
    idf.add(GlobalGeometryRules(
        starting_vertex_position="UpperLeftCorner",
        vertex_entry_direction="CounterClockWise",
        coordinate_system="World",
    ))

    # Output requests
    idf.add(OutputVariableDictionary(key_field="regular"))
    idf.add(OutputDiagnostics(
        diagnostics=[OutputDiagnosticsDiagnosticsItem(key="DisplayExtraWarnings")]
    ))
    idf.add(OutputTableSummaryReports(
        reports=[OutputTableSummaryReportsReportsItem(report_name="AllSummary")]
    ))
    # Note: OutputControlTableStyle has a 26.1-only field that breaks EnergyPlus 25.1.
    # Omit it; EnergyPlus defaults to Comma/CSV output for eplustbl.csv.
    for var in [
        "Zone Ideal Loads Heat Recovery Total Heating Energy",
        "Zone Ideal Loads Supply Air Total Cooling Energy",
        "Zone Lights Electricity Energy",
    ]:
        idf.add(OutputVariable(key_value="*", variable_name=var, reporting_frequency="Annual"))

    # Building & location
    idf.add(Building(
        name=building_name,
        north_axis=0.0,
        terrain="Suburbs",
        solar_distribution="FullInteriorAndExterior",
        maximum_number_of_warmup_days=25,
        minimum_number_of_warmup_days=1,
    ))
    loc = defaults.location
    idf.add(SiteLocation(
        name=loc.name,
        latitude=loc.latitude,
        longitude=loc.longitude,
        time_zone=loc.time_zone,
        elevation=loc.elevation,
    ))

    # Materials
    for mat in defaults.opaque_materials:
        idf.add(Material(
            name=mat.name,
            roughness=mat.roughness,
            thickness=mat.thickness,
            conductivity=mat.conductivity,
            density=mat.density,
            specific_heat=mat.specific_heat,
        ))
    for gla in defaults.glazing_materials:
        idf.add(WindowMaterialSimpleGlazingSystem(
            name=gla.name,
            u_factor=gla.u_factor,
            solar_heat_gain_coefficient=gla.solar_heat_gain_coefficient,
            visible_transmittance=gla.visible_transmittance,
        ))

    # Constructions
    for con in defaults.constructions:
        layers = {
            "outside_layer": con.layers[0] if len(con.layers) > 0 else None,
            **{f"layer_{i+2}": con.layers[i+1] for i in range(len(con.layers) - 1)},
        }
        idf.add(Construction(name=con.name, **layers))

    # Schedule type limits
    for st in defaults.schedule_types:
        idf.add(ScheduleTypeLimits(
            name=st.name,
            lower_limit_value=st.lower_limit,
            upper_limit_value=st.upper_limit,
            numeric_type=st.numeric_type,
            unit_type=st.unit_type,
        ))

    # Schedules
    for sched in defaults.schedules:
        idf.add(_make_schedule_compact(sched))

    # HVAC thermostat (shared across all zones)
    idf.add(HVACTemplateThermostat(
        name=defaults.hvac.thermostat_name,
        heating_setpoint_schedule_name=defaults.hvac.heating_setpoint_schedule_name,
        cooling_setpoint_schedule_name=defaults.hvac.cooling_setpoint_schedule_name,
    ))

    # ── Geometry ──────────────────────────────────────────────────────────────
    for box in boxes:
        _build_zone(idf, box, defaults, shared_wall_names)

    _patch_shared_walls(idf, shared_pairs)

    # ── Per-zone loads & HVAC ─────────────────────────────────────────────────
    for box in boxes:
        zn = box.ascii_name
        ppl = defaults.people
        idf.add(People(
            name=f"{zn}_People",
            zone_or_zonelist_or_space_or_spacelist_name=zn,
            number_of_people_schedule_name=ppl.number_of_people_schedule_name,
            number_of_people_calculation_method="People/Area",
            people_per_floor_area=ppl.people_per_floor_area,
            fraction_radiant=ppl.fraction_radiant,
            sensible_heat_fraction="Autocalculate",
            activity_level_schedule_name=ppl.activity_level_schedule_name,
        ))
        lgt = defaults.lights
        idf.add(Lights(
            name=f"{zn}_Lights",
            zone_or_zonelist_or_space_or_spacelist_name=zn,
            schedule_name=lgt.schedule_name,
            design_level_calculation_method="Watts/Area",
            watts_per_floor_area=lgt.watts_per_floor_area,
            return_air_fraction=lgt.return_air_fraction,
            fraction_radiant=lgt.fraction_radiant,
            fraction_visible=lgt.fraction_visible,
        ))
        idf.add(HVACTemplateZoneIdealLoadsAirSystem(
            zone_name=zn,
            template_thermostat_name=defaults.hvac.thermostat_name,
        ))

    # ── Save IDF + model JSON (for viewer) ────────────────────────────────────
    import json as _json
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    idf_path = output_dir / f"building_{timestamp}.idf"
    idf.save(idf_path)
    # Save original model dict alongside IDF so sim_viewer_app can pick it up
    model_json_path = output_dir / "model.json"
    model_json_path.write_text(
        _json.dumps(model_dict, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"[idf_converter] IDF saved → {idf_path}  ({len(raw_zones)} zones, {len(shared_pairs)} shared-wall pairs)")

    if not run_simulation:
        return str(output_dir)

    # ── Run EnergyPlus via idfpy.sim.simulate() ───────────────────────────────
    if not epw_path.exists():
        raise FileNotFoundError(f"EPW file not found: {epw_path}")

    results_dir = output_dir / f"results_{timestamp}"
    results_dir.mkdir(parents=True, exist_ok=True)

    result = simulate(
        idf_path,
        weather=epw_path,
        output_dir=results_dir,
        expand_objects=True,
        echo=False,  # suppress per-line EnergyPlus stdout
    )

    if result.success:
        eplustbl = results_dir / "eplustbl.csv"
        if eplustbl.exists():
            print(f"[idf_converter] Simulation complete → {results_dir}")
            return str(results_dir)
        candidates = list(results_dir.rglob("eplustbl.csv"))
        if candidates:
            return str(candidates[0].parent)

    print(f"[idf_converter] Simulation failed. err: {result.err}")
    return None
