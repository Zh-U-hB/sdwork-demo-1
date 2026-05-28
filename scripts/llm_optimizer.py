"""LLM-driven building energy optimization loop.

Reads EnergyPlus simulation results, asks an LLM to propose parameter
adjustments to reduce EUI, re-runs the simulation with the new parameters,
and repeats for a configurable number of iterations.

Public API
----------
::

    from scripts.llm_optimizer import run_llm_optimization, BRIDGE_CLUSTER_TUNABLE
    from scripts.generate_bridge_cluster import generate_bridge_cluster

    history = run_llm_optimization(
        initial_params=current_params,
        generator_fn=generate_bridge_cluster,
        tunable_spec=BRIDGE_CLUSTER_TUNABLE,
        max_iterations=5,
        convergence_threshold=2.0,   # MJ/m² — stop if EUI improves less than this
    )
    best = min(history, key=lambda r: r.eui)
    print(f"Best EUI: {best.eui:.1f}  params: {best.params}")
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from scripts.zone_partition import partition_model_by_floor
# ---------------------------------------------------------------------------
# Tunable-parameter specs per generator
# ---------------------------------------------------------------------------
# Keys that the LLM is allowed to modify.
# Position keys (*_x, *_y), site_size, building_name are NOT included to
# preserve the spatial strategy and site boundary.

BRIDGE_CLUSTER_TUNABLE: dict[str, dict] = {
    # --- vertical dimensions ---
    "max_floors":   {"min": 4,   "max": 10,  "type": "int",   "description": "Maximum floors allowed for any block"},
    "lobby_height": {"min": 3.0, "max": 9.0, "type": "float", "description": "Ground floor lobby height (m)"},
    "floor_height": {"min": 3.0, "max": 5.0, "type": "float", "description": "Typical office floor height (m)"},
    # --- per-block floor counts ---
    "west_floors":  {"min": 1,   "max": 10,  "type": "int",   "description": "Floors for the west block"},
    "east_floors":  {"min": 1,   "max": 10,  "type": "int",   "description": "Floors for the east block"},
    "north_floors": {"min": 1,   "max": 10,  "type": "int",   "description": "Floors for the north block"},
    # --- per-block footprint (width/length may affect compactness) ---
    "west_length":  {"min": 18.0, "max": 42.0, "type": "float", "description": "Length of west block along X (m)"},
    "west_width":   {"min": 16.0, "max": 36.0, "type": "float", "description": "Width of west block along Y (m)"},
    "east_length":  {"min": 18.0, "max": 42.0, "type": "float", "description": "Length of east block along X (m)"},
    "east_width":   {"min": 16.0, "max": 36.0, "type": "float", "description": "Width of east block along Y (m)"},
    "north_length": {"min": 18.0, "max": 42.0, "type": "float", "description": "Length of north block along X (m)"},
    "north_width":  {"min": 16.0, "max": 36.0, "type": "float", "description": "Width of north block along Y (m)"},
    # --- platform / terrace ---
    "terrace_depth":  {"min": 0.0,  "max": 6.0,  "type": "float", "description": "Upper-floor terrace setback per side (m)"},
    "platform_depth": {"min": 12.0, "max": 34.0, "type": "float", "description": "Depth of the shared ground platform (m)"},
    "platform_width": {"min": 10.0, "max": 28.0, "type": "float", "description": "Width of the shared ground platform (m)"},
}

# EnergyPlus defaults that LLM can also suggest overriding.
# These map to fields in ConverterDefaults (idf_defaults.py).
EP_DEFAULTS_TUNABLE: dict[str, dict] = {
    "lights_watts_per_floor_area": {
        "min": 5.0, "max": 20.0, "type": "float",
        "description": "Lighting power density (W/m²). Default 10.",
        "path": "lights.watts_per_floor_area",
    },
    "people_per_floor_area": {
        "min": 0.02, "max": 0.15, "type": "float",
        "description": "Occupant density (persons/m²). Default 0.05.",
        "path": "people.people_per_floor_area",
    },
    "window_wwr": {
        "min": 0.0, "max": 0.7, "type": "float",
        "description": "Window-to-wall ratio 0–0.7. Default 0 (no windows).",
        "path": "window.wwr",
    },
    "heating_setpoint": {
        "min": 16.0, "max": 24.0, "type": "float",
        "description": "Heating setpoint temperature (°C). Default 21.",
        "path": "hvac.heating_setpoint",
    },
    "cooling_setpoint": {
        "min": 22.0, "max": 30.0, "type": "float",
        "description": "Cooling setpoint temperature (°C). Default 26.",
        "path": "hvac.cooling_setpoint",
    },
}

# Combined spec used when building the system prompt
ALL_TUNABLE = {**BRIDGE_CLUSTER_TUNABLE, **EP_DEFAULTS_TUNABLE}


# ---------------------------------------------------------------------------
# Iteration record
# ---------------------------------------------------------------------------

@dataclass
class IterationRecord:
    """Result of one optimization iteration."""
    iteration: int
    params: dict[str, Any]
    ep_defaults_overrides: dict[str, Any]   # overrides applied to ConverterDefaults
    eui: float                              # MJ/m²
    total_site_gj: float
    conditioned_area_m2: float
    end_uses: list[dict]                    # [{"end_use": str, "total_gj": float}]
    llm_analysis: str                       # LLM's plain-text explanation
    result_dir: str | None
    error: str | None = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _compute_eui(sim_data: dict) -> float:
    total = sim_data["site_energy"].get("Total Site Energy", 0.0)
    area = sim_data["building_area"].get("Net Conditioned Building Area", 0.0)
    return total * 1000 / area if area > 0 else 0.0


def _end_use_pct(end_uses: list[dict], name: str, total_gj: float) -> float:
    for item in end_uses:
        if item["end_use"] == name:
            return item["total_gj"] / total_gj * 100 if total_gj else 0.0
    return 0.0


def format_sim_context(
    sim_data: dict,
    params: dict,
    ep_defaults_overrides: dict,
    iteration: int,
    history: list[IterationRecord],
) -> str:
    """Return a human-readable context string for the LLM prompt."""
    eui = _compute_eui(sim_data)
    total_gj = sim_data["site_energy"].get("Total Site Energy", 0.0)
    area = sim_data["building_area"].get("Net Conditioned Building Area", 0.0)
    end_uses = sim_data.get("end_uses", [])

    lines = [
        f"=== Iteration {iteration} Results ===",
        f"EUI: {eui:.1f} MJ/m²",
        f"Total site energy: {total_gj:.2f} GJ",
        f"Conditioned area: {area:.1f} m²",
        "",
        "End-use breakdown:",
    ]
    for item in end_uses:
        pct = item["total_gj"] / total_gj * 100 if total_gj else 0.0
        lines.append(f"  {item['end_use']}: {item['total_gj']:.2f} GJ ({pct:.1f}%)")

    if history:
        lines += ["", "EUI history (previous iterations):"]
        valid_history = [r for r in history if r.eui < 1e5]
        best_eui = min((r.eui for r in valid_history), default=float("inf"))
        for rec in history:
            if rec.eui >= 1e5:
                lines.append(f"  Iter {rec.iteration}: FAILED")
            else:
                mark = " * (best)" if rec.eui == best_eui else ""
                lines.append(f"  Iter {rec.iteration}: {rec.eui:.1f} MJ/m²{mark}")

    lines += ["", "Current geometry parameters:"]
    for k, v in params.items():
        if k not in ("building_name", "add_open_space_markers"):
            lines.append(f"  {k}: {v}")

    if ep_defaults_overrides:
        lines += ["", "Current EnergyPlus defaults overrides:"]
        for k, v in ep_defaults_overrides.items():
            lines.append(f"  {k}: {v}")

    return "\n".join(lines)


def _build_system_prompt(tunable_spec: dict) -> str:
    """Build the LLM system prompt with parameter specs."""
    spec_lines = []
    for key, spec in tunable_spec.items():
        desc = spec["description"]
        lo, hi = spec["min"], spec["max"]
        typ = spec["type"]
        spec_lines.append(f'  "{key}": {{"type": "{typ}", "min": {lo}, "max": {hi}, "description": "{desc}"}}')

    spec_block = "{\n" + ",\n".join(spec_lines) + "\n}"

    # Build prompt without .format() to avoid conflicts with JSON braces in spec_block
    return (
        "You are a building energy performance optimization expert. Your job is to\n"
        "analyze EnergyPlus simulation results and suggest parameter adjustments that\n"
        "will reduce the building's EUI (Energy Use Intensity, MJ/m²).\n"
        "\n"
        "TUNABLE PARAMETERS (you may only suggest changes to these):\n"
        + spec_block + "\n"
        "\n"
        "RULES:\n"
        "1. Return ONLY valid JSON. No markdown, no explanation outside the JSON.\n"
        '2. JSON format:\n'
        '   {\n'
        '     "analysis": "<1-3 sentence explanation of why EUI is high and what to change>",\n'
        '     "changes": {\n'
        '       "<param_name>": <new_value>,\n'
        '       ...\n'
        '     }\n'
        '   }\n'
        "3. Only include parameters you actually want to change in \"changes\".\n"
        "4. Stay within the min/max bounds for each parameter.\n"
        "5. Make incremental adjustments — do not change more than 3-4 parameters at once.\n"
        "6. Focus on the largest energy end-use first (usually Lighting or Cooling).\n"
        "7. Lighting is the dominant load in office buildings. If Interior Lighting > 50%\n"
        "   of total energy, reduce lights_watts_per_floor_area.\n"
        "8. For cooling-dominated buildings, consider increasing window_wwr only if it\n"
        "   allows for natural cooling strategies — but avoid high WWR in hot climates.\n"
        "9. If EUI is already improving, continue in the same direction with smaller steps.\n"
        "10. If EUI is stagnating (< 1 MJ/m² improvement in last 2 iterations), try a\n"
        "    different strategy.\n"
    )


def _parse_llm_json(text: str) -> dict:
    """Extract and parse JSON from LLM response, handling markdown fences."""
    # Strip markdown fences if present
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if fence:
        text = fence.group(1).strip()
    # Find first { ... }
    brace = re.search(r"\{[\s\S]*\}", text)
    if brace:
        text = brace.group(0)
    return json.loads(text)


# ---------------------------------------------------------------------------
# EP defaults overrides → ConverterDefaults
# ---------------------------------------------------------------------------

def _apply_ep_overrides(overrides: dict):
    """Return a ConverterDefaults instance with the given overrides applied."""
    from scripts.idf_defaults import make_default_settings

    defaults = make_default_settings()
    for key, value in overrides.items():
        spec = EP_DEFAULTS_TUNABLE.get(key, {})
        path = spec.get("path", "")
        if path == "lights.watts_per_floor_area":
            defaults.lights.watts_per_floor_area = float(value)
        elif path == "people.people_per_floor_area":
            defaults.people.people_per_floor_area = float(value)
        elif path == "window.wwr":
            defaults.window.wwr = float(value)
        elif path == "hvac.heating_setpoint":
            _patch_setpoint_schedule(defaults, "heating", float(value))
        elif path == "hvac.cooling_setpoint":
            _patch_setpoint_schedule(defaults, "cooling", float(value))
    return defaults


def _patch_setpoint_schedule(defaults, which: str, value: float) -> None:
    """Update heating or cooling setpoint schedule value in-place."""
    from scripts.idf_defaults import ScheduleDef
    target_name = f"{'Heating' if which == 'heating' else 'Cooling'}SetpointSchedule"
    new_data = [
        "Through: 12/31",
        "For: AllDays",
        f"Until: 24:00,{value:.1f}",
    ]
    for i, sched in enumerate(defaults.schedules):
        if sched.name == target_name:
            defaults.schedules[i] = ScheduleDef(
                name=sched.name,
                type_limits_name=sched.type_limits_name,
                data=new_data,
            )
            return


# ---------------------------------------------------------------------------
# Sanitise and validate LLM-proposed parameters
# ---------------------------------------------------------------------------

def _sanitize_params(
    proposed: dict,
    current_geometry: dict,
    current_ep_overrides: dict,
    tunable_spec: dict,
    generator_fn: Callable,
) -> tuple[dict, dict]:
    """Return (sanitized_geometry_params, sanitized_ep_overrides).

    Steps:
      1. Only keep keys in tunable_spec.
      2. Clamp to [min, max].
      3. Cast to correct type.
      4. Verify generator does not raise ValueError.
    """
    geo_params = dict(current_geometry)
    ep_overrides = dict(current_ep_overrides)

    for key, value in proposed.items():
        if key not in tunable_spec:
            continue
        spec = tunable_spec[key]
        lo, hi = spec["min"], spec["max"]
        if spec["type"] == "int":
            value = int(round(float(value)))
        else:
            value = float(value)
        value = max(lo, min(hi, value))

        from scripts.facade_params import FACADE_GEOMETRY_KEYS

        if key in FACADE_GEOMETRY_KEYS:
            geo_params[key] = value
        elif key in EP_DEFAULTS_TUNABLE:
            ep_overrides[key] = value
        else:
            geo_params[key] = value

    from scripts.facade_params import decode_generator_bools

    geo_params = decode_generator_bools(geo_params)

    # Validate that generator accepts the new geo params; revert both dicts on failure
    try:
        generator_fn(**geo_params)
    except ValueError:
        geo_params = dict(current_geometry)
        ep_overrides = dict(current_ep_overrides)

    return geo_params, ep_overrides


# ---------------------------------------------------------------------------
# Single LLM call → new parameters
# ---------------------------------------------------------------------------

def llm_optimize_params(
    params: dict,
    ep_defaults_overrides: dict,
    sim_data: dict,
    iteration: int,
    history: list[IterationRecord],
    tunable_spec: dict,
    generator_fn: Callable,
    llm,
) -> tuple[dict, dict, str]:
    """Call LLM and return (new_geo_params, new_ep_overrides, analysis_text).

    Returns current params unchanged on any error.
    """
    from langchain_core.messages import HumanMessage, SystemMessage

    context = format_sim_context(sim_data, params, ep_defaults_overrides, iteration, history)
    system_prompt = _build_system_prompt(tunable_spec)

    try:
        response = llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=context),
        ])
        raw = response.content if hasattr(response, "content") else str(response)
        # Handle list content (some models return list of dicts)
        if isinstance(raw, list):
            raw = " ".join(
                item.get("text", "") if isinstance(item, dict) else str(item)
                for item in raw
            )

        parsed = _parse_llm_json(raw)
        analysis = parsed.get("analysis", "")
        proposed = parsed.get("changes", {})

        new_geo, new_ep = _sanitize_params(
            proposed, params, ep_defaults_overrides, tunable_spec, generator_fn
        )
        return new_geo, new_ep, analysis

    except Exception as exc:
        return params, ep_defaults_overrides, f"[LLM call failed: {exc}]"


# ---------------------------------------------------------------------------
# Main optimization loop
# ---------------------------------------------------------------------------

def run_llm_optimization(
    initial_params: dict,
    generator_fn: Callable,
    tunable_spec: dict | None = None,
    max_iterations: int = 5,
    convergence_threshold: float = 2.0,
    initial_ep_overrides: dict | None = None,
    partition_enabled: bool = True,
    perimeter_depth: float = 4.0,
    weather_file: str | Path | None = None,
    output_base: str = "output/llm_opt",
    llm=None,
    progress_callback: Callable[[IterationRecord], None] | None = None,
) -> list[IterationRecord]:
    """Run the LLM-driven optimization loop.

    Parameters
    ----------
    initial_params:
        Generator keyword arguments for the first iteration.
    generator_fn:
        Callable matching the signature of ``generate_bridge_cluster`` etc.
    tunable_spec:
        Dict of tunable parameter specs.  Defaults to ``ALL_TUNABLE``.
    max_iterations:
        Maximum number of simulate-and-improve cycles.
    convergence_threshold:
        Stop early if EUI improvement over the last 2 iterations is less than
        this value (MJ/m²).
    initial_ep_overrides:
        Starting EnergyPlus defaults overrides (empty dict means factory defaults).
    output_base:
        Base directory for EnergyPlus result sub-folders.
    llm:
        LangChain chat model.  Created from env vars if ``None``.
    progress_callback:
        Optional callable invoked after each iteration with the IterationRecord.
        Useful for streaming progress to a Streamlit UI.

    Returns
    -------
    list[IterationRecord]
        One record per completed iteration, in order.
    """
    from scripts.ep_sim_utils import read_eplustbl, run_ep_simulation_direct
    from src.agent.llm import create_llm

    if llm is None:
        llm = create_llm()

    if tunable_spec is None:
        tunable_spec = ALL_TUNABLE

    params = dict(initial_params)
    ep_overrides = dict(initial_ep_overrides or {})
    history: list[IterationRecord] = []

    # Create one directory per LLM optimization run, and store per-iteration simulations inside.
    ts = time.strftime("%Y%m%d_%H%M%S")
    ns = time.time_ns() % 1_000_000_000
    run_dir = Path(output_base) / f"llm_{ts}_{ns:09d}"
    sims_dir = run_dir / "sims"
    sims_dir.mkdir(parents=True, exist_ok=True)

    run_meta = {
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "max_iterations": int(max_iterations),
        "convergence_threshold": float(convergence_threshold),
        "partition_enabled": bool(partition_enabled),
        "perimeter_depth": float(perimeter_depth),
        "weather_file": str(weather_file) if weather_file else None,
        "initial_params": dict(initial_params),
        "initial_ep_overrides": dict(initial_ep_overrides or {}),
        "tunable_keys": sorted(list((tunable_spec or ALL_TUNABLE).keys())),
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "run_meta.json").write_text(json.dumps(run_meta, indent=2, ensure_ascii=False), encoding="utf-8")

    for i in range(max_iterations):
        # --- Generate model and run simulation ---
        try:
            raw_model = generator_fn(**params)
        except ValueError as exc:
            record = IterationRecord(
                iteration=i, params=dict(params), ep_defaults_overrides=dict(ep_overrides),
                eui=float("inf"), total_site_gj=0.0, conditioned_area_m2=0.0,
                end_uses=[], llm_analysis="", result_dir=None,
                error=f"Generator error: {exc}",
            )
            history.append(record)
            if progress_callback:
                progress_callback(record)
            break

        # Partition first (simulation uses partitioned zones)
        model = raw_model
        if partition_enabled:
            model = partition_model_by_floor(
                raw_model,
                perimeter_depth=float(perimeter_depth),
                lobby_height=float(params.get("lobby_height", 6.0)),
                floor_height=float(params.get("floor_height", 4.0)),
            )

        ep_defaults = _apply_ep_overrides(ep_overrides)
        from scripts.facade_params import apply_facade_to_ep_defaults

        apply_facade_to_ep_defaults(ep_defaults, params)

        # Persist iteration settings (for reproducibility)
        iter_ns = time.time_ns() % 1_000_000_000
        iter_id = f"iter_{i:02d}_{iter_ns:09d}"
        iter_settings = {
            "iteration": i,
            "params": dict(params),
            "ep_defaults_overrides": dict(ep_overrides),
        }
        (sims_dir / f"{iter_id}.json").write_text(
            json.dumps(iter_settings, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        result_dir = run_ep_simulation_direct(
            model,
            building_name=params.get("building_name", "LLM_Opt"),
            defaults=ep_defaults,
            output_base=sims_dir,
            run_id=iter_id,
            weather_file=weather_file,
        )

        if result_dir is None:
            record = IterationRecord(
                iteration=i, params=dict(params), ep_defaults_overrides=dict(ep_overrides),
                eui=float("inf"), total_site_gj=0.0, conditioned_area_m2=0.0,
                end_uses=[], llm_analysis="", result_dir=None,
                error="Simulation failed or eplustbl.csv not found.",
            )
            history.append(record)
            if progress_callback:
                progress_callback(record)
            # Try to continue with same params
            continue

        sim_data = read_eplustbl(result_dir)
        if not sim_data.get("exists"):
            record = IterationRecord(
                iteration=i, params=dict(params), ep_defaults_overrides=dict(ep_overrides),
                eui=float("inf"), total_site_gj=0.0, conditioned_area_m2=0.0,
                end_uses=[], llm_analysis="", result_dir=result_dir,
                error="eplustbl.csv not found.",
            )
            history.append(record)
            if progress_callback:
                progress_callback(record)
            continue

        eui = _compute_eui(sim_data)
        total_gj = sim_data["site_energy"].get("Total Site Energy", 0.0)
        area = sim_data["building_area"].get("Net Conditioned Building Area", 0.0)

        # --- Ask LLM for next step ---
        new_geo, new_ep, analysis = llm_optimize_params(
            params=params,
            ep_defaults_overrides=ep_overrides,
            sim_data=sim_data,
            iteration=i,
            history=history,
            tunable_spec=tunable_spec,
            generator_fn=generator_fn,
            llm=llm,
        )

        record = IterationRecord(
            iteration=i,
            params=dict(params),
            ep_defaults_overrides=dict(ep_overrides),
            eui=eui,
            total_site_gj=total_gj,
            conditioned_area_m2=area,
            end_uses=sim_data.get("end_uses", []),
            llm_analysis=analysis,
            result_dir=result_dir,
        )
        history.append(record)
        if progress_callback:
            progress_callback(record)

        # --- Convergence check ---
        # Stop when the best EUI seen in the last 3 valid iterations has not
        # improved by at least convergence_threshold compared to the 4th-last best.
        valid = [r for r in history if r.eui < 1e5]
        if len(valid) >= 3:
            recent3 = sorted(valid, key=lambda r: r.iteration)[-3:]
            best_recent = min(r.eui for r in recent3)
            # Compare to the best before this window
            older = [r for r in valid if r not in recent3]
            best_older = min((r.eui for r in older), default=float("inf"))
            improvement = best_older - best_recent   # positive = getting better
            if improvement < convergence_threshold:
                break

        # --- Update params for next iteration ---
        params = new_geo
        ep_overrides = new_ep

    return history


# ---------------------------------------------------------------------------
# Convenience: best record from a history list
# ---------------------------------------------------------------------------

def best_record(history: list[IterationRecord]) -> IterationRecord | None:
    """Return the iteration with the lowest valid EUI."""
    valid = [r for r in history if r.eui < 1e5 and r.error is None]
    return min(valid, key=lambda r: r.eui) if valid else None
