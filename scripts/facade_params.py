"""Shared facade (windows + overhang) parameters for generate_20260528 workflows."""

from __future__ import annotations

from typing import Any

# Keys passed to generate_20260528() for geometry-level facade control.
FACADE_GEOMETRY_KEYS = frozenset({
    "window_enabled",
    "window_wwr",
    "window_module",
    "shading_enabled",
    "shading_depth",
})

FACADE_TUNABLE: dict[str, dict] = {
    "window_wwr": {
        "min": 0.0,
        "max": 0.8,
        "type": "float",
        "description": "Geometric window-to-wall ratio (module windows on each facade segment).",
    },
    "window_module": {
        "min": 0.5,
        "max": 3.0,
        "type": "float",
        "description": "Facade module length (m) for splitting each wall edge.",
    },
    "shading_depth": {
        "min": 0.0,
        "max": 2.0,
        "type": "float",
        "description": "Horizontal overhang depth (m) at each floor top.",
    },
    "window_enabled": {
        "min": 0,
        "max": 1,
        "type": "int",
        "description": "Enable geometric facade windows (0/1).",
    },
    "shading_enabled": {
        "min": 0,
        "max": 1,
        "type": "int",
        "description": "Enable horizontal overhang shading surfaces (0/1).",
    },
}


def decode_generator_bools(geo: dict[str, Any]) -> dict[str, Any]:
    """Decode 0/1 flags to bool for generate_20260528."""
    out = dict(geo)
    for key in (
        "window_enabled",
        "shading_enabled",
        "add_aerial_platforms",
        "add_open_space_markers",
    ):
        if key not in out:
            continue
        val = out[key]
        if isinstance(val, bool):
            continue
        out[key] = bool(int(val))
    return out


def apply_facade_to_ep_defaults(defaults: Any, geo: dict[str, Any]) -> None:
    """Align IDF global WWR with geometric facade WWR when present."""
    if not bool(geo.get("window_enabled", True)):
        defaults.window.wwr = 0.0
        return
    if "window_wwr" in geo:
        defaults.window.wwr = float(geo["window_wwr"])


def make_ep_defaults_for_geometry(
    geo: dict[str, Any],
    ep_overrides: dict[str, float] | None = None,
) -> Any:
    """Build ConverterDefaults with EP tunables + facade WWR sync."""
    from scripts.idf_defaults import make_default_settings
    from scripts.llm_optimizer import _apply_ep_overrides

    defaults = _apply_ep_overrides(ep_overrides or {})
    apply_facade_to_ep_defaults(defaults, geo)
    return defaults


def default_facade_params() -> dict[str, Any]:
    return {
        "window_enabled": True,
        "window_wwr": 0.4,
        "window_module": 1.0,
        "shading_enabled": True,
        "shading_depth": 0.5,
    }
