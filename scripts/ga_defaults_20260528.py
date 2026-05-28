"""Default fixed parameters and gene definitions for GA on generate_20260528."""

from __future__ import annotations

from typing import Any

from scripts.facade_params import default_facade_params

# Names evolved by GA (see DEFAULT_GENES_20260528 in ga_core_20260528.py).
GA_GENE_NAMES = frozenset({
    "shared_aspect_ratio",
    "shared_offset_angle",
    "shared_offset_distance",
    "boundary_shift",
    "window_wwr",
})

# Non-default overrides for GA runs (everything else uses generate_20260528 defaults).
GA_PARAM_OVERRIDES: dict[str, Any] = {
    "shading_depth": 1.0,
    "platform_edge_walk_distance": 10.0,
}


def default_ga_fixed_params(
    *,
    building_name: str = "GA_20260528",
    site_size: float = 100.0,
) -> dict[str, Any]:
    """Fixed generator kwargs for GA (excludes tunable genes)."""
    facade = default_facade_params()
    facade["shading_depth"] = GA_PARAM_OVERRIDES["shading_depth"]

    return {
        "building_name": building_name,
        "site_size": float(site_size),
        "total_area": 10000.0,
        "lobby_height": 6.0,
        "floor_height": 4.0,
        "setback_south": 15.0,
        "setback_west": 15.0,
        "setback_north": 10.0,
        "setback_east": 10.0,
        "boundary_shift": 40.0,
        "group_size": 2,
        "min_support_overlap_ratio": 0.5,
        "add_aerial_platforms": True,
        "platform_edge_walk_distance": GA_PARAM_OVERRIDES["platform_edge_walk_distance"],
        "add_open_space_markers": True,
        **facade,
        # Unified three-block controls (values filled from GA individual / seed).
        "shared_aspect_ratio": 1.0,
        "shared_offset_angle": 180.0,
        "shared_offset_distance": 2.0,
    }


def extract_ga_genes_seed(sidebar_params: dict[str, Any]) -> dict[str, Any]:
    """Map sidebar state to initial GA gene values (unified aspect/angle/distance)."""
    seed: dict[str, Any] = {}

    sar = sidebar_params.get("shared_aspect_ratio")
    if sar is not None:
        seed["shared_aspect_ratio"] = float(sar)
    else:
        seed["shared_aspect_ratio"] = float(sidebar_params.get("low_aspect_ratio", 1.0))

    sao = sidebar_params.get("shared_offset_angle")
    if sao is not None:
        seed["shared_offset_angle"] = float(sao)
    else:
        seed["shared_offset_angle"] = float(sidebar_params.get("low_offset_angle", 180.0))

    sod = sidebar_params.get("shared_offset_distance")
    if sod is not None:
        seed["shared_offset_distance"] = float(sod)
    else:
        seed["shared_offset_distance"] = float(sidebar_params.get("low_offset_distance", 2.0))

    if "boundary_shift" in sidebar_params:
        seed["boundary_shift"] = float(sidebar_params["boundary_shift"])
    if "window_wwr" in sidebar_params:
        seed["window_wwr"] = float(sidebar_params["window_wwr"])

    return seed


def build_ga_fixed_params(sidebar_params: dict[str, Any] | None = None) -> dict[str, Any]:
    """Full fixed dict for run_ga: defaults + overrides + optional gene seeds for baseline."""
    sidebar_params = sidebar_params or {}
    fixed = default_ga_fixed_params(
        building_name=str(sidebar_params.get("building_name", "GA_20260528")),
        site_size=float(sidebar_params.get("site_size", 100.0)),
    )
    fixed.update(extract_ga_genes_seed(sidebar_params))
    return fixed
