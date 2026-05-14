import json
from pathlib import Path

from langchain_core.tools import tool

from src.models.zone import Zone, Point3D, Dimensions, BuildingModel

_zones: list[Zone] = []
_building_name: str = "Unnamed Building"


def reset_store() -> None:
    global _zones, _building_name
    _zones = []
    _building_name = "Unnamed Building"


def get_zones() -> list[Zone]:
    return _zones


def set_building_name(name: str) -> None:
    global _building_name
    _building_name = name


@tool
def create_zone(
    name: str,
    origin_x: float = 0.0,
    origin_y: float = 0.0,
    origin_z: float = 0.0,
    length: float = 1.0,
    width: float = 1.0,
    height: float = 3.0,
) -> str:
    """Create a new building zone with its 3D geometry. All dimensions in meters.
    origin_x/y/z define the lower-left corner, length is along X-axis, width along Y-axis, height along Z-axis.
    Returns the created zone info as JSON."""
    existing = {z.name for z in _zones}
    if name in existing:
        return f"Error: Zone '{name}' already exists. Use update_zone to modify or delete_zone to remove it first."

    zone = Zone(
        name=name,
        origin=Point3D(x=origin_x, y=origin_y, z=origin_z),
        dimensions=Dimensions(length=length, width=width, height=height),
    )
    _zones.append(zone)
    return f"Created: {zone.model_dump_json()}"


@tool
def list_zones() -> str:
    """List all created zones with their geometry."""
    if not _zones:
        return "No zones created yet."
    model = BuildingModel(building_name=_building_name, zones=_zones)
    return model.model_dump_json(indent=2)


@tool
def update_zone(
    name: str,
    origin_x: float | None = None,
    origin_y: float | None = None,
    origin_z: float | None = None,
    length: float | None = None,
    width: float | None = None,
    height: float | None = None,
) -> str:
    """Update an existing zone's geometry. Only specify the fields you want to change.
    Returns the updated zone info as JSON."""
    for zone in _zones:
        if zone.name == name:
            if origin_x is not None:
                zone.origin.x = origin_x
            if origin_y is not None:
                zone.origin.y = origin_y
            if origin_z is not None:
                zone.origin.z = origin_z
            if length is not None:
                zone.dimensions.length = length
            if width is not None:
                zone.dimensions.width = width
            if height is not None:
                zone.dimensions.height = height
            return f"Updated: {zone.model_dump_json()}"
    return f"Error: Zone '{name}' not found. Available zones: {[z.name for z in _zones]}"


@tool
def delete_zone(name: str) -> str:
    """Delete a zone by name."""
    global _zones
    before = len(_zones)
    _zones = [z for z in _zones if z.name != name]
    if len(_zones) < before:
        return f"Deleted zone '{name}'. {len(_zones)} zones remaining."
    return f"Error: Zone '{name}' not found. Available zones: {[z.name for z in _zones]}"


@tool
def export_json(filepath: str) -> str:
    """Export all zones to a JSON file. Call this when done creating all zones.
    Returns the file path on success."""
    model = BuildingModel(building_name=_building_name, zones=_zones)
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(model.model_dump_json(indent=2), encoding="utf-8")
    return f"Exported {len(_zones)} zones to {filepath}"


ZONE_TOOLS = [create_zone, list_zones, update_zone, delete_zone, export_json]
