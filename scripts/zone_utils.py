"""Shared zone creation utilities for generation scripts.

Provides `add_zone` for rectangular box zones and `add_polygon_zone`
for arbitrary polygon (trapezoidal, L-shaped, etc.) floor plans.
"""

from __future__ import annotations


def add_zone(
    zones: list[dict],
    name: str,
    x: float,
    y: float,
    z: float,
    length: float,
    width: float,
    height: float,
    *,
    category: str = "mass",
) -> None:
    """Add a rectangular box zone (backward compatible with all generators)."""
    if length <= 0 or width <= 0 or height <= 0:
        return
    x0, y0, z0 = round(x, 3), round(y, 3), round(z, 3)
    x1, y1, z1 = round(x + length, 3), round(y + width, 3), round(z + height, 3)
    zones.append({
        "name": name,
        "category": category,
        "origin": {"x": x0, "y": y0, "z": z0},
        "dimensions": {
            "length": round(length, 3),
            "width": round(width, 3),
            "height": round(height, 3),
        },
        "points": [
            {"x": x0, "y": y0, "z": z0},
            {"x": x1, "y": y0, "z": z0},
            {"x": x1, "y": y1, "z": z0},
            {"x": x0, "y": y1, "z": z0},
            {"x": x0, "y": y0, "z": z1},
            {"x": x1, "y": y0, "z": z1},
            {"x": x1, "y": y1, "z": z1},
            {"x": x0, "y": y1, "z": z1},
        ],
    })


def add_polygon_zone(
    zones: list[dict],
    name: str,
    floor_polygon: list[tuple[float, float]],
    z: float,
    height: float,
    *,
    category: str = "mass",
) -> None:
    """Add an extruded-polygon zone (trapezoidal, L-shaped, etc.).

    Parameters
    ----------
    zones:
        List to append the new zone dict to.
    name:
        Zone name.
    floor_polygon:
        2D floor plan vertices ordered CCW when viewed from above.
        Must have at least 3 vertices.
    z:
        Base elevation in meters.
    height:
        Extrusion height in meters.
    category:
        Zone category string.
    """
    n = len(floor_polygon)
    if n < 3 or height <= 0:
        return

    z0 = round(z, 3)
    z1 = round(z + height, 3)

    # Bounding box for origin/dimensions (backward compat)
    xs = [p[0] for p in floor_polygon]
    ys = [p[1] for p in floor_polygon]
    x0, y0 = round(min(xs), 3), round(min(ys), 3)
    x1, y1 = round(max(xs), 3), round(max(ys), 3)

    # 3D points for visualization: bottom vertices then top vertices
    bottom = [{"x": round(px, 3), "y": round(py, 3), "z": z0} for px, py in floor_polygon]
    top = [{"x": p["x"], "y": p["y"], "z": z1} for p in bottom]

    zones.append({
        "name": name,
        "category": category,
        "origin": {"x": x0, "y": y0, "z": z0},
        "dimensions": {
            "length": round(x1 - x0, 3),
            "width": round(y1 - y0, 3),
            "height": round(height, 3),
        },
        "floor_polygon": [{"x": round(px, 3), "y": round(py, 3)} for px, py in floor_polygon],
        "points": bottom + top,
    })


def zone_floor_area(zone: dict) -> float:
    """Compute floor area using polygon (shoelace) or box dimensions."""
    poly = zone.get("floor_polygon")
    if poly and len(poly) >= 3:
        n = len(poly)
        area = 0.0
        for i in range(n):
            j = (i + 1) % n
            area += poly[i]["x"] * poly[j]["y"]
            area -= poly[j]["x"] * poly[i]["y"]
        return abs(area) / 2.0
    d = zone["dimensions"]
    return d["length"] * d["width"]
