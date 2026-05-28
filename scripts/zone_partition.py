"""Partition massing models into floor, exposed, perimeter, and interior zones.

This first implementation deliberately avoids optional geometry dependencies.
It supports the axis-aligned rectangular mass blocks used by the current
parametric generators, while preserving aerial-platform prism zones as their
own independent zones.  The output still uses the same JSON shape as the rest
of the project: each zone has origin, dimensions, and prism points. Rectangular
and trapezoid/triangle perimeter partitions are represented as vertical prisms.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path


EPS = 1e-6


@dataclass(frozen=True)
class Rect:
    x0: float
    y0: float
    x1: float
    y1: float

    @property
    def area(self) -> float:
        return max(0.0, self.x1 - self.x0) * max(0.0, self.y1 - self.y0)

    def overlaps(self, other: "Rect", tol: float = EPS) -> bool:
        return min(self.x1, other.x1) - max(self.x0, other.x0) > tol and min(self.y1, other.y1) - max(self.y0, other.y0) > tol

    def contains_rect(self, other: "Rect", tol: float = EPS) -> bool:
        return self.x0 <= other.x0 + tol and self.y0 <= other.y0 + tol and self.x1 >= other.x1 - tol and self.y1 >= other.y1 - tol


@dataclass
class FloorPlate:
    name: str
    source_zone: str
    source_category: str
    rect: Rect
    z: float
    height: float
    floor_index: int

    @property
    def building_key(self) -> str:
        return self.source_zone.split("_block_", 1)[0]


@dataclass
class Cell:
    rect: Rect
    category: str
    exposure: tuple[str, ...]


@dataclass
class ZonePart:
    polygon: list[tuple[float, float]]
    category: str
    exposure: tuple[str, ...]


def round3(value: float) -> float:
    return round(value, 3)


def add_rect_zone(
    zones: list[dict],
    name: str,
    rect: Rect,
    z: float,
    height: float,
    *,
    category: str,
    metadata: dict,
) -> None:
    if rect.area <= EPS or height <= EPS:
        return
    x0, y0, z0 = round3(rect.x0), round3(rect.y0), round3(z)
    x1, y1, z1 = round3(rect.x1), round3(rect.y1), round3(z + height)
    zones.append({
        "name": name,
        "category": category,
        "origin": {"x": x0, "y": y0, "z": z0},
        "dimensions": {
            "length": round3(x1 - x0),
            "width": round3(y1 - y0),
            "height": round3(height),
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
        "metadata": metadata,
    })


def add_polygon_zone(
    zones: list[dict],
    name: str,
    polygon: list[tuple[float, float]],
    z: float,
    height: float,
    *,
    category: str,
    metadata: dict,
) -> None:
    if len(polygon) < 3 or height <= EPS:
        return
    area = polygon_area(polygon)
    if abs(area) <= EPS:
        return
    if area < 0:
        polygon = list(reversed(polygon))

    xs = [point[0] for point in polygon]
    ys = [point[1] for point in polygon]
    z0, z1 = round3(z), round3(z + height)
    bottom = [{"x": round3(x), "y": round3(y), "z": z0} for x, y in polygon]
    top = [{"x": round3(x), "y": round3(y), "z": z1} for x, y in polygon]
    zones.append({
        "name": name,
        "category": category,
        "origin": {"x": round3(min(xs)), "y": round3(min(ys)), "z": z0},
        "dimensions": {
            "length": round3(max(xs) - min(xs)),
            "width": round3(max(ys) - min(ys)),
            "height": round3(height),
        },
        "points": bottom + top,
        "metadata": metadata,
    })


def add_prism_zone(
    zones: list[dict],
    name: str,
    points: list[dict],
    *,
    category: str,
    metadata: dict,
) -> None:
    if len(points) != 8:
        return
    xs = [p["x"] for p in points]
    ys = [p["y"] for p in points]
    zs = [p["z"] for p in points]
    zones.append({
        "name": name,
        "category": category,
        "origin": {"x": round3(min(xs)), "y": round3(min(ys)), "z": round3(min(zs))},
        "dimensions": {
            "length": round3(max(xs) - min(xs)),
            "width": round3(max(ys) - min(ys)),
            "height": round3(max(zs) - min(zs)),
        },
        "points": [{"x": round3(p["x"]), "y": round3(p["y"]), "z": round3(p["z"])} for p in points],
        "metadata": metadata,
    })


def zone_rect(zone: dict) -> Rect:
    points = zone.get("points") or []
    if len(points) >= 4:
        bottom = points[:4]
        return Rect(
            min(p["x"] for p in bottom),
            min(p["y"] for p in bottom),
            max(p["x"] for p in bottom),
            max(p["y"] for p in bottom),
        )
    origin = zone["origin"]
    dims = zone["dimensions"]
    return Rect(origin["x"], origin["y"], origin["x"] + dims["length"], origin["y"] + dims["width"])


def polygon_area(polygon: list[tuple[float, float]]) -> float:
    area = 0.0
    for i, (x0, y0) in enumerate(polygon):
        x1, y1 = polygon[(i + 1) % len(polygon)]
        area += x0 * y1 - x1 * y0
    return area / 2.0


def rect_polygon(rect: Rect) -> list[tuple[float, float]]:
    return [(rect.x0, rect.y0), (rect.x1, rect.y0), (rect.x1, rect.y1), (rect.x0, rect.y1)]


def side_band_polygon(rect: Rect, side: str, segment: tuple[float, float], depth: float) -> list[tuple[float, float]]:
    d = min(depth, (rect.x1 - rect.x0) / 2.0, (rect.y1 - rect.y0) / 2.0)
    s0, s1 = segment
    if side == "south":
        inner0 = (rect.x0 + d if abs(s0 - rect.x0) <= EPS else s0, rect.y0 + d)
        inner1 = (rect.x1 - d if abs(s1 - rect.x1) <= EPS else s1, rect.y0 + d)
        return [(s0, rect.y0), (s1, rect.y0), inner1, inner0]
    if side == "north":
        inner0 = (rect.x1 - d if abs(s1 - rect.x1) <= EPS else s1, rect.y1 - d)
        inner1 = (rect.x0 + d if abs(s0 - rect.x0) <= EPS else s0, rect.y1 - d)
        return [(s1, rect.y1), (s0, rect.y1), inner1, inner0]
    if side == "east":
        inner0 = (rect.x1 - d, rect.y0 + d if abs(s0 - rect.y0) <= EPS else s0)
        inner1 = (rect.x1 - d, rect.y1 - d if abs(s1 - rect.y1) <= EPS else s1)
        return [(rect.x1, s0), (rect.x1, s1), inner1, inner0]
    if side == "west":
        inner0 = (rect.x0 + d, rect.y1 - d if abs(s1 - rect.y1) <= EPS else s1)
        inner1 = (rect.x0 + d, rect.y0 + d if abs(s0 - rect.y0) <= EPS else s0)
        return [(rect.x0, s1), (rect.x0, s0), inner1, inner0]
    raise ValueError(side)


def corner_split_parts(rect: Rect, exposure: tuple[str, ...]) -> list[ZonePart] | None:
    exposure_set = set(exposure)
    if {"south_wall", "west_wall"} <= exposure_set:
        return [
            ZonePart([(rect.x0, rect.y0), (rect.x1, rect.y0), (rect.x1, rect.y1)], "perimeter_zone", ("south_wall",)),
            ZonePart([(rect.x0, rect.y0), (rect.x1, rect.y1), (rect.x0, rect.y1)], "perimeter_zone", ("west_wall",)),
        ]
    if {"south_wall", "east_wall"} <= exposure_set:
        return [
            ZonePart([(rect.x0, rect.y0), (rect.x1, rect.y0), (rect.x0, rect.y1)], "perimeter_zone", ("south_wall",)),
            ZonePart([(rect.x1, rect.y0), (rect.x1, rect.y1), (rect.x0, rect.y1)], "perimeter_zone", ("east_wall",)),
        ]
    if {"north_wall", "east_wall"} <= exposure_set:
        return [
            ZonePart([(rect.x1, rect.y1), (rect.x0, rect.y1), (rect.x0, rect.y0)], "perimeter_zone", ("north_wall",)),
            ZonePart([(rect.x1, rect.y1), (rect.x0, rect.y0), (rect.x1, rect.y0)], "perimeter_zone", ("east_wall",)),
        ]
    if {"north_wall", "west_wall"} <= exposure_set:
        return [
            ZonePart([(rect.x1, rect.y1), (rect.x0, rect.y1), (rect.x1, rect.y0)], "perimeter_zone", ("north_wall",)),
            ZonePart([(rect.x0, rect.y1), (rect.x0, rect.y0), (rect.x1, rect.y0)], "perimeter_zone", ("west_wall",)),
        ]
    return None


def floor_index_for_z(z: float, lobby_height: float, floor_height: float) -> int:
    if z < lobby_height - EPS:
        return 1
    return int(round((z - lobby_height) / floor_height)) + 2


def floor_bounds(max_height: float, lobby_height: float, floor_height: float) -> list[tuple[int, float, float]]:
    bounds = [(1, 0.0, lobby_height)]
    floor_index = 2
    z = lobby_height
    while z < max_height - EPS:
        bounds.append((floor_index, z, z + floor_height))
        floor_index += 1
        z += floor_height
    return bounds


def expand_mass_to_floor_plates(model: dict, lobby_height: float, floor_height: float) -> list[FloorPlate]:
    mass_zones = [z for z in model["zones"] if z.get("category") == "mass_block"]
    max_height = max((z["origin"]["z"] + z["dimensions"]["height"] for z in mass_zones), default=0.0)
    bounds = floor_bounds(max_height, lobby_height, floor_height)
    plates: list[FloorPlate] = []

    for zone in mass_zones:
        z0 = zone["origin"]["z"]
        z1 = z0 + zone["dimensions"]["height"]
        rect = zone_rect(zone)
        for floor_index, fz0, fz1 in bounds:
            overlap = min(z1, fz1) - max(z0, fz0)
            if overlap <= EPS:
                continue
            slice_z = max(z0, fz0)
            plates.append(FloorPlate(
                name=f"{zone['name']}_F{floor_index:02d}",
                source_zone=zone["name"],
                source_category=zone.get("category", ""),
                rect=rect,
                z=slice_z,
                height=overlap,
                floor_index=floor_index,
            ))
    return plates


def clipped_overlap_bounds(rect: Rect, other: Rect) -> tuple[float, float, float, float] | None:
    x0 = max(rect.x0, other.x0)
    x1 = min(rect.x1, other.x1)
    y0 = max(rect.y0, other.y0)
    y1 = min(rect.y1, other.y1)
    if x1 - x0 <= EPS or y1 - y0 <= EPS:
        return None
    return x0, x1, y0, y1


def add_bound(value: float, low: float, high: float, bounds: set[float]) -> None:
    if low + EPS < value < high - EPS:
        bounds.add(value)


def subtract_segments(segments_in: list[tuple[float, float]], cut0: float, cut1: float) -> list[tuple[float, float]]:
    result = []
    for s0, s1 in segments_in:
        a0, a1 = max(s0, cut0), min(s1, cut1)
        if a1 - a0 <= EPS:
            result.append((s0, s1))
            continue
        if a0 - s0 > EPS:
            result.append((s0, a0))
        if s1 - a1 > EPS:
            result.append((a1, s1))
    return result


def adjacent_plates(plate: FloorPlate, plates: list[FloorPlate], direction: str) -> list[FloorPlate]:
    if direction == "above":
        z = plate.z + plate.height
        return [p for p in plates if p.building_key == plate.building_key and abs(p.z - z) <= EPS]
    if direction == "below":
        z = plate.z
        return [p for p in plates if p.building_key == plate.building_key and abs(p.z + p.height - z) <= EPS]
    raise ValueError(direction)


def same_level_plates(plate: FloorPlate, plates: list[FloorPlate]) -> list[FloorPlate]:
    return [
        p
        for p in plates
        if p is not plate
        and p.building_key == plate.building_key
        and abs(p.z - plate.z) <= EPS
        and abs(p.height - plate.height) <= EPS
    ]


def rect_covered_by_any(rect: Rect, covers: list[FloorPlate]) -> bool:
    return any(p.rect.contains_rect(rect) for p in covers)


def side_segments(plate: FloorPlate, peers: list[FloorPlate]) -> dict[str, list[tuple[float, float]]]:
    rect = plate.rect
    segments = {
        "south": [(rect.x0, rect.x1)],
        "north": [(rect.x0, rect.x1)],
        "west": [(rect.y0, rect.y1)],
        "east": [(rect.y0, rect.y1)],
    }

    for peer in peers:
        other = peer.rect
        if abs(other.y1 - rect.y0) <= EPS:
            segments["south"] = subtract_segments(segments["south"], other.x0, other.x1)
        if abs(other.y0 - rect.y1) <= EPS:
            segments["north"] = subtract_segments(segments["north"], other.x0, other.x1)
        if abs(other.x1 - rect.x0) <= EPS:
            segments["west"] = subtract_segments(segments["west"], other.y0, other.y1)
        if abs(other.x0 - rect.x1) <= EPS:
            segments["east"] = subtract_segments(segments["east"], other.y0, other.y1)
    return segments


def cell_perimeter_exposures(cell: Rect, plate: FloorPlate, exposed_segments: dict[str, list[tuple[float, float]]], depth: float) -> list[str]:
    rect = plate.rect
    exposures: list[str] = []
    if cell.y0 < rect.y0 + depth - EPS:
        for s0, s1 in exposed_segments["south"]:
            if min(cell.x1, s1) - max(cell.x0, s0) > EPS:
                exposures.append("south_wall")
                break
    if cell.y1 > rect.y1 - depth + EPS:
        for s0, s1 in exposed_segments["north"]:
            if min(cell.x1, s1) - max(cell.x0, s0) > EPS:
                exposures.append("north_wall")
                break
    if cell.x0 < rect.x0 + depth - EPS:
        for s0, s1 in exposed_segments["west"]:
            if min(cell.y1, s1) - max(cell.y0, s0) > EPS:
                exposures.append("west_wall")
                break
    if cell.x1 > rect.x1 - depth + EPS:
        for s0, s1 in exposed_segments["east"]:
            if min(cell.y1, s1) - max(cell.y0, s0) > EPS:
                exposures.append("east_wall")
                break
    return exposures


def shield_segments_with_horizontal_zones(
    plate: FloorPlate,
    exposed_segments: dict[str, list[tuple[float, float]]],
    horizontal_cells: list[Cell],
) -> dict[str, list[tuple[float, float]]]:
    rect = plate.rect
    shielded = {side: list(segments) for side, segments in exposed_segments.items()}
    for cell in horizontal_cells:
        if abs(cell.rect.y0 - rect.y0) <= EPS:
            shielded["south"] = subtract_segments(shielded["south"], cell.rect.x0, cell.rect.x1)
        if abs(cell.rect.y1 - rect.y1) <= EPS:
            shielded["north"] = subtract_segments(shielded["north"], cell.rect.x0, cell.rect.x1)
        if abs(cell.rect.x0 - rect.x0) <= EPS:
            shielded["west"] = subtract_segments(shielded["west"], cell.rect.y0, cell.rect.y1)
        if abs(cell.rect.x1 - rect.x1) <= EPS:
            shielded["east"] = subtract_segments(shielded["east"], cell.rect.y0, cell.rect.y1)
    return shielded


def initial_grid_cells(plate: FloorPlate, plates: list[FloorPlate], perimeter_depth: float) -> list[Rect]:
    rect = plate.rect
    x_bounds = {rect.x0, rect.x1}
    y_bounds = {rect.y0, rect.y1}

    related = adjacent_plates(plate, plates, "above") + adjacent_plates(plate, plates, "below") + same_level_plates(plate, plates)
    for other in related:
        clipped = clipped_overlap_bounds(rect, other.rect)
        if clipped:
            ox0, ox1, oy0, oy1 = clipped
            add_bound(ox0, rect.x0, rect.x1, x_bounds)
            add_bound(ox1, rect.x0, rect.x1, x_bounds)
            add_bound(oy0, rect.y0, rect.y1, y_bounds)
            add_bound(oy1, rect.y0, rect.y1, y_bounds)
        add_bound(other.rect.x0, rect.x0, rect.x1, x_bounds)
        add_bound(other.rect.x1, rect.x0, rect.x1, x_bounds)
        add_bound(other.rect.y0, rect.y0, rect.y1, y_bounds)
        add_bound(other.rect.y1, rect.y0, rect.y1, y_bounds)

    for value in (rect.x0 + perimeter_depth, rect.x1 - perimeter_depth):
        add_bound(value, rect.x0, rect.x1, x_bounds)
    for value in (rect.y0 + perimeter_depth, rect.y1 - perimeter_depth):
        add_bound(value, rect.y0, rect.y1, y_bounds)

    exposed = side_segments(plate, same_level_plates(plate, plates))
    for side, segments in exposed.items():
        for s0, s1 in segments:
            if side in {"south", "north"}:
                add_bound(s0, rect.x0, rect.x1, x_bounds)
                add_bound(s1, rect.x0, rect.x1, x_bounds)
            else:
                add_bound(s0, rect.y0, rect.y1, y_bounds)
                add_bound(s1, rect.y0, rect.y1, y_bounds)

    xs = sorted(x_bounds)
    ys = sorted(y_bounds)
    cells = []
    for x0, x1 in zip(xs, xs[1:]):
        for y0, y1 in zip(ys, ys[1:]):
            if x1 - x0 > EPS and y1 - y0 > EPS:
                cells.append(Rect(x0, y0, x1, y1))
    return cells


def merge_cells(cells: list[Cell]) -> list[Cell]:
    buckets: dict[tuple[str, tuple[str, ...]], list[Rect]] = {}
    for cell in cells:
        buckets.setdefault((cell.category, cell.exposure), []).append(cell.rect)

    merged: list[Cell] = []
    for (category, exposure), rects in buckets.items():
        changed = True
        rects = list(rects)
        while changed:
            changed = False
            next_rects: list[Rect] = []
            used = [False] * len(rects)
            for i, a in enumerate(rects):
                if used[i]:
                    continue
                combined = a
                used[i] = True
                for j in range(i + 1, len(rects)):
                    if used[j]:
                        continue
                    b = rects[j]
                    horizontal = abs(combined.y0 - b.y0) <= EPS and abs(combined.y1 - b.y1) <= EPS and (abs(combined.x1 - b.x0) <= EPS or abs(b.x1 - combined.x0) <= EPS)
                    vertical = abs(combined.x0 - b.x0) <= EPS and abs(combined.x1 - b.x1) <= EPS and (abs(combined.y1 - b.y0) <= EPS or abs(b.y1 - combined.y0) <= EPS)
                    if horizontal or vertical:
                        combined = Rect(min(combined.x0, b.x0), min(combined.y0, b.y0), max(combined.x1, b.x1), max(combined.y1, b.y1))
                        used[j] = True
                        changed = True
                next_rects.append(combined)
            rects = next_rects
        merged.extend(Cell(rect=r, category=category, exposure=exposure) for r in rects)
    return merged


def classify_plate(plate: FloorPlate, plates: list[FloorPlate], perimeter_depth: float) -> list[ZonePart]:
    above = adjacent_plates(plate, plates, "above")
    below = adjacent_plates(plate, plates, "below")
    exposed_segments = side_segments(plate, same_level_plates(plate, plates))
    base_cells: list[tuple[Rect, tuple[str, ...]]] = []
    horizontal_cells: list[Cell] = []

    for rect in initial_grid_cells(plate, plates, perimeter_depth):
        top_exposed = bool(above) and not rect_covered_by_any(rect, above)
        bottom_exposed = plate.z > EPS and bool(below) and not rect_covered_by_any(rect, below)
        exposure: list[str] = []
        if top_exposed:
            exposure.append("top")
        if bottom_exposed:
            exposure.append("bottom")
        if exposure:
            horizontal_cells.append(Cell(rect, "horizontal_exposed_zone", tuple(exposure)))
            continue
        base_cells.append((rect, tuple()))

    effective_segments = shield_segments_with_horizontal_zones(plate, exposed_segments, horizontal_cells)
    cells: list[Cell] = list(horizontal_cells)
    side_band_parts: list[ZonePart] = []
    for side, segments in effective_segments.items():
        exposure = (f"{side}_wall",)
        for segment in segments:
            side_band_parts.append(ZonePart(side_band_polygon(plate.rect, side, segment, perimeter_depth), "perimeter_zone", exposure))

    for rect, _ in base_cells:
        perimeter = cell_perimeter_exposures(rect, plate, effective_segments, perimeter_depth)
        if perimeter:
            continue
        else:
            cells.append(Cell(rect, "interior_zone", tuple()))

    merged_cells = merge_cells(cells)
    horizontal_cells = [cell for cell in merged_cells if cell.category == "horizontal_exposed_zone"]
    if not horizontal_cells:
        parts: list[ZonePart] = list(side_band_parts)
        for cell in merged_cells:
            if cell.category == "interior_zone":
                parts.append(ZonePart(rect_polygon(cell.rect), cell.category, cell.exposure))
        return parts

    parts = list(side_band_parts)
    for cell in merged_cells:
        parts.append(ZonePart(rect_polygon(cell.rect), cell.category, cell.exposure))
    return parts


def partition_model_by_floor(
    model: dict,
    *,
    perimeter_depth: float = 4.0,
    lobby_height: float = 6.0,
    floor_height: float = 4.0,
) -> dict:
    plates = expand_mass_to_floor_plates(model, lobby_height, floor_height)
    out_zones: list[dict] = []
    partition_counts: dict[str, int] = {}

    for plate in plates:
        parts = classify_plate(plate, plates, perimeter_depth)
        seen_rect_keys: set[tuple[float, float, float, float, float, float, str]] = set()
        for index, part in enumerate(parts, start=1):
            partition_counts[part.category] = partition_counts.get(part.category, 0) + 1
            exposure_tag = "_".join(part.exposure) if part.exposure else "core"
            # The EnergyPlus converter currently supports adjacency splitting only for
            # axis-aligned rectangles (box zones). To keep the partition workflow
            # simulation-ready, we approximate any polygon part by its bounding
            # rectangle and dedupe identical rectangles (can happen for corner splits).
            xs = [p[0] for p in part.polygon]
            ys = [p[1] for p in part.polygon]
            rect = Rect(min(xs), min(ys), max(xs), max(ys))
            key = (round3(rect.x0), round3(rect.y0), round3(rect.x1), round3(rect.y1), round3(plate.z), round3(plate.height), part.category)
            if key in seen_rect_keys:
                continue
            seen_rect_keys.add(key)
            add_rect_zone(
                out_zones,
                f"{plate.name}_{part.category}_{exposure_tag}_{index:02d}",
                rect,
                plate.z,
                plate.height,
                category=part.category,
                metadata={
                    "source_zone": plate.source_zone,
                    "floor": plate.floor_index,
                    "partition_type": part.category,
                    "exposure": list(part.exposure),
                    "perimeter_depth": perimeter_depth,
                    "geometry_note": "polygon_part_approximated_to_rect_bbox",
                },
            )

    for zone in model["zones"]:
        if zone.get("category") == "aerial_platform":
            partition_counts["aerial_platform_zone"] = partition_counts.get("aerial_platform_zone", 0) + 1
            add_prism_zone(
                out_zones,
                f"{zone['name']}_zone",
                zone.get("points", []),
                category="aerial_platform_zone",
                metadata={
                    "source_zone": zone["name"],
                    "partition_type": "aerial_platform_zone",
                    "exposure": ["bridge"],
                },
            )

    return {
        "building_name": f"{model.get('building_name', 'Building')} Partitioned",
        "zones": out_zones,
        "metadata": {
            **model.get("metadata", {}),
            "partition": {
                "perimeter_depth": perimeter_depth,
                "lobby_height": lobby_height,
                "floor_height": floor_height,
                "source_zone_count": len(model.get("zones", [])),
                "floor_plate_count": len(plates),
                "partition_zone_count": len(out_zones),
                "counts": partition_counts,
                "note": "Axis-aligned rectangular partitioner; aerial platforms are preserved as independent prism zones.",
            },
        },
    }


def load_or_generate_model(input_path: str | None) -> dict:
    if input_path:
        return json.loads(Path(input_path).read_text(encoding="utf-8"))
    from scripts.generate_20260528 import generate_20260528

    return generate_20260528()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Partition model zones by floor, exposure, perimeter, and interior.")
    parser.add_argument("--input", default=None, help="Input model JSON. Defaults to generate_20260528().")
    parser.add_argument("--output", default="output/generate_20260528_partitioned.json")
    parser.add_argument("--perimeter-depth", type=float, default=4.0)
    parser.add_argument("--lobby-height", type=float, default=6.0)
    parser.add_argument("--floor-height", type=float, default=4.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = load_or_generate_model(args.input)
    partitioned = partition_model_by_floor(
        model,
        perimeter_depth=args.perimeter_depth,
        lobby_height=args.lobby_height,
        floor_height=args.floor_height,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(partitioned, indent=2, ensure_ascii=False), encoding="utf-8")

    counts = partitioned["metadata"]["partition"]["counts"]
    print(f"Wrote {len(partitioned['zones'])} partition zones to {output}")
    for key in sorted(counts):
        print(f"{key}: {counts[key]}")


if __name__ == "__main__":
    main()
