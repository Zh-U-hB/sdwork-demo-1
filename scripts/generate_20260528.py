"""Generate three setback-bound, offset grouped office blocks.

The model has three buildings with fixed floor counts: low/mid/high are
4/5/6 floors. Their single-floor areas are derived from a total floor area and
the 1.2:1.0:0.8 low-to-high footprint ratio. Each building is split into
two-floor vertical groups, and upper groups are offset from the bottom group by
an angle and distance while passing setback, collision, and support checks.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


EPS = 1e-6


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
    category: str = "mass_block",
) -> None:
    if length <= 0 or width <= 0 or height <= 0:
        return
    x0, y0, z0 = round(x, 3), round(y, 3), round(z, 3)
    x1 = round(x + length, 3)
    y1 = round(y + width, 3)
    z1 = round(z + height, 3)
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


def add_prism_zone(
    zones: list[dict],
    name: str,
    footprint: list[tuple[float, float]],
    z: float,
    height: float,
    *,
    category: str = "aerial_platform",
) -> None:
    if len(footprint) != 4 or height <= 0:
        return
    xs = [point[0] for point in footprint]
    ys = [point[1] for point in footprint]
    x0, y0, z0 = round(min(xs), 3), round(min(ys), 3), round(z, 3)
    x1, y1, z1 = round(max(xs), 3), round(max(ys), 3), round(z + height, 3)
    bottom = [{"x": round(x, 3), "y": round(y, 3), "z": z0} for x, y in footprint]
    top = [{"x": point["x"], "y": point["y"], "z": z1} for point in bottom]
    zones.append({
        "name": name,
        "category": category,
        "origin": {"x": x0, "y": y0, "z": z0},
        "dimensions": {
            "length": round(x1 - x0, 3),
            "width": round(y1 - y0, 3),
            "height": round(height, 3),
        },
        "points": bottom + top,
    })


def clamp(value: float, low: float, high: float) -> float:
    return min(max(value, low), high)


def floor_z(floor_index: int, lobby_height: float, floor_height: float) -> float:
    if floor_index == 1:
        return 0.0
    return lobby_height + (floor_index - 2) * floor_height


def grouped_floors(floors: int, group_size: int) -> list[tuple[int, int]]:
    groups = []
    start = 1
    while start <= floors:
        count = min(group_size, floors - start + 1)
        groups.append((start, count))
        start += count
    return groups


def group_height(start_floor: int, floor_count: int, lobby_height: float, floor_height: float) -> float:
    height = 0.0
    for floor_index in range(start_floor, start_floor + floor_count):
        height += lobby_height if floor_index == 1 else floor_height
    return height


def rect_from_area(area: float, aspect_ratio: float) -> tuple[float, float]:
    if area <= 0:
        raise ValueError("area must be positive")
    if aspect_ratio <= 0:
        raise ValueError("aspect_ratio must be positive")
    return math.sqrt(area * aspect_ratio), math.sqrt(area / aspect_ratio)


def offset_xy(angle_degrees: float, distance: float, group_index: int) -> tuple[float, float]:
    angle = math.radians(angle_degrees)
    scale = distance * group_index
    return math.cos(angle) * scale, math.sin(angle) * scale


def boundary_point(
    s: float,
    x_min: float,
    y_min: float,
    x_max: float,
    y_max: float,
) -> tuple[str, float, float]:
    length = x_max - x_min
    width = y_max - y_min
    perimeter = 2 * (length + width)
    s = s % perimeter
    if s < length:
        return "south", x_min + s, y_min
    s -= length
    if s < width:
        return "east", x_max, y_min + s
    s -= width
    if s < length:
        return "north", x_max - s, y_max
    s -= length
    return "west", x_min, y_max - s


def place_on_boundary(
    side: str,
    px: float,
    py: float,
    length: float,
    width: float,
    x_min: float,
    y_min: float,
    x_max: float,
    y_max: float,
) -> tuple[float, float]:
    if side == "south":
        return clamp(px - length / 2, x_min, x_max - length), y_min
    if side == "east":
        return x_max - length, clamp(py - width / 2, y_min, y_max - width)
    if side == "north":
        return clamp(px - length / 2, x_min, x_max - length), y_max - width
    if side == "west":
        return x_min, clamp(py - width / 2, y_min, y_max - width)
    raise ValueError(f"unsupported boundary side: {side}")


def offset_bounds(floors: int, group_size: int, offset_angle: float, offset_distance: float) -> tuple[float, float, float, float]:
    offsets = [
        offset_xy(offset_angle, offset_distance, group_index)
        for group_index, _ in enumerate(grouped_floors(floors, group_size))
    ]
    dxs = [dx for dx, _ in offsets]
    dys = [dy for _, dy in offsets]
    return min(dxs), max(dxs), min(dys), max(dys)


def clamp_origin_for_offset_groups(
    x: float,
    y: float,
    length: float,
    width: float,
    floors: int,
    group_size: int,
    offset_angle: float,
    offset_distance: float,
    x_min: float,
    y_min: float,
    x_max: float,
    y_max: float,
) -> tuple[float, float]:
    min_dx, max_dx, min_dy, max_dy = offset_bounds(floors, group_size, offset_angle, offset_distance)
    low_x = x_min - min_dx
    high_x = x_max - length - max_dx
    low_y = y_min - min_dy
    high_y = y_max - width - max_dy
    if low_x > high_x + EPS or low_y > high_y + EPS:
        raise ValueError("offset groups do not fit within buildable area")
    return clamp(x, low_x, high_x), clamp(y, low_y, high_y)


def overlap_1d(a0: float, a1: float, b0: float, b1: float) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))


def box_bounds(zone: dict) -> tuple[float, float, float, float, float, float]:
    origin = zone["origin"]
    dims = zone["dimensions"]
    x0 = origin["x"]
    y0 = origin["y"]
    z0 = origin["z"]
    return (
        x0,
        x0 + dims["length"],
        y0,
        y0 + dims["width"],
        z0,
        z0 + dims["height"],
    )


def plan_overlap_area(a: dict, b: dict) -> float:
    ax0, ax1, ay0, ay1, _, _ = box_bounds(a)
    bx0, bx1, by0, by1, _, _ = box_bounds(b)
    return overlap_1d(ax0, ax1, bx0, bx1) * overlap_1d(ay0, ay1, by0, by1)


def zone_rect(zone: dict) -> tuple[float, float, float, float]:
    x0, x1, y0, y1, _, _ = box_bounds(zone)
    return x0, y0, x1, y1


def rect_edges(rect: tuple[float, float, float, float]) -> list[dict]:
    x0, y0, x1, y1 = rect
    return [
        {"side": "south", "orientation": "h", "p0": (x0, y0), "p1": (x1, y0)},
        {"side": "east", "orientation": "v", "p0": (x1, y0), "p1": (x1, y1)},
        {"side": "north", "orientation": "h", "p0": (x1, y1), "p1": (x0, y1)},
        {"side": "west", "orientation": "v", "p0": (x0, y1), "p1": (x0, y0)},
    ]


def segment_distance(a: dict, b: dict) -> float:
    ax0, ay0 = a["p0"]
    ax1, ay1 = a["p1"]
    bx0, by0 = b["p0"]
    bx1, by1 = b["p1"]
    if a["orientation"] == "h":
        a_min, a_max = sorted((ax0, ax1))
        b_min, b_max = sorted((bx0, bx1))
        gap = max(0.0, max(a_min, b_min) - min(a_max, b_max))
        return math.hypot(gap, ay0 - by0)
    a_min, a_max = sorted((ay0, ay1))
    b_min, b_max = sorted((by0, by1))
    gap = max(0.0, max(a_min, b_min) - min(a_max, b_max))
    return math.hypot(ax0 - bx0, gap)


def nearest_parallel_edges(source_zone: dict, target_zone: dict) -> tuple[dict, dict]:
    source_edges = rect_edges(zone_rect(source_zone))
    target_edges = rect_edges(zone_rect(target_zone))
    candidates = [
        (segment_distance(source_edge, target_edge), source_edge, target_edge)
        for source_edge in source_edges
        for target_edge in target_edges
        if source_edge["orientation"] == target_edge["orientation"]
    ]
    _, source_edge, target_edge = min(candidates, key=lambda item: item[0])
    return source_edge, target_edge


def closest_point_from_edge_to_edge(point_edge: dict, reference_edge: dict) -> tuple[float, float]:
    p0 = point_edge["p0"]
    p1 = point_edge["p1"]
    r0 = reference_edge["p0"]
    r1 = reference_edge["p1"]
    if point_edge["orientation"] == "h":
        target_x = clamp((r0[0] + r1[0]) / 2, min(p0[0], p1[0]), max(p0[0], p1[0]))
        return target_x, p0[1]
    target_y = clamp((r0[1] + r1[1]) / 2, min(p0[1], p1[1]), max(p0[1], p1[1]))
    return p0[0], target_y


def move_along_edge(edge: dict, start: tuple[float, float], distance: float) -> tuple[float, float]:
    p0 = edge["p0"]
    p1 = edge["p1"]
    vx = p1[0] - p0[0]
    vy = p1[1] - p0[1]
    length = math.hypot(vx, vy)
    if length <= EPS:
        return start
    ux, uy = vx / length, vy / length
    forward = (start[0] + ux * distance, start[1] + uy * distance)
    backward = (start[0] - ux * distance, start[1] - uy * distance)

    def on_edge(point: tuple[float, float]) -> bool:
        x, y = point
        return (
            min(p0[0], p1[0]) - EPS <= x <= max(p0[0], p1[0]) + EPS
            and min(p0[1], p1[1]) - EPS <= y <= max(p0[1], p1[1]) + EPS
        )

    if on_edge(forward):
        return forward
    if on_edge(backward):
        return backward
    return p1 if math.dist(start, p1) >= math.dist(start, p0) else p0


def platform_footprint(
    source_edge: dict,
    target_start: tuple[float, float],
    target_end: tuple[float, float],
) -> list[tuple[float, float]]:
    source_p0 = source_edge["p0"]
    source_p1 = source_edge["p1"]
    axis = 0 if source_edge["orientation"] == "h" else 1
    target_points = sorted([target_start, target_end], key=lambda point: point[axis])
    if source_p0[axis] <= source_p1[axis]:
        target_for_p0, target_for_p1 = target_points[0], target_points[1]
    else:
        target_for_p0, target_for_p1 = target_points[1], target_points[0]
    return [source_p0, source_p1, target_for_p1, target_for_p0]


def zone_contains_floor(zone: dict, floor_index: int, lobby_height: float, floor_height: float) -> bool:
    z0 = zone["origin"]["z"]
    z1 = z0 + zone["dimensions"]["height"]
    floor_bottom = floor_z(floor_index, lobby_height, floor_height)
    floor_top = floor_bottom + (lobby_height if floor_index == 1 else floor_height)
    return z0 <= floor_bottom + EPS and z1 >= floor_top - EPS


def building_zone_at_floor(
    zones: list[dict],
    building_name: str,
    floor_index: int,
    lobby_height: float,
    floor_height: float,
) -> dict:
    for zone in zones:
        if zone.get("category") == "mass_block" and zone["name"].startswith(building_name):
            if zone_contains_floor(zone, floor_index, lobby_height, floor_height):
                return zone
    raise ValueError(f"cannot find {building_name} at floor {floor_index}")


def make_aerial_platform(
    zones: list[dict],
    *,
    name: str,
    source_building: str,
    target_building: str,
    floor_index: int,
    edge_walk_distance: float,
    lobby_height: float,
    floor_height: float,
) -> dict:
    source_zone = building_zone_at_floor(zones, source_building, floor_index, lobby_height, floor_height)
    target_zone = building_zone_at_floor(zones, target_building, floor_index, lobby_height, floor_height)
    source_edge, target_edge = nearest_parallel_edges(source_zone, target_zone)
    target_start = closest_point_from_edge_to_edge(target_edge, source_edge)
    target_end = move_along_edge(target_edge, target_start, edge_walk_distance)
    footprint = platform_footprint(source_edge, target_start, target_end)
    add_prism_zone(
        zones,
        name,
        footprint,
        floor_z(floor_index, lobby_height, floor_height),
        floor_height,
        category="aerial_platform",
    )
    return {
        "name": name,
        "floor": floor_index,
        "source_building": source_building,
        "target_building": target_building,
        "source_zone": source_zone["name"],
        "target_zone": target_zone["name"],
        "source_edge": source_edge["side"],
        "target_edge": target_edge["side"],
        "edge_walk_distance": edge_walk_distance,
        "footprint": [{"x": round(x, 3), "y": round(y, 3)} for x, y in footprint],
    }


def validate_setbacks(zones: list[dict], x_min: float, y_min: float, x_max: float, y_max: float) -> None:
    for zone in zones:
        x0, x1, y0, y1, _, _ = box_bounds(zone)
        if x0 < x_min - EPS or y0 < y_min - EPS or x1 > x_max + EPS or y1 > y_max + EPS:
            raise ValueError(f"{zone['name']} violates setback bounds")


def validate_collisions(zones: list[dict]) -> None:
    for index, a in enumerate(zones):
        ax0, ax1, ay0, ay1, az0, az1 = box_bounds(a)
        for b in zones[index + 1:]:
            bx0, bx1, by0, by1, bz0, bz1 = box_bounds(b)
            ox = overlap_1d(ax0, ax1, bx0, bx1)
            oy = overlap_1d(ay0, ay1, by0, by1)
            oz = overlap_1d(az0, az1, bz0, bz1)
            if ox > EPS and oy > EPS and oz > EPS:
                raise ValueError(f"{a['name']} collides with {b['name']}")


def validate_support(zones: list[dict], min_support_overlap_ratio: float) -> None:
    if not 0.0 <= min_support_overlap_ratio <= 1.0:
        raise ValueError("min_support_overlap_ratio must be between 0 and 1")

    for zone in zones:
        if zone.get("category") == "aerial_platform":
            continue
        x0, x1, y0, y1, z0, _ = box_bounds(zone)
        if z0 <= EPS:
            continue

        footprint = (x1 - x0) * (y1 - y0)
        supported_area = 0.0
        for lower in zones:
            if lower is zone:
                continue
            *_, lower_top = box_bounds(lower)
            if abs(lower_top - z0) <= EPS:
                supported_area += plan_overlap_area(zone, lower)

        if footprint <= EPS or supported_area / footprint + EPS < min_support_overlap_ratio:
            raise ValueError(
                f"{zone['name']} is insufficiently supported "
                f"({supported_area / footprint:.1%} < {min_support_overlap_ratio:.1%})"
            )


def validate_platform_attachments(zones: list[dict]) -> None:
    mass_zones = [zone for zone in zones if zone.get("category") == "mass_block"]
    for platform in [zone for zone in zones if zone.get("category") == "aerial_platform"]:
        px0, px1, py0, py1, pz0, pz1 = box_bounds(platform)
        contacts = 0
        for mass in mass_zones:
            mx0, mx1, my0, my1, mz0, mz1 = box_bounds(mass)
            if overlap_1d(pz0, pz1, mz0, mz1) <= EPS:
                continue
            vertical_touch = (
                abs(px0 - mx1) <= EPS or abs(px1 - mx0) <= EPS
            ) and overlap_1d(py0, py1, my0, my1) > EPS
            horizontal_touch = (
                abs(py0 - my1) <= EPS or abs(py1 - my0) <= EPS
            ) and overlap_1d(px0, px1, mx0, mx1) > EPS
            if vertical_touch or horizontal_touch:
                contacts += 1
        if contacts < 2:
            raise ValueError(f"{platform['name']} is not attached to two building masses")


def make_building_groups(
    zones: list[dict],
    *,
    name: str,
    floors: int,
    x: float,
    y: float,
    length: float,
    width: float,
    lobby_height: float,
    floor_height: float,
    group_size: int,
    offset_angle: float,
    offset_distance: float,
) -> None:
    for group_index, (start_floor, floor_count) in enumerate(grouped_floors(floors, group_size)):
        dx, dy = offset_xy(offset_angle, offset_distance, group_index)
        add_zone(
            zones,
            f"{name}_G{group_index + 1:02d}_F{start_floor:02d}_to_F{start_floor + floor_count - 1:02d}",
            x + dx,
            y + dy,
            floor_z(start_floor, lobby_height, floor_height),
            length,
            width,
            group_height(start_floor, floor_count, lobby_height, floor_height),
            category="mass_block",
        )


def generate_20260528(
    *,
    building_name: str = "Boundary Offset Three Block Office",
    site_size: float = 100.0,
    total_area: float = 10000.0,
    lobby_height: float = 6.0,
    floor_height: float = 4.0,
    setback_south: float = 15.0,
    setback_west: float = 15.0,
    setback_north: float = 10.0,
    setback_east: float = 10.0,
    low_aspect_ratio: float = 1.0,
    mid_aspect_ratio: float = 1.0,
    high_aspect_ratio: float = 1.0,
    boundary_shift: float = 40.0,
    group_size: int = 2,
    low_offset_angle: float = 45.0,
    mid_offset_angle: float = 180.0,
    high_offset_angle: float = 315.0,
    low_offset_distance: float = 2.0,
    mid_offset_distance: float = 2.0,
    high_offset_distance: float = 2.0,
    min_support_overlap_ratio: float = 0.5,
    add_aerial_platforms: bool = True,
    platform_edge_walk_distance: float = 5.0,
    add_open_space_markers: bool = True,
) -> dict:
    if site_size <= 0:
        raise ValueError("site_size must be positive")
    if total_area <= 0:
        raise ValueError("total_area must be positive")
    if group_size < 1:
        raise ValueError("group_size must be at least 1")
    if platform_edge_walk_distance <= 0:
        raise ValueError("platform_edge_walk_distance must be positive")

    buildings = [
        ("low_block", 4, 1.2, low_aspect_ratio, low_offset_angle, low_offset_distance),
        ("mid_block", 5, 1.0, mid_aspect_ratio, mid_offset_angle, mid_offset_distance),
        ("high_block", 6, 0.8, high_aspect_ratio, high_offset_angle, high_offset_distance),
    ]
    max_floors = max(b[1] for b in buildings)
    if lobby_height + (max_floors - 1) * floor_height >= 50:
        raise ValueError("building height must be under 50m")

    x_min = setback_west
    y_min = setback_south
    x_max = site_size - setback_east
    y_max = site_size - setback_north
    if x_max <= x_min or y_max <= y_min:
        raise ValueError("setbacks leave no buildable area")

    weighted_floors = sum(floors * area_ratio for _, floors, area_ratio, *_ in buildings)
    base_area = total_area / weighted_floors

    perimeter = 2 * ((x_max - x_min) + (y_max - y_min))
    spacing = perimeter / len(buildings)
    zones: list[dict] = []
    building_metadata = []

    for index, (name, floors, area_ratio, aspect_ratio, offset_angle, offset_distance) in enumerate(buildings):
        floor_area = base_area * area_ratio
        length, width = rect_from_area(floor_area, aspect_ratio)
        if length > x_max - x_min or width > y_max - y_min:
            raise ValueError(f"{name} footprint does not fit within buildable area")

        side, px, py = boundary_point(boundary_shift + spacing * index, x_min, y_min, x_max, y_max)
        x, y = place_on_boundary(side, px, py, length, width, x_min, y_min, x_max, y_max)
        x, y = clamp_origin_for_offset_groups(
            x,
            y,
            length,
            width,
            floors,
            group_size,
            offset_angle,
            offset_distance,
            x_min,
            y_min,
            x_max,
            y_max,
        )
        make_building_groups(
            zones,
            name=name,
            floors=floors,
            x=x,
            y=y,
            length=length,
            width=width,
            lobby_height=lobby_height,
            floor_height=floor_height,
            group_size=group_size,
            offset_angle=offset_angle,
            offset_distance=offset_distance,
        )
        building_metadata.append({
            "name": name,
            "floors": floors,
            "floor_area": round(floor_area, 3),
            "length": round(length, 3),
            "width": round(width, 3),
            "aspect_ratio": aspect_ratio,
            "boundary_side": side,
            "boundary_s": round((boundary_shift + spacing * index) % perimeter, 3),
            "base_origin": {"x": round(x, 3), "y": round(y, 3)},
            "offset_angle": offset_angle,
            "offset_distance": offset_distance,
        })

    validate_setbacks(zones, x_min, y_min, x_max, y_max)
    validate_collisions(zones)
    validate_support(zones, min_support_overlap_ratio)

    platform_metadata = []
    if add_aerial_platforms:
        platform_metadata.append(make_aerial_platform(
            zones,
            name="aerial_platform_high_to_low_F03",
            source_building="high_block",
            target_building="low_block",
            floor_index=3,
            edge_walk_distance=platform_edge_walk_distance,
            lobby_height=lobby_height,
            floor_height=floor_height,
        ))
        platform_metadata.append(make_aerial_platform(
            zones,
            name="aerial_platform_high_to_mid_F04",
            source_building="high_block",
            target_building="mid_block",
            floor_index=4,
            edge_walk_distance=platform_edge_walk_distance,
            lobby_height=lobby_height,
            floor_height=floor_height,
        ))
        validate_setbacks(zones, x_min, y_min, x_max, y_max)
        validate_platform_attachments(zones)

    if add_open_space_markers:
        add_zone(
            zones,
            "buildable_area_reference",
            x_min,
            y_min,
            0.0,
            x_max - x_min,
            y_max - y_min,
            0.12,
            category="open_space_reference",
        )

    return {
        "building_name": building_name,
        "zones": zones,
        "metadata": {
            "site_size": site_size,
            "total_area": total_area,
            "setbacks": {
                "south": setback_south,
                "west": setback_west,
                "north": setback_north,
                "east": setback_east,
            },
            "buildable_bounds": {
                "x_min": x_min,
                "y_min": y_min,
                "x_max": x_max,
                "y_max": y_max,
            },
            "boundary_shift": boundary_shift,
            "boundary_spacing": spacing,
            "group_size": group_size,
            "min_support_overlap_ratio": min_support_overlap_ratio,
            "platform_edge_walk_distance": platform_edge_walk_distance,
            "buildings": building_metadata,
            "aerial_platforms": platform_metadata,
        },
    }


def gross_area(model: dict) -> float:
    buildings = model.get("metadata", {}).get("buildings", [])
    if buildings:
        return sum(building["floor_area"] * building["floors"] for building in buildings)
    return sum(
        zone["dimensions"]["length"] * zone["dimensions"]["width"]
        for zone in model["zones"]
        if zone.get("category") == "mass_block"
    )


def model_height(model: dict) -> float:
    return max(
        zone["origin"]["z"] + zone["dimensions"]["height"]
        for zone in model["zones"]
        if zone.get("category") == "mass_block"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate three boundary-setback offset grouped office blocks.")
    parser.add_argument("--name", default="Boundary Offset Three Block Office")
    parser.add_argument("--output", default="output/generate_20260528.json")
    parser.add_argument("--site-size", type=float, default=100.0)
    parser.add_argument("--total-area", type=float, default=10000.0)
    parser.add_argument("--lobby-height", type=float, default=6.0)
    parser.add_argument("--floor-height", type=float, default=4.0)
    parser.add_argument("--setback-south", type=float, default=15.0)
    parser.add_argument("--setback-west", type=float, default=15.0)
    parser.add_argument("--setback-north", type=float, default=10.0)
    parser.add_argument("--setback-east", type=float, default=10.0)
    parser.add_argument("--low-aspect-ratio", type=float, default=1.0)
    parser.add_argument("--mid-aspect-ratio", type=float, default=1.0)
    parser.add_argument("--high-aspect-ratio", type=float, default=1.0)
    parser.add_argument("--boundary-shift", type=float, default=40.0)
    parser.add_argument("--group-size", type=int, default=2)
    parser.add_argument("--low-offset-angle", type=float, default=45.0)
    parser.add_argument("--mid-offset-angle", type=float, default=180.0)
    parser.add_argument("--high-offset-angle", type=float, default=315.0)
    parser.add_argument("--low-offset-distance", type=float, default=2.0)
    parser.add_argument("--mid-offset-distance", type=float, default=2.0)
    parser.add_argument("--high-offset-distance", type=float, default=2.0)
    parser.add_argument("--min-support-overlap-ratio", type=float, default=0.5)
    parser.add_argument("--no-aerial-platforms", action="store_true")
    parser.add_argument("--platform-edge-walk-distance", type=float, default=5.0)
    parser.add_argument("--no-open-space-markers", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = generate_20260528(
        building_name=args.name,
        site_size=args.site_size,
        total_area=args.total_area,
        lobby_height=args.lobby_height,
        floor_height=args.floor_height,
        setback_south=args.setback_south,
        setback_west=args.setback_west,
        setback_north=args.setback_north,
        setback_east=args.setback_east,
        low_aspect_ratio=args.low_aspect_ratio,
        mid_aspect_ratio=args.mid_aspect_ratio,
        high_aspect_ratio=args.high_aspect_ratio,
        boundary_shift=args.boundary_shift,
        group_size=args.group_size,
        low_offset_angle=args.low_offset_angle,
        mid_offset_angle=args.mid_offset_angle,
        high_offset_angle=args.high_offset_angle,
        low_offset_distance=args.low_offset_distance,
        mid_offset_distance=args.mid_offset_distance,
        high_offset_distance=args.high_offset_distance,
        min_support_overlap_ratio=args.min_support_overlap_ratio,
        add_aerial_platforms=not args.no_aerial_platforms,
        platform_edge_walk_distance=args.platform_edge_walk_distance,
        add_open_space_markers=not args.no_open_space_markers,
    )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(model, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {len(model['zones'])} zones to {output}")
    print(f"Target gross floor area: {args.total_area:.1f} sqm")
    print(f"Generated mass floor area: {gross_area(model):.1f} sqm")
    print(f"Building height: {model_height(model):.1f} m")
    print("Validation: OK")


if __name__ == "__main__":
    main()
