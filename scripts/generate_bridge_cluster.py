"""Generate a platform-connected cluster office massing inspired by stacked urban blocks.

The model uses three medium-size office blocks around a central public plaza.
All three blocks land on the ground. Instead of thin sky bridges, a one-story
public platform connects the buildings into a coherent campus while keeping
the total area within the assignment target.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


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


def floor_z(floor_index: int, lobby_height: float, floor_height: float) -> float:
    if floor_index == 1:
        return 0.0
    return lobby_height + (floor_index - 2) * floor_height


def inset_plate(
    x: float,
    y: float,
    length: float,
    width: float,
    floor_index: int,
    max_floors: int,
    terrace_depth: float,
) -> tuple[float, float, float, float]:
    """Create a gentle stepped profile for upper terraces."""
    if floor_index <= 2:
        inset = 0.0
    elif floor_index >= max_floors - 1:
        inset = terrace_depth
    else:
        inset = terrace_depth * 0.5
    return x + inset, y + inset, max(4.0, length - inset * 2), max(4.0, width - inset * 2)


def add_office_stack(
    zones: list[dict],
    prefix: str,
    x: float,
    y: float,
    length: float,
    width: float,
    floors: int,
    lobby_height: float,
    floor_height: float,
    terrace_depth: float,
    skip_ground: bool,
) -> None:
    start_floor = 2 if skip_ground else 1
    for floor_index in range(start_floor, floors + 1):
        z = floor_z(floor_index, lobby_height, floor_height)
        height = lobby_height if floor_index == 1 else floor_height
        px, py, pl, pw = inset_plate(x, y, length, width, floor_index, floors, terrace_depth)
        add_zone(
            zones,
            f"F{floor_index:02d}_{prefix}",
            px,
            py,
            z,
            pl,
            pw,
            height,
            category="mass_block",
        )


def add_platform(
    zones: list[dict],
    name: str,
    x: float,
    y: float,
    length: float,
    width: float,
    height: float,
) -> None:
    add_zone(zones, f"F01_{name}", x, y, 0.0, length, width, height, category="platform")


def generate_bridge_cluster(
    *,
    building_name: str = "Platform Cluster Office",
    site_size: float = 100.0,
    max_floors: int = 9,
    lobby_height: float = 6.0,
    floor_height: float = 4.0,
    floor_plate_efficiency: float = 1.0,
    west_x: float = 12.0,
    west_y: float = 18.0,
    west_length: float = 30.0,
    west_width: float = 24.0,
    west_floors: int = 6,
    east_x: float = 60.0,
    east_y: float = 26.0,
    east_length: float = 28.0,
    east_width: float = 24.0,
    east_floors: int = 7,
    north_x: float = 34.0,
    north_y: float = 64.0,
    north_length: float = 28.0,
    north_width: float = 22.0,
    north_floors: int = 5,
    terrace_depth: float = 3.0,
    platform_depth: float = 22.0,
    platform_width: float = 16.0,
    skip_west_ground: bool = False,
    skip_east_ground: bool = False,
    skip_north_ground: bool = False,
    add_open_space_markers: bool = True,
) -> dict:
    if max_floors > 10:
        raise ValueError("max_floors must be 10 or less")
    if lobby_height + (max_floors - 1) * floor_height >= 50:
        raise ValueError("building height must be under 50m")
    if not 0.4 <= floor_plate_efficiency <= 1.0:
        raise ValueError("floor_plate_efficiency must be between 0.4 and 1.0")

    blocks = [
        ("west_block", west_x, west_y, west_length, west_width, west_floors),
        ("east_block", east_x, east_y, east_length, east_width, east_floors),
        ("north_block", north_x, north_y, north_length, north_width, north_floors),
    ]
    for name, x, y, length, width, floors in blocks:
        if floors > max_floors:
            raise ValueError(f"{name} floors exceed max_floors")
        if x < 0 or y < 0 or x + length > site_size or y + width > site_size:
            raise ValueError(f"{name} exceeds site bounds")
        if length < 18 or width < 16:
            raise ValueError(f"{name} is too small for a major office block")

    zones: list[dict] = []
    add_office_stack(zones, "west_block", west_x, west_y, west_length, west_width, west_floors, lobby_height, floor_height, terrace_depth, skip_west_ground)
    add_office_stack(zones, "east_block", east_x, east_y, east_length, east_width, east_floors, lobby_height, floor_height, terrace_depth, skip_east_ground)
    add_office_stack(zones, "north_block", north_x, north_y, north_length, north_width, north_floors, lobby_height, floor_height, terrace_depth, skip_north_ground)

    platform_x = max(0.0, min(north_x, west_x + west_length - platform_width * 0.5))
    platform_y = west_y + west_width
    add_platform(
        zones,
        "shared_ground_platform",
        platform_x,
        platform_y,
        max(platform_width, east_x - platform_x),
        max(platform_depth, north_y - platform_y),
        lobby_height,
    )

    add_zone(
        zones,
        "central_open_plaza_reference",
        34.0,
        34.0,
        0.0,
        28.0,
        26.0,
        0.12,
        category="open_space_reference",
    )
    if add_open_space_markers:
        add_zone(
            zones,
            "southwest_urban_entry_reference",
            8.0,
            8.0,
            0.0,
            30.0,
            16.0,
            0.12,
            category="open_space_reference",
        )
        add_zone(
            zones,
            "northeast_digital_block_route_reference",
            62.0,
            62.0,
            0.0,
            28.0,
            16.0,
            0.12,
            category="open_space_reference",
        )

    model = {"building_name": building_name, "zones": zones}
    model["metadata"] = {
        "site_size": site_size,
        "max_floors": max_floors,
        "floor_plate_efficiency": floor_plate_efficiency,
        "platform_strategy": "one-story shared ground platform connecting the three grounded buildings",
        "gross_area_adjustment": "zone box floor areas are used directly when efficiency is 1.0",
    }
    return model


def gross_area(model: dict) -> float:
    efficiency = model.get("metadata", {}).get("floor_plate_efficiency", 1.0)
    return sum(
        z["dimensions"]["length"] * z["dimensions"]["width"] * efficiency
        for z in model["zones"]
        if z.get("category") != "open_space_reference" and z["dimensions"]["height"] > 1.0
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a three-block platform-connected office massing.")
    parser.add_argument("--name", default="Platform Cluster Office")
    parser.add_argument("--output", default="output/platform_cluster_office.json")
    parser.add_argument("--site-size", type=float, default=100.0)
    parser.add_argument("--max-floors", type=int, default=9)
    parser.add_argument("--lobby-height", type=float, default=6.0)
    parser.add_argument("--floor-height", type=float, default=4.0)
    parser.add_argument("--floor-plate-efficiency", type=float, default=1.0)
    parser.add_argument("--west-floors", type=int, default=6)
    parser.add_argument("--east-floors", type=int, default=7)
    parser.add_argument("--north-floors", type=int, default=5)
    parser.add_argument("--platform-depth", type=float, default=22.0)
    parser.add_argument("--platform-width", type=float, default=16.0)
    parser.add_argument("--terrace-depth", type=float, default=3.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = generate_bridge_cluster(
        building_name=args.name,
        site_size=args.site_size,
        max_floors=args.max_floors,
        lobby_height=args.lobby_height,
        floor_height=args.floor_height,
        floor_plate_efficiency=args.floor_plate_efficiency,
        west_floors=args.west_floors,
        east_floors=args.east_floors,
        north_floors=args.north_floors,
        platform_depth=args.platform_depth,
        platform_width=args.platform_width,
        terrace_depth=args.terrace_depth,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(model, indent=2, ensure_ascii=False), encoding="utf-8")
    height = max(
        z["origin"]["z"] + z["dimensions"]["height"]
        for z in model["zones"]
        if z.get("category") != "open_space_reference" and z["dimensions"]["height"] > 1.0
    )
    area = gross_area(model)
    print(f"Wrote {len(model['zones'])} zones to {output}")
    print(f"Adjusted gross floor area: {area:.1f} sqm")
    print(f"Building height: {height:.1f} m")
    print(f"Area target status: {'OK' if 9000 <= area <= 11000 else 'OUT_OF_RANGE'}")


if __name__ == "__main__":
    main()
