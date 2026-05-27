"""Generate a corner-block-to-cantilevered-L office massing JSON.

Ground floors contain two or three large public blocks concentrated near the
northwest and southeast site corners, preserving a southwest-to-northeast open
route. Upper floors gradually bridge and merge into a cantilevered L-shaped
office mass.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


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


def add_rect_zone(
    zones: list[dict],
    name: str,
    rect: tuple[float, float, float, float],
    z: float,
    height: float,
    *,
    category: str = "mass",
) -> None:
    x, y, length, width = rect
    add_zone(zones, name, x, y, z, length, width, height, category=category)


def inset_rect(x: float, y: float, length: float, width: float, inset: float) -> tuple[float, float, float, float]:
    return x + inset, y + inset, max(0.0, length - inset * 2), max(0.0, width - inset * 2)


def blend_rect(
    start: tuple[float, float, float, float],
    end: tuple[float, float, float, float],
    t: float,
) -> tuple[float, float, float, float]:
    return tuple(lerp(a, b, t) for a, b in zip(start, end))


def l_rects(
    *,
    x: float,
    y: float,
    arm_width: float,
    east_length: float,
    north_length: float,
) -> dict[str, tuple[float, float, float, float]]:
    return {
        "corner": (x, y, arm_width, arm_width),
        "east_arm": (x + arm_width, y, east_length - arm_width, arm_width),
        "north_arm": (x, y + arm_width, arm_width, north_length - arm_width),
    }


def generate_corner_path_l(
    *,
    building_name: str = "Corner Path L Office",
    site_size: float = 100.0,
    floors: int = 9,
    lobby_height: float = 6.0,
    floor_height: float = 4.0,
    ground_blocks: int = 3,
    min_ground_block_length: float = 20.0,
    min_ground_block_width: float = 16.0,
    northwest_x: float = 10.0,
    northwest_y: float = 66.0,
    northwest_length: float = 28.0,
    northwest_width: float = 22.0,
    southeast_x: float = 64.0,
    southeast_y: float = 10.0,
    southeast_length: float = 24.0,
    southeast_width: float = 22.0,
    third_block_x: float = 58.0,
    third_block_y: float = 42.0,
    third_block_length: float = 20.0,
    third_block_width: float = 16.0,
    l_origin_x: float = 18.0,
    l_origin_y: float = 18.0,
    l_arm_width: float = 11.5,
    l_east_length: float = 48.0,
    l_north_length: float = 48.0,
    merge_start_floor: int = 2,
    solid_l_start_floor: int = 7,
    cantilever: float = 8.0,
    open_route_width: float = 18.0,
    add_open_space_markers: bool = True,
) -> dict:
    if floors < 4:
        raise ValueError("floors must be at least 4")
    if lobby_height + (floors - 1) * floor_height >= 50.0:
        raise ValueError("height must be under 50m")
    if ground_blocks not in {2, 3}:
        raise ValueError("ground_blocks must be 2 or 3")
    if not (1 <= merge_start_floor < solid_l_start_floor <= floors):
        raise ValueError("merge_start_floor must be below solid_l_start_floor")

    ground_rects = {
        "nw_public_hall": (northwest_x, northwest_y, northwest_length, northwest_width),
        "se_public_hall": (southeast_x, southeast_y, southeast_length, southeast_width),
    }
    if ground_blocks == 3:
        ground_rects["east_shared_hall"] = (third_block_x, third_block_y, third_block_length, third_block_width)

    for name, (_, _, length, width) in ground_rects.items():
        if length < min_ground_block_length or width < min_ground_block_width:
            raise ValueError(f"{name} is smaller than the minimum large-space block size")

    for name, (x, y, length, width) in ground_rects.items():
        if x < 0 or y < 0 or x + length > site_size or y + width > site_size:
            raise ValueError(f"{name} exceeds site bounds")

    final_l = l_rects(
        x=l_origin_x - cantilever,
        y=l_origin_y - cantilever,
        arm_width=l_arm_width,
        east_length=l_east_length + cantilever,
        north_length=l_north_length + cantilever,
    )
    for name, (x, y, length, width) in final_l.items():
        if x < 0 or y < 0 or x + length > site_size or y + width > site_size:
            raise ValueError(f"final L {name} exceeds site bounds")

    zones: list[dict] = []

    for floor in range(floors):
        floor_no = floor + 1
        z = 0.0 if floor == 0 else lobby_height + (floor - 1) * floor_height
        height = lobby_height if floor == 0 else floor_height
        floor_tag = f"F{floor_no:02d}"

        if floor_no < merge_start_floor:
            for name, rect in ground_rects.items():
                add_rect_zone(zones, f"{floor_tag}_{name}", rect, z, height)
            continue

        if floor_no >= solid_l_start_floor:
            for part_name, rect in final_l.items():
                add_rect_zone(zones, f"{floor_tag}_cantilever_l_{part_name}", rect, z, height)
            continue

        t = (floor_no - merge_start_floor) / max(1, solid_l_start_floor - merge_start_floor)
        eased = t * t * (3 - 2 * t)

        transition_l = l_rects(
            x=lerp(l_origin_x, l_origin_x - cantilever, eased),
            y=lerp(l_origin_y, l_origin_y - cantilever, eased),
            arm_width=lerp(l_arm_width * 0.78, l_arm_width, eased),
            east_length=lerp(l_east_length * 0.68, l_east_length + cantilever, eased),
            north_length=lerp(l_north_length * 0.68, l_north_length + cantilever, eased),
        )

        nw_start = inset_rect(*ground_rects["nw_public_hall"], inset=lerp(3.0, 0.0, eased))
        se_start = inset_rect(*ground_rects["se_public_hall"], inset=lerp(3.0, 0.0, eased))
        corner_rect = blend_rect(nw_start, transition_l["north_arm"], eased)
        east_rect = blend_rect(se_start, transition_l["east_arm"], eased)

        add_rect_zone(zones, f"{floor_tag}_merging_northwest_to_north_arm", corner_rect, z, height)
        add_rect_zone(zones, f"{floor_tag}_merging_southeast_to_east_arm", east_rect, z, height)

        if ground_blocks == 3:
            third_start = inset_rect(*ground_rects["east_shared_hall"], inset=lerp(2.0, 0.0, eased))
            add_rect_zone(
                zones,
                f"{floor_tag}_merging_third_to_l_corner",
                blend_rect(third_start, transition_l["corner"], eased),
                z,
                height,
            )
        else:
            add_rect_zone(zones, f"{floor_tag}_light_bridge_l_corner", transition_l["corner"], z, height)

        bridge_width = lerp(5.0, l_arm_width * 0.75, eased)
        add_zone(
            zones,
            f"{floor_tag}_elevated_diagonal_bridge",
            lerp(34.0, transition_l["corner"][0] + l_arm_width * 0.45, eased),
            lerp(34.0, transition_l["corner"][1] + l_arm_width * 0.45, eased),
            z,
            lerp(30.0, 36.0, eased),
            bridge_width,
            height,
        )

    if add_open_space_markers:
        add_zone(
            zones,
            "site_sw_gateway_open_route_reference",
            8.0,
            34.0,
            0.0,
            34.0,
            open_route_width,
            0.12,
            category="open_space_reference",
        )
        add_zone(
            zones,
            "site_central_activity_plaza_reference",
            35.0,
            34.0,
            0.0,
            30.0,
            28.0,
            0.12,
            category="open_space_reference",
        )
        add_zone(
            zones,
            "site_ne_digital_block_open_route_reference",
            56.0,
            62.0,
            0.0,
            34.0,
            open_route_width,
            0.12,
            category="open_space_reference",
        )

    return {"building_name": building_name, "zones": zones}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate large corner ground blocks merging upward into a cantilevered L massing."
    )
    parser.add_argument("--name", default="Corner Path L Office")
    parser.add_argument("--output", default="output/corner_path_l_office.json")
    parser.add_argument("--site-size", type=float, default=100.0)
    parser.add_argument("--floors", type=int, default=9)
    parser.add_argument("--lobby-height", type=float, default=6.0)
    parser.add_argument("--floor-height", type=float, default=4.0)
    parser.add_argument("--ground-blocks", type=int, choices=[2, 3], default=3)
    parser.add_argument("--min-ground-block-length", type=float, default=20.0)
    parser.add_argument("--min-ground-block-width", type=float, default=16.0)
    parser.add_argument("--northwest-x", type=float, default=10.0)
    parser.add_argument("--northwest-y", type=float, default=66.0)
    parser.add_argument("--northwest-length", type=float, default=28.0)
    parser.add_argument("--northwest-width", type=float, default=22.0)
    parser.add_argument("--southeast-x", type=float, default=64.0)
    parser.add_argument("--southeast-y", type=float, default=10.0)
    parser.add_argument("--southeast-length", type=float, default=24.0)
    parser.add_argument("--southeast-width", type=float, default=22.0)
    parser.add_argument("--third-block-x", type=float, default=58.0)
    parser.add_argument("--third-block-y", type=float, default=42.0)
    parser.add_argument("--third-block-length", type=float, default=20.0)
    parser.add_argument("--third-block-width", type=float, default=16.0)
    parser.add_argument("--l-origin-x", type=float, default=18.0)
    parser.add_argument("--l-origin-y", type=float, default=18.0)
    parser.add_argument("--l-arm-width", type=float, default=11.5)
    parser.add_argument("--l-east-length", type=float, default=48.0)
    parser.add_argument("--l-north-length", type=float, default=48.0)
    parser.add_argument("--merge-start-floor", type=int, default=2)
    parser.add_argument("--solid-l-start-floor", type=int, default=7)
    parser.add_argument("--cantilever", type=float, default=8.0)
    parser.add_argument("--open-route-width", type=float, default=18.0)
    parser.add_argument("--no-open-space-markers", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = generate_corner_path_l(
        building_name=args.name,
        site_size=args.site_size,
        floors=args.floors,
        lobby_height=args.lobby_height,
        floor_height=args.floor_height,
        ground_blocks=args.ground_blocks,
        min_ground_block_length=args.min_ground_block_length,
        min_ground_block_width=args.min_ground_block_width,
        northwest_x=args.northwest_x,
        northwest_y=args.northwest_y,
        northwest_length=args.northwest_length,
        northwest_width=args.northwest_width,
        southeast_x=args.southeast_x,
        southeast_y=args.southeast_y,
        southeast_length=args.southeast_length,
        southeast_width=args.southeast_width,
        third_block_x=args.third_block_x,
        third_block_y=args.third_block_y,
        third_block_length=args.third_block_length,
        third_block_width=args.third_block_width,
        l_origin_x=args.l_origin_x,
        l_origin_y=args.l_origin_y,
        l_arm_width=args.l_arm_width,
        l_east_length=args.l_east_length,
        l_north_length=args.l_north_length,
        merge_start_floor=args.merge_start_floor,
        solid_l_start_floor=args.solid_l_start_floor,
        cantilever=args.cantilever,
        open_route_width=args.open_route_width,
        add_open_space_markers=not args.no_open_space_markers,
    )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(model, indent=2, ensure_ascii=False), encoding="utf-8")

    mass_zones = [z for z in model["zones"] if z.get("category") != "open_space_reference"]
    total_area = sum(z["dimensions"]["length"] * z["dimensions"]["width"] for z in mass_zones)
    total_height = args.lobby_height + (args.floors - 1) * args.floor_height
    print(f"Wrote {len(model['zones'])} zones to {output}")
    print(f"Approx gross floor area: {total_area:.1f} sqm")
    print(f"Building height: {total_height:.1f} m")
    print(f"Ground blocks: {args.ground_blocks}, min block size: {args.min_ground_block_length} x {args.min_ground_block_width} m")


if __name__ == "__main__":
    main()
