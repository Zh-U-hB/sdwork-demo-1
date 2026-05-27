"""Generate a structurally supported corner-block-to-L office massing JSON.

This variant keeps the ground-level urban idea: two or three large blocks near
the northwest and southeast corners with an open southwest-to-northeast public
route. Unlike the more expressive cantilever version, most upper L-shaped mass
is supported by stacked lower/mid-level blocks. Cantilevers are allowed only as
small edge projections.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.generate_corner_path_l import add_rect_zone, add_zone, blend_rect, inset_rect, lerp, l_rects


def generate_supported_corner_path_l(
    *,
    building_name: str = "Supported Corner Path L Office",
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
    l_arm_width: float = 10.0,
    l_east_length: float = 36.0,
    l_north_length: float = 36.0,
    merge_start_floor: int = 2,
    solid_l_start_floor: int = 7,
    cantilever: float = 3.0,
    support_band_width: float = 8.0,
    support_start_floor: int = 2,
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
    if not (2 <= support_start_floor <= solid_l_start_floor):
        raise ValueError("support_start_floor must be between 2 and solid_l_start_floor")
    if cantilever > 5.0:
        raise ValueError("cantilever should stay at or below 5m in the supported version")

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

    supported_l = l_rects(
        x=l_origin_x,
        y=l_origin_y,
        arm_width=l_arm_width,
        east_length=l_east_length,
        north_length=l_north_length,
    )
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

    support_rects = {
        "north_support_band": (
            l_origin_x,
            l_origin_y + l_arm_width,
            support_band_width,
            l_north_length - l_arm_width,
        ),
        "east_support_band": (
            l_origin_x + l_arm_width,
            l_origin_y,
            l_east_length - l_arm_width,
            support_band_width,
        ),
        "l_corner_support_core": (
            l_origin_x,
            l_origin_y,
            l_arm_width,
            l_arm_width,
        ),
    }

    zones: list[dict] = []

    for floor in range(floors):
        floor_no = floor + 1
        z = 0.0 if floor == 0 else lobby_height + (floor - 1) * floor_height
        height = lobby_height if floor == 0 else floor_height
        floor_tag = f"F{floor_no:02d}"

        if floor_no == 1:
            for name, rect in ground_rects.items():
                add_rect_zone(zones, f"{floor_tag}_{name}", rect, z, height)
            continue

        if support_start_floor <= floor_no < merge_start_floor:
            for name, rect in ground_rects.items():
                add_rect_zone(zones, f"{floor_tag}_{name}", rect, z, height)

        if floor_no >= support_start_floor and floor_no < solid_l_start_floor:
            support_t = (floor_no - support_start_floor) / max(1, solid_l_start_floor - support_start_floor)
            eased_support = support_t * support_t * (3 - 2 * support_t)
            for name, target in support_rects.items():
                if name == "north_support_band":
                    start = inset_rect(*ground_rects["nw_public_hall"], inset=2.0)
                elif name == "east_support_band":
                    start = inset_rect(*ground_rects["se_public_hall"], inset=2.0)
                else:
                    start = inset_rect(
                        *ground_rects.get("east_shared_hall", ground_rects["se_public_hall"]),
                        inset=3.0,
                    )
                add_rect_zone(
                    zones,
                    f"{floor_tag}_{name}",
                    blend_rect(start, target, eased_support),
                    z,
                    height,
                    category="support_mass",
                )

        if floor_no >= merge_start_floor and floor_no < solid_l_start_floor:
            merge_t = (floor_no - merge_start_floor) / max(1, solid_l_start_floor - merge_start_floor)
            eased = merge_t * merge_t * (3 - 2 * merge_t)
            transition_l = l_rects(
                x=lerp(l_origin_x + 2.0, l_origin_x, eased),
                y=lerp(l_origin_y + 2.0, l_origin_y, eased),
                arm_width=lerp(l_arm_width * 0.82, l_arm_width, eased),
                east_length=lerp(l_east_length * 0.74, l_east_length, eased),
                north_length=lerp(l_north_length * 0.74, l_north_length, eased),
            )
            add_rect_zone(zones, f"{floor_tag}_supported_l_corner", transition_l["corner"], z, height)
            add_rect_zone(zones, f"{floor_tag}_supported_l_east_arm", transition_l["east_arm"], z, height)
            add_rect_zone(zones, f"{floor_tag}_supported_l_north_arm", transition_l["north_arm"], z, height)
            continue

        if floor_no >= solid_l_start_floor:
            for part_name, supported_rect in supported_l.items():
                add_rect_zone(zones, f"{floor_tag}_supported_l_{part_name}", supported_rect, z, height)

            if cantilever > 0:
                # Small, explicit edge cantilevers. Most floor area remains over support.
                add_rect_zone(
                    zones,
                    f"{floor_tag}_minor_west_cantilever",
                    (l_origin_x - cantilever, l_origin_y, cantilever, l_arm_width),
                    z,
                    height,
                    category="minor_cantilever",
                )
                add_rect_zone(
                    zones,
                    f"{floor_tag}_minor_south_cantilever",
                    (l_origin_x, l_origin_y - cantilever, l_east_length, cantilever),
                    z,
                    height,
                    category="minor_cantilever",
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
        description="Generate supported large corner ground blocks merging into a mostly-supported L massing."
    )
    parser.add_argument("--name", default="Supported Corner Path L Office")
    parser.add_argument("--output", default="output/supported_corner_path_l_office.json")
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
    parser.add_argument("--l-arm-width", type=float, default=10.0)
    parser.add_argument("--l-east-length", type=float, default=36.0)
    parser.add_argument("--l-north-length", type=float, default=36.0)
    parser.add_argument("--merge-start-floor", type=int, default=3)
    parser.add_argument("--solid-l-start-floor", type=int, default=7)
    parser.add_argument("--cantilever", type=float, default=3.0)
    parser.add_argument("--support-band-width", type=float, default=8.0)
    parser.add_argument("--support-start-floor", type=int, default=2)
    parser.add_argument("--open-route-width", type=float, default=18.0)
    parser.add_argument("--no-open-space-markers", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = generate_supported_corner_path_l(
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
        support_band_width=args.support_band_width,
        support_start_floor=args.support_start_floor,
        open_route_width=args.open_route_width,
        add_open_space_markers=not args.no_open_space_markers,
    )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(model, indent=2, ensure_ascii=False), encoding="utf-8")

    mass_zones = [z for z in model["zones"] if z.get("category") != "open_space_reference"]
    total_area = sum(z["dimensions"]["length"] * z["dimensions"]["width"] for z in mass_zones)
    total_height = args.lobby_height + (args.floors - 1) * args.floor_height
    cantilever_area = sum(
        z["dimensions"]["length"] * z["dimensions"]["width"]
        for z in mass_zones
        if z.get("category") == "minor_cantilever"
    )
    print(f"Wrote {len(model['zones'])} zones to {output}")
    print(f"Approx gross floor area: {total_area:.1f} sqm")
    print(f"Building height: {total_height:.1f} m")
    print(f"Minor cantilever area ratio: {cantilever_area / total_area:.1%}")


if __name__ == "__main__":
    main()
