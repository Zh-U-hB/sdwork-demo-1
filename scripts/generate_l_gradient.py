"""Generate a parametric office massing JSON.

The massing transitions from scattered lower blocks to a complete L-shaped
office volume at the top. Output matches the existing BuildingModel JSON shape.
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
) -> None:
    if length <= 0 or width <= 0 or height <= 0:
        return
    x0, y0, z0 = round(x, 3), round(y, 3), round(z, 3)
    x1 = round(x + length, 3)
    y1 = round(y + width, 3)
    z1 = round(z + height, 3)
    zones.append({
        "name": name,
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


def make_segments(origin: float, total: float, count: int, gap: float) -> list[tuple[float, float]]:
    """Split a line into equal segments with gaps."""
    if count <= 1:
        return [(origin, total)]
    segment = (total - gap * (count - 1)) / count
    if segment <= 0:
        raise ValueError(f"make_segments: gap {gap} × {count-1} exceeds total {total}")
    return [(origin + i * (segment + gap), segment) for i in range(count)]


def generate_l_gradient(
    *,
    building_name: str = "Gradient L Office",
    site_size: float = 100.0,
    floors: int = 11,
    lobby_height: float = 5.5,
    floor_height: float = 4.0,
    base_x: float = 18.0,
    base_y: float = 16.0,
    arm_width: float = 13.5,
    horizontal_length: float = 58.0,
    vertical_length: float = 54.0,
    scatter_gap: float = 8.0,
    min_fragment_scale: float = 0.62,
    merge_power: float = 1.35,
    bridge_start_floor: int = 4,
    top_solid_floors: int = 3,
    add_courtyard_marker: bool = True,
) -> dict:
    if floors < 2:
        raise ValueError("floors must be at least 2")
    if lobby_height + (floors - 1) * floor_height >= 50.0:
        raise ValueError("height must be under 50m")
    if base_x + horizontal_length > site_size or base_y + vertical_length > site_size:
        raise ValueError("L shape exceeds site bounds")
    if arm_width >= min(horizontal_length, vertical_length):
        raise ValueError("arm_width is too large for the L shape")
    if top_solid_floors < 1 or top_solid_floors >= floors:
        raise ValueError("top_solid_floors must be between 1 and floors - 1")

    zones: list[dict] = []

    h_parts = make_segments(base_x + arm_width, horizontal_length - arm_width, 3, scatter_gap)
    v_parts = make_segments(base_y + arm_width, vertical_length - arm_width, 3, scatter_gap)

    for floor in range(floors):
        t = floor / (floors - 1)
        merge = t**merge_power
        loose = 1.0 - merge
        height = lobby_height if floor == 0 else floor_height
        z = 0.0 if floor == 0 else lobby_height + (floor - 1) * floor_height
        scale = lerp(min_fragment_scale, 1.0, merge)
        floor_tag = f"F{floor + 1:02d}"

        if floor >= floors - top_solid_floors:
            add_zone(zones, f"{floor_tag}_l_corner", base_x, base_y, z, arm_width, arm_width, height)
            add_zone(
                zones,
                f"{floor_tag}_l_east_arm",
                base_x + arm_width,
                base_y,
                z,
                horizontal_length - arm_width,
                arm_width,
                height,
            )
            add_zone(
                zones,
                f"{floor_tag}_l_north_arm",
                base_x,
                base_y + arm_width,
                z,
                arm_width,
                vertical_length - arm_width,
                height,
            )
            continue

        # L corner block: it grows from a compact public block into the full corner.
        corner_size = arm_width * lerp(0.75, 1.0, merge)
        corner_shift = arm_width * (1.0 - corner_size / arm_width) * 0.5 * loose
        add_zone(
            zones,
            f"{floor_tag}_corner_core",
            base_x + corner_shift,
            base_y + corner_shift,
            z,
            corner_size,
            corner_size,
            height,
        )

        # Horizontal arm fragments slide and stretch until they become continuous.
        for idx, (x, length) in enumerate(h_parts, start=1):
            x_offset = loose * scatter_gap * (idx - 2)
            y_offset = loose * scatter_gap * (0.6 if idx % 2 else -0.35)
            part_length = length * scale
            part_width = arm_width * lerp(0.72, 1.0, merge)
            add_zone(
                zones,
                f"{floor_tag}_east_arm_{idx}",
                x + x_offset,
                base_y + y_offset,
                z,
                part_length,
                part_width,
                height,
            )

        # Vertical arm fragments slide and stretch until they become continuous.
        for idx, (y, width) in enumerate(v_parts, start=1):
            x_offset = loose * scatter_gap * (-0.45 if idx % 2 else 0.65)
            y_offset = loose * scatter_gap * (idx - 2)
            part_length = arm_width * lerp(0.72, 1.0, merge)
            part_width = width * scale
            add_zone(
                zones,
                f"{floor_tag}_north_arm_{idx}",
                base_x + x_offset,
                y + y_offset,
                z,
                part_length,
                part_width,
                height,
            )

        # Mid-level bridges make the aggregation legible without filling all gaps at once.
        if floor + 1 >= bridge_start_floor:
            bridge_merge = ((floor + 1 - bridge_start_floor) / max(1, floors - bridge_start_floor)) ** 1.2
            bridge_width = arm_width * lerp(0.22, 0.65, bridge_merge)
            add_zone(
                zones,
                f"{floor_tag}_east_bridge",
                base_x + arm_width,
                base_y + (arm_width - bridge_width) * 0.5,
                z,
                horizontal_length - arm_width,
                bridge_width,
                height,
            )
            add_zone(
                zones,
                f"{floor_tag}_north_bridge",
                base_x + (arm_width - bridge_width) * 0.5,
                base_y + arm_width,
                z,
                bridge_width,
                vertical_length - arm_width,
                height,
            )

    if add_courtyard_marker:
        add_zone(
            zones,
            "site_inner_courtyard_reference",
            base_x + arm_width,
            base_y + arm_width,
            0.0,
            max(4.0, horizontal_length - arm_width),
            max(4.0, vertical_length - arm_width),
            0.15,
        )

    return {"building_name": building_name, "zones": zones}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a scattered-to-complete L-shaped office massing JSON."
    )
    parser.add_argument("--name", default="Gradient L Office")
    parser.add_argument("--output", default="output/gradient_l_office.json")
    parser.add_argument("--site-size", type=float, default=100.0)
    parser.add_argument("--floors", type=int, default=11)
    parser.add_argument("--lobby-height", type=float, default=5.5)
    parser.add_argument("--floor-height", type=float, default=4.0)
    parser.add_argument("--base-x", type=float, default=18.0)
    parser.add_argument("--base-y", type=float, default=16.0)
    parser.add_argument("--arm-width", type=float, default=13.5)
    parser.add_argument("--horizontal-length", type=float, default=58.0)
    parser.add_argument("--vertical-length", type=float, default=54.0)
    parser.add_argument("--scatter-gap", type=float, default=8.0)
    parser.add_argument("--min-fragment-scale", type=float, default=0.62)
    parser.add_argument("--merge-power", type=float, default=1.35)
    parser.add_argument("--bridge-start-floor", type=int, default=4)
    parser.add_argument("--top-solid-floors", type=int, default=3)
    parser.add_argument("--no-courtyard-marker", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = generate_l_gradient(
        building_name=args.name,
        site_size=args.site_size,
        floors=args.floors,
        lobby_height=args.lobby_height,
        floor_height=args.floor_height,
        base_x=args.base_x,
        base_y=args.base_y,
        arm_width=args.arm_width,
        horizontal_length=args.horizontal_length,
        vertical_length=args.vertical_length,
        scatter_gap=args.scatter_gap,
        min_fragment_scale=args.min_fragment_scale,
        merge_power=args.merge_power,
        bridge_start_floor=args.bridge_start_floor,
        top_solid_floors=args.top_solid_floors,
        add_courtyard_marker=not args.no_courtyard_marker,
    )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(model, indent=2, ensure_ascii=False), encoding="utf-8")
    total_area = sum(
        z["dimensions"]["length"] * z["dimensions"]["width"]
        for z in model["zones"]
        if z["dimensions"]["height"] > 1.0
    )
    total_height = args.lobby_height + (args.floors - 1) * args.floor_height
    print(f"Wrote {len(model['zones'])} zones to {output}")
    print(f"Approx gross floor area: {total_area:.1f} sqm")
    print(f"Building height: {total_height:.1f} m")


if __name__ == "__main__":
    main()
