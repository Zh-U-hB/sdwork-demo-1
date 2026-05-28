"""Partitioned models must not overload wall fenestration area."""

from __future__ import annotations

import re
from pathlib import Path

from scripts.facade_params import default_facade_params
from scripts.generate_20260528 import generate_20260528
from scripts.idf_converter import convert_and_run
from scripts.zone_partition import partition_model_by_floor


def test_partitioned_idf_wall_opening_fraction(tmp_path: Path) -> None:
    params = {
        "building_name": "TestPart",
        "site_size": 100.0,
        "total_area": 8000.0,
        "lobby_height": 6.0,
        "floor_height": 4.0,
        "setback_south": 15.0,
        "setback_west": 15.0,
        "setback_north": 10.0,
        "setback_east": 10.0,
        "boundary_shift": 40.0,
        "group_size": 2,
        "low_aspect_ratio": 1.0,
        "mid_aspect_ratio": 1.0,
        "high_aspect_ratio": 1.0,
        "low_offset_angle": 45.0,
        "mid_offset_angle": 180.0,
        "high_offset_angle": 315.0,
        "low_offset_distance": 2.0,
        "mid_offset_distance": 2.0,
        "high_offset_distance": 2.0,
        "min_support_overlap_ratio": 0.5,
        **default_facade_params(),
    }
    raw = generate_20260528(**params)
    model = partition_model_by_floor(raw, perimeter_depth=4.0, lobby_height=6.0, floor_height=4.0)

    convert_and_run(model, output_dir=tmp_path, run_simulation=False)
    text = next(tmp_path.glob("building_*.idf")).read_text(encoding="utf-8", errors="replace")

    wall_blocks = re.split(r"\n(?=BuildingSurface:Detailed,)", text)
    fen_blocks = re.split(r"\n(?=FenestrationSurface:Detailed,)", text)
    wall_names = {
        re.search(r"^\s+(\S+),", block, re.M).group(1)
        for block in wall_blocks[1:]
        if "Wall" in block and "Outdoors" in block
    }
    openings_per_wall: dict[str, int] = {}
    for block in fen_blocks[1:]:
        parent = re.search(r"building_surface_name\n\s+(\S+),", block, re.I)
        if parent:
            openings_per_wall[parent.group(1)] = openings_per_wall.get(parent.group(1), 0) + 1

    overloaded = [w for w, n in openings_per_wall.items() if n > 40]
    assert not overloaded, f"walls with excessive windows: {overloaded[:3]}"
