"""IDF export: JSON windows, shading, adiabatic ground."""

from __future__ import annotations

from pathlib import Path

from scripts.facade_params import default_facade_params
from scripts.generate_20260528 import generate_20260528
from scripts.idf_converter import convert_and_run
from scripts.zone_partition import partition_model_by_floor


def _minimal_model() -> dict:
    params = {
        "building_name": "TestFacade",
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
    return generate_20260528(**params)


def test_idf_includes_json_windows_shading_and_adiabatic_ground(tmp_path: Path) -> None:
    model = _minimal_model()
    assert any(z.get("windows") for z in model["zones"])
    assert any(z.get("shading_surfaces") for z in model["zones"])

    out = convert_and_run(model, output_dir=tmp_path, run_simulation=False)
    assert out is not None
    idf_files = list(tmp_path.glob("building_*.idf"))
    assert idf_files
    text = idf_files[0].read_text(encoding="utf-8", errors="replace")

    assert "FenestrationSurface:Detailed" in text
    assert "WindowConstruction" in text
    assert "SimpleGlazing" in text
    assert "Shading:Building:Detailed" in text
    assert text.count("Adiabatic") >= 1
    assert "outside_boundary_condition\n    Ground" not in text
    assert text.count("FenestrationSurface:Detailed") >= 10
    assert text.count("Shading:Building:Detailed") >= 1
