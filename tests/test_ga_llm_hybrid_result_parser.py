"""Tests for eplustbl objective extraction."""

from __future__ import annotations

from ga_llm_hybrid.energyplus.result_parser import objectives_from_eplustbl, pick_objectives
from ga_llm_hybrid.config import ObjectiveDef


def test_objectives_from_eplustbl_eui_and_peaks():
    sim = {
        "exists": True,
        "path": "/tmp/eplustbl.csv",
        "site_energy": {"Total Site Energy": 500.0},
        "building_area": {"Net Conditioned Building Area": 1000.0},
        "demand_end_uses": [
            {"end_use": "Cooling", "demand_w": 120000.0},
            {"end_use": "Heating", "demand_w": 80000.0},
        ],
    }
    obj = objectives_from_eplustbl(sim)
    assert obj["eui_mj_m2"] == 500.0
    assert obj["peak_cooling_load"] == 120000.0
    assert obj["peak_heating_load"] == 80000.0


def test_pick_objectives_filters_config():
    full = {"eui_mj_m2": 200.0, "peak_cooling_load": 50.0}
    defs = [
        ObjectiveDef(name="eui_mj_m2", weight=1.0),
        ObjectiveDef(name="peak_cooling_load", weight=0.5),
    ]
    picked = pick_objectives(full, defs)
    assert set(picked.keys()) == {"eui_mj_m2", "peak_cooling_load"}
