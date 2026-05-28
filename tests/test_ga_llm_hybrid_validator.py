"""Tests for LLM output validation."""

from __future__ import annotations

from ga_llm_hybrid.config import ParamDef
from ga_llm_hybrid.core.parameter_space import ParameterSpace
from ga_llm_hybrid.llm.validator import validate_llm_analysis


def test_clamp_hard_limit():
    defs = [ParamDef(name="shade", type="continuous", range=[0.0, 2.0])]
    space = ParameterSpace(defs, seed=1)
    raw = {
        "important_parameters": {
            "high_impact": [
                {
                    "name": "shade",
                    "sensitivity": 0.4,
                    "recommended_range": [10.0, 12.0],
                    "confidence": "high",
                }
            ]
        }
    }
    out = validate_llm_analysis(raw, space)
    rec = out["important_parameters"]["high_impact"][0]["recommended_range"]
    assert rec[0] >= -1.0
    assert rec[1] <= 3.0


def test_reject_low_confidence_morris_mismatch():
    defs = [ParamDef(name="glazing", type="continuous", range=[0.2, 0.8])]
    space = ParameterSpace(defs, seed=2)
    raw = {
        "important_parameters": {
            "high_impact": [
                {
                    "name": "glazing",
                    "sensitivity": 0.9,
                    "recommended_range": [0.3, 0.5],
                    "confidence": "high",
                }
            ]
        }
    }
    morris = {"glazing": {"eui_mj_m2": 0.01}}
    out = validate_llm_analysis(raw, space, morris=morris)
    assert out["important_parameters"]["high_impact"] == []


def test_limit_narrow_range():
    defs = [ParamDef(name="wwr", type="continuous", range=[0.0, 1.0])]
    space = ParameterSpace(defs, seed=3)
    space.update_range("wwr", new_min=0.0, new_max=1.0)
    raw = {
        "exploration_guidance": {
            "narrow_range": [
                {"param": "wwr", "new_min": 0.45, "new_max": 0.55, "confidence": "high"}
            ]
        }
    }
    out = validate_llm_analysis(raw, space)
    item = out["exploration_guidance"]["narrow_range"][0]
    span = item["new_max"] - item["new_min"]
    assert span >= 0.7
