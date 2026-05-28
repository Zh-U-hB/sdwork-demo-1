"""Tests for LLM JSON parsing with mock responses."""

from __future__ import annotations

from ga_llm_hybrid.config import ParamDef
from ga_llm_hybrid.core.individual import Individual
from ga_llm_hybrid.core.parameter_space import ParameterSpace
from ga_llm_hybrid.llm.analyzer import LLMAnalyzer, parse_llm_json


def test_parse_llm_json_with_markdown_fence():
    text = '```json\n{"seed_solutions": []}\n```'
    assert parse_llm_json(text) == {"seed_solutions": []}


def test_analyzer_writes_files(tmp_path):
    defs = [ParamDef(name="x", type="continuous", range=[0.0, 1.0])]
    space = ParameterSpace(defs, seed=1)

    class _MockLLM:
        def invoke(self, messages):
            class _R:
                content = '{"convergence_assessment": {"remaining_potential": "medium"}}'

            return _R()

    analyzer = LLMAnalyzer(space, llm=_MockLLM())
    pop = [
        Individual(
            id=1,
            generation=0,
            genes={"x": 0.5},
            params={"x": 0.5},
            objectives={"eui_mj_m2": 100},
            fitness=100,
            feasible=True,
        )
    ]
    result = analyzer.analyze(pop, pop, {}, tmp_path)
    assert "convergence_assessment" in result
    assert (tmp_path / "llm_prompt.txt").exists()
    assert (tmp_path / "llm_analysis.json").exists()
