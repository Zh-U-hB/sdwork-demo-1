"""LLM analysis module with robust JSON parsing."""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

from ga_llm_hybrid.core.individual import Individual
from ga_llm_hybrid.core.parameter_space import ParameterSpace
from ga_llm_hybrid.llm.prompts import build_analysis_prompt

logger = logging.getLogger(__name__)


class LLMAnalyzer:
    """Call LLM to analyze GA results and suggest search guidance."""

    def __init__(
        self,
        space: ParameterSpace,
        building_context: dict[str, str] | None = None,
        llm: Any | None = None,
    ) -> None:
        self.space = space
        self.building_context = building_context or {}
        self._llm = llm

    def _get_llm(self) -> Any:
        if self._llm is not None:
            return self._llm
        from src.agent.llm import create_llm
        from src.config import LLMConfig

        return create_llm(LLMConfig())

    def analyze(
        self,
        population: list[Individual],
        pareto: list[Individual],
        morris: dict[str, dict[str, float]] | None,
        round_dir: Path,
    ) -> dict[str, Any]:
        """Run LLM analysis; persist prompt/response."""
        prompt = self._build_prompt(population, pareto, morris)
        (round_dir / "llm_prompt.txt").write_text(prompt, encoding="utf-8")

        try:
            llm = self._get_llm()
            from langchain_core.messages import HumanMessage

            resp = llm.invoke([HumanMessage(content=prompt)])
            text = _response_to_text(resp)
        except Exception as exc:
            logger.warning("LLM call failed: %s; using empty analysis", exc)
            text = "{}"

        (round_dir / "llm_response.txt").write_text(text, encoding="utf-8")
        parsed = parse_llm_json(text)
        (round_dir / "llm_analysis.json").write_text(
            json.dumps(parsed, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        return parsed

    def _build_prompt(
        self,
        population: list[Individual],
        pareto: list[Individual],
        morris: dict[str, dict[str, float]] | None,
    ) -> str:
        n = len(population)
        summary = (
            f"- 总模拟方案数：{n}\n"
            f"- 当前 Pareto 前沿方案数：{len(pareto)}\n"
            f"- 可行解比例：{sum(1 for p in population if p.feasible) / max(n, 1):.1%}"
        )
        morris_lines = ["param | sensitivity"]
        if morris:
            for pname, sens in morris.items():
                val = max(sens.values()) if sens else 0
                morris_lines.append(f"{pname} | {val:.4f}")
        else:
            morris_lines.append("(未计算)")

        pareto_lines = ["id | fitness | params_summary"]
        for p in pareto[:12]:
            ps = ", ".join(f"{k}={v:.3g}" if isinstance(v, float) else f"{k}={v}" for k, v in list(p.params.items())[:6])
            fit = p.fitness if p.fitness is not None else 1e9
            pareto_lines.append(f"{p.id} | {fit:.2f} | {ps}")

        return build_analysis_prompt(
            self.building_context,
            self.space.summary_table(),
            summary,
            "\n".join(morris_lines),
            "\n".join(pareto_lines),
        )


def _response_to_text(resp: Any) -> str:
    """Normalize LangChain / Anthropic message content to plain text."""
    content = resp.content if hasattr(resp, "content") else resp
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                if block.get("type") == "thinking":
                    continue
                if "text" in block:
                    parts.append(str(block["text"]))
                elif "thinking" not in block:
                    parts.append(str(block))
            else:
                text_attr = getattr(block, "text", None)
                block_type = getattr(block, "type", None)
                if block_type == "thinking":
                    continue
                if text_attr is not None:
                    parts.append(str(text_attr))
        return "\n".join(parts)
    return str(content)


def parse_llm_json(text: str) -> dict[str, Any]:
    """Tolerant JSON extraction from LLM output."""
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    decoder = json.JSONDecoder()
    start = 0
    while True:
        brace = text.find("{", start)
        if brace < 0:
            break
        try:
            obj, _ = decoder.raw_decode(text[brace:])
            if isinstance(obj, dict) and obj:
                return obj
        except json.JSONDecodeError:
            pass
        start = brace + 1
    logger.warning("Failed to parse LLM JSON; returning empty dict")
    return {}
