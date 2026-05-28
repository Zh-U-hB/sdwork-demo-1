"""Prompt templates for LLM analysis."""

from __future__ import annotations

from typing import Any


def build_analysis_prompt(
    building_context: dict[str, str],
    param_table: str,
    population_summary: str,
    morris_table: str,
    pareto_table: str,
) -> str:
    """Build the structured analysis prompt for the LLM."""
    ctx = building_context
    return f"""## 建筑信息
- 气候区：{ctx.get('climate', '夏热冬冷（上海）')}
- 建筑类型：{ctx.get('building_type', '办公建筑')}
- 建筑面积：{ctx.get('floor_area', '2000 m²')}
- 层数：{ctx.get('floors', '3')}
- 优化目标：{ctx.get('objectives', '最小化年能耗和热不舒适小时数')}

## 当前参数空间
{param_table}

## 种群统计摘要
{population_summary}

## 各参数与目标的相关性（Morris 敏感性）
{morris_table}

## Pareto 前沿方案
{pareto_table}

## 分析任务

请基于以上数据完成：重要参数识别、参数范围优化建议、未探索方向、综合推荐。

请以 JSON 格式输出（仅 JSON，无 markdown 代码块）：

{{
  "important_parameters": {{
    "high_impact": [{{"name": "...", "sensitivity": 0.0, "recommended_range": [0, 1], "reason": "..."}}],
    "low_impact": [{{"name": "...", "suggested_fix_value": "..."}}],
    "new_parameters": [{{"name": "...", "type": "boolean", "reason": "..."}}]
  }},
  "exploration_guidance": {{
    "narrow_range": [{{"param": "...", "new_min": 0, "new_max": 1, "confidence": "high"}}],
    "new_directions": [{{"description": "...", "estimated_improvement": "...", "reason": "..."}}]
  }},
  "seed_solutions": [{{"param1": 0, "reason": "..."}}],
  "convergence_assessment": {{
    "is_plateauing": false,
    "remaining_potential": "medium",
    "recommended_next_action": "continue_narrowing"
  }}
}}
"""
