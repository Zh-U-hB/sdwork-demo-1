"""Generate final optimization report."""

from __future__ import annotations

from typing import Any

from ga_llm_hybrid.core.individual import Individual


def build_final_report(
    rounds: int,
    final_pareto: list[Individual],
    llm_assessments: list[dict[str, Any]],
) -> dict[str, Any]:
    """Assemble final report dict."""
    best = min(final_pareto, key=lambda x: x.fitness or 1e9) if final_pareto else None
    return {
        "total_rounds": rounds,
        "pareto_size": len(final_pareto),
        "best": {
            "id": best.id if best else None,
            "fitness": best.fitness if best else None,
            "params": best.params if best else {},
            "objectives": best.objectives if best else {},
        },
        "llm_convergence_history": [
            a.get("convergence_assessment", {}) for a in llm_assessments
        ],
    }
