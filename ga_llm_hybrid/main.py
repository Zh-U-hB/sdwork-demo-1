"""CLI entry for GA + LLM hybrid optimization."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Allow running as script from repo root
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from ga_llm_hybrid.orchestrator import run_optimization


def main() -> None:
    parser = argparse.ArgumentParser(description="GA + LLM hybrid building energy optimizer")
    parser.add_argument(
        "-c",
        "--config",
        default="configs/ga_llm_hybrid_arch.yaml",
        help="Path to YAML/JSON config",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="output/ga_llm_hybrid",
        help="Output directory",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    report = run_optimization(args.config, args.output)
    print("Optimization complete.")
    print(f"  Rounds: {report.get('total_rounds')}")
    print(f"  Best fitness: {report.get('best', {}).get('fitness')}")
    print(f"  Output: {args.output}")


if __name__ == "__main__":
    main()
