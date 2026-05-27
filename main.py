"""CLI entry point for the 3D architectural modeling agent.

Usage:
    python main.py --description "A 100sqm house with living room(6x5m), bedroom(4x4m), kitchen(3x3m), height 3m"
    python main.py --description "..." --name "My Villa" --output output/villa.json
"""

import argparse
import asyncio
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


def main():
    parser = argparse.ArgumentParser(
        description="LLM Agent for 3D architectural modeling - generates JSON for Rhino/Grasshopper"
    )
    parser.add_argument(
        "--description", "-d",
        required=True,
        help="Natural language description of the building",
    )
    parser.add_argument(
        "--name", "-n",
        default="Unnamed Building",
        help="Building name (default: Unnamed Building)",
    )
    parser.add_argument(
        "--output", "-o",
        default="output/building.json",
        help="Output JSON file path (default: output/building.json)",
    )
    parser.add_argument(
        "--interactive", "-i",
        action="store_true",
        help="Run in interactive mode (chat with the agent)",
    )
    parser.add_argument(
        "--simulate", "-s",
        action="store_true",
        help=(
            "After zone generation, pass the model to EnergyPlus-Agent for IDF "
            "generation and simulation. Requires ENERGYPLUS_AGENT_PATH in .env."
        ),
    )
    parser.add_argument(
        "--idf-output",
        default=None,
        help=(
            "Output path for the generated IDF file "
            "(default: same as --output but with .idf extension)"
        ),
    )

    args = parser.parse_args()

    from src.agent.graph import run_agent

    if args.interactive:
        print("Interactive mode not yet implemented. Use --description for batch mode.")
        return

    print(f"Building: {args.name}")
    print(f"Description: {args.description}")
    print(f"Output (JSON): {args.output}")
    if args.simulate:
        idf_path = args.idf_output or str(Path(args.output).with_suffix(".idf"))
        print(f"Output (IDF):  {idf_path}")
    print("-" * 50)
    print("Agent is working...")

    result = asyncio.run(run_agent(
        building_description=args.description,
        building_name=args.name,
        output_path=args.output,
        idf_output_path=args.idf_output,
        run_energyplus=args.simulate,
    ))

    # Print the final assistant message
    messages = result.get("messages", [])
    if messages:
        last_msg = messages[-1]
        content = last_msg.content if hasattr(last_msg, "content") else str(last_msg)
        if isinstance(content, str):
            print(content)
        elif isinstance(content, list):
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    print(item["text"])

    print(f"\nDone. Zone JSON saved to: {args.output}")
    if args.simulate:
        sim_result = result.get("simulation_result", "")
        idf_path = result.get("idf_output_path", "")
        if idf_path:
            print(f"IDF saved to: {idf_path}")
        if sim_result and "[EnergyPlus]" not in sim_result:
            print(f"Simulation result: {sim_result}")


if __name__ == "__main__":
    main()
