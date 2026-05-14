"""CLI entry point for the 3D architectural modeling agent.

Usage:
    python main.py --description "A 100sqm house with living room(6x5m), bedroom(4x4m), kitchen(3x3m), height 3m"
    python main.py --description "..." --name "My Villa" --output output/villa.json
"""

import argparse
import asyncio

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

    args = parser.parse_args()

    from src.agent.graph import run_agent

    if args.interactive:
        print("Interactive mode not yet implemented. Use --description for batch mode.")
        return

    print(f"Building: {args.name}")
    print(f"Description: {args.description}")
    print(f"Output: {args.output}")
    print("-" * 50)
    print("Agent is working...")

    result = asyncio.run(run_agent(
        building_description=args.description,
        building_name=args.name,
        output_path=args.output,
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

    print(f"\nDone. Output saved to: {args.output}")


if __name__ == "__main__":
    main()
