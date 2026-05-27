from pathlib import Path

from langchain_core.messages import AIMessage

from src.agent.state import AgentState
from src.agent.tools import get_zones
from src.models.zone import BuildingModel


def export_node(state: AgentState) -> dict:
    """Export zones to JSON file and return the result."""
    zones = state.get("zones") or get_zones()
    building_name = state.get("building_name", "Unnamed Building")

    output_path = state.get("output_path", "output/building.json")
    model = BuildingModel(building_name=building_name, zones=zones)

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(model.model_dump_json(indent=2), encoding="utf-8")

    summary = (
        f"Building model exported to {output_path}\n{len(zones)} zones created:\n" +
        "\n".join(f"  - {z.name}: origin({z.origin.x}, {z.origin.y}, {z.origin.z}), "
                  f"{z.dimensions.length}x{z.dimensions.width}x{z.dimensions.height}m"
                  for z in zones)
    )
    return {
        "zones": zones,
        "messages": [AIMessage(content=summary)],
    }
