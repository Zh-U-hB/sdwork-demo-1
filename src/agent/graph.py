import os

from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver

from src.agent.state import AgentState
from src.agent.nodes.intake import intake_node
from src.agent.nodes.zone import build_zone_agent_node
from src.agent.nodes.export import export_node
from src.agent.nodes.energyplus import energyplus_node
from src.agent.tools import reset_store, set_building_name, get_zones
from src.agent.llm import create_llm


def _energyplus_configured() -> bool:
    """Return True if at least one EnergyPlus transport config is present."""
    transport = os.getenv("ENERGYPLUS_AGENT_TRANSPORT", "stdio").lower()
    if transport == "stdio":
        return bool(os.getenv("ENERGYPLUS_AGENT_PATH", "").strip())
    return bool(os.getenv("ENERGYPLUS_AGENT_URL", "").strip())


def build_graph(run_energyplus: bool = False):
    """Build the LangGraph agent pipeline.

    When run_energyplus=True *and* the EnergyPlus-Agent is configured in .env,
    a 4th node is appended:  intake -> zone_agent -> export -> energyplus
    Otherwise the graph ends after export.
    """
    llm = create_llm()
    zone_agent = build_zone_agent_node(llm)

    graph = StateGraph(AgentState)

    def _intake(state: AgentState) -> dict:
        return intake_node(state, llm)

    graph.add_node("intake", _intake)

    async def _zone_agent(state: AgentState) -> dict:
        reset_store()
        set_building_name(state.get("building_name", "Unnamed Building"))

        intake_msgs = state.get("messages", [])
        summary_text = ""
        for msg in reversed(intake_msgs):
            if hasattr(msg, "content") and msg.content:
                summary_text = str(msg.content)
                break

        output_path = state.get("output_path", "output/building.json")
        task = f"""
Building description summary:
{summary_text}

Your task: Create all zones described above using the create_zone tool.
After all zones are created, call export_json with filepath="{output_path}".
"""
        result = await zone_agent.ainvoke({"messages": [HumanMessage(content=task)]})
        return {"zones": get_zones()}

    graph.add_node("zone_agent", _zone_agent)
    graph.add_node("export", export_node)

    graph.add_edge(START, "intake")
    graph.add_edge("intake", "zone_agent")
    graph.add_edge("zone_agent", "export")

    if run_energyplus:
        graph.add_node("energyplus", energyplus_node)
        graph.add_edge("export", "energyplus")
        graph.add_edge("energyplus", END)
    else:
        graph.add_edge("export", END)

    return graph.compile(checkpointer=MemorySaver())


async def run_agent(
    building_description: str,
    building_name: str = "Unnamed Building",
    output_path: str = "output/building.json",
    idf_output_path: str | None = None,
    run_energyplus: bool = False,
) -> dict:
    """Run the agent end-to-end and return the result state.

    Args:
        building_description: Natural-language description of the building.
        building_name:        Display name stored in the output JSON.
        output_path:          Path for the zone geometry JSON file.
        idf_output_path:      Path for the generated IDF file.
                              Defaults to output_path with .idf extension.
        run_energyplus:       When True, append the EnergyPlus simulation node.
    """
    graph = build_graph(run_energyplus=run_energyplus)

    from pathlib import Path
    resolved_idf = idf_output_path or str(Path(output_path).with_suffix(".idf"))

    initial_state: AgentState = {
        "messages": [HumanMessage(content=building_description)],
        "building_name": building_name,
        "zones": [],
        "building_description": building_description,
        "output_path": output_path,
        "idf_output_path": resolved_idf,
    }

    config = {"configurable": {"thread_id": "main"}}
    result = await graph.ainvoke(initial_state, config)
    return result
