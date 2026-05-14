from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver

from src.agent.state import AgentState
from src.agent.nodes.intake import intake_node
from src.agent.nodes.zone import build_zone_agent_node
from src.agent.nodes.export import export_node
from src.agent.tools import reset_store, set_building_name, get_zones
from src.agent.llm import create_llm


def build_graph():
    """Build the 3-node LangGraph agent: intake -> zone_agent -> export."""
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
    graph.add_edge("export", END)

    return graph.compile(checkpointer=MemorySaver())


async def run_agent(building_description: str, building_name: str = "Unnamed Building",
                    output_path: str = "output/building.json") -> dict:
    """Run the agent end-to-end and return the result state."""
    graph = build_graph()

    initial_state: AgentState = {
        "messages": [HumanMessage(content=building_description)],
        "building_name": building_name,
        "zones": [],
        "building_description": building_description,
        "output_path": output_path,
    }

    config = {"configurable": {"thread_id": "main"}}
    result = await graph.ainvoke(initial_state, config)
    return result
