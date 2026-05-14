from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.language_models import BaseChatModel

from src.agent.state import AgentState

INTAKE_SYSTEM_PROMPT = """You are an architectural designer AI. Your task is to analyze the user's building description and produce a structured summary that will be used by a downstream agent to create 3D zones.

From the user's description, extract:
1. **Building name** - a descriptive name for the building
2. **Zones** - each room or functional area with:
   - Suggested position (origin_x, origin_y) relative to adjacent rooms
   - Dimensions (length, width, height) in meters
   - How they connect/adjacent to each other

**Spatial planning rules:**
- Place zones adjacent to each other along X or Y axis (not diagonally)
- The first zone should start at origin (0, 0, 0)
- Adjacent zones should share a wall — the next zone's origin should align with the previous zone's boundary
- All zones on the same floor should have origin_z = 0
- Standard floor height is 3.0m unless specified otherwise
- Default dimensions when not specified: length=5m, width=5m, height=3m

**IMPORTANT: Output ONLY the structured building summary in plain text, NOT JSON.
Include exact coordinates for each zone so the downstream agent can create them precisely."""


def intake_node(state: AgentState, llm: BaseChatModel) -> dict:
    """Parse user's building description into a structured summary."""
    user_input = state.get("building_description", "")

    response = llm.invoke([
        SystemMessage(content=INTAKE_SYSTEM_PROMPT),
        HumanMessage(content=f"Building description:\n{user_input}"),
    ])

    return {
        "messages": [response],
    }
