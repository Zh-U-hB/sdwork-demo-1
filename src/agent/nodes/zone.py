from langgraph.prebuilt import create_react_agent
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from src.agent.state import AgentState
from src.agent.tools import ZONE_TOOLS, reset_store, set_building_name, export_json

ZONE_SYSTEM_PROMPT = """You are a 3D building modeling agent. Your task is to create building zones using the provided tools.

**Available tools:**
- `create_zone`: Create a zone with name, origin (x,y,z), and dimensions (length,width,height) in meters
- `list_zones`: List all created zones to check your progress
- `update_zone`: Modify an existing zone's properties
- `delete_zone`: Remove a zone
- `export_json`: Export the final building model to a JSON file

**Critical rules:**
1. Create ALL zones mentioned in the building description — do not skip any
2. Use coordinates that make spatial sense — adjacent rooms should be placed next to each other
3. First room at origin (0, 0, 0), subsequent rooms placed along X or Y axis
4. All dimensions in meters, all coordinates in meters
5. Use `list_zones` to verify your work before finishing
6. When all zones are created, call `export_json` with the file path provided

**Workflow:**
1. Read the building description carefully
2. Create each zone one by one using `create_zone`
3. Use `list_zones` to verify all zones are created correctly
4. Call `export_json` with the exact file path provided to save the final model
"""


def build_zone_agent_node(llm: BaseChatModel):
    """Build a ReAct agent node for zone creation."""
    agent = create_react_agent(
        model=llm,
        tools=ZONE_TOOLS,
        prompt=ZONE_SYSTEM_PROMPT,
    )
    return agent
