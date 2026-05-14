"""Conversational agent for multi-turn building model refinement."""

from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.language_models import BaseChatModel

from src.agent.tools import ZONE_TOOLS, reset_store, get_zones, set_building_name, export_json

CHAT_SYSTEM_PROMPT = """You are an architectural 3D modeling assistant. You help users create and refine building models through conversation.

**Available tools:**
- `create_zone`: Create a room/zone with name, origin (x,y,z), and dimensions (length,width,height) in meters
- `list_zones`: Show all current zones
- `update_zone`: Modify an existing zone's position or size
- `delete_zone`: Remove a zone
- `export_json`: Save the model to a JSON file for Rhino/Grasshopper

**Rules:**
1. When the user describes a building, create ALL mentioned zones with appropriate coordinates
2. Place adjacent rooms next to each other along X or Y axis (share walls)
3. First room at origin (0,0,0). All rooms on same floor have z=0 unless multi-story
4. Standard floor height is 3.0m unless specified otherwise
5. After every change, call export_json to save, then list_zones to confirm
6. When the user asks to modify (resize, move, add, remove), use the appropriate tool
7. Always confirm what you've done in a concise response
8. If the user asks a question about the current model, use list_zones first

**Multi-story buildings:**
- Ground floor: z=0
- Upper floors: z = sum of floor heights below (e.g., 2F at z=3.0 if 1F height=3.0m)

Be conversational and helpful. Keep responses in the same language as the user."""


def build_chat_agent(llm: BaseChatModel):
    """Build a conversational ReAct agent for multi-turn building modeling."""
    return create_react_agent(
        model=llm,
        tools=ZONE_TOOLS,
        prompt=CHAT_SYSTEM_PROMPT,
    )


def _dicts_to_messages(history: list[dict]) -> list:
    """Convert Streamlit-style history dicts to LangChain messages."""
    out = []
    for m in history:
        if m["role"] == "user":
            out.append(HumanMessage(content=m["content"]))
        else:
            out.append(AIMessage(content=m["content"]))
    return out


async def chat_turn(agent, user_message: str, history: list[dict] | None = None,
                    output_path: str = "output/building.json") -> dict:
    """Run one conversation turn with full history context.
    Returns updated zones and agent response text."""
    msgs = _dicts_to_messages(history or [])
    msgs.append(HumanMessage(content=user_message))

    result = await agent.ainvoke({"messages": msgs})

    messages = result.get("messages", [])
    response_text = ""
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and msg.content:
            response_text = str(msg.content)
            break

    return {
        "zones": get_zones(),
        "response": response_text,
    }
