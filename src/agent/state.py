from typing import Annotated, NotRequired, TypedDict

from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage

from src.models.zone import Zone


class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    building_name: str
    zones: list[Zone]
    building_description: str
    output_path: str
    # EnergyPlus integration (optional — only populated when simulation is enabled)
    idf_output_path: NotRequired[str]
    simulation_result: NotRequired[str]
