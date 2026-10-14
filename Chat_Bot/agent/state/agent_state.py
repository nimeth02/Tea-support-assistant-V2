from typing import TypedDict, Annotated, Optional
import operator

class AgentState(TypedDict):
    """State passed between the nodes of the agent's LangGraph workflow."""
    userId:int
    query: str
    intent: str
    response: str
    history: Annotated[list, operator.add]
    image_path: Optional[str]