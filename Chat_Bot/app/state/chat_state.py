from typing import TypedDict, Annotated
import operator

class ChatState(TypedDict):
    """State for the LangGraph workflow."""
    userId:int
    query: str
    intent: str
    response: str
    history: Annotated[list, operator.add]