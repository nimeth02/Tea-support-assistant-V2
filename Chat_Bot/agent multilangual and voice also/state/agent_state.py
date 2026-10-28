from typing import TypedDict, Annotated, Optional
import operator

class AgentState(TypedDict):
    """State passed between the nodes of the agent's LangGraph workflow."""
    userId:int
    query: str              # the question in English - what the classifier and experts read
    intent: str
    response: str           # the answer; English until translate_response runs
    history: Annotated[list, operator.add]
    image_path: Optional[str]
    # Multilingual additions
    language: str           # what the user asked for: "auto", "en", "si" or "ta"
    detected_language: str  # the language actually used for this chat ("en", "si", "ta")
    original_query: str     # the question exactly as the user typed or spoke it
