from typing import Optional

from langchain.agents import AgentState
from pydantic import BaseModel
from typing_extensions import NotRequired


class ChatContext(BaseModel):
    """Fixed facts about the current chat turn, handed to the agents with `context=`.

    Tools read it through `runtime.context` (a ToolRuntime argument the LLM never sees), so the
    LLM never has to pass - or can get wrong - the farmer's id or the photo path.
    It is a Pydantic model, so a bad value (e.g. user_id=None) fails right when the turn starts.
    """

    user_id: int
    image_path: Optional[str] = None


class TeaAgentState(AgentState):
    """State schema of the orchestrator agent: what it carries while it works on one message.

    AgentState already holds `messages` - the farmer's message, every expert call and answer,
    and the final reply. We add `intent`: which experts were used (e.g. "disease+fertilize"),
    filled in when the agent finishes and read by the backend for its metrics.

    LangChain requires a state schema to be a TypedDict that extends AgentState.
    """

    intent: NotRequired[str]
