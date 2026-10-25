"""State and context objects, one per framework that needs them.

Each expert runs in a different framework, so each has its own way of carrying "facts about this
question" (who the farmer is, whether a photo is attached) to its tools. Those facts are never
written by the LLM: the tools read them from here, so the model can't get the farmer's id wrong.

  ChatContext        LangChain  - fertilizer agent tools read it through `runtime.context`
  OrchestratorContext LangChain - the orchestrator's A2A tools read it the same way
  TeaAgentState      LangChain  - the orchestrator's state: messages + which experts it used
  DiseaseTurnState   CrewAI     - shared by the disease agent's tools for one question
  HelpState          LangGraph  - flows through the help agent's retrieve -> answer graph
"""
from typing import Any, Dict, List, Optional

from langchain.agents import AgentState
from pydantic import BaseModel, Field
from typing_extensions import NotRequired, TypedDict


class ChatContext(BaseModel):
    user_id: int = Field(gt=0)
    image_path: Optional[str] = None


class OrchestratorContext(ChatContext):
    trace_id: str = "-"


class TeaAgentState(AgentState):
    """AgentState holds `messages`; `intent` = experts used, e.g. "disease+fertilize"."""

    intent: NotRequired[str]


class DiseaseTurnState(BaseModel):
    user_id: int = 0
    query: str = ""
    image_path: Optional[str] = None
    tools_used: List[str] = Field(default_factory=list)  # filled in by the tools as they run


class HelpState(TypedDict, total=False):
    question: str
    records: List[Dict[str, Any]]
    answer: str
