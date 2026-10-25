"""The one contract every expert follows, whatever framework it is built with.

The A2A server only knows this interface, not CrewAI/LangChain/LangGraph. That is what lets
experts written in different frameworks sit behind the same protocol: swap the inside of an
expert (say, rewrite the help agent in CrewAI) and nothing outside it changes.
"""
from dataclasses import dataclass, field
from typing import List, Optional, Protocol


@dataclass
class ExpertAnswer:
    text: str
    tools_used: List[str] = field(default_factory=list)
    model: Optional[str] = None       # which LLM finally answered (shows fallbacks in use)
    degraded: bool = False            # True when this is a fallback "sorry" reply, not a real answer


class Expert(Protocol):
    name: str        # short id, e.g. "disease"; also the intent label and metrics label
    framework: str   # "crewai" | "langchain" | "langgraph"

    async def answer(self, question: str, user_id: int, image_path: Optional[str] = None) -> ExpertAnswer:
        ...
