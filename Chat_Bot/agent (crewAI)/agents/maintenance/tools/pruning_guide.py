"""PruningGuide tool: pruning instructions (placeholder answer for now)."""
from typing import Type

from pydantic import BaseModel

from agent.agents.shared_tools.base import TeaTool
from agent.agents.shared_tools.schemas import SearchQuery


def pruning_guide(query: str) -> str:
    """Provide pruning instructions."""
    return "Prune tea plants lightly every 6 months during early spring."


class PruningGuideTool(TeaTool):
    name: str = "PruningGuide"
    description: str = "Provides pruning instructions based on plant age or season."
    args_schema: Type[BaseModel] = SearchQuery

    def _run(self, query: str) -> str:
        self.note_use()
        return pruning_guide(query)
