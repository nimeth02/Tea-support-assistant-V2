"""PruningGuide tool: pruning instructions (placeholder answer for now)."""
from langchain_core.tools import Tool

from agent.agents.shared_tools.schemas import SearchQuery


def pruning_guide(query: str) -> str:
    """Provide pruning instructions."""
    return "Prune tea plants lightly every 6 months during early spring."


pruning_guide_tool = Tool(
    name="PruningGuide",
    func=pruning_guide,
    args_schema=SearchQuery,
    description="Provides pruning instructions based on plant age or season.",
)
