"""PruningGuide tool: pruning instructions (placeholder answer for now)."""
from langchain.tools import Tool


def pruning_guide(query: str) -> str:
    """Provide pruning instructions."""
    return "Prune tea plants lightly every 6 months during early spring."


pruning_guide_tool = Tool(
    name="PruningGuide",
    func=pruning_guide,
    description="Provides pruning instructions based on plant age or season.",
)
