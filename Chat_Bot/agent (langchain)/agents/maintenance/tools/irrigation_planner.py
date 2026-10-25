"""IrrigationPlanner tool: irrigation schedule suggestions (placeholder answer for now)."""
from langchain_core.tools import Tool

from agent.agents.shared_tools.schemas import SearchQuery


def irrigation_planner(query: str) -> str:
    """Suggest an irrigation schedule."""
    return "Irrigate weekly during the growing season."


irrigation_planner_tool = Tool(
    name="IrrigationPlanner",
    func=irrigation_planner,
    args_schema=SearchQuery,
    description="Suggests irrigation schedules based on season or soil conditions.",
)
