"""IrrigationPlanner tool: irrigation schedule suggestions (placeholder answer for now)."""
from typing import Type

from pydantic import BaseModel

from agent.agents.shared_tools.base import TeaTool
from agent.agents.shared_tools.schemas import SearchQuery


def irrigation_planner(query: str) -> str:
    """Suggest an irrigation schedule."""
    return "Irrigate weekly during the growing season."


class IrrigationPlannerTool(TeaTool):
    name: str = "IrrigationPlanner"
    description: str = "Suggests irrigation schedules based on season or soil conditions."
    args_schema: Type[BaseModel] = SearchQuery

    def _run(self, query: str) -> str:
        self.note_use()
        return irrigation_planner(query)
