"""Tools available to the maintenance expert, one module per tool."""
from typing import List

from agent.agents.maintenance.tools.irrigation_planner import IrrigationPlannerTool
from agent.agents.maintenance.tools.pruning_guide import PruningGuideTool
from agent.agents.shared_tools import TeaTool
from agent.state.agent_state import TeaChatState

EXPERT = "maintenance"  # intent label recorded when any of these tools runs


def get_maintenance_tools(chat_state: TeaChatState) -> List[TeaTool]:
    return [
        PruningGuideTool(chat_state=chat_state, expert=EXPERT),
        IrrigationPlannerTool(chat_state=chat_state, expert=EXPERT),
    ]


__all__ = ["get_maintenance_tools", "PruningGuideTool", "IrrigationPlannerTool"]
