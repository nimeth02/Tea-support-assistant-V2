"""Tools available to the fertilizer expert: its own tools plus shared ones."""
from typing import List

from agent.agents.fertilize.tools.dosage_calculator import DosageCalculatorTool
from agent.agents.fertilize.tools.fertilize_history import FertilizeHistoryTool
from agent.agents.fertilize.tools.tea_land_data import TeaLandDataTool
from agent.agents.shared_tools import TeaTool, WeatherForecastTool
from agent.state.agent_state import TeaChatState

EXPERT = "fertilize"  # intent label recorded when any of these tools runs


def get_fertilize_tools(chat_state: TeaChatState) -> List[TeaTool]:
    """The fertilizer tools, bound to this turn's state (they read the farmer's user_id from it)."""
    return [
        WeatherForecastTool(chat_state=chat_state, expert=EXPERT),  # shared
        DosageCalculatorTool(chat_state=chat_state, expert=EXPERT),
        TeaLandDataTool(chat_state=chat_state, expert=EXPERT),
        FertilizeHistoryTool(chat_state=chat_state, expert=EXPERT),
    ]


__all__ = [
    "get_fertilize_tools",
    "DosageCalculatorTool",
    "TeaLandDataTool",
    "FertilizeHistoryTool",
]
