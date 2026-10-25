"""Tools available to the FertilizeAgent: its own tools plus shared ones."""
from agent.agents.fertilize.tools.dosage_calculator import dosage_calculator_tool
from agent.agents.fertilize.tools.fertilize_history import fertilize_history_tool
from agent.agents.fertilize.tools.tea_land_data import tea_land_data_tool
from agent.agents.shared_tools import weather_forecast_tool

FERTILIZE_TOOLS = [
    weather_forecast_tool,      # shared
    dosage_calculator_tool,
    tea_land_data_tool,
    fertilize_history_tool,
]

__all__ = [
    "FERTILIZE_TOOLS",
    "dosage_calculator_tool",
    "tea_land_data_tool",
    "fertilize_history_tool",
]
