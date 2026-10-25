"""LangChain tools of the fertilizer expert.

Each tool is a thin LangChain wrapper around a plain function. Tools that need the farmer get
their user_id from `runtime.context` (a ToolRuntime argument LangChain injects and hides from
the LLM), so the model never has to type the id and can't swap it for another farmer's.
"""
from langchain.tools import ToolRuntime, tool

from agent.agents.fertilize.tools.dosage_calculator import dosage_calculator_by_user
from agent.agents.fertilize.tools.fertilize_history import fertilize_history_by_user
from agent.agents.fertilize.tools.tea_land_data import tea_land_data_by_user
from agent.agents.shared_tools import weather_forecast
from agent.agents.shared_tools.schemas import LocationInput
from agent.state.agent_state import ChatContext


@tool("TeaLandDataByUser")
def tea_land_data_tool(runtime: ToolRuntime[ChatContext]):
    """Retrieves the farmer's tea land data (location, land size, slope, number of plants). Takes no input - it already knows the farmer."""
    return tea_land_data_by_user(runtime.context.user_id)


@tool("GetFertilizeHistoryByUser")
def fertilize_history_tool(runtime: ToolRuntime[ChatContext]) -> str:
    """Fetches the farmer's fertilization history (dates and amounts), newest first. Takes no input - it already knows the farmer."""
    return fertilize_history_by_user(runtime.context.user_id)


@tool("DosageCalculator")
def dosage_calculator_tool(runtime: ToolRuntime[ChatContext]) -> str:
    """Calculates the recommended fertilizer dosage in kg for the farmer's land. Takes no input - it already knows the farmer."""
    return dosage_calculator_by_user(runtime.context.user_id)


@tool("WeatherForecast", args_schema=LocationInput)
def weather_forecast_tool(location: str):
    """Fetches a 14-day weather forecast (daily rain chance and precipitation in mm) for a location."""
    return weather_forecast(location)


FERTILIZE_TOOLS = [tea_land_data_tool, fertilize_history_tool, dosage_calculator_tool, weather_forecast_tool]

__all__ = ["FERTILIZE_TOOLS"]
