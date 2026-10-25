"""Tools shared by more than one agent, one module per tool.

Rules:
  * Keep descriptions generic; put "when to call it" guidance in each agent's prompt.
  * Never import from a sibling agent package (disease/, fertilize/, ...) here;
    the dependency direction is agent -> shared_tools, never the other way.
  * Every tool is a TeaTool (shared_tools/base.py), built per chat turn with that turn's state.
"""
from agent.agents.shared_tools.base import TeaTool
from agent.agents.shared_tools.weather_forecast import WeatherForecastTool
from agent.agents.shared_tools.web_search import WebSearchTool

__all__ = ["TeaTool", "WeatherForecastTool", "WebSearchTool"]
