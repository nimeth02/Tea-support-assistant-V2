"""Tools shared by more than one agent, one module per tool.

Rules:
  * Keep descriptions generic; put "when to call it" guidance in each agent's prompt.
  * Never import from a sibling agent package (disease/, fertilize/, ...) here;
    the dependency direction is agent -> shared_tools, never the other way.
  * A tool that needs a dependency (DB connection, index, ...) should expose a factory
    that takes it explicitly instead of a module-level singleton.
"""
from agent.agents.shared_tools.weather_forecast import weather_forecast_tool
from agent.agents.shared_tools.web_search import web_search_tool

__all__ = ["weather_forecast_tool", "web_search_tool"]
