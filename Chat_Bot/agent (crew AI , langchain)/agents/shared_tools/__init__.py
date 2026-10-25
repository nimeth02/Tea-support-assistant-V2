"""Tool logic shared by more than one expert, written as plain Python functions.

Why plain functions: the experts use different frameworks (CrewAI, LangChain, LangGraph), and
each framework has its own "tool" class. Keeping the real work framework-free means it is
written and tested once; each expert wraps it in a thin adapter for its own framework
(see agents/disease/tools and agents/fertilize/tools).

Rules:
  * No framework imports here.
  * Never import from a sibling expert package (disease/, fertilize/, ...).
"""
from agent.agents.shared_tools.weather_forecast import weather_forecast
from agent.agents.shared_tools.web_search import duckduckgo_search

__all__ = ["weather_forecast", "duckduckgo_search"]
