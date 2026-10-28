"""WeatherForecast tool (shared): 14-day rain forecast for a location (weatherapi.com).

Generic on purpose: agent-specific guidance (e.g. "look up the farmer's land location
first") belongs in the calling agent's prompt, not here.
"""
from typing import Dict, List

import requests
from langchain_core.tools import Tool

from agent.agents.shared_tools.schemas import LocationInput

from agent.config import settings


def weather_forecast(location: str) -> List[Dict]:
    """Return daily rain chance and precipitation for the next 14 days."""
    print("_weather_forecast")
    print(location)
    url = f"{settings.WHETHER_BASE_URL}/forecast.json"
    params = {"key": settings.WHETHER_API, "q": location, "days": 14}
    response = requests.get(url, params=params, timeout=10)
    response.raise_for_status()
    data = response.json()

    daily = data.get("forecast", {}).get("forecastday", [])
    forecast = []
    for day in daily:
        day_info = day.get("day", {})
        forecast.append(
            {
                "date": day.get("date"),
                "chance_of_rain": day_info.get("daily_chance_of_rain"),
                "total_precip_mm": day_info.get("totalprecip_mm"),
            }
        )
    print(forecast)
    return forecast


weather_forecast_tool = Tool(
    name="WeatherForecast",
    func=weather_forecast,
    args_schema=LocationInput,
    description=(
        "Fetches a 14-day weather forecast (daily rain chance and precipitation in mm) for a location. "
        "Input: location name (string, e.g., 'Kandy')."
    ),
)
