"""14-day rain forecast for a location (weatherapi.com). Framework-free; see __init__.py."""
import logging
from typing import Dict, List, Union

import requests

from agent.config import settings

logger = logging.getLogger("agent.tools.weather")


def weather_forecast(location: str) -> Union[List[Dict], str]:
    """Return daily rain chance and precipitation for the next 14 days, or an error message."""
    if not settings.WHETHER_API:
        return "Weather forecast is not available (no weather API key configured)."
    try:
        response = requests.get(
            f"{settings.WHETHER_BASE_URL}/forecast.json",
            params={"key": settings.WHETHER_API, "q": location, "days": 14},
            timeout=10,
        )
        response.raise_for_status()
    except requests.RequestException as e:
        logger.warning("weather_forecast_failed location=%s error=%s", location, e)
        return f"Could not get the weather forecast for {location} right now."

    days = response.json().get("forecast", {}).get("forecastday", [])
    return [
        {
            "date": day.get("date"),
            "chance_of_rain": day.get("day", {}).get("daily_chance_of_rain"),
            "total_precip_mm": day.get("day", {}).get("totalprecip_mm"),
        }
        for day in days
    ]
