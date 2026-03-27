"""
Access National Weather Service data for US locations, including forecasts and alerts.
"""

import sys
import logging

import asyncio

import inspect
from typing import Any, Annotated, Callable
from pydantic import Field

import httpx

from weather_helpers import make_nws_request, format_alert

LOG_FORMAT = "\033[32m%(levelname)s\033[0m:    %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger("weather")


# Constants
NWS_API_BASE = "https://api.weather.gov"

# Implementing tool execution
async def get_alerts(
    state: Annotated[str, Field(description="Two-letter US state code (e.g. CA, NY)")]
) -> str:
    """Get weather alerts for a US state.
    
    Args:
        state: Two-letter US state code (e.g. CA, NY)
    """
    url = f"{NWS_API_BASE}/alerts/active?area={state.upper()}"
    data = await make_nws_request(url)
    if not data or "features" not in data:
        return "Unable to fetch alerts or no alerts found."
    
    features = data["features"]
    if not features:
        return "No active alerts for this state."
    
    alerts = [format_alert(feature) for feature in features]
    return "\n---\n".join(alerts)


async def get_forecast(
    latitude: Annotated[float, Field(description="The latitude of the location.")],
    longitude: Annotated[float, Field(description="The longitude of the location.")]
) -> str:
    """Get weather forecast for a location.

    Args:
        latitude: The latitude of the location.
        longitude: The longitude of the location.
    """

    # First get the forecast grid endpoint
    points_url = f"{NWS_API_BASE}/points/{latitude},{longitude}"
    points_data = await make_nws_request(points_url)

    if not points_data:
        return "Unable to fetch forecast data for this location."

    # Get the forecast URL from the points response
    forecast_url = points_data["properties"]["forecast"]
    forecast_data = await make_nws_request(forecast_url)

    if not forecast_data:
        return "Unable to fetch detailed forecast."
    
    # Format the periods into a readable forecast string
    periods = forecast_data["properties"]["periods"]
    forecasts = []
    for period in periods[:5]: # Only show next 5 periods
        forecast = f"""
{period["name"]}:
Temperature: {period["temperature"]}°{period["temperatureUnit"]}
Wind: {period["windSpeed"]} {period["windDirection"]}
Forecast: {period["detailedForecast"]}
"""
        forecasts.append(forecast)
    
    return "\n---\n".join(forecasts)



# def main():
#     result = asyncio.run(get_forecast(34.0522, -118.2437))
#     print(result[:200])
#     logging.info("Example forecast retrieval complete.")

# if __name__ == "__main__":
#     main()



