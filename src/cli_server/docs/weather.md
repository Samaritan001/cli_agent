Access National Weather Service data for US locations, including forecasts and alerts.
/home/samaritan011/mcp/cli_agent/src/cli_server/weather_py_server/weather_cli.py
------------------------------
Function: get_alerts
async def get_alerts(state: Annotated[str, Field(description='Two-letter US state code (e.g. CA, NY)')]) -> str:
Description: Get weather alerts for a US state.

Args:
    state: Two-letter US state code (e.g. CA, NY)
------------------------------
Function: get_forecast
async def get_forecast(latitude: Annotated[float, Field(description='The latitude of the location.')], longitude: Annotated[float, Field(description='The longitude of the location.')]) -> str:
Description: Get weather forecast for a location.

Args:
    latitude: The latitude of the location.
    longitude: The longitude of the location.
