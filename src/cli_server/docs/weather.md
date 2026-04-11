Access National Weather Service data for US locations, including forecasts and alerts.
/home/samaritan011/mcp/cli_agent/src_ts/cli_server/weather_ts_server/weather_cli.ts
------------------------------
Function: getAlerts
async function getAlerts(state: string): Promise<string>
Description: Get weather alerts for a US state.

Args:
    state: Two-letter US state code (e.g. CA, NY)
------------------------------
Function: getForecast
async function getForecast(latitude: number, longitude: number): Promise<string>
Description: Get weather forecast for a location.

Args:
    latitude: The latitude of the location.
    longitude: The longitude of the location.
