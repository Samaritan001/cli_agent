/**
 * Access National Weather Service data for US locations, including forecasts and alerts.
 */

import { makeNwsRequest, formatAlert } from "./weather_helpers";

const NWS_API_BASE = "https://api.weather.gov";

export async function getAlerts(state: string): Promise<string> {
  const url = `${NWS_API_BASE}/alerts/active?area=${state.toUpperCase()}`;
  const data = await makeNwsRequest(url);
  if (!data || !data.features) return "Unable to fetch alerts or no alerts found.";
  const features = data.features as any[];
  if (!features.length) return "No active alerts for this state.";
  return features.map(formatAlert).join("\n---\n");
}

export async function getForecast(latitude: number, longitude: number): Promise<string> {
  const lat = Math.round(latitude * 10000) / 10000;
  const lon = Math.round(longitude * 10000) / 10000;

  const pointsUrl = `${NWS_API_BASE}/points/${lat},${lon}`;
  const pointsData = await makeNwsRequest(pointsUrl);
  if (!pointsData) return "Unable to fetch forecast data for this location.";

  const forecastUrl = pointsData?.properties?.forecast;
  if (!forecastUrl) return "Unable to fetch detailed forecast.";

  const forecastData = await makeNwsRequest(forecastUrl);
  if (!forecastData) return "Unable to fetch detailed forecast.";

  const periods: any[] = forecastData?.properties?.periods ?? [];
  const chunks = periods.slice(0, 5).map((period) => {
    return `\n${period.name}:\nTemperature: ${period.temperature}°${period.temperatureUnit}\nWind: ${period.windSpeed} ${period.windDirection}\nForecast: ${period.detailedForecast}\n`;
  });
  return chunks.join("\n---\n");
}

