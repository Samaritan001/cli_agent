import logging

from typing import Any, Annotated
from pydantic import Field

import httpx

# Constants
USER_AGENT = "weather-app/1.0"

# Helper functions
async def make_nws_request(url: str) -> dict[str, Any] | None:
    """Make a request to the NWS API with proper error handling."""
    headers = {"User-Agent": USER_AGENT, "Accept": "application/geo+json"}
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(url, headers=headers, timeout=30.0)
            response.raise_for_status()
            return response.json()
        except Exception:
            logging.exception(f"Error making request to NWS API: {url}")
            return None

def format_alert(feature: dict) -> str:
    """Format an alert feature into a readable string."""
    properties = feature["properties"]
    event = properties.get("event", "Unknown")
    area = properties.get("areaDesc", "Unknown")
    severity = properties.get("severity", "Unknown")
    description = properties.get("description", "No description available")
    instructions = properties.get("instructions", "No specific instructions provided")
    return f"""
Event: {event}
Area: {area}\nSeverity: {severity}
Severity: {severity}
Description: {description}
Instructions: {instructions}
    """