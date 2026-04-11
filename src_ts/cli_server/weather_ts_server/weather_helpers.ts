const USER_AGENT = "weather-app/1.0";

export async function makeNwsRequest(url: string): Promise<any | null> {
  const res = await fetch(url, {
    headers: {
      "user-agent": USER_AGENT,
      accept: "application/geo+json",
    },
  });
  if (!res.ok) return null;
  return await res.json();
}

export function formatAlert(feature: any): string {
  const p = feature?.properties ?? {};
  const event = p.event ?? "Unknown";
  const area = p.areaDesc ?? "Unknown";
  const severity = p.severity ?? "Unknown";
  const description = p.description ?? "No description available";
  const instructions = p.instructions ?? "No specific instructions provided";
  return `\nEvent: ${event}\nArea: ${area}\nSeverity: ${severity}\nDescription: ${description}\nInstructions: ${instructions}\n`;
}

