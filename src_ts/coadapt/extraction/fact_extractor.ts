/**
 * Derives short factual bullets from a user/assistant pair via LLM JSON or heuristic fallback.
 */
import OpenAI from "openai";

const FACTS_SCHEMA = `{"facts":["string"]}`;

function heuristicFacts(userText: string, assistantText: string): string[] {
  const out: string[] = [];
  const u = userText.trim().split(/\s+/).slice(0, 12).join(" ");
  if (u) out.push(`User asked (excerpt): ${u.slice(0, 120)}`);
  const sentences = assistantText
    .split(/(?<=[.!?])\s+/)
    .map((s) => s.trim())
    .filter(Boolean);
  for (const s of sentences.slice(0, 2)) {
    if (s.length > 8) out.push(`Assistant: ${s.slice(0, 200)}`);
    if (out.length >= 3) break;
  }
  return out.slice(0, 3);
}

/**
 * Optional LLM-assisted fact bullets (requires OPENAI_API_KEY unless skipped).
 * Falls back to heuristic excerpts when API is missing or fails.
 */
export async function extractFacts(userText: string, assistantText: string): Promise<string[]> {
  if (process.env.COADAPT_SKIP_FACTS === "1") return [];
  const key = process.env.OPENAI_API_KEY;
  if (!key) return heuristicFacts(userText, assistantText);

  const model = process.env.COADAPT_FACT_MODEL ?? "gpt-4o-mini";
  try {
    const openai = new OpenAI({ apiKey: key });
    const res = await openai.chat.completions.create({
      model,
      max_tokens: 300,
      messages: [
        {
          role: "system",
          content:
            "Extract at most 3 short factual bullets about what happened in this exchange. " +
            `Reply with JSON only, shape ${FACTS_SCHEMA}. Use empty array if nothing factual.`,
        },
        {
          role: "user",
          content: `User:\n${userText.slice(0, 6000)}\n\nAssistant:\n${assistantText.slice(0, 6000)}`,
        },
      ],
      response_format: { type: "json_object" },
    });
    const raw = res.choices[0]?.message?.content ?? "{}";
    const parsed = JSON.parse(raw) as { facts?: unknown };
    const facts = parsed.facts;
    if (!Array.isArray(facts)) return heuristicFacts(userText, assistantText);
    return facts
      .filter((x): x is string => typeof x === "string")
      .map((s) => s.trim())
      .filter(Boolean)
      .slice(0, 3);
  } catch {
    return heuristicFacts(userText, assistantText);
  }
}
