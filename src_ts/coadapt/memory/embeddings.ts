/**
 * Text embeddings for memory retrieval: OpenAI `text-embedding-3-small` at 256 dims when configured,
 * otherwise deterministic hash vectors; cosine similarity helpers.
 */
import crypto from "node:crypto";
import OpenAI from "openai";

const DIM = 256;

let openaiClient: OpenAI | null | undefined;

function getOpenAI(): OpenAI | null {
  if (process.env.COADAPT_USE_HASH_ONLY === "1") return null;
  if (!process.env.OPENAI_API_KEY) return null;
  if (openaiClient === undefined) {
    openaiClient = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });
  }
  return openaiClient;
}

/** Deterministic embedding when OpenAI is off or unavailable. */
export function hashEmbedding(text: string): number[] {
  const v = new Array<number>(DIM).fill(0);
  const norm = text.trim().toLowerCase();
  for (let i = 0; i < norm.length; i++) {
    const slice = norm.slice(i, Math.min(i + 8, norm.length));
    const h = crypto.createHash("sha256").update(slice).digest();
    for (let b = 0; b < DIM; b++) {
      v[b] += b < h.length ? h[b] / 255 : 0;
    }
  }
  const mag = Math.sqrt(v.reduce((s, x) => s + x * x, 0)) || 1;
  return v.map((x) => x / mag);
}

function l2Normalize(e: number[]): number[] {
  const mag = Math.sqrt(e.reduce((s, x) => s + x * x, 0)) || 1;
  return e.map((x) => x / mag);
}

export function getEmbeddingBackendLabel(): "openai" | "hash" {
  return getOpenAI() ? "openai" : "hash";
}

export async function embedText(text: string): Promise<number[]> {
  const trimmed = text.slice(0, 12000);
  const client = getOpenAI();
  if (client) {
    try {
      const res = await client.embeddings.create({
        model: "text-embedding-3-small",
        input: trimmed,
        dimensions: DIM,
      });
      const e = res.data[0]?.embedding;
      if (e && e.length === DIM) return l2Normalize(e);
    } catch {
      // fall through
    }
  }
  return hashEmbedding(trimmed);
}

export function cosine(a: number[], b: number[]): number {
  let s = 0;
  for (let i = 0; i < Math.min(a.length, b.length); i++) s += a[i] * b[i];
  return s;
}

export { DIM };
