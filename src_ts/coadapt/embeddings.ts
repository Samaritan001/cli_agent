import crypto from "node:crypto";

const DIM = 256;

/** Deterministic embedding for Phase 1 (no extra API calls; consistent dimension). */
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

export async function embedText(text: string): Promise<number[]> {
  return hashEmbedding(text);
}

export function cosine(a: number[], b: number[]): number {
  let s = 0;
  for (let i = 0; i < Math.min(a.length, b.length); i++) s += a[i] * b[i];
  return s;
}

export { DIM };
