/**
 * Persists conversation chunks with embeddings; retrieves top-k by cosine similarity to a query.
 */
import fs from "node:fs";
import path from "node:path";
import { cosine, embedText } from "./embeddings";

export type MemoryChunk = {
  id: string;
  text: string;
  embedding: number[];
  createdAt: string;
};

export type MemoryStoreData = {
  version: 1;
  chunks: MemoryChunk[];
};

const MAX_CHUNKS = 500;

export class MemoryStore {
  private chunks: MemoryChunk[] = [];
  private readonly filePath: string;

  constructor(filePath: string) {
    this.filePath = filePath;
    this.load();
  }

  private load(): void {
    if (!fs.existsSync(this.filePath)) return;
    try {
      const raw = JSON.parse(fs.readFileSync(this.filePath, "utf8")) as MemoryStoreData;
      if (raw.version === 1 && Array.isArray(raw.chunks)) this.chunks = raw.chunks.slice(-MAX_CHUNKS);
    } catch {
      this.chunks = [];
    }
  }

  private persist(): void {
    fs.mkdirSync(path.dirname(this.filePath), { recursive: true });
    const out: MemoryStoreData = { version: 1, chunks: this.chunks.slice(-MAX_CHUNKS) };
    fs.writeFileSync(this.filePath, JSON.stringify(out), "utf8");
  }

  async retrieveWithScores(query: string, k: number): Promise<{ chunk: MemoryChunk; score: number }[]> {
    if (this.chunks.length === 0 || !query.trim()) return [];
    const q = await embedText(query);
    const scored = this.chunks.map((c) => ({ chunk: c, score: cosine(q, c.embedding) }));
    scored.sort((a, b) => b.score - a.score);
    return scored.slice(0, k);
  }

  async retrieve(query: string, k: number): Promise<MemoryChunk[]> {
    const hits = await this.retrieveWithScores(query, k);
    return hits.map((h) => h.chunk);
  }

  async addTurnChunk(
    userText: string,
    assistantText: string,
    opts?: { facts?: string[] }
  ): Promise<void> {
    let text = `User: ${userText.slice(0, 4000)}\nAssistant: ${assistantText.slice(0, 4000)}`;
    if (opts?.facts && opts.facts.length > 0) {
      text += `\nFacts:\n${opts.facts.map((f) => `- ${f}`).join("\n")}`;
    }
    const embedding = await embedText(text);
    const chunk: MemoryChunk = {
      id: `${Date.now()}-${Math.random().toString(36).slice(2, 9)}`,
      text: text.slice(0, 12000),
      embedding,
      createdAt: new Date().toISOString(),
    };
    this.chunks.push(chunk);
    if (this.chunks.length > MAX_CHUNKS) this.chunks = this.chunks.slice(-MAX_CHUNKS);
    this.persist();
  }
}
