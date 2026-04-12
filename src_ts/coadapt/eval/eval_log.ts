import fs from "node:fs";
import path from "node:path";

export type EvalEvent =
  | {
      kind: "turn_start";
      ts: string;
      sessionId: string;
      userTurnIndex: number;
      rulesApplied: { id: string; message: string }[];
      profileHashBefore: string;
      profileHashAfter: string;
    }
  | {
      kind: "context_built";
      ts: string;
      sessionId: string;
      userTurnIndex: number;
      embeddingBackend: "openai" | "hash";
      retrieval: { chunkId: string; score: number }[];
    }
  | {
      kind: "turn_end";
      ts: string;
      sessionId: string;
      userTurnIndex: number;
      turnId: string;
      profileHash: string;
      toolCallsCount: number;
      extractedFactsCount: number;
    };

export function evalLogPath(dataDir: string): string {
  return path.join(dataDir, "eval.jsonl");
}

export function appendEvalLog(dataDir: string, event: EvalEvent): void {
  const p = evalLogPath(dataDir);
  fs.mkdirSync(path.dirname(p), { recursive: true });
  fs.appendFileSync(p, `${JSON.stringify(event)}\n`, "utf8");
}
