/**
 * Append-only structured events (`turn_start`, `context_built`, `turn_end`) to `eval.jsonl` for analytics.
 */
import fs from "node:fs";
import path from "node:path";
import type { TurnMetrics } from "./metrics";

export type EvalEvent =
  | {
      kind: "turn_start";
      ts: string;
      sessionId: string;
      userTurnIndex: number;
      rulesApplied: { id: string; message: string }[];
      profileHashBefore: string;
      profileHashAfter: string;
      /** Learning: reward applied to previous turn's bandit arm when user sends the next message. */
      banditRewardPreviousArm?: number;
      banditPreviousArmIndex?: number;
      /** Heuristic: user text looks like a correction / disagreement. */
      correctionSignal?: boolean;
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
      banditArmId?: string;
      banditArmIndex?: number;
      turnMetrics?: TurnMetrics;
    };

export function evalLogPath(dataDir: string): string {
  return path.join(dataDir, "eval.jsonl");
}

export function appendEvalLog(dataDir: string, event: EvalEvent): void {
  const p = evalLogPath(dataDir);
  fs.mkdirSync(path.dirname(p), { recursive: true });
  fs.appendFileSync(p, `${JSON.stringify(event)}\n`, "utf8");
}
