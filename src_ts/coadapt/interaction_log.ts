import fs from "node:fs";
import path from "node:path";
import type { InteractionTurn } from "./schemas";

export function appendInteractionLog(filePath: string, turn: InteractionTurn): void {
  fs.mkdirSync(path.dirname(filePath), { recursive: true });
  fs.appendFileSync(filePath, `${JSON.stringify(turn)}\n`, "utf8");
}

export function ensureLogDir(dir: string): void {
  fs.mkdirSync(dir, { recursive: true });
}

export function interactionLogPath(dataDir: string, sessionId: string): string {
  return path.join(dataDir, "logs", `${sessionId}.jsonl`);
}
