import fs from "node:fs";
import type { EvalEvent } from "./eval_log";

export type EvalReplayReport = {
  path: string;
  turnEndEvents: number;
  profileHashChanges: number;
  warnings: string[];
};

/**
 * Read `.cli_agent/eval.jsonl` and summarize profile stability (Phase 1 eval v0).
 * Does not replay model calls — only inspects logged hashes and rule signals.
 */
export function analyzeEvalLog(evalJsonlPath: string): EvalReplayReport {
  const warnings: string[] = [];
  if (!fs.existsSync(evalJsonlPath)) {
    return {
      path: evalJsonlPath,
      turnEndEvents: 0,
      profileHashChanges: 0,
      warnings: ["file not found"],
    };
  }
  const raw = fs.readFileSync(evalJsonlPath, "utf8");
  const lines = raw.trim().split("\n").filter(Boolean);
  const hashes: string[] = [];
  for (const line of lines) {
    try {
      const ev = JSON.parse(line) as EvalEvent;
      if (ev.kind === "turn_end") hashes.push(ev.profileHash);
    } catch {
      warnings.push("skipped invalid json line");
    }
  }
  let changes = 0;
  for (let i = 1; i < hashes.length; i++) {
    if (hashes[i] !== hashes[i - 1]) changes += 1;
  }
  if (hashes.length > 6 && changes >= hashes.length - 1) {
    warnings.push("profile hash changes on almost every turn — possible oscillation or noisy rules");
  }
  return {
    path: evalJsonlPath,
    turnEndEvents: hashes.length,
    profileHashChanges: changes,
    warnings,
  };
}
