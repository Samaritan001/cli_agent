/**
 * Offline analysis of `eval.jsonl`: profile-hash churn from `turn_end`, and bandit reward stats from
 * `turn_start` (`banditRewardPreviousArm`) when learning is enabled — not LLM replay.
 */
import fs from "node:fs";
import type { EvalEvent } from "./eval_log";

export type EvalReplayReport = {
  path: string;
  turnEndEvents: number;
  profileHashChanges: number;
  /** `turn_start` events that include `banditRewardPreviousArm` (learning). */
  banditRewardEvents: number;
  /** Mean of logged bandit rewards (same definition as live `computeBanditReward`). */
  meanBanditReward?: number;
  sumBanditReward?: number;
  /** `turn_start` events with correctionSignal === true. */
  correctionSignalCount: number;
  warnings: string[];
};

/**
 * Read `.cli_agent/eval.jsonl`: profile stability, bandit rewards, correction signals.
 * Does not replay model calls — only inspects logged hashes and rule signals.
 */
export function analyzeEvalLog(evalJsonlPath: string): EvalReplayReport {
  const warnings: string[] = [];
  if (!fs.existsSync(evalJsonlPath)) {
    return {
      path: evalJsonlPath,
      turnEndEvents: 0,
      profileHashChanges: 0,
      banditRewardEvents: 0,
      correctionSignalCount: 0,
      warnings: ["file not found"],
    };
  }
  const raw = fs.readFileSync(evalJsonlPath, "utf8");
  const lines = raw.trim().split("\n").filter(Boolean);
  const hashes: string[] = [];
  let banditRewardEvents = 0;
  let sumBanditReward = 0;
  let correctionSignalCount = 0;
  for (const line of lines) {
    try {
      const ev = JSON.parse(line) as EvalEvent;
      if (ev.kind === "turn_end") hashes.push(ev.profileHash);
      if (ev.kind === "turn_start") {
        if (ev.correctionSignal === true) correctionSignalCount += 1;
        if (typeof ev.banditRewardPreviousArm === "number") {
          banditRewardEvents += 1;
          sumBanditReward += ev.banditRewardPreviousArm;
        }
      }
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
    banditRewardEvents,
    correctionSignalCount,
    ...(banditRewardEvents > 0
      ? {
          meanBanditReward: sumBanditReward / banditRewardEvents,
          sumBanditReward,
        }
      : {}),
    warnings,
  };
}
