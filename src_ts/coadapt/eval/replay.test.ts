/**
 * Unit tests for offline eval log analysis (`analyzeEvalLog`).
 */
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { afterEach, describe, expect, it } from "vitest";
import { appendEvalLog, evalLogPath } from "./eval_log";
import { analyzeEvalLog } from "./replay";

describe("analyzeEvalLog", () => {
  let dir: string;

  afterEach(() => {
    if (dir && fs.existsSync(dir)) fs.rmSync(dir, { recursive: true, force: true });
  });

  it("reports missing file", () => {
    const p = path.join(os.tmpdir(), `eval-missing-${Date.now()}.jsonl`);
    const r = analyzeEvalLog(p);
    expect(r.turnEndEvents).toBe(0);
    expect(r.banditRewardEvents).toBe(0);
    expect(r.correctionSignalCount).toBe(0);
    expect(r.warnings.some((w) => w.includes("not found"))).toBe(true);
  });

  it("counts turn_end events and hash changes", () => {
    dir = fs.mkdtempSync(path.join(os.tmpdir(), "coadapt-eval-"));
    appendEvalLog(dir, {
      kind: "turn_end",
      ts: "t1",
      sessionId: "s",
      userTurnIndex: 0,
      turnId: "00000000-0000-4000-8000-000000000001",
      profileHash: "aaaaaaaaaaaaaaaa",
      toolCallsCount: 0,
      extractedFactsCount: 0,
    });
    appendEvalLog(dir, {
      kind: "turn_end",
      ts: "t2",
      sessionId: "s",
      userTurnIndex: 1,
      turnId: "00000000-0000-4000-8000-000000000002",
      profileHash: "bbbbbbbbbbbbbbbb",
      toolCallsCount: 1,
      extractedFactsCount: 0,
    });
    const p = evalLogPath(dir);
    const r = analyzeEvalLog(p);
    expect(r.turnEndEvents).toBe(2);
    expect(r.profileHashChanges).toBe(1);
    expect(r.banditRewardEvents).toBe(0);
    expect(r.correctionSignalCount).toBe(0);
  });

  it("aggregates bandit rewards from turn_start", () => {
    dir = fs.mkdtempSync(path.join(os.tmpdir(), "coadapt-eval-br-"));
    appendEvalLog(dir, {
      kind: "turn_start",
      ts: "t0",
      sessionId: "s",
      userTurnIndex: 1,
      rulesApplied: [],
      profileHashBefore: "a",
      profileHashAfter: "b",
      banditRewardPreviousArm: 0.6,
      banditPreviousArmIndex: 0,
    });
    appendEvalLog(dir, {
      kind: "turn_start",
      ts: "t1",
      sessionId: "s",
      userTurnIndex: 2,
      rulesApplied: [],
      profileHashBefore: "b",
      profileHashAfter: "c",
      banditRewardPreviousArm: 0.8,
      banditPreviousArmIndex: 1,
    });
    const r = analyzeEvalLog(evalLogPath(dir));
    expect(r.banditRewardEvents).toBe(2);
    expect(r.sumBanditReward).toBeCloseTo(1.4, 5);
    expect(r.meanBanditReward).toBeCloseTo(0.7, 5);
  });
});
