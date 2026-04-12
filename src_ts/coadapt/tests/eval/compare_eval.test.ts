import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { afterEach, describe, expect, it } from "vitest";
import { compareEvalLogs } from "../../eval/compare_eval";

describe("compareEvalLogs", () => {
  let dir: string;

  afterEach(() => {
    if (dir && fs.existsSync(dir)) fs.rmSync(dir, { recursive: true, force: true });
  });

  it("computes delta mean bandit reward when both logs have rewards", () => {
    dir = fs.mkdtempSync(path.join(os.tmpdir(), "coadapt-cmp-"));
    const a = path.join(dir, "a.jsonl");
    const b = path.join(dir, "b.jsonl");
    fs.writeFileSync(
      a,
      `${JSON.stringify({
        kind: "turn_start",
        ts: "t",
        sessionId: "s",
        userTurnIndex: 1,
        rulesApplied: [],
        profileHashBefore: "x",
        profileHashAfter: "y",
        banditRewardPreviousArm: 0.8,
        banditPreviousArmIndex: 0,
      })}\n`
    );
    fs.writeFileSync(
      b,
      `${JSON.stringify({
        kind: "turn_start",
        ts: "t",
        sessionId: "s",
        userTurnIndex: 1,
        rulesApplied: [],
        profileHashBefore: "x",
        profileHashAfter: "y",
        banditRewardPreviousArm: 0.5,
        banditPreviousArmIndex: 0,
      })}\n`
    );
    const r = compareEvalLogs(a, b);
    expect(r.deltaMeanBanditReward).toBeCloseTo(0.3, 5);
  });
});
