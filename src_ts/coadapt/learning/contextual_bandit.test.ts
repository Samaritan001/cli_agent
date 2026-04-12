import fs from "node:fs";
import os from "node:os";
import { afterEach, describe, expect, it, vi } from "vitest";
import { defaultUserProfile } from "../profile/defaults";
import { buildContextFeatures } from "./context_features";
import { ContextualLinUCBBandit, solveLinearSystem } from "./contextual_bandit";

describe("solveLinearSystem", () => {
  it("solves 2×2", () => {
    const A = [
      [2, 1],
      [1, 3],
    ];
    const x = solveLinearSystem(A, [1, 2]);
    expect(x).not.toBeNull();
    expect(x![0]).toBeCloseTo(0.2, 5);
    expect(x![1]).toBeCloseTo(0.6, 5);
  });
});

describe("ContextualLinUCBBandit", () => {
  let tmp: string;
  let randomSpy: ReturnType<typeof vi.spyOn>;

  afterEach(() => {
    randomSpy?.mockRestore();
    if (tmp && fs.existsSync(tmp)) fs.unlinkSync(tmp);
  });

  it("persists updates and prefers arm with higher observed reward (no exploration)", () => {
    tmp = `${os.tmpdir()}/linucb-${Date.now()}.json`;
    randomSpy = vi.spyOn(Math, "random").mockReturnValue(0.99);
    const b = new ContextualLinUCBBandit(tmp);
    const x = buildContextFeatures(defaultUserProfile(), { userMessageLen: 100, completedUserTurns: 0 });
    const a0 = b.selectArm(x);
    b.update(a0, 0.2, x);
    const a1 = b.selectArm(x);
    b.update(a1, 0.95, x);
    const a2 = b.selectArm(x);
    expect(a2).toBe(a1);
    const raw = JSON.parse(fs.readFileSync(tmp, "utf8")) as { version: number; arms: { updateCount: number }[] };
    expect(raw.version).toBe(2);
    expect(raw.arms.some((arm) => arm.updateCount >= 1)).toBe(true);
  });
});
