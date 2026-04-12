import fs from "node:fs";
import os from "node:os";
import { afterEach, describe, expect, it } from "vitest";
import { EpsilonGreedyBandit } from "../../learning/bandit";

describe("EpsilonGreedyBandit", () => {
  let tmp: string;

  afterEach(() => {
    if (tmp && fs.existsSync(tmp)) fs.unlinkSync(tmp);
  });

  it("updates running mean reward", () => {
    tmp = `${os.tmpdir()}/bandit-${Date.now()}.json`;
    const b = new EpsilonGreedyBandit(tmp, 0);
    b.update(0, 1);
    b.update(0, 0);
    const arms = b.getArmsSnapshot();
    expect(arms[0]!.count).toBe(2);
    expect(arms[0]!.meanReward).toBeCloseTo(0.5, 5);
  });
});
