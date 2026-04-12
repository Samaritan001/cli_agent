import { describe, expect, it } from "vitest";
import { computeBanditReward } from "./reward";

describe("computeBanditReward", () => {
  it("returns mid-range for moderate signals", () => {
    const r = computeBanditReward({
      nextUserMessageLen: 200,
      msSinceLastTurnEnd: 60_000,
      prevAssistantLen: 400,
    });
    expect(r).toBeGreaterThan(0.2);
    expect(r).toBeLessThanOrEqual(1);
  });

  it("penalizes very short follow-up user messages", () => {
    const low = computeBanditReward({
      nextUserMessageLen: 2,
      msSinceLastTurnEnd: 10_000,
      prevAssistantLen: 100,
    });
    const high = computeBanditReward({
      nextUserMessageLen: 200,
      msSinceLastTurnEnd: 10_000,
      prevAssistantLen: 100,
    });
    expect(high).toBeGreaterThan(low);
  });
});
