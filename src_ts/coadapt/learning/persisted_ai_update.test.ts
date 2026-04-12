import { describe, expect, it } from "vitest";
import { defaultAIProfile } from "../profile/defaults";
import { nudgePersistedAiFromBanditReward } from "./persisted_ai_update";

describe("nudgePersistedAiFromBanditReward", () => {
  it("moves style toward terse arm when reward is high", () => {
    const a = defaultAIProfile();
    const out = nudgePersistedAiFromBanditReward(a, "terse", 1);
    expect(out.style.conciseVsExploratory).toBeLessThan(a.style.conciseVsExploratory);
    expect(out.reasoning.depthOfExplanation).toBeLessThan(a.reasoning.depthOfExplanation);
  });

  it("no-op when reward is zero", () => {
    const a = defaultAIProfile();
    const out = nudgePersistedAiFromBanditReward(a, "detailed", 0);
    expect(out).toBe(a);
  });
});
