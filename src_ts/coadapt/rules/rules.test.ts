/**
 * Unit tests for rule-based profile updates (`applyRuleBasedUpdates`).
 */
import { describe, expect, it } from "vitest";
import { defaultAIProfile, defaultUserProfile } from "../profile/defaults";
import { applyRuleBasedUpdates } from "./rules";

describe("applyRuleBasedUpdates", () => {
  it("does not change profiles on first user turn signals", () => {
    const u = defaultUserProfile();
    const a = defaultAIProfile();
    const out = applyRuleBasedUpdates(u, a, {
      completedUserTurns: 0,
      prevAssistantLen: 0,
      currentUserLen: 10,
      hasPreviousAssistant: false,
    });
    expect(out.user).toEqual(u);
    expect(out.ai).toEqual(a);
    expect(out.log).toEqual([]);
    expect(out.appliedRules).toEqual([]);
  });

  it("increases depth after enough completed turns", () => {
    const u = defaultUserProfile();
    const a = defaultAIProfile();
    const beforeDetail = u.cognitiveStyle.detailVsHighLevel;
    const out = applyRuleBasedUpdates(u, a, {
      completedUserTurns: 2,
      prevAssistantLen: 100,
      currentUserLen: 50,
      hasPreviousAssistant: true,
    });
    expect(out.user.cognitiveStyle.detailVsHighLevel).toBeGreaterThan(beforeDetail);
    expect(out.ai.reasoning.depthOfExplanation).toBeGreaterThan(a.reasoning.depthOfExplanation);
    expect(out.log.some((l) => l.includes("followups"))).toBe(true);
    expect(out.appliedRules.some((r) => r.id === "followups_depth")).toBe(true);
  });

  it("shifts toward concise when user sends short reply after long assistant", () => {
    const u = defaultUserProfile();
    const a = defaultAIProfile();
    const before = a.style.conciseVsExploratory;
    const out = applyRuleBasedUpdates(u, a, {
      completedUserTurns: 1,
      prevAssistantLen: 800,
      currentUserLen: 20,
      hasPreviousAssistant: true,
    });
    expect(out.ai.style.conciseVsExploratory).toBeLessThan(before);
    expect(out.log.some((l) => l.includes("long_reply_short_user"))).toBe(true);
    expect(out.appliedRules.some((r) => r.id === "long_reply_short_user")).toBe(true);
  });
});
