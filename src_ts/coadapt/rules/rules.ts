/**
 * Phase 1 rule-based updates: bounded nudges to user/AI profiles from simple behavioral signals.
 */
import {
  AIProfileSchema,
  UserProfileSchema,
  type AIProfile,
  type UserProfile,
} from "../profile/schemas";

export type RuleSignals = {
  /** Completed user turns before this one (0 = first message in session). */
  completedUserTurns: number;
  prevAssistantLen: number;
  currentUserLen: number;
  hasPreviousAssistant: boolean;
};

/** Max absolute change per field in one rule application (stability). */
export const MAX_DELTA = 0.08;

function clamp01(n: number): number {
  return Math.max(0, Math.min(1, n));
}

function nudgeToward(value: number, target: number, maxDelta: number): number {
  const d = target - value;
  const step = Math.sign(d) * Math.min(Math.abs(d), maxDelta);
  return clamp01(value + step);
}

/**
 * Phase 1 rule-based updates:
 * - More follow-ups → slightly more depth/detail preference.
 * - Long assistant reply followed by very short user message → reduce verbosity / exploratory tone.
 */
export function applyRuleBasedUpdates(
  user: UserProfile,
  ai: AIProfile,
  signals: RuleSignals
): {
  user: UserProfile;
  ai: AIProfile;
  log: string[];
  appliedRules: { id: string; message: string }[];
} {
  const log: string[] = [];
  const appliedRules: { id: string; message: string }[] = [];
  let u = UserProfileSchema.parse(JSON.parse(JSON.stringify(user)) as UserProfile);
  let a = AIProfileSchema.parse(JSON.parse(JSON.stringify(ai)) as AIProfile);

  if (signals.completedUserTurns >= 2) {
    const before = u.cognitiveStyle.detailVsHighLevel;
    u.cognitiveStyle.detailVsHighLevel = clamp01(u.cognitiveStyle.detailVsHighLevel + 0.04);
    a.reasoning.depthOfExplanation = clamp01(a.reasoning.depthOfExplanation + 0.04);
    if (u.cognitiveStyle.detailVsHighLevel !== before) {
      const msg = "followups: increased detail/depth preference (+0.04)";
      log.push(msg);
      appliedRules.push({ id: "followups_depth", message: msg });
    }
  }

  if (signals.hasPreviousAssistant && signals.prevAssistantLen > 600 && signals.currentUserLen < 40) {
    a.style.conciseVsExploratory = nudgeToward(a.style.conciseVsExploratory, 0, MAX_DELTA);
    u.preferences.responseLength = nudgeToward(u.preferences.responseLength, 0, MAX_DELTA);
    const msg = "long_reply_short_user: shifted toward concise responses";
    log.push(msg);
    appliedRules.push({ id: "long_reply_short_user", message: msg });
  }

  return { user: u, ai: a, log, appliedRules };
}
