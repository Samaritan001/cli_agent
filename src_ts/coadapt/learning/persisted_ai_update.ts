/**
 * Progressively updates the **persisted** AI profile toward bandit arm targets (plan §3.2, §3.3).
 * Reward scales step size (confidence-weighted); clamps keep updates bounded (KL-style trust region).
 */
import { AIProfileSchema, type AIProfile } from "../profile/schemas";
import type { BanditArmId } from "./strategies";
import { getArmKnobTargets } from "./strategies";

/** Max fractional move toward targets in one turn when reward = 1 (env override). */
const DEFAULT_LEARN_RATE = 0.09;

function learnRate(): number {
  const raw = process.env.COADAPT_AI_PROFILE_LEARN_RATE;
  if (raw === undefined || raw === "") return DEFAULT_LEARN_RATE;
  const n = Number(raw);
  return Number.isFinite(n) && n > 0 && n <= 0.25 ? n : DEFAULT_LEARN_RATE;
}

function clamp01(n: number): number {
  return Math.max(0, Math.min(1, n));
}

/**
 * After rules, nudge persisted `ai` toward the **rewarded** arm's targets.
 * Step size = `learnRate * reward` (higher observed reward → stronger pull toward that strategy).
 */
export function nudgePersistedAiFromBanditReward(
  ai: AIProfile,
  rewardedArmId: BanditArmId,
  reward: number
): AIProfile {
  const r = Math.max(0, Math.min(1, reward));
  if (r <= 0) return ai;

  const t = getArmKnobTargets(rewardedArmId);
  const step = Math.min(0.12, learnRate() * r);

  const next: AIProfile = {
    ...ai,
    style: {
      conciseVsExploratory: clamp01(
        ai.style.conciseVsExploratory + step * (t.conciseVsExploratory - ai.style.conciseVsExploratory)
      ),
    },
    reasoning: {
      ...ai.reasoning,
      depthOfExplanation: clamp01(
        ai.reasoning.depthOfExplanation + step * (t.depthOfExplanation - ai.reasoning.depthOfExplanation)
      ),
    },
    initiative: {
      reactiveVsProactive: clamp01(
        ai.initiative.reactiveVsProactive + step * (t.reactiveVsProactive - ai.initiative.reactiveVsProactive)
      ),
    },
    uncertaintyHandling: { ...ai.uncertaintyHandling },
    personality: { ...ai.personality },
  };
  return AIProfileSchema.parse(next);
}
