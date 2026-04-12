/**
 * Discrete response strategies (bandit arms): targets for AI profile knobs and prompt instructions.
 * Blending uses bounded coupling toward arm targets (KL-style bounded policy).
 */
import { AIProfileSchema, type AIProfile } from "../profile/schemas";

export const BANDIT_ARM_IDS = ["terse", "balanced", "detailed"] as const;
export type BanditArmId = (typeof BANDIT_ARM_IDS)[number];

export function isBanditArmId(s: string): s is BanditArmId {
  return (BANDIT_ARM_IDS as readonly string[]).includes(s);
}

/** Soft targets per arm (0–1) for key AI knobs — shared by prompt blend and persisted-profile learning. */
export type ArmKnobTargets = {
  conciseVsExploratory: number;
  depthOfExplanation: number;
  reactiveVsProactive: number;
};

const ARM_TARGETS: Record<BanditArmId, ArmKnobTargets> = {
  terse: { conciseVsExploratory: 0.25, depthOfExplanation: 0.3, reactiveVsProactive: 0.35 },
  balanced: { conciseVsExploratory: 0.5, depthOfExplanation: 0.5, reactiveVsProactive: 0.5 },
  detailed: { conciseVsExploratory: 0.72, depthOfExplanation: 0.82, reactiveVsProactive: 0.55 },
};

export function getArmKnobTargets(armId: BanditArmId): ArmKnobTargets {
  return { ...ARM_TARGETS[armId] };
}

/** Max blend weight toward arm-specific targets (baseline keeps (1-alpha)). */
export const DEFAULT_ARM_BLEND = 0.28;

function clamp01(n: number): number {
  return Math.max(0, Math.min(1, n));
}

/**
 * Blend persisted AI profile with arm targets for **this turn's** prompt only (bounded coupling).
 */
export function blendAIProfileForArm(
  ai: AIProfile,
  armId: BanditArmId,
  alpha: number = DEFAULT_ARM_BLEND
): AIProfile {
  const t = ARM_TARGETS[armId];
  const blended: AIProfile = {
    ...ai,
    style: {
      conciseVsExploratory: clamp01((1 - alpha) * ai.style.conciseVsExploratory + alpha * t.conciseVsExploratory),
    },
    reasoning: {
      ...ai.reasoning,
      depthOfExplanation: clamp01(
        (1 - alpha) * ai.reasoning.depthOfExplanation + alpha * t.depthOfExplanation
      ),
    },
    initiative: {
      reactiveVsProactive: clamp01(
        (1 - alpha) * ai.initiative.reactiveVsProactive + alpha * t.reactiveVsProactive
      ),
    },
    uncertaintyHandling: { ...ai.uncertaintyHandling },
    personality: { ...ai.personality },
  };
  return AIProfileSchema.parse(blended);
}

export function getStrategyInstruction(armId: BanditArmId): string {
  switch (armId) {
    case "terse":
      return "Response strategy (this turn): favor concise answers; avoid long preambles.";
    case "detailed":
      return "Response strategy (this turn): favor thorough explanations and explicit reasoning steps when helpful.";
    default:
      return "Response strategy (this turn): balanced length and depth unless the user asks otherwise.";
  }
}

export function armIndexToId(index: number): BanditArmId {
  const i = Math.max(0, Math.min(BANDIT_ARM_IDS.length - 1, Math.floor(index)));
  return BANDIT_ARM_IDS[i]!;
}

export function armIdToIndex(id: string): number {
  const i = BANDIT_ARM_IDS.indexOf(id as BanditArmId);
  return i >= 0 ? i : 1;
}
