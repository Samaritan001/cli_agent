/**
 * Optional slow reversion of numeric profile sliders toward neutral (0.5).
 * Controlled by COADAPT_PROFILE_DECAY (0 = disabled, typical 0.001–0.02 per turn).
 */
import { UserProfileSchema, AIProfileSchema, type AIProfile, type UserProfile } from "./schemas";

function clamp01(n: number): number {
  return Math.max(0, Math.min(1, n));
}

function decayTowardMid(v: number, rate: number): number {
  const mid = 0.5;
  return clamp01(v + rate * (mid - v));
}

function decayRate(): number {
  const raw = process.env.COADAPT_PROFILE_DECAY;
  if (raw === undefined || raw === "" || raw === "0") return 0;
  const n = Number(raw);
  return Number.isFinite(n) && n > 0 && n <= 0.15 ? n : 0;
}

/**
 * Apply decay to all 0–1 user and AI fields after other updates, before save.
 */
export function applyTemporalDecay(user: UserProfile, ai: AIProfile): { user: UserProfile; ai: AIProfile } {
  const rate = decayRate();
  if (rate <= 0) return { user, ai };

  const u = UserProfileSchema.parse(JSON.parse(JSON.stringify(user)) as UserProfile);
  const a = AIProfileSchema.parse(JSON.parse(JSON.stringify(ai)) as AIProfile);

  u.preferences.responseLength = decayTowardMid(u.preferences.responseLength, rate);
  u.cognitiveStyle.analyticalVsIntuitive = decayTowardMid(u.cognitiveStyle.analyticalVsIntuitive, rate);
  u.cognitiveStyle.detailVsHighLevel = decayTowardMid(u.cognitiveStyle.detailVsHighLevel, rate);
  u.behaviorPatterns.interactionFrequency = decayTowardMid(u.behaviorPatterns.interactionFrequency, rate);
  u.behaviorPatterns.modalityTextVsVoice = decayTowardMid(u.behaviorPatterns.modalityTextVsVoice, rate);
  u.emotionalTraits.riskTolerance = decayTowardMid(u.emotionalTraits.riskTolerance, rate);
  u.emotionalTraits.patience = decayTowardMid(u.emotionalTraits.patience, rate);
  u.emotionalTraits.ambiguityTolerance = decayTowardMid(u.emotionalTraits.ambiguityTolerance, rate);
  u.meta.explorationVsExploitation = decayTowardMid(u.meta.explorationVsExploitation, rate);
  u.meta.consistencyVsNovelty = decayTowardMid(u.meta.consistencyVsNovelty, rate);

  a.style.conciseVsExploratory = decayTowardMid(a.style.conciseVsExploratory, rate);
  a.reasoning.fastVsSlowThinking = decayTowardMid(a.reasoning.fastVsSlowThinking, rate);
  a.reasoning.depthOfExplanation = decayTowardMid(a.reasoning.depthOfExplanation, rate);
  a.initiative.reactiveVsProactive = decayTowardMid(a.initiative.reactiveVsProactive, rate);
  a.uncertaintyHandling.hedgingVsDecisive = decayTowardMid(a.uncertaintyHandling.hedgingVsDecisive, rate);
  a.personality.skepticism = decayTowardMid(a.personality.skepticism, rate);
  a.personality.empathy = decayTowardMid(a.personality.empathy, rate);
  a.personality.literalism = decayTowardMid(a.personality.literalism, rate);

  return { user: u, ai: a };
}
