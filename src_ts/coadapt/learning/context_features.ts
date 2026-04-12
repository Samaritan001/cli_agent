/**
 * Hand-crafted context vector for the contextual bandit (plan §6.3): user knobs + session signals.
 * No embeddings — fixed dimension for LinUCB.
 */
import type { UserProfile } from "../profile/schemas";

/** Must match `ContextualLinUCBBandit` persisted `d`. */
export const CONTEXT_FEATURE_DIM = 8;

export type ContextFeatureSignals = {
  userMessageLen: number;
  /** Completed assistant turns before this user turn (`CoAdaptSession.completedTurns`). */
  completedUserTurns: number;
};

/**
 * Normalized features in [0,1] plus bias. Order is stable for persisted LinUCB weights.
 */
export function buildContextFeatures(user: UserProfile, signals: ContextFeatureSignals): number[] {
  const len = Math.min(1, Math.max(0, signals.userMessageLen / 2000));
  const turn = Math.min(1, Math.max(0, signals.completedUserTurns / 20));
  const x: number[] = [
    1,
    user.preferences.responseLength,
    user.cognitiveStyle.detailVsHighLevel,
    user.cognitiveStyle.analyticalVsIntuitive,
    user.behaviorPatterns.interactionFrequency,
    user.meta.explorationVsExploitation,
    len,
    turn,
  ];
  if (x.length !== CONTEXT_FEATURE_DIM) {
    throw new Error(`context_features: expected ${CONTEXT_FEATURE_DIM} dims, got ${x.length}`);
  }
  return x;
}
