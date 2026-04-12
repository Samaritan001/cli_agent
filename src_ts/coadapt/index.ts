/**
 * Co-adaptation package: session, profiles, memory, rules, learning (bandit), evaluation, extraction.
 * Import from here in app code.
 */
export { CoAdaptSession, type CoAdaptSessionOptions } from "./session/co_adapt_session";
export { MemoryStore, type MemoryChunk } from "./memory/memory_store";
export { applyRuleBasedUpdates, MAX_DELTA, type RuleSignals } from "./rules/rules";
export { loadProfiles, saveProfiles, type PersistedProfiles } from "./profile/profile_store";
export { hashProfiles } from "./profile/profile_hash";
export { applyUserInference } from "./profile/inference";
export { appendEvalLog, evalLogPath, type EvalEvent } from "./eval/eval_log";
export { analyzeEvalLog, type EvalReplayReport } from "./eval/replay";
export {
  computeTurnMetrics,
  computePhase2TurnMetrics,
  type TurnMetrics,
  type Phase2TurnMetrics,
} from "./eval/metrics";
export { extractFacts } from "./extraction/fact_extractor";
export { getEmbeddingBackendLabel } from "./memory/embeddings";
export {
  UserProfileSchema,
  AIProfileSchema,
  InteractionTurnSchema,
  type UserProfile,
  type AIProfile,
  type InteractionTurn,
} from "./profile/schemas";
export { defaultUserProfile, defaultAIProfile } from "./profile/defaults";

export {
  EpsilonGreedyBandit,
  type BanditArmStats,
  type BanditState,
} from "./learning/bandit";
export {
  ContextualLinUCBBandit,
  type ContextualBanditStateV2,
  type LinucbArmState,
  solveLinearSystem,
} from "./learning/contextual_bandit";
export { CONTEXT_FEATURE_DIM, buildContextFeatures, type ContextFeatureSignals } from "./learning/context_features";
export { computeBanditReward, type BanditRewardSignals } from "./learning/reward";
export {
  BANDIT_ARM_IDS,
  type BanditArmId,
  type ArmKnobTargets,
  blendAIProfileForArm,
  getArmKnobTargets,
  getStrategyInstruction,
  armIndexToId,
  armIdToIndex,
  isBanditArmId,
  DEFAULT_ARM_BLEND,
} from "./learning/strategies";
export { nudgePersistedAiFromBanditReward } from "./learning/persisted_ai_update";
