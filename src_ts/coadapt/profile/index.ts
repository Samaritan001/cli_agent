/** Profile layer: schemas, persistence, inference, decay, rule-based updates. */
export { applyRuleBasedUpdates, MAX_DELTA, type RuleSignals } from "./rules";
export { loadProfiles, saveProfiles, type PersistedProfiles } from "./profile_store";
export { hashProfiles } from "./profile_hash";
export { applyUserInference } from "./inference";
export { mergeUserInferenceLlm, shouldRunUserInferenceLlm } from "./inference_llm";
export { applyTemporalDecay } from "./temporal_decay";
export { detectCorrectionSignal } from "./behavioral_signals";
export {
  UserProfileSchema,
  AIProfileSchema,
  InteractionTurnSchema,
  type UserProfile,
  type AIProfile,
  type InteractionTurn,
} from "./schemas";
export { defaultUserProfile, defaultAIProfile } from "./defaults";
