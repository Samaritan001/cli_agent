/**
 * Co-adaptation package public surface: session orchestration, profiles, memory, rules,
 * evaluation logging, replay analysis, and fact extraction. Import from here in app code.
 */
export { CoAdaptSession, type CoAdaptSessionOptions } from "./session/co_adapt_session";
export { MemoryStore, type MemoryChunk } from "./memory/memory_store";
export { applyRuleBasedUpdates, MAX_DELTA, type RuleSignals } from "./rules/rules";
export { loadProfiles, saveProfiles, type PersistedProfiles } from "./profile/profile_store";
export { hashProfiles } from "./profile/profile_hash";
export { appendEvalLog, evalLogPath, type EvalEvent } from "./eval/eval_log";
export { analyzeEvalLog, type EvalReplayReport } from "./eval/replay";
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
