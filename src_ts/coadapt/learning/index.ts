/** Learning: bandit, reward, strategies, contextual features, persisted AI nudges. */
export {
  EpsilonGreedyBandit,
  type BanditArmStats,
  type BanditState,
} from "./bandit";
export {
  ContextualLinUCBBandit,
  type ContextualBanditStateV2,
  type LinucbArmState,
  solveLinearSystem,
} from "./contextual_bandit";
export { CONTEXT_FEATURE_DIM, buildContextFeatures, type ContextFeatureSignals } from "./context_features";
export { computeBanditReward, type BanditRewardSignals } from "./reward";
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
} from "./strategies";
export { nudgePersistedAiFromBanditReward } from "./persisted_ai_update";
