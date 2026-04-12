/**
 * User-turn-start pipeline: bandit reward for the *previous* arm, rules, learning on profiles,
 * temporal decay, then bandit selection for the *next* arm. Pure orchestration — `CoAdaptSession`
 * holds state and I/O (save, logs).
 */
import { buildContextFeatures } from "../learning/context_features";
import type { ContextualLinUCBBandit } from "../learning/contextual_bandit";
import { computeBanditReward } from "../learning/reward";
import { nudgePersistedAiFromBanditReward } from "../learning/persisted_ai_update";
import { armIndexToId } from "../learning/strategies";
import { applyUserInference } from "../profile/inference";
import { mergeUserInferenceLlm, shouldRunUserInferenceLlm } from "../profile/inference_llm";
import { applyTemporalDecay } from "../profile/temporal_decay";
import type { AIProfile, UserProfile } from "../profile/schemas";
import { applyRuleBasedUpdates, type RuleSignals } from "../profile/rules";

export type BanditRewardStepResult = {
  banditRewardPreviousArm?: number;
  banditPreviousArmIndex?: number;
  lastUserWaitMs: number | null;
};

/** Update LinUCB from the prior turn’s arm and observed reward; compute wait time for logging. */
export function applyBanditRewardForPreviousTurn(params: {
  learningEnabled: boolean;
  bandit: ContextualLinUCBBandit | null;
  lastTurnEndTs: number | null;
  lastChosenArmIndex: number | null;
  lastBanditContext: number[] | null;
  userText: string;
  prevAssistantLen: number;
}): BanditRewardStepResult {
  const {
    learningEnabled,
    bandit,
    lastTurnEndTs,
    lastChosenArmIndex,
    lastBanditContext,
    userText,
    prevAssistantLen,
  } = params;

  if (
    learningEnabled &&
    bandit &&
    lastTurnEndTs != null &&
    lastChosenArmIndex != null &&
    lastBanditContext != null
  ) {
    const lastUserWaitMs = Date.now() - lastTurnEndTs;
    const reward = computeBanditReward({
      nextUserMessageLen: userText.length,
      msSinceLastTurnEnd: lastUserWaitMs,
      prevAssistantLen,
    });
    bandit.update(lastChosenArmIndex, reward, lastBanditContext);
    return {
      banditRewardPreviousArm: reward,
      banditPreviousArmIndex: lastChosenArmIndex,
      lastUserWaitMs,
    };
  }
  return {
    lastUserWaitMs: lastTurnEndTs != null ? Date.now() - lastTurnEndTs : null,
  };
}

async function applyLearningUserAndAi(params: {
  learningEnabled: boolean;
  user: UserProfile;
  ai: AIProfile;
  userText: string;
  turnIdx: number;
  banditRewardPreviousArm?: number;
  banditPreviousArmIndex?: number;
}): Promise<{ user: UserProfile; ai: AIProfile }> {
  let { user, ai } = params;
  if (!params.learningEnabled) return { user, ai };

  user = applyUserInference(user, params.userText);
  if (shouldRunUserInferenceLlm(params.userText, params.turnIdx)) {
    user = await mergeUserInferenceLlm(user, params.userText);
  }
  if (
    params.banditRewardPreviousArm !== undefined &&
    params.banditPreviousArmIndex !== undefined
  ) {
    ai = nudgePersistedAiFromBanditReward(
      ai,
      armIndexToId(params.banditPreviousArmIndex),
      params.banditRewardPreviousArm
    );
  }
  return { user, ai };
}

/** Rules → optional learning nudges → temporal decay. */
export async function runTurnStartProfilePipeline(params: {
  learningEnabled: boolean;
  persistedUser: UserProfile;
  persistedAi: AIProfile;
  userText: string;
  completedUserTurns: number;
  prevAssistantLen: number;
  turnIdx: number;
  banditRewardPreviousArm?: number;
  banditPreviousArmIndex?: number;
}): Promise<{
  user: UserProfile;
  ai: AIProfile;
  log: string[];
  appliedRules: { id: string; message: string }[];
}> {
  const signals: RuleSignals = {
    completedUserTurns: params.completedUserTurns,
    prevAssistantLen: params.prevAssistantLen,
    currentUserLen: params.userText.length,
    hasPreviousAssistant: params.completedUserTurns > 0,
  };
  const rules = applyRuleBasedUpdates(params.persistedUser, params.persistedAi, signals);
  const learned = await applyLearningUserAndAi({
    learningEnabled: params.learningEnabled,
    user: rules.user,
    ai: rules.ai,
    userText: params.userText,
    turnIdx: params.turnIdx,
    banditRewardPreviousArm: params.banditRewardPreviousArm,
    banditPreviousArmIndex: params.banditPreviousArmIndex,
  });
  const decayed = applyTemporalDecay(learned.user, learned.ai);
  return {
    user: decayed.user,
    ai: decayed.ai,
    log: rules.log,
    appliedRules: rules.appliedRules,
  };
}

/** Choose next bandit arm and store context for the following reward update. */
export function selectBanditArmForNextTurn(params: {
  learningEnabled: boolean;
  bandit: ContextualLinUCBBandit | null;
  user: UserProfile;
  userText: string;
  completedUserTurns: number;
}): { lastBanditContext: number[] | null; lastChosenArmIndex: number | null } {
  if (!params.learningEnabled || !params.bandit) {
    return { lastBanditContext: null, lastChosenArmIndex: null };
  }
  const ctx = buildContextFeatures(params.user, {
    userMessageLen: params.userText.length,
    completedUserTurns: params.completedUserTurns,
  });
  return {
    lastBanditContext: ctx,
    lastChosenArmIndex: params.bandit.selectArm(ctx),
  };
}
