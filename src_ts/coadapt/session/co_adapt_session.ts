/**
 * Orchestrates one co-adaptation session: applies rules at turn start, builds profile+memory
 * context for the model, then logs turns, updates vector memory, and writes eval/interaction JSONL.
 *
 * Optional learning (COADAPT_LEARNING=1 or COADAPT_PHASE2=1): contextual LinUCB over arms, reward,
 * heuristic + optional LLM user inference, temporal decay, and turn metrics — see `profile/`, `learning/`, `eval/metrics.ts`.
 */
import crypto from "node:crypto";
import path from "node:path";
import process from "node:process";
import { computeTurnMetrics } from "../eval/metrics";
import { appendEvalLog } from "../eval/eval_log";
import { extractFacts } from "../extraction/fact_extractor";
import { buildContextFeatures } from "../learning/context_features";
import { ContextualLinUCBBandit } from "../learning/contextual_bandit";
import { computeBanditReward } from "../learning/reward";
import { nudgePersistedAiFromBanditReward } from "../learning/persisted_ai_update";
import {
  armIndexToId,
  blendAIProfileForArm,
  getStrategyInstruction,
  type BanditArmId,
} from "../learning/strategies";
import { appendInteractionLog, interactionLogPath } from "../logs/interaction_log";
import { getEmbeddingBackendLabel } from "../memory/embeddings";
import { MemoryStore } from "../memory/memory_store";
import { defaultAIProfile, defaultUserProfile } from "../profile/defaults";
import { detectCorrectionSignal } from "../profile/behavioral_signals";
import { mergeUserInferenceLlm, shouldRunUserInferenceLlm } from "../profile/inference_llm";
import { applyUserInference } from "../profile/inference";
import { applyTemporalDecay } from "../profile/temporal_decay";
import { hashProfiles } from "../profile/profile_hash";
import { loadProfiles, saveProfiles, type PersistedProfiles } from "../profile/profile_store";
import type { AIProfile, InteractionTurn, UserProfile } from "../profile/schemas";
import { applyRuleBasedUpdates } from "../rules/rules";
import { logInfo } from "../../shared/logging";

function formatProfilesForPrompt(user: UserProfile, ai: AIProfile): string {
  const lines: string[] = [];
  lines.push("User model (inferred, 0=low … 1=high):");
  lines.push(`- responseLength: ${user.preferences.responseLength.toFixed(2)}`);
  lines.push(`- detailVsHighLevel: ${user.cognitiveStyle.detailVsHighLevel.toFixed(2)}`);
  lines.push(`- analyticalVsIntuitive: ${user.cognitiveStyle.analyticalVsIntuitive.toFixed(2)}`);
  lines.push(`AI profile (control knobs):`);
  lines.push(`- conciseVsExploratory: ${ai.style.conciseVsExploratory.toFixed(2)}`);
  lines.push(`- depthOfExplanation: ${ai.reasoning.depthOfExplanation.toFixed(2)}`);
  lines.push(`- reactiveVsProactive: ${ai.initiative.reactiveVsProactive.toFixed(2)}`);
  return lines.join("\n");
}

export type CoAdaptSessionOptions = {
  /** Defaults to `<cwd>/.cli_agent` or `CLI_AGENT_DATA_DIR`. */
  dataDir?: string;
  sessionId?: string;
  /** Top-k memory chunks to inject. */
  memoryTopK?: number;
  /** Enable bandit + inference + turn metrics (overrides env when set). */
  learning?: boolean;
  /** @deprecated Use `learning` */
  phase2?: boolean;
};

function isLearningEnabled(opts?: { learning?: boolean; phase2?: boolean }): boolean {
  if (opts?.learning === true || opts?.phase2 === true) return true;
  if (opts?.learning === false || opts?.phase2 === false) return false;
  return process.env.COADAPT_LEARNING === "1" || process.env.COADAPT_PHASE2 === "1";
}

export class CoAdaptSession {
  private readonly dataDir: string;
  private readonly sessionId: string;
  private readonly profilesPath: string;
  private readonly memory: MemoryStore;
  private readonly memoryTopK: number;
  private readonly learningEnabled: boolean;
  private bandit: ContextualLinUCBBandit | null = null;

  private persisted: PersistedProfiles;
  private completedTurns = 0;
  private prevAssistantLen = 0;
  /** Wall time of last afterTurn (ms). */
  private lastTurnEndTs: number | null = null;
  /** Arm index chosen for the current turn (prompt blend); updated each onUserTurnStart. */
  private lastChosenArmIndex: number | null = null;
  /** Context vector used when `lastChosenArmIndex` was selected (for LinUCB update). */
  private lastBanditContext: number[] | null = null;
  /** User wait ms since last turn end, measured at onUserTurnStart (for logging). */
  private lastUserWaitMs: number | null = null;
  /** Increments each `onUserTurnStart` (for LLM inference throttle). */
  private userTurnStarts = 0;

  constructor(opts?: CoAdaptSessionOptions) {
    this.dataDir = opts?.dataDir ?? process.env.CLI_AGENT_DATA_DIR ?? path.join(process.cwd(), ".cli_agent");
    this.sessionId = opts?.sessionId ?? "default";
    this.memoryTopK = opts?.memoryTopK ?? 3;
    this.learningEnabled = isLearningEnabled({ learning: opts?.learning, phase2: opts?.phase2 });
    this.profilesPath = path.join(this.dataDir, "profiles.json");
    this.memory = new MemoryStore(path.join(this.dataDir, "memory.json"));
    this.persisted = loadProfiles(this.profilesPath);
    if (this.learningEnabled) {
      this.bandit = new ContextualLinUCBBandit(path.join(this.dataDir, "bandit.json"));
    }
  }

  get user(): UserProfile {
    return this.persisted.user;
  }

  get ai(): AIProfile {
    return this.persisted.ai;
  }

  /** Call at the start of each user turn, before adding the message to model history. Applies rules from the previous turn. */
  async onUserTurnStart(userText: string): Promise<void> {
    const correctionSignal = detectCorrectionSignal(userText);
    const turnIdx = this.userTurnStarts;
    this.userTurnStarts += 1;

    let banditRewardPreviousArm: number | undefined;
    let banditPreviousArmIndex: number | undefined;

    if (
      this.learningEnabled &&
      this.bandit &&
      this.lastTurnEndTs != null &&
      this.lastChosenArmIndex != null &&
      this.lastBanditContext != null
    ) {
      this.lastUserWaitMs = Date.now() - this.lastTurnEndTs;
      const reward = computeBanditReward({
        nextUserMessageLen: userText.length,
        msSinceLastTurnEnd: this.lastUserWaitMs,
        prevAssistantLen: this.prevAssistantLen,
      });
      this.bandit.update(this.lastChosenArmIndex, reward, this.lastBanditContext);
      banditRewardPreviousArm = reward;
      banditPreviousArmIndex = this.lastChosenArmIndex;
      logInfo("coadapt", `learning bandit: reward=${reward.toFixed(3)} arm=${this.lastChosenArmIndex}`);
    } else {
      this.lastUserWaitMs = this.lastTurnEndTs != null ? Date.now() - this.lastTurnEndTs : null;
    }

    const profileHashBefore = hashProfiles(this.persisted.user, this.persisted.ai);
    let { user, ai, log, appliedRules } = applyRuleBasedUpdates(this.persisted.user, this.persisted.ai, {
      completedUserTurns: this.completedTurns,
      prevAssistantLen: this.prevAssistantLen,
      currentUserLen: userText.length,
      hasPreviousAssistant: this.completedTurns > 0,
    });

    if (this.learningEnabled) {
      user = applyUserInference(user, userText);
      if (shouldRunUserInferenceLlm(userText, turnIdx)) {
        user = await mergeUserInferenceLlm(user, userText);
      }
      if (
        banditRewardPreviousArm !== undefined &&
        banditPreviousArmIndex !== undefined
      ) {
        ai = nudgePersistedAiFromBanditReward(ai, armIndexToId(banditPreviousArmIndex), banditRewardPreviousArm);
      }
    }

    const decayed = applyTemporalDecay(user, ai);
    user = decayed.user;
    ai = decayed.ai;

    this.persisted = {
      ...this.persisted,
      user,
      ai,
      updatedAt: new Date().toISOString(),
    };
    saveProfiles(this.profilesPath, this.persisted);
    const profileHashAfter = hashProfiles(this.persisted.user, this.persisted.ai);

    if (this.learningEnabled && this.bandit) {
      const ctx = buildContextFeatures(user, {
        userMessageLen: userText.length,
        completedUserTurns: this.completedTurns,
      });
      this.lastBanditContext = ctx;
      this.lastChosenArmIndex = this.bandit.selectArm(ctx);
      logInfo("coadapt", `learning bandit: selected arm=${this.lastChosenArmIndex}`);
    }

    appendEvalLog(this.dataDir, {
      kind: "turn_start",
      ts: new Date().toISOString(),
      sessionId: this.sessionId,
      userTurnIndex: this.completedTurns,
      rulesApplied: appliedRules,
      profileHashBefore,
      profileHashAfter,
      ...(correctionSignal ? { correctionSignal: true } : {}),
      ...(banditRewardPreviousArm !== undefined
        ? { banditRewardPreviousArm, banditPreviousArmIndex }
        : {}),
    });
    for (const line of log) logInfo("coadapt", `rule: ${line}`);
  }

  /** Build system augmentation: profiles + retrieved memory for the last user message. */
  async buildContextBlock(lastUserMessage: string): Promise<string> {
    const hits = await this.memory.retrieveWithScores(lastUserMessage, this.memoryTopK);
    const mem = hits.map((h) => h.chunk);
    appendEvalLog(this.dataDir, {
      kind: "context_built",
      ts: new Date().toISOString(),
      sessionId: this.sessionId,
      userTurnIndex: this.completedTurns,
      embeddingBackend: getEmbeddingBackendLabel(),
      retrieval: hits.map((h) => ({ chunkId: h.chunk.id, score: h.score })),
    });
    const parts: string[] = [];

    let aiForPrompt = this.persisted.ai;
    if (this.learningEnabled && this.lastChosenArmIndex != null) {
      const armId = armIndexToId(this.lastChosenArmIndex);
      aiForPrompt = blendAIProfileForArm(this.persisted.ai, armId);
      parts.push(getStrategyInstruction(armId));
    }

    parts.push(formatProfilesForPrompt(this.persisted.user, aiForPrompt));
    if (mem.length > 0) {
      parts.push("Retrieved prior turns (may be partial):");
      mem.forEach((c, i) => {
        parts.push(`[${i + 1}] ${c.text.slice(0, 1200)}`);
      });
    }
    return parts.join("\n");
  }

  /** After assistant output for this user message is finalized (including tool loop). */
  async afterTurn(
    userText: string,
    assistantText: string,
    meta?: { toolCallsCount?: number }
  ): Promise<void> {
    const turnIndex = this.completedTurns;
    const turnId = crypto.randomUUID();
    const facts = await extractFacts(userText, assistantText);
    await this.memory.addTurnChunk(userText, assistantText, { facts: facts.length > 0 ? facts : undefined });

    const profileHash = hashProfiles(this.persisted.user, this.persisted.ai);

    let turnMetrics: ReturnType<typeof computeTurnMetrics> | undefined;
    let banditArmId: BanditArmId | undefined;
    let banditArmIndex: number | undefined;

    if (this.learningEnabled && this.lastChosenArmIndex != null) {
      banditArmIndex = this.lastChosenArmIndex;
      const armId = armIndexToId(banditArmIndex);
      banditArmId = armId;
      const displayAi = blendAIProfileForArm(this.persisted.ai, armId);
      turnMetrics = computeTurnMetrics({
        userMessageLen: userText.length,
        baselineAi: this.persisted.ai,
        displayAi,
      });
    }

    const turn: InteractionTurn = {
      ts: new Date().toISOString(),
      sessionId: this.sessionId,
      userText,
      assistantText,
      userTurnIndex: turnIndex,
      turnId,
      profileHashAtEnd: profileHash,
      extractedFacts: facts.length > 0 ? facts : undefined,
      toolCallsCount: meta?.toolCallsCount ?? 0,
      ...(banditArmId !== undefined
        ? {
            banditArmId,
            banditArmIndex,
            msSincePreviousTurnEnd: this.lastUserWaitMs ?? undefined,
            turnMetrics,
          }
        : {}),
    };
    appendInteractionLog(interactionLogPath(this.dataDir, this.sessionId), turn);
    appendEvalLog(this.dataDir, {
      kind: "turn_end",
      ts: new Date().toISOString(),
      sessionId: this.sessionId,
      userTurnIndex: turnIndex,
      turnId,
      profileHash,
      toolCallsCount: meta?.toolCallsCount ?? 0,
      extractedFactsCount: facts.length,
      ...(banditArmId !== undefined ? { banditArmId, banditArmIndex, turnMetrics } : {}),
    });

    this.prevAssistantLen = assistantText.length;
    this.completedTurns += 1;
    this.lastTurnEndTs = Date.now();
    logInfo("coadapt", `turn ${this.completedTurns} logged; memory + eval updated`);
  }

  /** Reset session counters (profiles, memory, bandit file persist on disk). */
  resetSessionCounters(): void {
    this.completedTurns = 0;
    this.prevAssistantLen = 0;
    this.lastTurnEndTs = null;
    this.lastChosenArmIndex = null;
    this.lastBanditContext = null;
    this.lastUserWaitMs = null;
    this.userTurnStarts = 0;
  }

  /** Dev / tests: reload profiles from disk. */
  reloadProfiles(): void {
    this.persisted = loadProfiles(this.profilesPath);
  }

  /** Dev / tests: restore defaults on disk. */
  resetProfilesToDefaults(): void {
    this.persisted = {
      version: 1,
      user: defaultUserProfile(),
      ai: defaultAIProfile(),
      updatedAt: new Date().toISOString(),
    };
    saveProfiles(this.profilesPath, this.persisted);
  }
}
