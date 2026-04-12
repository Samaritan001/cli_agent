/**
 * Orchestrates one co-adaptation session: applies rules at turn start, builds profile+memory
 * context for the model, then logs turns, updates vector memory, and writes eval/interaction JSONL.
 */
import crypto from "node:crypto";
import path from "node:path";
import process from "node:process";
import { extractFacts } from "../extraction/fact_extractor";
import { appendEvalLog } from "../eval/eval_log";
import { getEmbeddingBackendLabel } from "../memory/embeddings";
import { MemoryStore } from "../memory/memory_store";
import { appendInteractionLog, interactionLogPath } from "../logs/interaction_log";
import { defaultAIProfile, defaultUserProfile } from "../profile/defaults";
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
};

export class CoAdaptSession {
  private readonly dataDir: string;
  private readonly sessionId: string;
  private readonly profilesPath: string;
  private readonly memory: MemoryStore;
  private readonly memoryTopK: number;

  private persisted: PersistedProfiles;
  private completedTurns = 0;
  private prevAssistantLen = 0;

  constructor(opts?: CoAdaptSessionOptions) {
    this.dataDir = opts?.dataDir ?? process.env.CLI_AGENT_DATA_DIR ?? path.join(process.cwd(), ".cli_agent");
    this.sessionId = opts?.sessionId ?? "default";
    this.memoryTopK = opts?.memoryTopK ?? 3;
    this.profilesPath = path.join(this.dataDir, "profiles.json");
    this.memory = new MemoryStore(path.join(this.dataDir, "memory.json"));
    this.persisted = loadProfiles(this.profilesPath);
  }

  get user(): UserProfile {
    return this.persisted.user;
  }

  get ai(): AIProfile {
    return this.persisted.ai;
  }

  /** Call at the start of each user turn, before adding the message to model history. Applies rules from the previous turn. */
  onUserTurnStart(userText: string): void {
    const profileHashBefore = hashProfiles(this.persisted.user, this.persisted.ai);
    const { user, ai, log, appliedRules } = applyRuleBasedUpdates(this.persisted.user, this.persisted.ai, {
      completedUserTurns: this.completedTurns,
      prevAssistantLen: this.prevAssistantLen,
      currentUserLen: userText.length,
      hasPreviousAssistant: this.completedTurns > 0,
    });
    this.persisted = {
      ...this.persisted,
      user,
      ai,
      updatedAt: new Date().toISOString(),
    };
    saveProfiles(this.profilesPath, this.persisted);
    const profileHashAfter = hashProfiles(this.persisted.user, this.persisted.ai);
    appendEvalLog(this.dataDir, {
      kind: "turn_start",
      ts: new Date().toISOString(),
      sessionId: this.sessionId,
      userTurnIndex: this.completedTurns,
      rulesApplied: appliedRules,
      profileHashBefore,
      profileHashAfter,
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
    parts.push(formatProfilesForPrompt(this.persisted.user, this.persisted.ai));
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
    });

    this.prevAssistantLen = assistantText.length;
    this.completedTurns += 1;
    logInfo("coadapt", `turn ${this.completedTurns} logged; memory + eval updated`);
  }

  /** Reset session counters (profiles and memory persist). */
  resetSessionCounters(): void {
    this.completedTurns = 0;
    this.prevAssistantLen = 0;
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
