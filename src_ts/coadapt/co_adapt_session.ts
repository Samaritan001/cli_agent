import path from "node:path";
import process from "node:process";
import { appendInteractionLog, interactionLogPath } from "./interaction_log";
import { loadProfiles, saveProfiles, type PersistedProfiles } from "./profile_store";
import { MemoryStore } from "./memory_store";
import { applyRuleBasedUpdates } from "./rules";
import { defaultAIProfile, defaultUserProfile } from "./defaults";
import type { AIProfile, InteractionTurn, UserProfile } from "./schemas";
import { logInfo } from "../shared/logging";

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
    const { user, ai, log } = applyRuleBasedUpdates(this.persisted.user, this.persisted.ai, {
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
    for (const line of log) logInfo("coadapt", `rule: ${line}`);
  }

  /** Build system augmentation: profiles + retrieved memory for the last user message. */
  async buildContextBlock(lastUserMessage: string): Promise<string> {
    const mem = await this.memory.retrieve(lastUserMessage, this.memoryTopK);
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
  async afterTurn(userText: string, assistantText: string): Promise<void> {
    const turn: InteractionTurn = {
      ts: new Date().toISOString(),
      sessionId: this.sessionId,
      userText,
      assistantText,
      userTurnIndex: this.completedTurns,
    };
    appendInteractionLog(interactionLogPath(this.dataDir, this.sessionId), turn);
    await this.memory.addTurnChunk(userText, assistantText);
    this.prevAssistantLen = assistantText.length;
    this.completedTurns += 1;
    logInfo("coadapt", `turn ${this.completedTurns} logged; memory updated`);
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
