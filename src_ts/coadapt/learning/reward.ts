/**
 * Proxy reward for the bandit after the user sends the *next* message (engagement / return).
 */

function clamp01(n: number): number {
  return Math.max(0, Math.min(1, n));
}

export type BanditRewardSignals = {
  /** Length of the user message that follows the assistant turn (engagement proxy). */
  nextUserMessageLen: number;
  /** Milliseconds from previous turn end (afterTurn) to this user message. */
  msSinceLastTurnEnd: number;
  /** Length of the assistant reply that preceded this user message. */
  prevAssistantLen: number;
};

/**
 * Combine normalized message length, recency of reply, and a short-message penalty.
 * Reward is in [0, 1] for epsilon-greedy updates.
 */
export function computeBanditReward(s: BanditRewardSignals): number {
  const engage = Math.min(1, s.nextUserMessageLen / 900);
  const recency = 1 / (1 + s.msSinceLastTurnEnd / 480_000);
  const shortPenalty = s.nextUserMessageLen < 4 ? 0.15 : 1;
  const assistantOk = s.prevAssistantLen > 15 ? 1 : 0.45;
  return clamp01(0.45 * engage + 0.35 * recency + 0.1 * shortPenalty + 0.1 * assistantOk);
}
