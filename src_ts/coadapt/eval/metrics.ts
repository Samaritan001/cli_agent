/**
 * Turn-level proxies: engagement (message shape), alignment (display vs baseline AI profile).
 */
import type { AIProfile } from "../profile/schemas";

export type TurnMetrics = {
  engagementProxy: number;
  alignmentProxy: number;
  consistencyProxy: number;
};

function clamp01(n: number): number {
  return Math.max(0, Math.min(1, n));
}

/**
 * Computed at turn end: user message length as engagement; blend vs baseline as alignment.
 * consistencyProxy reuses alignment for now (single-session coherence placeholder).
 */
export function computeTurnMetrics(opts: {
  userMessageLen: number;
  baselineAi: AIProfile;
  displayAi: AIProfile;
}): TurnMetrics {
  const engagementProxy = clamp01(opts.userMessageLen / 2000);
  const diffs: number[] = [];
  diffs.push(Math.abs(opts.baselineAi.style.conciseVsExploratory - opts.displayAi.style.conciseVsExploratory));
  diffs.push(Math.abs(opts.baselineAi.reasoning.depthOfExplanation - opts.displayAi.reasoning.depthOfExplanation));
  diffs.push(Math.abs(opts.baselineAi.initiative.reactiveVsProactive - opts.displayAi.initiative.reactiveVsProactive));
  const meanDiff = diffs.reduce((a, b) => a + b, 0) / diffs.length;
  const alignmentProxy = clamp01(1 - meanDiff);
  return {
    engagementProxy,
    alignmentProxy,
    consistencyProxy: alignmentProxy,
  };
}

/** @deprecated Use {@link computeTurnMetrics} */
export const computePhase2TurnMetrics = computeTurnMetrics;
/** @deprecated Use {@link TurnMetrics} */
export type Phase2TurnMetrics = TurnMetrics;
