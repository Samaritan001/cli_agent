/**
 * Zod schemas and TypeScript types for user profile, AI profile, and interaction-turn records.
 */
import { z } from "zod";

/** Bounded 0–1 sliders; lists hold optional human-readable tags. */
export const UserProfileSchema = z.object({
  preferences: z.object({
    topics: z.array(z.string()).max(32).default([]),
    responseLength: z.number().min(0).max(1),
    formatPreferences: z.array(z.string()).max(16).default([]),
  }),
  cognitiveStyle: z.object({
    analyticalVsIntuitive: z.number().min(0).max(1),
    detailVsHighLevel: z.number().min(0).max(1),
  }),
  behaviorPatterns: z.object({
    interactionFrequency: z.number().min(0).max(1),
    modalityTextVsVoice: z.number().min(0).max(1),
    taskTypes: z.array(z.string()).max(32).default([]),
  }),
  emotionalTraits: z.object({
    riskTolerance: z.number().min(0).max(1),
    patience: z.number().min(0).max(1),
    ambiguityTolerance: z.number().min(0).max(1),
  }),
  meta: z.object({
    explorationVsExploitation: z.number().min(0).max(1),
    consistencyVsNovelty: z.number().min(0).max(1),
  }),
});

export const AIProfileSchema = z.object({
  style: z.object({
    conciseVsExploratory: z.number().min(0).max(1),
  }),
  reasoning: z.object({
    fastVsSlowThinking: z.number().min(0).max(1),
    depthOfExplanation: z.number().min(0).max(1),
  }),
  initiative: z.object({
    reactiveVsProactive: z.number().min(0).max(1),
  }),
  uncertaintyHandling: z.object({
    hedgingVsDecisive: z.number().min(0).max(1),
  }),
  personality: z.object({
    skepticism: z.number().min(0).max(1),
    empathy: z.number().min(0).max(1),
    literalism: z.number().min(0).max(1),
  }),
});

export type UserProfile = z.infer<typeof UserProfileSchema>;
export type AIProfile = z.infer<typeof AIProfileSchema>;

export const InteractionTurnSchema = z.object({
  ts: z.string(),
  sessionId: z.string(),
  userText: z.string(),
  assistantText: z.string(),
  userTurnIndex: z.number().int().nonnegative(),
  turnId: z.string().uuid().optional(),
  profileHashAtEnd: z.string().optional(),
  extractedFacts: z.array(z.string()).max(20).optional(),
  toolCallsCount: z.number().int().nonnegative().optional(),
  /** Phase 2: bandit arm used for this turn's blended prompt (if enabled). */
  banditArmId: z.string().optional(),
  banditArmIndex: z.number().int().nonnegative().optional(),
  /** ms from previous turn's afterTurn to this user message (Phase 2 engagement signal). */
  msSincePreviousTurnEnd: z.number().nonnegative().optional(),
  turnMetrics: z
    .object({
      engagementProxy: z.number(),
      alignmentProxy: z.number(),
      consistencyProxy: z.number(),
    })
    .optional(),
});

export type InteractionTurn = z.infer<typeof InteractionTurnSchema>;
