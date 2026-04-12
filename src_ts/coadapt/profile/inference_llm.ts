/**
 * Optional LLM-assisted user-profile nudges (OpenAI JSON), merged after heuristics.
 * Throttled by env; requires OPENAI_API_KEY. On failure, returns input unchanged.
 */
import OpenAI from "openai";
import { z } from "zod";
import { UserProfileSchema, type UserProfile } from "./schemas";

const MAX_LLM_DELTA = 0.06;

const DeltaSchema = z
  .object({
    preferences: z
      .object({
        responseLength: z.number().min(0).max(1).optional(),
      })
      .optional(),
    cognitiveStyle: z
      .object({
        analyticalVsIntuitive: z.number().min(0).max(1).optional(),
        detailVsHighLevel: z.number().min(0).max(1).optional(),
      })
      .optional(),
    behaviorPatterns: z
      .object({
        interactionFrequency: z.number().min(0).max(1).optional(),
      })
      .optional(),
    emotionalTraits: z
      .object({
        patience: z.number().min(0).max(1).optional(),
      })
      .optional(),
    meta: z
      .object({
        explorationVsExploitation: z.number().min(0).max(1).optional(),
      })
      .optional(),
  })
  .passthrough();

function clamp01(n: number): number {
  return Math.max(0, Math.min(1, n));
}

function nudgeToward(current: number, target: number, maxDelta: number): number {
  const d = clamp01(target) - current;
  const step = Math.sign(d) * Math.min(Math.abs(d), maxDelta);
  return clamp01(current + step);
}

function mergeDelta(base: UserProfile, delta: z.infer<typeof DeltaSchema>): UserProfile {
  let u = UserProfileSchema.parse(JSON.parse(JSON.stringify(base)) as UserProfile);
  const d = delta;
  if (d.preferences?.responseLength !== undefined) {
    u.preferences.responseLength = nudgeToward(u.preferences.responseLength, d.preferences.responseLength, MAX_LLM_DELTA);
  }
  if (d.cognitiveStyle?.analyticalVsIntuitive !== undefined) {
    u.cognitiveStyle.analyticalVsIntuitive = nudgeToward(
      u.cognitiveStyle.analyticalVsIntuitive,
      d.cognitiveStyle.analyticalVsIntuitive,
      MAX_LLM_DELTA
    );
  }
  if (d.cognitiveStyle?.detailVsHighLevel !== undefined) {
    u.cognitiveStyle.detailVsHighLevel = nudgeToward(
      u.cognitiveStyle.detailVsHighLevel,
      d.cognitiveStyle.detailVsHighLevel,
      MAX_LLM_DELTA
    );
  }
  if (d.behaviorPatterns?.interactionFrequency !== undefined) {
    u.behaviorPatterns.interactionFrequency = nudgeToward(
      u.behaviorPatterns.interactionFrequency,
      d.behaviorPatterns.interactionFrequency,
      MAX_LLM_DELTA
    );
  }
  if (d.emotionalTraits?.patience !== undefined) {
    u.emotionalTraits.patience = nudgeToward(u.emotionalTraits.patience, d.emotionalTraits.patience, MAX_LLM_DELTA);
  }
  if (d.meta?.explorationVsExploitation !== undefined) {
    u.meta.explorationVsExploitation = nudgeToward(
      u.meta.explorationVsExploitation,
      d.meta.explorationVsExploitation,
      MAX_LLM_DELTA
    );
  }
  return u;
}

function llmThrottleEvery(): number {
  const raw = process.env.COADAPT_USER_INFERENCE_LLM_EVERY;
  const n = raw !== undefined && raw !== "" ? Number(raw) : 3;
  return Number.isFinite(n) && n >= 1 ? Math.floor(n) : 3;
}

function llmMinLen(): number {
  const raw = process.env.COADAPT_USER_INFERENCE_LLM_MIN_LEN;
  const n = raw !== undefined && raw !== "" ? Number(raw) : 40;
  return Number.isFinite(n) && n >= 0 ? Math.floor(n) : 40;
}

/**
 * Whether this turn should call the LLM (env + key + throttle + length).
 */
export function shouldRunUserInferenceLlm(userText: string, turnStartIndex: number): boolean {
  if (process.env.COADAPT_USER_INFERENCE_LLM !== "1") return false;
  if (!process.env.OPENAI_API_KEY) return false;
  if (userText.trim().length < llmMinLen()) return false;
  const every = llmThrottleEvery();
  if (turnStartIndex % every !== 0) return false;
  return true;
}

/**
 * LLM suggests target values for a subset of fields; we nudge toward them with MAX_LLM_DELTA caps.
 */
export async function mergeUserInferenceLlm(user: UserProfile, userText: string): Promise<UserProfile> {
  const key = process.env.OPENAI_API_KEY;
  if (!key) return user;

  const model = process.env.COADAPT_USER_INFERENCE_MODEL ?? "gpt-4o-mini";
  const schemaHint =
    '{"preferences":{"responseLength":0.5},"cognitiveStyle":{"detailVsHighLevel":0.5},"behaviorPatterns":{"interactionFrequency":0.5},"emotionalTraits":{"patience":0.5},"meta":{"explorationVsExploitation":0.5}}';
  try {
    const openai = new OpenAI({ apiKey: key });
    const res = await openai.chat.completions.create({
      model,
      max_tokens: 200,
      messages: [
        {
          role: "system",
          content:
            "Infer a latent user model from the latest user message only. " +
            "Reply with JSON only: optional keys preferences.responseLength, cognitiveStyle.analyticalVsIntuitive, cognitiveStyle.detailVsHighLevel, " +
            "behaviorPatterns.interactionFrequency, emotionalTraits.patience, meta.explorationVsExploitation. " +
            "Each value must be a number between 0 and 1. Omit keys you are uncertain about. " +
            `Example shape: ${schemaHint}`,
        },
        { role: "user", content: userText.slice(0, 8000) },
      ],
      response_format: { type: "json_object" },
    });
    const raw = res.choices[0]?.message?.content ?? "{}";
    const parsed = JSON.parse(raw) as unknown;
    const delta = DeltaSchema.safeParse(parsed);
    if (!delta.success) return user;
    return mergeDelta(user, delta.data);
  } catch {
    return user;
  }
}
