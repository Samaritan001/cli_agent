/**
 * Heuristic user-profile nudges from the current user message (keyword / length signals).
 * Merged with rule outputs under a small max delta for stability.
 */
import { UserProfileSchema, type UserProfile } from "./schemas";

const MAX_INFERENCE_DELTA = 0.06;

function clamp01(n: number): number {
  return Math.max(0, Math.min(1, n));
}

function nudgeField(current: number, target: number, maxDelta: number): number {
  const d = clamp01(target) - current;
  const step = Math.sign(d) * Math.min(Math.abs(d), maxDelta);
  return clamp01(current + step);
}

/**
 * Keyword / length heuristics only (no extra LLM call). Returns an updated user profile.
 */
export function applyUserInference(user: UserProfile, userText: string): UserProfile {
  let u = UserProfileSchema.parse(JSON.parse(JSON.stringify(user)) as UserProfile);
  const t = userText.trim();
  const len = t.length;

  if (/\b(tldr|brief|short|concise|be quick)\b/i.test(t)) {
    u.preferences.responseLength = nudgeField(u.preferences.responseLength, 0.25, MAX_INFERENCE_DELTA);
  }
  if (/\b(explain|why|how|detail|elaborate|thorough)\b/i.test(t)) {
    u.cognitiveStyle.detailVsHighLevel = nudgeField(u.cognitiveStyle.detailVsHighLevel, 0.75, MAX_INFERENCE_DELTA);
    u.preferences.responseLength = nudgeField(u.preferences.responseLength, 0.65, MAX_INFERENCE_DELTA * 0.5);
  }
  if (len > 600) {
    u.preferences.responseLength = nudgeField(u.preferences.responseLength, 0.62, MAX_INFERENCE_DELTA * 0.5);
    u.behaviorPatterns.interactionFrequency = nudgeField(u.behaviorPatterns.interactionFrequency, 0.65, MAX_INFERENCE_DELTA * 0.5);
  }

  return u;
}
