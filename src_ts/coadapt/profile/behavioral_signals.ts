/**
 * Lightweight behavioral proxies from raw user text (no extra model calls).
 */

/** User may be correcting or disagreeing with the prior assistant turn. */
export function detectCorrectionSignal(userText: string): boolean {
  const t = userText.trim();
  if (t.length < 4) return false;
  return /\b(wrong|incorrect|not what|mistake|actually|misunderstood|misread)\b/i.test(t);
}
