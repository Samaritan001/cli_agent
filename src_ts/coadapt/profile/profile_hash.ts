import crypto from "node:crypto";
import type { AIProfile, UserProfile } from "./schemas";

/** Stable fingerprint for eval / replay (16 hex chars). */
export function hashProfiles(user: UserProfile, ai: AIProfile): string {
  const payload = JSON.stringify({ user, ai });
  return crypto.createHash("sha256").update(payload).digest("hex").slice(0, 16);
}
