import { describe, expect, it, vi, afterEach } from "vitest";
import { defaultUserProfile, defaultAIProfile } from "./defaults";
import { applyTemporalDecay } from "./temporal_decay";

describe("applyTemporalDecay", () => {
  afterEach(() => {
    vi.unstubAllEnvs();
  });

  it("no-ops when COADAPT_PROFILE_DECAY is unset", () => {
    vi.stubEnv("COADAPT_PROFILE_DECAY", "");
    const u = defaultUserProfile();
    u.preferences.responseLength = 0.9;
    const out = applyTemporalDecay(u, defaultAIProfile());
    expect(out.user.preferences.responseLength).toBe(0.9);
  });

  it("pulls sliders toward 0.5 when decay > 0", () => {
    vi.stubEnv("COADAPT_PROFILE_DECAY", "0.1");
    const u = defaultUserProfile();
    u.preferences.responseLength = 0.9;
    const out = applyTemporalDecay(u, defaultAIProfile());
    expect(out.user.preferences.responseLength).toBeLessThan(0.9);
    expect(out.user.preferences.responseLength).toBeGreaterThan(0.5);
  });
});
