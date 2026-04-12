import { describe, expect, it, vi, afterEach } from "vitest";
import { defaultUserProfile } from "./defaults";
import { shouldRunUserInferenceLlm } from "./inference_llm";

describe("shouldRunUserInferenceLlm", () => {
  afterEach(() => {
    vi.unstubAllEnvs();
  });

  it("returns false when disabled", () => {
    vi.stubEnv("COADAPT_USER_INFERENCE_LLM", "0");
    expect(shouldRunUserInferenceLlm("hello world ".repeat(10), 0)).toBe(false);
  });

  it("returns false without API key", () => {
    vi.stubEnv("COADAPT_USER_INFERENCE_LLM", "1");
    vi.stubEnv("OPENAI_API_KEY", "");
    expect(shouldRunUserInferenceLlm("hello world ".repeat(10), 0)).toBe(false);
  });

  it("returns false when text too short", () => {
    vi.stubEnv("COADAPT_USER_INFERENCE_LLM", "1");
    vi.stubEnv("OPENAI_API_KEY", "sk-test");
    expect(shouldRunUserInferenceLlm("short", 0)).toBe(false);
  });

  it("returns true on throttle turn with key and length", () => {
    vi.stubEnv("COADAPT_USER_INFERENCE_LLM", "1");
    vi.stubEnv("OPENAI_API_KEY", "sk-test");
    vi.stubEnv("COADAPT_USER_INFERENCE_LLM_EVERY", "2");
    const long = "a".repeat(50);
    expect(shouldRunUserInferenceLlm(long, 0)).toBe(true);
    expect(shouldRunUserInferenceLlm(long, 1)).toBe(false);
    expect(shouldRunUserInferenceLlm(long, 2)).toBe(true);
  });
});
