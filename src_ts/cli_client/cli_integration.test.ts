/**
 * Integration-style test for the same co-adapt path the CLI uses (`cli_client.ts`):
 * `onUserTurnStart` → `buildContextBlock` → `afterTurn`, with persisted user + AI profiles.
 *
 * `CLIClient` constructs `CoAdaptSession()` with default options; learning is enabled when
 * `COADAPT_LEARNING=1` (or `COADAPT_PHASE2=1`). Here we pass `{ dataDir, learning: true }`
 * so the test is hermetic and does not depend on env.
 *
 * "Two profiles" = the paired **user** model + **AI** control profile in `profiles.json`.
 */
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { CoAdaptSession } from "../coadapt";
import { loadProfiles } from "../coadapt/profile/profile_store";

describe("CLI-equivalent co-adapt (user + AI profiles)", () => {
  let tmp: string;
  let randomSpy: ReturnType<typeof vi.spyOn>;

  beforeEach(() => {
    tmp = fs.mkdtempSync(path.join(os.tmpdir(), "cli-agent-coadapt-"));
    // Avoid epsilon exploration so arm index stays tied to greedy best (deterministic).
    randomSpy = vi.spyOn(Math, "random").mockReturnValue(0.99);
  });

  afterEach(() => {
    randomSpy.mockRestore();
    fs.rmSync(tmp, { recursive: true, force: true });
  });

  it("persists user inference and reward-scaled AI nudge across turns", async () => {
    const profilesPath = path.join(tmp, "profiles.json");
    const session = new CoAdaptSession({ dataDir: tmp, learning: true });

    const baseline = loadProfiles(profilesPath);
    expect(baseline.user.cognitiveStyle.detailVsHighLevel).toBe(0.5);
    expect(baseline.ai.style.conciseVsExploratory).toBe(0.5);

    const user1 = "Please explain in detail why caching matters for performance.";
    session.onUserTurnStart(user1);
    const ctx = await session.buildContextBlock(user1);
    expect(ctx).toContain("User model");
    expect(ctx).toContain("AI profile");

    const assistant1 =
      "Caching stores copies of data closer to consumers so reads are faster. This reply is long enough for reward signals.";
    await session.afterTurn(user1, assistant1);

    const afterTurn1 = loadProfiles(profilesPath);
    expect(afterTurn1.user.cognitiveStyle.detailVsHighLevel).toBeGreaterThan(
      baseline.user.cognitiveStyle.detailVsHighLevel
    );

    const user2 =
      "Thanks — say more about invalidation strategies and tradeoffs we should consider in production.";
    session.onUserTurnStart(user2);
    const afterTurn2Start = loadProfiles(profilesPath);
    expect(afterTurn2Start.ai.style.conciseVsExploratory).toBeLessThan(baseline.ai.style.conciseVsExploratory);

    await session.afterTurn(user2, "Common approaches include TTL, event-driven invalidation, and versioning.");

    const banditPath = path.join(tmp, "bandit.json");
    expect(fs.existsSync(banditPath)).toBe(true);
    const banditRaw = JSON.parse(fs.readFileSync(banditPath, "utf8")) as { arms: { count: number }[] };
    expect(banditRaw.arms.some((a) => a.count >= 1)).toBe(true);
  });

  it("rule followups_depth applies after enough completed turns (user + AI)", async () => {
    const profilesPath = path.join(tmp, "profiles.json");
    const session = new CoAdaptSession({ dataDir: tmp, learning: false });

    session.onUserTurnStart("Hello.");
    await session.afterTurn("Hello.", "Hi there, how can I help?");

    session.onUserTurnStart("One more thing.");
    await session.afterTurn("One more thing.", "Sure.");

    const before = loadProfiles(profilesPath);

    session.onUserTurnStart("Third message.");
    const afterRule = loadProfiles(profilesPath);

    expect(afterRule.user.cognitiveStyle.detailVsHighLevel).toBeGreaterThan(
      before.user.cognitiveStyle.detailVsHighLevel
    );
    expect(afterRule.ai.reasoning.depthOfExplanation).toBeGreaterThan(before.ai.reasoning.depthOfExplanation);

    await session.afterTurn("Third message.", "Acknowledged.");
  });
});
