/**
 * Compare two `eval.jsonl` runs (e.g. learning on vs off) for Phase 2 verification.
 */
import type { EvalReplayReport } from "./replay";
import { analyzeEvalLog } from "./replay";

export type CompareEvalResult = {
  pathA: string;
  pathB: string;
  reportA: EvalReplayReport;
  reportB: EvalReplayReport;
  /** meanBanditReward A − B (undefined if either side has no reward events). */
  deltaMeanBanditReward?: number;
};

export function compareEvalLogs(pathA: string, pathB: string): CompareEvalResult {
  const reportA = analyzeEvalLog(pathA);
  const reportB = analyzeEvalLog(pathB);
  let deltaMeanBanditReward: number | undefined;
  if (
    reportA.banditRewardEvents > 0 &&
    reportB.banditRewardEvents > 0 &&
    reportA.meanBanditReward !== undefined &&
    reportB.meanBanditReward !== undefined
  ) {
    deltaMeanBanditReward = reportA.meanBanditReward - reportB.meanBanditReward;
  }
  return { pathA, pathB, reportA, reportB, deltaMeanBanditReward };
}
