/** Evaluation: metrics, eval.jsonl, replay, compare, per-session interaction JSONL helpers. */
export { appendInteractionLog, ensureLogDir, interactionLogPath } from "./interaction_log";
export { appendEvalLog, evalLogPath, type EvalEvent } from "./eval_log";
export { analyzeEvalLog, type EvalReplayReport } from "./replay";
export { compareEvalLogs, type CompareEvalResult } from "./compare_eval";
export {
  computeTurnMetrics,
  computePhase2TurnMetrics,
  type TurnMetrics,
  type Phase2TurnMetrics,
} from "./metrics";
