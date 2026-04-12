# Co-adaptation (`coadapt/`)

Joint **user profile** + **AI profile** + **memory** + optional **learning** (contextual bandit, rewards, inference) for the CLI agent. Persistence defaults to `CLI_AGENT_DATA_DIR` or `<cwd>/.cli_agent`.

---

## Package entry

| File | Summary |
|------|---------|
| **`index.ts`** | Public API: re-exports `memory`, `profile`, `eval`, `learning`, plus `CoAdaptSession`. |

---

## `session/` — orchestration

| File | Summary |
|------|---------|
| **`co_adapt_session.ts`** | `CoAdaptSession`: loads profiles/memory/bandit, `onUserTurnStart`, `buildContextBlock`, `afterTurn`, counters, dev reset helpers. |
| **`turn_start.ts`** | Turn-start pipeline: bandit reward for previous arm, rules + learning profile updates + decay, next-arm selection. |
| **`format_profiles_for_prompt.ts`** | Renders compact user + AI lines for the model system augmentation. |

---

## `profile/` — user/AI model

| File | Summary |
|------|---------|
| **`schemas.ts`** | Zod schemas and TS types for `UserProfile`, `AIProfile`, `InteractionTurn`. |
| **`defaults.ts`** | Neutral default profiles (0.5 sliders). |
| **`profile_store.ts`** | Load/save `profiles.json`, deep merge with defaults, clamp numerics. |
| **`profile_hash.ts`** | Short stable hash of both profiles for eval logs. |
| **`rules.ts`** | Phase 1 **rule-based** nudges to user + AI from dialogue signals (`applyRuleBasedUpdates`). |
| **`inference.ts`** | Heuristic user-profile nudges from keywords/length. |
| **`inference_llm.ts`** | Optional OpenAI JSON merge after heuristics (throttled). |
| **`temporal_decay.ts`** | Optional pull of all sliders toward 0.5 each turn (`COADAPT_PROFILE_DECAY`). |
| **`behavioral_signals.ts`** | Lightweight text signals (e.g. correction keywords for eval). |
| **`index.ts`** | Barrel re-export for this folder. |
| **`PROFILE_TEMPLATES.md`** | Human-readable field reference and default JSON. |

---

## `learning/` — bandit + reward + strategies

| File | Summary |
|------|---------|
| **`strategies.ts`** | Bandit arm IDs, knob targets, `blendAIProfileForArm`, strategy instructions. |
| **`reward.ts`** | `computeBanditReward` from next message length, timing, assistant length. |
| **`context_features.ts`** | Hand-crafted context vector for LinUCB. |
| **`contextual_bandit.ts`** | Disjoint LinUCB, `bandit.json` v2 persistence, `solveLinearSystem`. |
| **`bandit.ts`** | Legacy ε-greedy bandit (v1 file format); kept for tests/compat. |
| **`persisted_ai_update.ts`** | Reward-scaled nudge of persisted AI profile toward arm targets. |
| **`index.ts`** | Barrel re-export. |

---

## `eval/` — metrics + logs (structured + interaction helpers)

| File | Summary |
|------|---------|
| **`eval_log.ts`** | Append structured events to `eval.jsonl` (`turn_start`, `context_built`, `turn_end`). |
| **`interaction_log.ts`** | Append turns to `<dataDir>/logs/<sessionId>.jsonl`; path helpers. |
| **`metrics.ts`** | `computeTurnMetrics` (engagement / alignment / consistency proxies). |
| **`replay.ts`** | `analyzeEvalLog` — hash churn, bandit reward stats, correction counts. |
| **`compare_eval.ts`** | Compare two `eval.jsonl` files (e.g. mean reward delta). |
| **`index.ts`** | Barrel re-export. |

---

## `memory/` — vector store + facts

| File | Summary |
|------|---------|
| **`memory_store.ts`** | Chunk storage, embed-on-write, `retrieveWithScores`. |
| **`embeddings.ts`** | Embedding backend selection (OpenAI vs hash fallback). |
| **`fact_extractor.ts`** | Optional LLM (or heuristic) fact bullets per turn for memory chunks. |
| **`index.ts`** | Barrel: store, embeddings, `extractFacts`. |

---

## `bin/` — small CLIs

| File | Summary |
|------|---------|
| **`replay_cli.ts`** | Read `eval.jsonl`, print `analyzeEvalLog` JSON. |
| **`compare_eval_cli.ts`** | Compare two eval paths; print `compareEvalLogs` JSON. |

---

## `tests/` — Vitest (mirrors implementation areas)

| Path | Summary |
|------|---------|
| **`tests/learning/`** | Bandit, reward, contextual bandit, persisted AI update. |
| **`tests/eval/`** | Replay analysis, eval compare. |
| **`tests/profile/`** | Rules, temporal decay, LLM inference gates. |
| **`tests/cli_client/`** | End-to-end `CoAdaptSession` integration. |
| **`tests/README.md`** | Short map of test layout. |

---

## Data flow (one user message)

```text
onUserTurnStart(userText)
  → bandit.update(previous arm, reward)     [if learning + prior turn]
  → applyRuleBasedUpdates
  → applyUserInference (+ optional LLM merge)
  → nudgePersistedAiFromBanditReward        [if learning + reward]
  → applyTemporalDecay
  → saveProfiles
  → bandit.selectArm(context)               [next arm for this turn]

buildContextBlock(lastUserMessage)
  → retrieve memory
  → blend AI profile for selected arm + strategy line + profile text

afterTurn(user, assistant)
  → extractFacts → memory chunk; interaction JSONL + eval JSONL; turn metrics
```

---

## Configuration

- **`project_plan.md` §6.3** — Phase 2 env table and verification steps.  
- **`profile/PROFILE_TEMPLATES.md`** — profile fields and optional runtime env notes.

---

## Scripts (repo root)

- `npm run replay:eval` — summarize one `eval.jsonl`
- `npm run eval:compare` — compare two eval logs
- `npm test` — `src_ts/coadapt/tests/**/*.test.ts` (see root `vitest.config.ts`)
