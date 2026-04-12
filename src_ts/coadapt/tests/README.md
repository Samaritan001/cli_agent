# Co-adapt tests

Vitest files mirror implementation areas under `coadapt/`:

- **`learning/`** — bandit, reward, contextual LinUCB, persisted AI nudge  
- **`eval/`** — `analyzeEvalLog`, `compareEvalLogs`  
- **`profile/`** — rules, temporal decay, LLM inference throttle  
- **`cli_client/`** — full `CoAdaptSession` flow (same order as `cli_client.ts`)

Imports use `../../<module>/...` to reach source files. Rule tests live next to other profile tests in **`tests/profile/`**.
