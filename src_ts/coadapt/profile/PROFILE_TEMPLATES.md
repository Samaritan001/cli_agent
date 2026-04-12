# Profile templates (co-adapt)

Authoritative shapes live in `schemas.ts` (`UserProfileSchema`, `AIProfileSchema`). On disk, both sit under `profiles.json` with `version`, `updatedAt`, and clamped numeric fields (see `profile_store.ts`).

**Convention:** Unless noted, numeric fields are **0–1** sliders (0 = low / one pole, 1 = high / other pole).

---

## User profile (`user`)

Latent model of the human: preferences, cognition, behavior, affect, and meta-learning style.

| Section | Field | Meaning (low → high) |
|--------|--------|----------------------|
| **preferences** | `topics` | Free-form tags (max 32). |
| | `responseLength` | Prefers shorter → longer replies. |
| | `formatPreferences` | Tags for desired formats (max 16). |
| **cognitiveStyle** | `analyticalVsIntuitive` | Intuitive → analytical. |
| | `detailVsHighLevel` | High-level → detail-oriented. |
| **behaviorPatterns** | `interactionFrequency` | Lower → higher interaction cadence. |
| | `modalityTextVsVoice` | Voice-leaning → text-leaning (or vice versa per product convention). |
| | `taskTypes` | Task tags (max 32). |
| **emotionalTraits** | `riskTolerance` | Risk-averse → risk-seeking. |
| | `patience` | Impatient → patient. |
| | `ambiguityTolerance` | Needs closure → tolerates ambiguity. |
| **meta** | `explorationVsExploitation` | Exploit known → explore novelty. |
| | `consistencyVsNovelty` | Consistency → novelty. |

### Default user JSON

```json
{
  "preferences": {
    "topics": [],
    "responseLength": 0.5,
    "formatPreferences": []
  },
  "cognitiveStyle": {
    "analyticalVsIntuitive": 0.5,
    "detailVsHighLevel": 0.5
  },
  "behaviorPatterns": {
    "interactionFrequency": 0.5,
    "modalityTextVsVoice": 0.5,
    "taskTypes": []
  },
  "emotionalTraits": {
    "riskTolerance": 0.5,
    "patience": 0.5,
    "ambiguityTolerance": 0.5
  },
  "meta": {
    "explorationVsExploitation": 0.5,
    "consistencyVsNovelty": 0.5
  }
}
```

---

## AI profile (`ai`)

Control knobs for the assistant (“model soul”): style, reasoning, initiative, uncertainty, personality.

| Section | Field | Meaning (low → high) |
|--------|--------|----------------------|
| **style** | `conciseVsExploratory` | Concise → exploratory / expansive. |
| **reasoning** | `fastVsSlowThinking` | Fast / heuristic → slow / deliberate. |
| | `depthOfExplanation` | Shallow → deep explanations. |
| **initiative** | `reactiveVsProactive` | Reactive → proactive. |
| **uncertaintyHandling** | `hedgingVsDecisive` | Hedging → decisive. |
| **personality** | `skepticism` | Low → high skepticism. |
| | `empathy` | Low → high empathy. |
| | `literalism` | Loose → literal. |

### Default AI JSON

```json
{
  "style": {
    "conciseVsExploratory": 0.5
  },
  "reasoning": {
    "fastVsSlowThinking": 0.5,
    "depthOfExplanation": 0.5
  },
  "initiative": {
    "reactiveVsProactive": 0.5
  },
  "uncertaintyHandling": {
    "hedgingVsDecisive": 0.5
  },
  "personality": {
    "skepticism": 0.5,
    "empathy": 0.5,
    "literalism": 0.5
  }
}
```

---

## Persisted file (`profiles.json`)

Wrapper written by `saveProfiles`:

```json
{
  "version": 1,
  "updatedAt": "2026-04-12T12:00:00.000Z",
  "user": { },
  "ai": { }
}
```

Fill `user` and `ai` with objects matching the templates above. Partial files are deep-merged with defaults on load.

---

## Optional runtime (Phase 2)

| Env | Effect |
|-----|--------|
| `COADAPT_PROFILE_DECAY` | Small positive number (e.g. `0.002`): each save pulls numeric sliders toward `0.5`. `0` or unset = off. |
| `COADAPT_USER_INFERENCE_LLM=1` | After heuristics, merge optional OpenAI JSON inference (`OPENAI_API_KEY` required). Throttled — see `inference_llm.ts`. |

See `project_plan.md` §6.3 for the full env table and eval workflow (`replay:eval`, `eval:compare`).
