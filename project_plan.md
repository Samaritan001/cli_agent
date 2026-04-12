# Proposed System Design

## Overview

This document describes a **co-adaptive intelligence framework** in which two artifacts evolve together through interaction:

| Artifact | Role |
| -------- | ---- |
| **User Profile** | A latent model of the human: preferences, style, and behavior. |
| **AI Profile (“Model Soul”)** | A policy layer that shapes how the assistant reasons and responds. |

Neither profile is fixed at deploy time. They are **updated jointly** using memory, signals from dialogue, and explicit or implicit feedback.

The architecture rests on three layers:

1. **Memory Layer** — durable, retrievable context beyond a single context window.  
2. **Profile Layer** — interpretable control knobs for personalization.  
3. **Learning Loop** — constrained updates that tie user behavior to profile changes.

---

## 1. Memory Layer

Inspired by structured memory architectures (e.g., HINDSIGHT), the system maintains a **temporal, entity-aware memory graph** (full graph features are phased; see [Implementation plan](#6-implementation-plan)).

### Key features

- **Narrative fact extraction** from conversations.  
- **Graph structure** (target end state) with:
  - Temporal links  
  - Semantic links  
  - Entity links  
  - Causal links  

### Purpose

- Preserve long-term context.  
- Enable structured retrieval for prompting and learning.  
- Ground both user and AI profiles in observed facts, not only the latest turn.

---

## 2. Profile Layer

### 2.1 User Profile (latent representation)

A structured, extensible representation of the user:

```yaml
UserProfile:
  Preferences:
    - topics
    - response length
    - format preferences

  CognitiveStyle:
    - analytical vs intuitive
    - detail-oriented vs high-level

  BehaviorPatterns:
    - interaction frequency
    - modality usage (text, voice, etc.)
    - task types

  EmotionalTraits:
    - risk tolerance
    - patience
    - ambiguity tolerance

  Meta:
    - exploration vs exploitation
    - consistency vs novelty preference
```

### 2.2 AI Profile (“Model Soul”)

Defines how the assistant behaves and reasons:

```yaml
AIProfile:
  Style:
    - concise vs exploratory

  Reasoning:
    - fast vs slow thinking
    - depth of explanation

  Initiative:
    - reactive vs proactive

  UncertaintyHandling:
    - hedging vs decisive

  Personality:
    - skepticism
    - empathy
    - literalism
```

### Design principles

- Profiles are **interpretable and modular** (inspectable fields, not an opaque vector).  
- Profiles act as **control knobs** for generation and tool use.  
- Profiles **evolve over time**; defaults are only a starting point.

---

## 3. Co-Adaptation Learning Loop

### Core loop

```text
User Behavior
   ↓
[Infer User Profile]
   ↓
[Optimize AI Profile]
   ↓
AI Behavior
   ↓
[Measure User Response]
   ↓
[Update Both Profiles with Constraints]
```

### 3.1 User profile update

**Inputs:** interaction logs; behavioral signals (e.g., clicks, follow-ups, dwell time); explicit or implicit feedback.

**Methods (in order of maturity):** heuristic rules; bandit optimization; inverse reinforcement learning (IRL).

**Goal:** infer what reward function best explains the user’s behavior—i.e., what they appear to optimize for.

### 3.2 AI profile update

**Inputs:** current user profile; task outcomes; user feedback signals.

**Goal:** optimize a policy of the form **π(action | state, user_profile)** under stability constraints.

### 3.3 Stability mechanisms (critical)

To limit drift and harmful feedback loops:

- **KL-style constraints** on how far profiles or policies move per update.  
- **Confidence-weighted** updates (weak evidence → small change).  
- **Temporal decay** (forgetting stale preferences).  
- **Bounded parameters** (numeric clamps, valid ranges).

---

## 4. Multi-Level Representation Strategy

Three levels of abstraction are maintained in parallel:

| Level | Role |
| ----- | ---- |
| **Raw data** | Short-term adaptation, full-fidelity logs. |
| **Summaries** | Long-term structured memory (chunks, facts, later graph nodes). |
| **Profiles** | Compressed latent representation for control and personalization. |

---

## 5. Product phases (capabilities)

High-level capability milestones:

### Phase 1 — MVP

- Simple memory (e.g., vector store).  
- Fixed profile schema.  
- Rule-based profile updates.

**Example rules:**

- More follow-ups → increase depth preference.  
- User disengagement after long replies → reduce verbosity.

### Phase 2 — Learning system

- Bandit optimization over discrete response strategies.  
- Learned user-profile inference (supervised or LLM-assisted), merged with rules.  
- Evaluation metrics: engagement, consistency, satisfaction proxies.

### Phase 3 — Co-adaptive intelligence

- Inverse reinforcement learning (IRL).  
- Joint optimization of user and AI profiles.  
- Dynamic policy adaptation with strong evaluation harness.

---

## 6. Implementation plan

This section turns the design into **build order**, **concrete artifacts**, and **exit criteria** for the `cli_agent` codebase (orchestrator, client, model assembly).

### 6.1 Architecture mapping

| Design concept | Implementation artifact |
| -------------- | ----------------------- |
| Raw data | Append-only interaction logs: messages, tool calls, timestamps; later UI/behavioral signals. |
| Summaries / memory | Phase 1: vector DB + optional “facts” store; Phase 2+: richer edges (entity, temporal, causal). |
| Profiles | Versioned documents matching the fixed schemas; metadata: `updated_at`, confidence, bounds. |
| Learning loop | Post-turn (or batched) job: read log → update profiles under constraints → inject into next prompt. |

**Dependency order:** persistence and schemas → memory ingestion and retrieval → profile **read** path in the model loop → profile **write** path (rules) → metrics → bandits → IRL / joint optimization.

### 6.2 Phase 1 — MVP (implementation checklist)

**Objective:** End-to-end path: *conversation → memory write → profile adjustment → next-turn behavior* without a full ML training stack.

1. **Data model and storage**  
   - Implement `UserProfile` and `AIProfile` as typed models (e.g., Pydantic); prefer bounded numeric fields where possible for safe updates.  
   - Persist defaults plus per-user or per-session overrides; fix an identity key early (`user_id`, workspace, or session).  
   - Define interaction log schema: `turn_id`, role, content, optional tool calls, timestamps.

2. **Memory (minimal)**  
   - Chunk recent turns and optional extracted facts; embed via the project’s embedding path.  
   - Metadata: session, turn range, source.  
   - Expose `retrieve(query, k)` for prompt assembly.

3. **Fact extraction (lightweight)**  
   - LLM-assisted extraction to bullet “facts” per window or turn; embed into the vector store. Defer full graph DB to later phases.

4. **Rule-based profile updates**  
   - Encode observable CLI signals: follow-up depth, reply length vs user reply length, etc.  
   - Apply **stability from day one:** max delta per turn, confidence weights, optional decay toward defaults.

5. **Agent integration**  
   - Where system instructions and tools are assembled, merge: base instruction + compact profile snippet + retrieved memory block. Keep injected text **small and structured**.

6. **Evaluation v0**  
   - Log profile version per turn, retrieval usage, rule firings.  
   - Support offline replay to detect oscillation or runaway updates.

**Phase 1 exit criteria:** Stable runs; no silent profile explosion; retrieval visibly affects answers; rules covered by unit tests.

### 6.3 Phase 2 — Learning system (implementation checklist)

**Objective:** Data-driven knobs with safety rails.

1. **Richer behavioral signals** in logs (time to next message, tool success, regenerations if available).  
2. **Discrete “response strategies”** as bandit arms (e.g., terse vs detailed, reactive vs proactive).  
3. **Contextual bandit** with context = compressed profile + recent summary; reward = engagement / task / negative proxies.  
4. **KL or bounded coupling** between bandit output and baseline policy.  
5. **Learned user-profile inference** merged with rules and constraints.  
6. **Automated metrics:** engagement proxies, alignment with declared `AIProfile`, adaptation speed after a deliberate preference change.

**Phase 2 exit criteria:** Measurable improvement on at least one proxy vs Phase 1; no runaway personalization.

### 6.4 Phase 3 — Co-adaptive intelligence (implementation checklist)

**Objective:** Joint learning—only after logging and evaluation are trustworthy.

1. **IRL / reward modeling** with clear action space and logged outcomes.  
2. **Joint optimization** of user model and AI policy with coupled constraints (e.g., separate learning rates, shared KL budget).  
3. **Memory graph:** promote chunks to nodes and edges (entity, temporal, causal) for precision retrieval and optional explanations.

### 6.5 Cross-cutting engineering

Implement incrementally across phases:

| Mechanism | Intent |
| --------- | ------ |
| KL / trust region | Cap movement of profile or policy per session. |
| Confidence weighting | Scale updates by evidence strength. |
| Temporal decay | Slow reversion toward prior or default. |
| Bounded parameters | Clamps and schema validation on every write. |

### 6.6 Repository alignment (`cli_agent`)

- Prefer a **single source of truth** for profiles and memory in one stack (e.g., Python `src/`) unless product requires TS parity.  
- **Natural hook:** model and tool assembly (e.g., managers + client) for “profile + memory + tools + base system prompt.”  
- **Tests:** extend existing patterns under `src/tests/` for rules, bounds, and prompt assembly.

### 6.7 Suggested sprints

| Sprint | Focus |
| ------ | ----- |
| 1 | Profile schemas, persistence, defaults, logging schema. |
| 2 | Vector memory, retrieval API, prompt integration. |
| 3 | Rule engine, stability clamps, metrics logging. |
| 4 | Fact extraction pipeline, offline replay / evaluation scripts. |
| 5+ | Bandits, learned inference, then IRL and joint optimization (Phases 2–3). |

---

## 7. Evaluation framework

### User profile quality

- Predictive accuracy of short-horizon behavior.  
- Stability over time (no unnecessary churn).  
- Generalization across tasks.

### AI performance

- Task success.  
- Personalization alignment.  
- Consistency with the active `AIProfile`.

### System-level

- Adaptation speed after a real preference shift.  
- Robustness to noisy or ambiguous signals.  
- Long-term satisfaction proxies (where available).

---

## Key insight

This system moves the assistant from a **stateless responder** to a **co-evolving partner** that learns:

- **Who the user is** (user profile).  
- **How it should think** (AI profile).

---

## Summary

- **Memory** provides structure and grounding.  
- **Profiles** provide interpretable personalization.  
- **Learning loop** provides adaptation under constraints.

Together, these enable **continuous co-evolution between human and AI**, implemented in staged milestones from rule-based MVP to joint optimization.
