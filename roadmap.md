# Project Pipeline: Memory-Guided User/AI Co-Adaptive Personalization

## 1. Motivation

Users often cannot fully specify what kind of assistant they need, how they think, or what interaction style best supports them. The proposed system treats personalization as a co-adaptive loop: the assistant learns a structured user model from interaction data, uses that model to update an AI behavior profile, and then observes whether the adapted assistant behavior improves future interaction quality.

The central hypothesis is that long-term personalization should not rely only on raw chat history or manually written preferences. Instead, each interaction session should produce structured memory, session summaries, performance evaluations, and profile updates.

## 2. High-Level Architecture

The prototype architecture contains five core components:

1. **Client/User**: sends task requests, feedback, documents, or multimodal context.
2. **Main Server**: coordinates the profile system, memory engine, tool manager, LLM model wrapper, and agent session.
3. **Memory Engine**: stores, retrieves, links, consolidates, and recalls memories.
4. **Tool Manager**: registers, activates, validates, prunes, and summarizes tools.
5. **LLM Model Wrapper**: manages system instructions, user messages, tool responses, generation, and response parsing.
6. **Orchestrator**: starts, stops, activates, and lists server instances.

## 3. Session-Level Pipeline

Each user--LLM interaction session follows this pipeline:

```text
User Request
  -> Load relevant user profile + AI profile
  -> Retrieve relevant memory nodes
  -> Run LLM / agent session
  -> Use tools if needed
  -> Generate response
  -> Summarize session
  -> Extract new memories
  -> Link memories across temporal / semantic / entity / causal dimensions
  -> Evaluate assistant performance
  -> Update user profile
  -> Update AI behavior profile
  -> Store session summary + evaluation + profile diffs
```

## 4. Memory System

### 4.1 Memory Unit

The system should store memories as narrative facts rather than disconnected chat snippets. A memory node should preserve enough context to be useful later.

Example memory node schema:

```json
{
  "id": "mem_001",
  "content": "The user is developing an AI personalization research project centered on memory and profile co-adaptation.",
  "source_session_id": "session_2026_05_11_001",
  "timestamp_utc": "2026-05-11T09:00:00Z",
  "sequence_id": "session_2026_05_11_001",
  "position": 12,
  "delta_t_prev": 45,
  "entities": ["AI personalization", "memory system", "user profile"],
  "sentiment": {
    "valence": "positive",
    "confidence": 0.72,
    "evidence": "user expresses project intent and asks for conversion into tex/md"
  },
  "importance": 0.86,
  "privacy_level": "project_memory",
  "expiration_policy": "retain_until_superseded"
}
```

### 4.2 Four Memory Link Layers

The system uses four graph edge types:

1. **Temporal links** connect events that happen near each other in the same task or session.
2. **Semantic links** connect memories with similar meanings or embeddings.
3. **Entity links** connect memories that mention the same canonical people, projects, tools, datasets, or concepts.
4. **Causal links** connect memories where one fact explains, motivates, blocks, or changes another.

Example temporal edge:

```json
{
  "from": "mem_001",
  "to": "mem_002",
  "type": "temporal",
  "delta_t": 120,
  "weight": 0.82
}
```

Example causal edge:

```json
{
  "from": "mem_001",
  "to": "mem_009",
  "type": "causal",
  "relation": "motivates",
  "rationale": "The user wants better personalization, so the system introduces a structured user profile update stage.",
  "confidence": 0.78
}
```

### 4.3 Memory Operations

#### Retain

Extract narrative facts, entities, sentiment, causal relations, task goals, user preferences, and assistant performance signals from each completed session.

#### Recall

At the start of each prompt, retrieve memories using multiple channels:

- semantic similarity
- entity overlap
- temporal recency
- explicit time references
- active project/task context
- high-importance memories
- causal neighbors of retrieved nodes

#### Reflect

Generate a session summary and a lightweight evaluation of what worked, what failed, and what should change next time.

#### Consolidate

Periodically merge duplicate memories, compress stale memory clusters, update higher-level mental models, and delete low-value or unsafe memory nodes.

## 5. Profile System

The profile system contains two main profiles and one optional tool-level profile.

### 5.1 User Profile

The user profile is a durable representation of the user across tasks. It should not simply repeat memory nodes. Instead, it should abstract patterns across many sessions.

Recommended dimensions:

| Dimension                     | Description                                                            | Example                                                                      |
| ----------------------------- | ---------------------------------------------------------------------- | ---------------------------------------------------------------------------- |
| Identity and role context     | Stable self-described background, roles, projects, affiliations        | Researcher working on AI personalization                                     |
| Goals and active projects     | Long-running objectives and current priorities                         | Build a project plan and prototype architecture                              |
| Domain interests              | Topics the user repeatedly cares about                                 | personalization, memory, user modeling, agentic AI                           |
| Knowledge and expertise       | Estimated familiarity with concepts, methods, tools                    | understands LLMs; asks for clarification on bandits                          |
| Preferences                   | Explicit and implicit likes/dislikes                                   | wants structured markdown/LaTeX project docs                                 |
| Communication style           | Preferred language, density, tone, formatting                          | often uses mixed English/Chinese; likes comprehensive but actionable answers |
| Cognitive/work style          | How the user reasons and plans                                         | iterative, research-oriented, architecture-first                             |
| Emotional/affective signals   | Frustration, uncertainty, enthusiasm, patience                         | asks feasibility questions and wants grounded planning                       |
| Decision/risk profile         | Tolerance for uncertainty, desire for citations, need for alternatives | prefers feasibility evaluation before implementation                         |
| Constraints                   | Deadlines, resources, privacy, compute, deployment limitations         | local storage and cloud sync are design considerations                       |
| Tool and workflow preferences | Preferred tools, file formats, coding environment                      | markdown, TeX, server/tool abstractions                                      |
| Feedback history              | What assistant behaviors were accepted, rejected, or corrected         | prefers profile dimensions to be comprehensive and research-backed           |
| Privacy and consent rules     | What can be stored, forgotten, or surfaced                             | profile/memory updates should support human-in-the-loop control              |

Suggested user profile schema:

```json
{
  "user_id": "local_user",
  "last_updated": "2026-05-11T09:00:00Z",
  "confidence_overall": 0.74,
  "profile": {
    "identity_role_context": [],
    "active_projects": [],
    "goals": [],
    "domain_interests": [],
    "knowledge_expertise": [],
    "preferences": {
      "explicit": [],
      "implicit": []
    },
    "communication_style": [],
    "cognitive_work_style": [],
    "affective_patterns": [],
    "decision_risk_profile": [],
    "constraints": [],
    "tool_workflow_preferences": [],
    "feedback_history": [],
    "privacy_consent_rules": []
  },
  "evidence_index": [
    {
      "claim": "The user prefers markdown/TeX research artifacts.",
      "supporting_memory_ids": ["mem_001", "mem_014"],
      "confidence": 0.91
    }
  ]
}
```

### 5.2 AI Profile / Model Soul

The AI profile is the adaptive policy layer. It specifies how the assistant should behave for this user, based on the user profile and prior feedback.

Recommended dimensions:

| Dimension               | Description                                                             |
| ----------------------- | ----------------------------------------------------------------------- |
| Response style          | concise, exploratory, rigorous, skeptical, tutorial-like                |
| Reasoning depth         | fast answer vs. deep analysis                                           |
| Initiative level        | reactive vs. proactive suggestions                                      |
| Uncertainty handling    | decisive vs. hedged with evidence                                       |
| Citation behavior       | when to cite papers, docs, or web sources                               |
| Format behavior         | preference for diagrams, tables, markdown, TeX, schemas                 |
| Tool-use policy         | when to retrieve memory, search literature, generate files, or run code |
| Safety/privacy behavior | when to ask consent, redact, or avoid saving sensitive information      |
| Evaluation policy       | how to judge whether the answer satisfied the user                      |

Example:

```json
{
  "response_style": "structured, research-oriented, implementation-aware",
  "reasoning_depth": "deep for architecture/research questions; concise for definitions",
  "initiative_level": "moderate-high",
  "uncertainty_handling": "state assumptions and cite sources when factual",
  "format_behavior": ["markdown", "LaTeX", "tables", "schemas", "pipelines"],
  "tool_use_policy": {
    "use_memory_retrieval": true,
    "use_literature_search_for_research_claims": true,
    "generate_artifacts_when_user_requests_md_or_tex": true
  }
}
```

### 5.3 Tool User Profile

Each tool or server may keep a smaller profile specific to that tool. For example, a writing tool may learn the user’s formatting preferences, while a coding tool may learn preferred languages, libraries, and testing style.

Tool profile should be:

- smaller than the global user profile
- more operational and less psychological
- scoped to one tool/server
- removed from the active context after the tool call ends

## 6. Profile Update Strategy

### 6.1 What should update the profile?

Use both raw conversation and session summaries:

- **Raw conversation** is useful for extracting exact evidence, quotes, correction signals, and task details.
- **Session summary** is useful for compressing task-level outcomes and reducing noise.
- **Evaluation summary** is useful for learning which assistant behavior worked or failed.

Recommended update flow:

```text
Raw interaction
  -> Session summary
  -> Memory extraction
  -> Assistant performance evaluation
  -> Candidate profile diff
  -> Evidence check against raw interaction
  -> Human-visible changelog or approval step
  -> Commit profile update
```

### 6.2 Update Frequency

Recommended default:

1. **Every completed task/session**: extract memories and create a session summary.
2. **Every completed task/session or strong signal**: propose small profile diffs.
3. **Periodically**: consolidate profile claims and remove contradictions.
4. **Immediately**: update profile when the user explicitly states a durable preference.

### 6.3 Fixed Template vs. Free-Form Profile

Use a hybrid format:

- fixed top-level schema for reliable retrieval, comparison, evaluation, and tool routing
- natural-language descriptions inside each field for expressive nuance
- confidence scores and evidence links for accountability
- profile diffs rather than full rewrites for stability

This avoids both extremes: rigid numeric-only profiles lose nuance, while unconstrained summaries are hard to evaluate and update safely.

## 7. Co-Adaptive Loop

The system implements the following loop:

```text
User behavior
  -> update memory graph
  -> update user profile
  -> update AI behavior profile
  -> adapt response/tool/routing policy
  -> observe user reaction and task success
  -> refine user profile and AI profile
```

The user profile answers: **Who is this user, what do they need, and how do they work?**

The AI profile answers: **How should the assistant behave for this user right now?**

## 8. Evaluation

Evaluation should happen at three levels.

### 8.1 Memory Evaluation

- retrieval precision: were recalled memories relevant?
- retrieval recall: did the system miss important memories?
- contradiction rate: did retrieved memories conflict with newer facts?
- graph quality: did temporal/entity/semantic/causal links help retrieval?

### 8.2 User Profile Evaluation

- predictive accuracy: does the profile predict user preferences and feedback?
- evidence grounding: is every claim supported by memory/session evidence?
- freshness: are stale attributes updated or downgraded?
- stability: does the profile avoid overreacting to one-off behavior?
- usefulness: does injecting the profile improve task success?

### 8.3 AI Profile Evaluation

- answer helpfulness
- formatting match
- reasoning-depth match
- user correction rate
- user satisfaction or preference ranking
- task completion quality
- tool-use appropriateness

## 9. Security and Privacy

Important risks:

- prompt injection during memory extraction
- storing sensitive or incorrect user claims
- tool output injection
- excessive profiling without user control
- profile drift from noisy observations
- unsafe tool permissions
- insecure local/cloud synchronization

Recommended controls:

- human-in-the-loop profile changelog
- evidence-linked profile claims
- confidence and expiration fields
- redaction of sensitive data
- tool permission boundaries
- sandboxed tool execution
- encryption for local and cloud storage
- clear delete/forget mechanisms

## 10. Minimal Viable Prototype

A practical MVP can be built in four stages.

### Stage 1: Session Summarization and Memory Extraction

- summarize completed sessions
- extract memory nodes
- store temporal, entity, semantic, and causal links
- retrieve relevant memories for the next prompt

### Stage 2: User Profile Diffing

- create fixed-schema user profile
- generate candidate profile diffs after each session
- attach evidence memory IDs
- manually review and commit diffs

### Stage 3: AI Profile Adaptation

- map user profile fields to AI behavior settings
- inject AI profile as a system-level policy layer
- evaluate response quality and user correction signals

### Stage 4: Tool and Routing Personalization

- create tool-specific user profiles
- adapt tool selection and tool prompts
- evaluate whether personalized routing improves task success and cost

## 11. Core Research Questions

1. Should user profile updates be generated from raw conversation, session summaries, or both?
2. What is the right update frequency for memory, user profile, and AI profile?
3. How fixed should the user profile schema be?
4. How can temporal, semantic, entity, and causal memory links improve profile updates?
5. How can the system avoid overfitting to recent user behavior?
6. How should user profile changes be evaluated?
7. Can AI profile adaptation improve task success compared with static personalization?
8. How should human-in-the-loop controls be designed for trust and privacy?

## 12. Suggested Claim

This project proposes a memory-guided co-adaptive personalization architecture for LLM agents. The system builds a structured memory graph from user--assistant sessions, updates a durable user profile from evidence-linked session summaries, and adapts an AI behavior profile that controls response style, reasoning depth, tool use, and uncertainty handling. The key contribution is the separation between a **User Model** and an **AI Model Soul**, connected through an auditable profile update loop.
