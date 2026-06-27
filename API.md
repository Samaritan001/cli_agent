# CLI Agent API Reference

This document describes the public interfaces exposed by the core `cli_client` modules. All modules live under `src/cli_client/` unless noted otherwise.

**Import paths** (pytest / `pythonpath = ["src"]`):

```python
from cli_client.model import ClientLanguageModel, LanguageModelConfig
from cli_client.managers import ToolManualManager
from cli_client.memory_engine import MemoryEngine
from cli_client.memory_config import MemoryNode, MemoryConfig
from cli_client.cli_client import CLIClient
from cli_client.logging_config import configure_logging, get_logger
```

All `cli_client` modules use the package layout above. Run entry points with `src` on `PYTHONPATH` (e.g. `uv run src/cli_client/cli_client.py` from the project root).

---

## Table of Contents

1. [Model (`model.py`)](#1-model-modelpy)
2. [Model Format (`model_format.py`)](#2-model-format-model_formatpy)
3. [Memory Config (`memory_config.py`)](#3-memory-config-memory_configpy)
4. [Memory Engine (`memory_engine.py`)](#4-memory-engine-memory_enginepy)
5. [Tool Manager (`managers.py`)](#5-tool-manager-managerspy)
6. [Client (`cli_client.py`)](#6-client-cli_clientpy)
7. [Logging (`logging_config.py`)](#7-logging-logging_configpy)
8. [Shared Types & Conventions](#8-shared-types--conventions)

---

## <a name="1-model-modelpy"></a>1. Model (`model.py`)

Multi-provider language-model wrapper supporting OpenAI, Anthropic, and Google Gemini. Handles conversation history, provider-specific message formatting, tool-call parsing, and optional memory injection.

### `LanguageModelConfig`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `model_type` | `str` | `"google"` | Provider alias: `openai`/`gpt`, `anthropic`/`claude`, `google`/`gemini` |
| `model_name` | `Optional[str]` | provider default | Specific model id (e.g. `gpt-5.4-nano`) |
| `max_tokens` | `int` | `512` | Max completion tokens (capped at 2048) |

### `BaseLanguageModel`

General-purpose LLM client without tool definitions. Suitable for one-off or memory-side requests.

| Method | Signature | Description |
|--------|-----------|-------------|
| `__init__` | `(config: Optional[LanguageModelConfig] = None)` | Loads API key from env, initializes provider client and empty history |
| `update_system_instruction` | `(instruction: str) -> None` | Sets the system prompt |
| `get_history` | `(window: Optional[int] = None) -> List[Dict]` | Returns full or last-`window` history entries |
| `set_history` | `(history: List[Dict]) -> None` | Replaces conversation history |
| `get_latest_message_id` | `() -> Optional[int]` | Most recently assigned message id |
| `add_user_message` | `(content: str) -> int` | Appends a user turn; returns a session-scoped message id |
| `generate_response` | `(max_tokens: Optional[int] = None)` | Calls the provider API; appends assistant reply to history |
| `agenerate_response` | `(max_tokens: Optional[int] = None)` | Async wrapper via `asyncio.to_thread` |
| `parse_response` | `(response) -> Dict` | Normalizes provider response to `{"content": str, "tool_calls": List[Dict]}` |
| `to_messages` | `(tool_info=None, memory="", history_window=None)` | Builds provider payload via `_build_format_context` → `model_format` adapter |
| `from_response` | `(response) -> Dict` | Parses provider response into normalized dict |

**Notable attributes:** `system_instruction`, `config`, `_history`, `_message_ids`, `_client`, `_format`.

**Parsed tool call shape:**

```python
{"id": str, "command": str, "arguments": dict}
```

### `ClientLanguageModel`

Extends `BaseLanguageModel` for the interactive agent. Loads `tool_definitions_{model_type}.json` and `system_instruction.md` at init.

| Method | Signature | Description |
|--------|-----------|-------------|
| `add_tool_response` | `(content: str, command: str, tool_call_id: str) -> int` | Records a tool result in provider-specific format |
| `generate_response` | `(tool_info, *, max_tokens=None, memory_nodes=None, history_window=None)` | Generates with tools, optional recalled memory, and truncated history |
| `agenerate_response` | `(tool_info, *, max_tokens=None, memory_nodes=None, history_window=None)` | Async variant; same keyword-only optional parameters as sync |
| `get_history` | `(window: Optional[int] = None) -> List[Dict]` | Inherited; returns full or last-`window` history entries |

Inherits `to_messages` / `from_response` from `BaseLanguageModel` (no formatter overrides; memory and `history_window` are passed via `FormatContext`).

### Module-level helpers

| Function | Signature | Description |
|----------|-----------|-------------|
| `llm_side_request` | `(query, system_instruction="", config=None, history=None) -> Dict` | Stateless single-turn LLM call; returns parsed `{"content", "tool_calls"}` |
| `llm_side_request_async` | same (async) | Async variant |

---

## <a name="2-model-format-model_formatpy"></a>2. Model Format (`model_format.py`)

Provider-specific adapters selected by `get_model_format(model_type)`. Formatting lives in adapters only; `BaseLanguageModel` and `ClientLanguageModel` share the same `to_messages` / `from_response` surface and do not override per-provider methods.

### `FormatContext`

| Field | Type | Description |
|-------|------|-------------|
| `system_instruction` | `str` | System prompt |
| `history` | `List` | Provider-formatted conversation turns |
| `tool_info` | `Optional[Dict[str, str]]` | `tool_summaries` / `tool_manuals` from `ToolManualManager` |
| `memory` | `str` | Serialized recalled memory nodes |
| `history_window` | `Optional[int]` | Truncate history to last N turns |

### `get_model_format(model_type: str) -> ModelFormatAdapter`

Returns `OpenAIFormatAdapter`, `AnthropicFormatAdapter`, or `GoogleFormatAdapter`.

Each adapter implements:

| Method | Description |
|--------|-------------|
| `format_user_message(content)` | Provider-specific user turn dict |
| `format_tool_response(content, command, tool_call_id)` | Provider-specific tool result dict |
| `to_messages(ctx: FormatContext)` | Request payload (messages list or `(system, messages)` tuple) |
| `from_response(response)` | Normalized `{"content", "tool_calls"}` |
| `generate(client, model_name, ctx, max_tokens, tools=None)` | Provider API call |
| `record_assistant_turn(history, response)` | Append assistant turn to history |

---

## <a name="3-memory-config-memory_configpy"></a>3. Memory Config (`memory_config.py`)

Data classes used by `MemoryEngine` and passed into the language model during recall.

### `MemoryNode`

| Field | Type | Description |
|-------|------|-------------|
| `id` | `int` | Unique node id (FAISS-compatible int64) |
| `text` | `str` | Stored memory content |
| `fact_type` | `str` | `"EXPERIENCE"` or `"OBSERVATION"` |
| `timestamp` | `datetime` | UTC creation time |
| `embedding` | `np.ndarray` | Semantic vector (384-dim with default config) |
| `entities` | `Dict[str, int]` | Entity name → term frequency in text |
| `doc_length` | `int` | Token/word count for BM25 and budget |
| `causes` | `Dict[int, int]` | Incoming causal edges `{source_id: level 0–3}` |
| `effects` | `Dict[int, int]` | Outgoing causal edges `{target_id: level 0–3}` |
| `message_id_range` | `List[int]` | `[start_id, end_id]` linking memory to conversation turns |

### `MemoryConfig`

Configuration for indexing, recall fusion, and summarization limits.

| Category | Fields | Defaults (high level) |
|----------|--------|------------------------|
| Storage | `memory_dir` | `"./memory_docs"` |
| Indexing | `entity_N`, `faiss_dim`, `faiss_links_per_node`, `cause_window` | 5, 384, 32, 3 |
| Semantic recall | `semantic_K` | 5 |
| Entity recall (BM25) | `k1`, `b`, `entity_K` | 1.5, 0.75, 5 |
| Causal recall | `cause_effect_K` | 5 |
| Fusion | `k_rrf`, `logical_weights` | 60, `{entity:1, semantic:1, cause:1}` |
| Temporal | `temporal_decay_lambda` | 0.1 |
| Output limits | `max_recall`, `min_recall_score`, `memory_token_limit` | 20, 0.5, 2000 |

`min_recall_score` is applied in `recall()` after temporal boosting: scores are normalized to `[0, 1]` against the top candidate, and nodes below the threshold are excluded before neural reranking.
| Summarization | `summary_token_limit` | 500 |

---

## <a name="4-memory-engine-memory_enginepy"></a>4. Memory Engine (`memory_engine.py`)

Graph-backed long-term memory with semantic (FAISS HNSW), entity (BM25 inverted index), temporal, and causal links. LLM calls for entity extraction, causality, reranking, and summarization go through `llm_side_request` / `llm_side_request_async`.

### `MemoryEngine`

#### Construction

```python
MemoryEngine(config: Optional[MemoryConfig] = None, model_config: Optional[LanguageModelConfig] = None)
```

Initializes empty storage, loads `BAAI/bge-small-en-v1.5` embedder, and creates a FAISS `IndexIDMap(IndexHNSWFlat)`.

#### Lifecycle

| Method | Description |
|--------|-------------|
| `load_memory()` | Hydrates nodes from `memory_dir/*.json`; rebuilds indices |
| `aload_memory()` | Async, lock-protected `load_memory` |
| `save_memory()` | Persists new nodes (reverse temporal walk, stops at first existing file) |
| `asave_memory()` | Async, lock-protected `save_memory` |

#### Core pipeline

| Method | Returns | Description |
|--------|---------|-------------|
| `remember(context, message_id_range=None)` | `int` (node id) | **Stage 1 — Remember:** embed, extract entities, link causes, store node |
| `aremember(context, message_id_range)` | `int` | Async `remember` |
| `recall(query, earliest_history_id=0)` | `(latest_message_id, List[MemoryNode])` | **Stage 2 — Recall:** multi-channel retrieval → RRF → temporal boost → `min_recall_score` filter → neural rerank → token budget |
| `arecall(query, earliest_history_id=0)` | same | Async `recall` |
| `reflect(node_ids)` | `str` (stub) | **Stage 3 — Reflect:** not yet implemented (`pass`) |

`earliest_history_id` excludes nodes whose `message_id_range[1]` overlaps preserved live history. `CLIClient` passes this as `earliest_history_id` to `arecall`.

#### LLM sub-operations

| Method | Returns | Description |
|--------|---------|-------------|
| `extract_entities(context)` | `List[str]` | Named-entity extraction via side LLM |
| `aextract_entities(context)` | `List[str]` | Async variant |
| `identify_causes(current_memory, memory_window)` | `Dict[int, int]` | Causal levels for recent node ids |
| `aidentify_causes(current_memory, memory_window)` | `Dict[int, int]` | Async variant |
| `neural_rerank(candidate_ids, query)` | `List[int]` | LLM-based reordering; falls back to input order on parse failure |
| `aneural_rerank(candidate_ids, query)` | `List[int]` | Async variant |
| `summarize_memory_buffer(history)` | `str` | Summarizes a conversation window into session notes (intended for buffer flush) |
| `asummarize_memory_buffer(history)` | `str` | Async variant via `asyncio.to_thread` |

#### Retrieval helpers

| Method | Description |
|--------|-------------|
| `entity_bm25(query)` | BM25 scoring over entity index; returns top `entity_K` node ids |
| `rrf(ranked_lists)` | Reciprocal Rank Fusion across `semantic` / `entity` / `cause` lists |
| `temporal_boost(rrf_results)` | Applies exponential recency decay to fused scores |

#### Node access

| Method | Description |
|--------|-------------|
| `get_node(node_id) -> Optional[MemoryNode]` | Lookup by id |
| `has_node(node_id) -> bool` | Whether id exists |

#### Public mutable state

`config`, `model_config`, `_nodes`, `_temporal_stream`, `_entity_index`, `_semantic_index`, `embedding_model`, `avg_dl`, `latest_message_id`.

#### Static validators (internal but testable)

`_safe_json_loads`, `_validate_entities_payload`, `_validate_causalities_payload`, `_validate_ranks_payload`.

---

## <a name="5-tool-manager-managerspy"></a>5. Tool Manager (`managers.py`)

Manages tool server summaries and which full manuals are injected into the LLM context (“active loadout”).

### `ToolManualManager`

#### Registration & activation

| Method | Description |
|--------|-------------|
| `register_summary(summaries: Dict[str, str])` | Bulk-register `{server_name: summary}` |
| `register_tool(name, manual)` | Store full manual in registry (not yet active) |
| `check_tool(name) -> bool` | Whether a manual exists in registry |
| `activate_tool(name) -> bool` | Add server to active loadout; returns `False` if unknown |
| `prune_tool(name)` | Remove one server from active loadout |
| `flush_loadout()` | Clear all active manuals |

#### Context export

| Method | Returns | Description |
|--------|---------|-------------|
| `get_all_summaries()` | `str` | Formatted text of every registered summary |
| `get_all_manuals()` | `str` | Formatted text of **active** manuals only |
| `get_tool_info()` | `Dict[str, str]` | `{"tool_summaries": str, "tool_manuals": str}` for the LLM |

---

## <a name="6-client-cli_clientpy"></a>6. Client (`cli_client.py`)

Top-level interactive agent that wires the language model, tool manager, memory engine, and orchestrator HTTP API.

### `CLIClient`

#### Construction

```python
CLIClient(model_type="google", model_name=None, test_flag=False)
```

| Parameter | Description |
|-----------|-------------|
| `model_type` | Passed to `LanguageModelConfig` |
| `model_name` | Optional override model id |
| `test_flag` | If `False`, calls orchestrator `list_available_servers` on init |

**Composed instances:**

- `tool_manager: ToolManualManager`
- `language_model: ClientLanguageModel`
- `memory_engine: MemoryEngine` (uses `MEMORY_DIR = "./memory_docs"`)
- `history_window: int = 10` — recent turns kept in direct LLM context
- `summary_force_after_turns: int = 10` — force memory flush after N turns
- `summary_min_chars: int = 500` — char threshold contributing to importance score
- `summary_importance_threshold: float = 1.5` — minimum importance to trigger summarization

#### Main loop

| Method | Description |
|--------|-------------|
| `agent_loop()` | Async REPL: load memory → read user input → recall → generate → tool loop → optional summarize → save on exit |
| `tool_calling(tool_calls: List[Dict])` | Dispatches orchestrator commands; updates tool manager and model history |
| `should_summarize(user_input, assistant_output, had_tool_calls, pending_turns) -> bool` | Heuristic for flushing conversation buffer to memory |

#### Private helpers

| Method | Description |
|--------|-------------|
| `_list_available_servers()` | POST `list_available_servers` to orchestrator |
| `_flush_memory_buffer(end_message_id=None)` | `asummarize_memory_buffer` on buffered history, then `aremember`; defaults to latest message id |

---

### Orchestrator contract

`ORCHESTRATOR_URL = "http://127.0.0.1:8000/orchestrate"`

| Command | Effect on client |
|---------|------------------|
| `list_available_servers` | `register_summary` |
| `activate_server` | `register_tool` + `activate_tool` |
| `stop_server` | `prune_tool` |
| `execute_server_code` | Tool result appended to history |

### Test helpers (module-level)

| Function | Description |
|----------|-------------|
| `generate_list_command()` | JSON snippet for list command |
| `generate_activate_command(server_name)` | JSON snippet for activate |
| `generate_execute_commands(server_names, languages, codes)` | JSON array of execute payloads |
| `generate_stop_command(server_name)` | JSON snippet for stop |

---

## <a name="7-logging-logging_configpy"></a>7. Logging (`logging_config.py`)

Library modules acquire loggers via `get_logger(name)` and do **not** call `logging.basicConfig` at import time.

| Function | Description |
|----------|-------------|
| `get_logger(name: str) -> logging.Logger` | Returns a named logger (`"model"`, `"memory_engine"`, `"cli_client"`, etc.) |
| `configure_logging(level=logging.INFO, fmt=LOG_FORMAT)` | Idempotent root setup; call from entry points (e.g. `cli_client.py` `__main__`) |

---

## <a name="8-shared-types--conventions"></a>8. Shared Types & Conventions

### Normalized LLM output

```python
{
    "content": str,           # Assistant text (may be JSON for memory tasks)
    "tool_calls": [
        {"id": str, "command": str, "arguments": dict}
    ]
}
```

### Tool info dict (manager → model)

```python
{
    "tool_summaries": str,   # All registered server summaries
    "tool_manuals": str      # Active servers' full manuals
}
```

### Recall return tuple

```python
(latest_message_id: int, nodes: List[MemoryNode])
```

`latest_message_id` tracks the highest conversation message id stored in memory; used to align history windows between live context and recalled facts.

### Sync / async pairs

`MemoryEngine` and `BaseLanguageModel` / `ClientLanguageModel` expose async wrappers (`aremember`, `arecall`, `asummarize_memory_buffer`, `agenerate_response`, etc.) that delegate to their sync counterparts via `asyncio.to_thread` (with lock protection where noted). `ClientLanguageModel.generate_response` and `agenerate_response` share the same keyword-only optional parameters (`max_tokens`, `memory_nodes`, `history_window`).

### Provider environment variables

| Provider | Env var |
|----------|---------|
| OpenAI | `OPENAI_API_KEY` |
| Anthropic | `ANTHROPIC_API_KEY` |
| Google | `GOOGLE_API_KEY` |

---

## Related: Orchestrator HTTP API (`src/orchestrator.py`)

Not part of `cli_client`, but consumed by `CLIClient`.

### `AIOrchestrator` response types

Internal methods return a discriminated union:

**Success** (`OrchestratorSuccessResponse`):

```python
{"status": int, "result": Optional[str], "info": str}
```

**Error** (`OrchestratorErrorResponse`):

```python
{"status": int, "detail": str}
```

| Method | Return type | Notes |
|--------|-------------|-------|
| `list_servers()` | `OrchestratorSuccessResponse` | `result` is JSON-encoded summary map |
| `activate_server(name, fetch_manual)` | `OrchestratorResponse` | `result` is manual text when `fetch_manual=True`, else `None` |
| `stop_server(name)` | `OrchestratorResponse` | |
| `execute(name, language, code)` | `OrchestratorResponse` | `result` is stdout on success |

### HTTP endpoint

**Endpoint:** `POST /orchestrate`

**Body (`CommandRequest`):**

| Field | Type | Description |
|-------|------|-------------|
| `command` | `str` | `list_available_servers`, `activate_server`, `stop_server`, `execute_server_code` |
| `id` | `str` | Correlation id echoed in response |
| `server_name` | `Optional[str]` | Target server |
| `fetch_manual` | `Optional[bool]` | Whether to return manual on activate (default `True`) |
| `language` | `Optional[str]` | Code language for execute |
| `code` | `Optional[str]` | Code to run in container |

**Response:** success or error shape above, plus echoed `id`. Unknown `command` values return `400` with `detail`.
