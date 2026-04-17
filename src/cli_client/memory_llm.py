"""
Thin abstraction for memory-related LLM calls so multiple providers can be wired in later.

Implement MemoryLLMBackend (or duck-type the same methods) and pass an instance to MemoryEngine.

Environment (optional; all features default off — enable explicitly):

- MEMORY_LLM_PROVIDER: none | openai | anthropic | google (default: none)
- MEMORY_LLM_MODEL: optional model id for the chosen provider
- MEMORY_LLM_EXTRACT_ENTITIES: 1/true/yes or 0/false/no
- MEMORY_LLM_IDENTIFY_CAUSES: same
- MEMORY_LLM_NEURAL_RERANK: same
- MEMORY_LLM_REFLECT: same (controls reflect_synthesize)

API keys use the same vars as the main client: OPENAI_API_KEY, ANTHROPIC_API_KEY, GOOGLE_API_KEY.

Use build_memory_llm_from_env() or MemoryLLMClient.from_env() to construct a flagged wrapper
around a backend (still NullMemoryLLM until provider-specific calls are implemented).
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Protocol, runtime_checkable

from dotenv import load_dotenv


def _parse_bool(s: Optional[str], default: bool = False) -> bool:
    if s is None or s.strip() == "":
        return default
    return s.strip().lower() in ("1", "true", "yes", "on")


API_KEY_ENV = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "google": "GOOGLE_API_KEY",
}


@dataclass(frozen=True)
class MemoryLLMFeatures:
    extract_entities: bool = False
    identify_causes: bool = False
    neural_rerank: bool = False
    reflect_synthesize: bool = False


@dataclass(frozen=True)
class MemoryLLMConfig:
    """Resolved config for memory LLM; suitable for logging or future provider clients."""

    provider: str  # "none" | "openai" | "anthropic" | "google"
    model_name: Optional[str]
    features: MemoryLLMFeatures
    api_key: Optional[str]


def load_memory_llm_config_from_env() -> MemoryLLMConfig:
    load_dotenv()
    provider = (os.getenv("MEMORY_LLM_PROVIDER") or "none").strip().lower()
    if provider in ("gpt",):
        provider = "openai"
    if provider in ("claude",):
        provider = "anthropic"
    if provider in ("gemini",):
        provider = "google"
    if provider not in ("none", "openai", "anthropic", "google"):
        provider = "none"

    model_name = os.getenv("MEMORY_LLM_MODEL")
    if model_name is not None:
        model_name = model_name.strip() or None

    features = MemoryLLMFeatures(
        extract_entities=_parse_bool(os.getenv("MEMORY_LLM_EXTRACT_ENTITIES"), False),
        identify_causes=_parse_bool(os.getenv("MEMORY_LLM_IDENTIFY_CAUSES"), False),
        neural_rerank=_parse_bool(os.getenv("MEMORY_LLM_NEURAL_RERANK"), False),
        reflect_synthesize=_parse_bool(os.getenv("MEMORY_LLM_REFLECT"), False),
    )

    key_env = API_KEY_ENV.get(provider)
    api_key = os.getenv(key_env, "") if key_env else None
    if api_key is not None and api_key.strip() == "":
        api_key = None

    return MemoryLLMConfig(
        provider=provider,
        model_name=model_name,
        features=features,
        api_key=api_key,
    )


@runtime_checkable
class MemoryLLMBackend(Protocol):
    """Structural protocol: add OpenAI / Anthropic / Google adapters without changing MemoryEngine."""

    def extract_entities(self, text: str) -> List[str]:
        """Return entity strings to index (e.g. people, places)."""
        ...

    def identify_causes(self, context: str, temporal_node_ids: List[int]) -> Dict[int, int]:
        """Map existing memory node id -> causality level (e.g. 1–5) for edges into the new node."""
        ...

    def neural_rerank(self, candidate_ids: List[int], query: str) -> List[int]:
        """Reorder candidate_ids by relevance to query; may return a subset or full list."""
        ...

    def reflect_synthesize(self, facts: List[str]) -> str:
        """Turn raw node texts into a consolidated observation string."""
        ...


class NullMemoryLLM:
    """Default backend: no LLM; identity / empty behavior."""

    def extract_entities(self, text: str) -> List[str]:
        return []

    def identify_causes(self, context: str, temporal_node_ids: List[int]) -> Dict[int, int]:
        return {}

    def neural_rerank(self, candidate_ids: List[int], query: str) -> List[int]:
        return list(candidate_ids)

    def reflect_synthesize(self, facts: List[str]) -> str:
        return "; ".join(facts)


class MemoryLLMClient:
    """
    Feature-flagged facade over a MemoryLLMBackend.

    When a feature is disabled, the engine gets the same behavior as NullMemoryLLM for that call.
    When enabled, delegates to ``inner`` (placeholder NullMemoryLLM until provider-specific impl).
    """

    def __init__(self, config: MemoryLLMConfig, inner: Optional[MemoryLLMBackend] = None) -> None:
        self.config = config
        self._inner: MemoryLLMBackend = inner if inner is not None else NullMemoryLLM()

    @classmethod
    def from_env(cls, inner: Optional[MemoryLLMBackend] = None) -> "MemoryLLMClient":
        return cls(load_memory_llm_config_from_env(), inner=inner)

    def extract_entities(self, text: str) -> List[str]:
        if not self.config.features.extract_entities:
            return []
        return self._inner.extract_entities(text)

    def identify_causes(self, context: str, temporal_node_ids: List[int]) -> Dict[int, int]:
        if not self.config.features.identify_causes:
            return {}
        return self._inner.identify_causes(context, temporal_node_ids)

    def neural_rerank(self, candidate_ids: List[int], query: str) -> List[int]:
        if not self.config.features.neural_rerank:
            return list(candidate_ids)
        return self._inner.neural_rerank(candidate_ids, query)

    def reflect_synthesize(self, facts: List[str]) -> str:
        if not self.config.features.reflect_synthesize:
            return "; ".join(facts)
        return self._inner.reflect_synthesize(facts)


def build_memory_llm_from_env(inner: Optional[MemoryLLMBackend] = None) -> MemoryLLMBackend:
    """Factory: env-backed MemoryLLMClient wrapping ``inner`` (default NullMemoryLLM)."""
    return MemoryLLMClient.from_env(inner=inner)
