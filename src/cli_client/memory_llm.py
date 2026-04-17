"""
Thin abstraction for memory-related LLM calls so multiple providers can be wired in later.

Implement MemoryLLMBackend (or duck-type the same methods) and pass an instance to MemoryEngine.
"""
from __future__ import annotations

from typing import Dict, List, Protocol, runtime_checkable


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
