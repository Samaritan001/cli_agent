"""
Provider-backed MemoryLLMBackend: implements extract_entities via chat APIs.

Other protocol methods delegate to NullMemoryLLM until later steps.
"""
from __future__ import annotations

import json
import logging
from typing import Any, Dict, List

from memory_llm import MemoryLLMConfig, NullMemoryLLM

logger = logging.getLogger(__name__)

_DEFAULT_MODELS = {
    "openai": "gpt-5.4-nano",
    "anthropic": "claude-4-6-opus",
    "google": "gemini-3.1-flash-lite-preview",
}


def _model_name(cfg: MemoryLLMConfig) -> str:
    if cfg.model_name:
        return cfg.model_name
    return _DEFAULT_MODELS.get(cfg.provider, _DEFAULT_MODELS["openai"])


def _strip_json_fence(raw: str) -> str:
    s = raw.strip()
    if s.startswith("```"):
        lines = s.split("\n")
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        s = "\n".join(lines)
    return s.strip()


def parse_entities_response(raw: str, max_entities: int) -> List[str]:
    """Parse model output into a deduped list of entity strings (case-insensitive dedupe)."""
    try:
        s = _strip_json_fence(raw)
        data: Any = json.loads(s)
    except (json.JSONDecodeError, TypeError) as e:
        logger.warning("memory_llm entities JSON parse failed: %s", e)
        return []

    if isinstance(data, list):
        entities = data
    elif isinstance(data, dict) and "entities" in data:
        entities = data["entities"]
    else:
        return []

    out: List[str] = []
    seen: set[str] = set()
    for e in entities:
        if not isinstance(e, str):
            continue
        t = e.strip()
        if not t:
            continue
        key = t.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(t)
        if len(out) >= max_entities:
            break
    return out


def _entities_system_prompt(max_entities: int) -> str:
    return (
        "You extract named entities for memory indexing. "
        "Return ONLY valid JSON with this exact shape: "
        '{"entities": ["Entity1", "Entity2"]}. '
        f"Include at most {max_entities} entities. "
        "Prefer people, organizations, locations, and key proper nouns; "
        "use short surface forms as they appear in the text; "
        "no duplicate meanings; use an empty array if there are none."
    )


def _anthropic_message_text(resp: Any) -> str:
    parts: List[str] = []
    for block in getattr(resp, "content", []) or []:
        text = getattr(block, "text", None)
        if text:
            parts.append(text)
    return "".join(parts)


def _google_response_text(resp: Any) -> str:
    t = getattr(resp, "text", None)
    if t:
        return str(t)
    try:
        cands = getattr(resp, "candidates", None) or []
        if not cands:
            return ""
        parts = getattr(cands[0].content, "parts", None) or []
        return "".join(getattr(p, "text", "") or "" for p in parts)
    except (IndexError, AttributeError):
        return ""


class ProviderMemoryLLM:
    """Routes memory LLM calls to OpenAI / Anthropic / Google; only extract_entities is implemented."""

    def __init__(self, config: MemoryLLMConfig) -> None:
        self._config = config
        self._null = NullMemoryLLM()

    def extract_entities(self, text: str) -> List[str]:
        cfg = self._config
        if cfg.provider == "none" or not cfg.api_key:
            return []
        if not text.strip():
            return []

        model = _model_name(cfg)
        system = _entities_system_prompt(cfg.max_entities)
        user = f"Text:\n{text}"

        try:
            if cfg.provider == "openai":
                raw = self._openai_extract(system, user, model)
            elif cfg.provider == "anthropic":
                raw = self._anthropic_extract(system, user, model)
            elif cfg.provider == "google":
                raw = self._google_extract(system, user, model)
            else:
                return []
        except Exception as e:
            logger.warning("memory_llm extract_entities failed (%s): %s", cfg.provider, e)
            return []

        return parse_entities_response(raw, cfg.max_entities)

    def _openai_extract(self, system: str, user: str, model: str) -> str:
        from openai import OpenAI

        client = OpenAI(api_key=self._config.api_key)
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            response_format={"type": "json_object"},
            max_completion_tokens=256,
        )
        choice = resp.choices[0]
        content = choice.message.content
        return content if content else ""

    def _anthropic_extract(self, system: str, user: str, model: str) -> str:
        from anthropic import Anthropic

        client = Anthropic(api_key=self._config.api_key)
        resp = client.messages.create(
            model=model,
            max_tokens=256,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return _anthropic_message_text(resp)

    def _google_extract(self, system: str, user: str, model: str) -> str:
        from google import genai

        client = genai.Client(api_key=self._config.api_key)
        prompt = f"{system}\n\n{user}\n\nRespond with JSON only, no markdown."
        resp = client.models.generate_content(model=model, contents=prompt)
        return _google_response_text(resp)

    def identify_causes(self, context: str, temporal_node_ids: List[int]) -> Dict[int, int]:
        return self._null.identify_causes(context, temporal_node_ids)

    def neural_rerank(self, candidate_ids: List[int], query: str) -> List[int]:
        return self._null.neural_rerank(candidate_ids, query)

    def reflect_synthesize(self, facts: List[str]) -> str:
        return self._null.reflect_synthesize(facts)
