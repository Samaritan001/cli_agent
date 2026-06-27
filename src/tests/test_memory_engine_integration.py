"""
Integration tests: remember + recall with a fake side LLM (no real API).

Uses fastembed + FAISS (may download model on first run).
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict

import pytest

import cli_client.memory_config as mem_cfg
import cli_client.memory_engine as mem_eng

_MARK = "XYZZYINTEG"


def _fake_llm(query: str, system_instruction: str = "", **kwargs: Any) -> Dict[str, str]:
    s = system_instruction
    if "You extract named entities" in s:
        if _MARK in query:
            return {"content": json.dumps({"entities": [_MARK]})}
        return {"content": json.dumps({"entities": []})}
    if "Causal Logic Engine" in s:
        ids = [int(x) for x in re.findall(r'"memory_id":\s*(\d+)', query)]
        if not ids:
            return {"content": json.dumps({"causalities": []})}
        return {
            "content": json.dumps(
                {"causalities": [{"memory_id": ids[0], "level": 1}]}
            )
        }
    if "Semantic Relevance Auditor" in s:
        ids = [int(x) for x in re.findall(r'"memory_id":\s*(\d+)', query)]
        n = len(ids)
        if n == 0:
            return {"content": json.dumps({"ranks": []})}
        ranks = [{"memory_id": ids[i], "rank": i + 1} for i in range(n)]
        return {"content": json.dumps({"ranks": ranks})}
    return {"content": "{}"}


async def _fake_llm_async(
    query: str, system_instruction: str = "", **kwargs: Any
) -> Dict[str, str]:
    return _fake_llm(query, system_instruction, **kwargs)


@pytest.fixture
def engine(monkeypatch) -> mem_eng.MemoryEngine:
    cfg = mem_cfg.MemoryConfig(
        memory_dir="/tmp",
        cause_window=3,
        semantic_K=5,
        entity_K=5,
        cause_effect_K=5,
        max_recall=20,
        memory_token_limit=100_000,
    )
    monkeypatch.setattr(mem_eng, "llm_side_request", _fake_llm)
    monkeypatch.setattr(mem_eng, "llm_side_request_async", _fake_llm_async)
    return mem_eng.MemoryEngine(config=cfg)


def test_recall_with_no_nodes(engine: mem_eng.MemoryEngine):
    latest, nodes = engine.recall("anything")
    assert latest == 0
    assert nodes == []


def test_remember_and_recall_roundtrip(engine: mem_eng.MemoryEngine):
    t1 = f"User met Ada at conference {_MARK}"
    t2 = f"User and {_MARK} project shipped on Tuesday"
    n1 = engine.remember(t1, message_id_range=[0, 0])
    n2 = engine.remember(t2, message_id_range=[0, 0])
    assert engine.has_node(n1) and engine.has_node(n2)
    assert engine.get_node(n1).effects.get(n2) == 1

    q = f"tell me about {_MARK}"
    _latest, out = engine.recall(q, earliest_history_id=0)
    texts = {n.text for n in out}
    assert t1 in texts
    assert t2 in texts


@pytest.mark.asyncio
async def test_aremember_arecall_async_flow(engine: mem_eng.MemoryEngine):
    t = f"async test {_MARK} note"
    n = await engine.aremember(t, [0, 0])
    assert engine.has_node(n)
    _latest, nodes = await engine.arecall(f"query {_MARK}")
    assert any(t in m.text for m in nodes)
