"""Unit tests: MemoryEngine JSON payload validators (no FAISS / embed)."""

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

import cli_client.memory_config as mem_cfg
import cli_client.memory_engine as mem_eng
from cli_client.memory_engine import MemoryEngine


def _make_engine():
    with patch.object(mem_eng, "TextEmbedding", return_value=MagicMock()), patch.object(
        mem_eng.faiss, "IndexIDMap", return_value=MagicMock()
    ):
        return MemoryEngine(config=mem_cfg.MemoryConfig(memory_dir="/tmp"))


def test_validate_entities():
    p = {"entities": ["a", "a", "b", "c"]}
    assert MemoryEngine._validate_entities_payload(p, 10) == ["a", "b", "c"]
    assert MemoryEngine._validate_entities_payload(p, 2) == ["a", "b"]
    assert MemoryEngine._validate_entities_payload({}, 5) == []


def test_validate_causalities():
    p = {
        "causalities": [
            {"memory_id": 1, "level": 2},
            {"memory_id": 99, "level": 5},
        ]
    }
    assert MemoryEngine._validate_causalities_payload(p, {1, 2}) == {1: 2}
    out = MemoryEngine._validate_causalities_payload(
        {"causalities": [{"memory_id": 1, "level": 9}]},
        {1},
    )
    assert out[1] == 3


def test_validate_ranks():
    p = {
        "ranks": [
            {"memory_id": 2, "rank": 1},
            {"memory_id": 1, "rank": 2},
        ]
    }
    assert MemoryEngine._validate_ranks_payload(p, {1, 2}) == [2, 1]
    # duplicate rank: first wins; second row skipped (not a full permutation)
    out_dup = MemoryEngine._validate_ranks_payload(
        {"ranks": [{"memory_id": 1, "rank": 1}, {"memory_id": 2, "rank": 1}]},
        {1, 2},
    )
    assert out_dup == [1]


def test_summarize_memory_buffer_returns_stripped_text(monkeypatch):
    history = [{"role": "user", "content": "hi"}]

    def fake_llm(query, system_instruction="", config=None, history=None):
        assert history == [{"role": "user", "content": "hi"}]
        assert "session memory" in system_instruction
        return {"content": "  # Session Title\nUpdated notes\n  "}

    monkeypatch.setattr(mem_eng, "llm_side_request", fake_llm)
    engine = _make_engine()
    assert engine.summarize_memory_buffer(history) == "# Session Title\nUpdated notes"


@pytest.mark.asyncio
async def test_asummarize_memory_buffer_delegates_to_sync(monkeypatch):
    engine = _make_engine()
    monkeypatch.setattr(
        engine,
        "summarize_memory_buffer",
        lambda history: "async notes",
    )
    assert await engine.asummarize_memory_buffer([{"role": "user", "content": "hi"}]) == "async notes"


def test_summarize_memory_buffer_empty_on_blank_response(monkeypatch):
    monkeypatch.setattr(mem_eng, "llm_side_request", lambda *args, **kwargs: {"content": "   "})
    engine = _make_engine()
    assert engine.summarize_memory_buffer([]) == ""


def test_summary_system_prompt_uses_config():
    engine = _make_engine()
    engine.config.summary_token_limit = 42
    assert "~42" in engine._summary_system_prompt()


def test_recall_applies_min_recall_score(monkeypatch):
    engine = _make_engine()
    engine.config.min_recall_score = 1.0

    np = __import__("numpy")
    now = datetime.now(timezone.utc)
    for nid, text in [(1, "alpha"), (2, "beta")]:
        engine._nodes[nid] = mem_cfg.MemoryNode(
            id=nid,
            text=text,
            fact_type="EXPERIENCE",
            timestamp=now,
            embedding=np.zeros(384, dtype="float32"),
            entities={},
            doc_length=1,
            causes={},
            effects={},
        )

    monkeypatch.setattr(engine, "entity_bm25", lambda q: [1, 2])
    monkeypatch.setattr(
        engine.embedding_model,
        "embed",
        lambda texts: iter([np.zeros(384, dtype="float32")]),
    )
    engine._semantic_index.search = MagicMock(
        return_value=(np.array([[0.1, 0.2]]), np.array([[1, 2]]))
    )
    monkeypatch.setattr(engine, "temporal_boost", lambda rrf: {1: 1.0, 2: 0.2})
    monkeypatch.setattr(engine, "neural_rerank", lambda ids, q: ids)

    _, nodes = engine.recall("test")
    assert len(nodes) == 1
    assert nodes[0].id == 1
