"""Unit tests: MemoryEngine JSON payload validators (no FAISS / embed)."""

import os
import sys

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "cli_client"))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from memory_engine import MemoryEngine  # noqa: E402


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
