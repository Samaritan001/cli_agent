"""Tests for memory entity JSON parsing (no API calls)."""

import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "cli_client")))

from memory_llm_provider import parse_entities_response


def test_parse_entities_object():
    raw = '{"entities": ["UCLA", "User", "UCLA"]}'
    assert parse_entities_response(raw, 10) == ["UCLA", "User"]


def test_parse_entities_list():
    raw = '["a", "b"]'
    assert parse_entities_response(raw, 10) == ["a", "b"]


def test_parse_entities_fence():
    raw = '```json\n{"entities": ["X"]}\n```'
    assert parse_entities_response(raw, 10) == ["X"]


def test_parse_entities_max():
    raw = '{"entities": ["a", "b", "c"]}'
    assert parse_entities_response(raw, 2) == ["a", "b"]
