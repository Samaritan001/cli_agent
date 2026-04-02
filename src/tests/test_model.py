import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest
from unittest.mock import patch, MagicMock
from cli_client.model import LanguageModel


@patch("cli_client.model.os.getenv", return_value="fake_api_key")
def test_language_model_init(mock_getenv):
    # Test Provider Initialization
    model = LanguageModel(model_type="openai")
    assert model.model_name == "gpt-5"
    
    # Test Adding Messages
    model.add_user_message("Hello")
    assert len(model.history) == 1
    assert model.history[0]["role"] == "user"

    # Test Tool Response formatting
    model.add_tool_response("Success", "math", "call_123")
    assert len(model.history) == 2
    assert model.history[1]["role"] == "tool"

@patch("cli_client.model.os.getenv", return_value="fake_api_key")
def test_to_openai_formatting(mock_getenv):
    model = LanguageModel(model_type="openai")
    tool_info = {
        "tool_summaries": "Math Tool",
        "tool_manuals": "How to use Math Tool"
    }
    msgs = model.to_openai(tool_info)
    assert len(msgs) == 3 # Base system, System (summaries), System (manuals)
    assert "Math Tool" in msgs[1]["content"]

if __name__ == "__main__":
    test_language_model_init()
    test_to_openai_formatting()
    print("✅ All tests passed!")