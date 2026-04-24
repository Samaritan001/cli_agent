import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest
from unittest.mock import patch

from cli_client.model import BaseLanguageModel, LanguageModelConfig, DEFAULT_MODELS


@patch("cli_client.model.os.getenv", return_value="fake_api_key")
def test_language_model_init(mock_getenv):
    model = BaseLanguageModel(LanguageModelConfig(model_type="openai"))
    assert model.config.model_name == DEFAULT_MODELS["openai"]

    model.add_user_message("Hello")
    assert len(model.history) == 1
    assert model.history[0]["role"] == "user"


@patch("cli_client.model.os.getenv", return_value="fake_api_key")
def test_to_openai_formatting(mock_getenv):
    model = BaseLanguageModel(LanguageModelConfig(model_type="openai"))
    tool_info = {
        "tool_summaries": "Math Tool",
        "tool_manuals": "How to use Math Tool",
    }
    msgs = model.to_openai(tool_info)
    assert len(msgs) == 3
    assert "Math Tool" in msgs[1]["content"]
