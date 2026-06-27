import pytest
from unittest.mock import patch, MagicMock, AsyncMock

from cli_client.model import BaseLanguageModel, ClientLanguageModel, LanguageModelConfig, DEFAULT_MODELS


@patch("cli_client.model.os.getenv", return_value="fake_api_key")
def test_language_model_init(mock_getenv):
    model = BaseLanguageModel(LanguageModelConfig(model_type="openai"))
    assert model.config.model_name == DEFAULT_MODELS["openai"]

    model.add_user_message("Hello")
    history = model.get_history()
    assert len(history) == 1
    assert history[0]["role"] == "user"


@patch("cli_client.model.os.getenv", return_value="fake_api_key")
def test_to_messages_formatting(mock_getenv):
    model = BaseLanguageModel(LanguageModelConfig(model_type="openai"))
    tool_info = {
        "tool_summaries": "Math Tool",
        "tool_manuals": "How to use Math Tool",
    }
    msgs = model.to_messages(tool_info)
    assert len(msgs) == 3
    assert "Math Tool" in msgs[1]["content"]


@pytest.mark.asyncio
@patch("cli_client.model.os.getenv", return_value="fake_api_key")
async def test_client_agenerate_response_forwards_kwargs(mock_getenv):
    model = ClientLanguageModel(
        config=LanguageModelConfig(model_type="openai"),
        tool_definitions=[],
        system_instruction="test",
    )
    sentinel_nodes = [MagicMock()]
    model.generate_response = MagicMock(return_value="ok")

    result = await model.agenerate_response(
        {"tool_summaries": "", "tool_manuals": ""},
        max_tokens=128,
        memory_nodes=sentinel_nodes,
        history_window=5,
    )

    assert result == "ok"
    model.generate_response.assert_called_once_with(
        {"tool_summaries": "", "tool_manuals": ""},
        max_tokens=128,
        memory_nodes=sentinel_nodes,
        history_window=5,
    )
