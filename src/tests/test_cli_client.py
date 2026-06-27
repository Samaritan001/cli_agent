import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from cli_client.cli_client import CLIClient

@pytest.fixture
def mock_httpx_post():
    with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
        mock_response = MagicMock()
        mock_response.text = '{"status": 200, "result": "Success", "id": "call_1"}'
        mock_post.return_value = mock_response
        yield mock_post

@pytest.mark.asyncio
@patch("cli_client.cli_client.MemoryEngine")
@patch("cli_client.cli_client.ClientLanguageModel")
async def test_should_summarize_uses_client_thresholds(MockLanguageModel, MockMemoryEngine):
    client = CLIClient(test_flag=True)
    client.summary_min_chars = 100
    client.summary_importance_threshold = 2.0
    client.summary_force_after_turns = 5

    assert client.should_summarize("short", "reply", had_tool_calls=False, pending_turns=1) is False
    assert client.should_summarize("x" * 60, "y" * 60, had_tool_calls=True, pending_turns=1) is True
    assert client.should_summarize("short", "reply", had_tool_calls=False, pending_turns=5) is True


@pytest.mark.asyncio
@patch("cli_client.cli_client.MemoryEngine")
@patch("cli_client.cli_client.ClientLanguageModel")
async def test_flush_memory_buffer_without_args_uses_latest_id(MockLanguageModel, MockMemoryEngine):
    client = CLIClient(test_flag=True)
    client.memory_buffer_start_id = 10
    client.language_model.get_latest_message_id.return_value = 15
    client.language_model.get_history.return_value = [{"role": "user", "content": "hi"}]
    client.memory_engine.asummarize_memory_buffer = AsyncMock(return_value="")

    await client._flush_memory_buffer()

    client.language_model.get_latest_message_id.assert_called_once()
    client.language_model.get_history.assert_called_once_with(window=6)
    client.memory_engine.aremember.assert_not_called()
    assert client.memory_buffer_start_id == 0


@pytest.mark.asyncio
@patch("cli_client.cli_client.MemoryEngine")
@patch("cli_client.cli_client.ClientLanguageModel")
async def test_flush_memory_buffer_with_explicit_end_id(MockLanguageModel, MockMemoryEngine):
    client = CLIClient(test_flag=True)
    client.memory_buffer_start_id = 3
    client.language_model.get_history.return_value = [{"role": "user", "content": "hi"}]
    client.memory_engine.asummarize_memory_buffer = AsyncMock(return_value="session notes")
    client.memory_engine.aremember = AsyncMock()

    await client._flush_memory_buffer(end_message_id=7)

    client.language_model.get_latest_message_id.assert_not_called()
    client.memory_engine.aremember.assert_awaited_once_with(
        context="session notes",
        message_id_range=[3, 7],
    )


@pytest.mark.asyncio
@patch("cli_client.cli_client.MemoryEngine")
@patch("cli_client.cli_client.ClientLanguageModel")
async def test_tool_calling(MockLanguageModel, MockMemoryEngine, mock_httpx_post):
    client = CLIClient(test_flag=True)
    # Mocking a tool call from the LLM
    # Shape expected by tool_callings: top-level id + command, arguments for orchestrator body
    tool_calls = [
        {
            "id": "call_1",
            "command": "execute_server_code",
            "arguments": {
                "server_name": "math_server",
                "language": "python",
                "code": "print(1+1)",
            },
        }
    ]
    
    await client.tool_calling(tool_calls)
    
    # Ensure network request was made
    assert mock_httpx_post.call_count == 1
    
    # Ensure tool manager/language model were updated
    client.language_model.add_tool_response.assert_called_once()

if __name__ == "__main__":
    mock_httpx_post = AsyncMock()
    test_tool_calling(
        MockLanguageModel=MagicMock(),
        MockMemoryEngine=MagicMock(),
        mock_httpx_post=mock_httpx_post,
    )
    print("✅ All tests passed!")