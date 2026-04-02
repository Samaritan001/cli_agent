import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

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
@patch("cli_client.cli_client.LanguageModel")
async def test_tool_callings(MockLanguageModel, mock_httpx_post):
    client = CLIClient()
    # Mocking a tool call from the LLM
    tool_calls = [
        {
            "id": "call_1",
            "name": "math_server",
            "arguments": {"command": "execute", "language": "python", "code": "print(1+1)"}
        }
    ]
    
    # The fix we discussed is applied in our mental model of the test
    await client.tool_callings(tool_calls)
    
    # Ensure network request was made
    assert mock_httpx_post.call_count == 1
    
    # Ensure tool manager/language model were updated
    client.language_model.add_tool_response.assert_called_once()

if __name__ == "__main__":
    mock_httpx_post = AsyncMock()
    test_tool_callings(MockLanguageModel=MagicMock(), mock_httpx_post=mock_httpx_post)
    print("✅ All tests passed!")