import pytest
from unittest.mock import patch, MagicMock, AsyncMock, mock_open
from orchestrator import AIOrchestrator


@pytest.fixture
def mock_docker():
    with patch("docker.from_env") as mock_env:
        mock_client = MagicMock()
        mock_container = MagicMock()
        mock_client.containers.run.return_value = mock_container
        mock_env.return_value = mock_client
        yield mock_client


@pytest.fixture
def mock_fs():
    m_open = mock_open(read_data="Test Summary\nTest CLI Path\nManual Content")
    with patch("builtins.open", m_open), patch("os.listdir", return_value=["test_tool.md"]):
        yield


@pytest.mark.asyncio
async def test_orchestrator_list_servers(mock_docker, mock_fs):
    orchestrator = AIOrchestrator()
    result = await orchestrator.list_servers()
    assert result["status"] == 200
    assert "test_tool" in result["result"]


@pytest.mark.asyncio
async def test_orchestrator_activate_unknown_server(mock_docker, mock_fs):
    orchestrator = AIOrchestrator()
    result = await orchestrator.activate_server("missing", fetch_manual=False)
    assert result["status"] == 404
    assert "detail" in result
    assert "result" not in result


@pytest.mark.asyncio
@patch("orchestrator.AIOrchestrator._wait_for_server", return_value=True)
async def test_orchestrator_activate_and_execute(mock_wait, mock_docker, mock_fs):
    orchestrator = AIOrchestrator()

    result = await orchestrator.activate_server("test_tool", fetch_manual=False)
    assert result["status"] == 200
    assert "test_tool" in orchestrator.active_servers

    with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"stdout": "Hello World"}
        mock_post.return_value = mock_resp

        exec_result = await orchestrator.execute("test_tool", "python", "print('Hello World')")
        assert exec_result["status"] == 200
        assert exec_result["result"] == "Hello World"
