# Idea
Use the idea of CLI as foundation, integrating the server structure of MCP and the efficiency attributes of Agent Skills, dedicated to build a high performance AI Agent architecture with good reliability, security, and compatibility.

# Notes
1. Add API keys to .env
2. Look at pyproject.toml for uv environment package setup. Use `uv add` command.
3. To run a test, first `uv run orchestrator.py`, then `uv run cli_client.py`

# TODO
1. Add tool registration (json) to and test openai and anthropic APIs.
2. Improve server manual description by adding examples.
3. Rewrite the architecture in typescript.
4. Increase compatibility of current server with skills in the market.
