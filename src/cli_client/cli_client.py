import json
import requests
import re
import logging
from fastapi import FastAPI, HTTPException, Response, status

LOG_FORMAT = "\033[32m%(levelname)s\033[0m:    %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger("manual_generator")

SERVER_URL = "http://127.0.0.1:8000/orchestrate"

def clean_json_string(raw_string):
    """Cleans potential Markdown artifacts from LLM response."""
    # Removes ```json ... ``` blocks
    cleaned = re.sub(r"```json\s?|```", "", raw_string).strip()
    return cleaned

def execute_tool(json_payload):
    data = json.loads(clean_json_string(json_payload))
    response = requests.post(SERVER_URL, json=data)
    # logging.info(f"Raw Response:\n{response.status_code} - {response.text}")
    # logging.info("huhuhuhun")
    if response.status_code != status.HTTP_200_OK:
        raise HTTPException(status_code=response.status_code, detail=f"{response.json()['detail']}")
    if not response.text:
        return {"info": "Server returned successfully with an empty body"}
    return response.json()

def mock_agent_loop(llm_output):
    print("--- Agent received code suggestion ---")
    try:
        result = execute_tool(llm_output)
    except Exception as e:
        print(f"❌ Error occurred while executing tool: {e}")
        return

    print("✅ Success!")
    print(result)

# Example usage
if __name__ == "__main__":
    # Simulated LLM output
    llm_json_list = """
    ```json
    {
        "command": "list"
    }
    ```
    """
    llm_json_activate = """
    ```json
    {
        "command": "activate",
        "server_name": "weather"
    }
    ```
    """
    llm_json_execute = """
    ```json
    {
        "command": "execute",
        "server_name": "weather",
        "language": "python",
        "code": "import asyncio; print('Hello from the AI Agent!'); from weather import get_forecast, get_alerts; print(asyncio.run(get_forecast(34.0522, -118.2437))); print('#'*30); print(asyncio.run(get_alerts('MA')))"
    }
    ```
    """
    llm_json_stop = """
    ```json
    {
        "command": "stop",
        "server_name": "weather"
    }
    ```
    """
    mock_agent_loop(llm_json_list)
    mock_agent_loop(llm_json_activate)
    mock_agent_loop(llm_json_execute)
    mock_agent_loop(llm_json_stop)
