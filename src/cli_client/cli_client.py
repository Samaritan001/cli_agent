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
    """Cleans Markdown artifacts and ensures only the first JSON object is parsed."""
    # 1. Remove markdown code block markers with any language tag
    cleaned = re.sub(r"```[a-z]*\s?|```", "", raw_string).strip()
    
    # 2. If there is still extra text after the closing brace, 
    # isolate just the first JSON object found.
    start_idx = cleaned.find('{')
    end_idx = cleaned.rfind('}')
    if start_idx != -1 and end_idx != -1:
        return cleaned[start_idx:end_idx + 1]
    
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

def generate_list_command():
    return """
    ```json
    {
        "command": "list"
    }
    ```
    """

def generate_activate_command(server_name):
    return f"""
    ```json
    {{
        "command": "activate",
        "server_name": "{server_name}"
    }}
    ```
    """

def generate_execute_command(server_name, language, code):
    return f"""
    ```json
    {{
        "command": "execute",
        "server_name": "{server_name}",
        "language": "{language}",
        "code": "{code}"
    }}
    ```
    """

def generate_stop_command(server_name):
    return f"""
    ```json
    {{
        "command": "stop",
        "server_name": "{server_name}"
    }}
    ```
    """

# Example usage
if __name__ == "__main__":
    # Simulated LLM output
    commands = [
        generate_list_command(),
        generate_activate_command("weather"),
        generate_activate_command("weather"),
        generate_execute_command("weather", "python", "print('Hello from the AI Agent!')"),
        generate_execute_command("weather", "python", "import asyncio; print(asyncio.run(get_forecast(34.0522, -118.2437)))"),
        generate_execute_command("weather", "python", "import asyncio; print(asyncio.run(get_alerts('MA')))"),
        generate_stop_command("weather"),
        generate_stop_command("weather")
    ]
    
    for command in commands:
        mock_agent_loop(command)

