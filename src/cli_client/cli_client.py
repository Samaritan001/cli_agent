import json
import httpx
import asyncio
from fastapi import FastAPI, HTTPException, Response, status

import re
import logging

from typing import Dict, Any

from managers import AgentContextManager

LOG_FORMAT = "\033[32m%(levelname)s\033[0m:    %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger("manual_generator")

SERVER_URL = "http://127.0.0.1:8000/orchestrate"

def clean_json_string(raw_string):
    """Cleans Markdown artifacts and ensures valid JSON (Object or Array) is parsed."""
    # 1. Remove markdown code block markers
    cleaned = re.sub(r"```[a-z]*\s?|```", "", raw_string).strip()
    
    # 2. Find the bounds of the JSON structure (either { } or [ ])
    # We find the first occurrence of either { or [
    match_start = re.search(r'[\[\{]', cleaned)
    # We find the last occurrence of either } or ]
    match_end = re.search(r'[\}\]](?=[^\}\]]*$)', cleaned)

    if match_start and match_end:
        start_idx = match_start.start()
        end_idx = match_end.start()
        return cleaned[start_idx:end_idx + 1]
    
    return cleaned

class CLIClient:
    def __init__(self):
        self.next_tool_call_id = 0
        self.memory = []
        self.context_manager = AgentContextManager()

    async def tool_callings(self, json_payload: str):
        call_records = {}   # call_id to call_details

        # Tool calls preprocess
        data = json.loads(clean_json_string(json_payload))
        if isinstance(data, dict):
            data = [data]
        for call in data:
            if "command" not in call:
                continue
            id = self.next_tool_call_id
            call["id"] = id
            call_records[id] = call
            self.next_tool_call_id += 1
            if call["command"] == "activate":
                call["fetch_manual"] = not self.context_manager.check_tool(call["server_name"])

        async with httpx.AsyncClient() as client:
            calls = [client.post(SERVER_URL, json=call) for call in data if "command" in call]
            call_responses = await asyncio.gather(*calls)

        # Tool calls reponse postprocess
        for call_resp in call_responses:
            call_resp = json.loads(call_resp.text)
            id = call_resp.get("id")
            if call_resp["status"] != status.HTTP_200_OK:
                error_status = call_resp["status"]
                detail = call_resp["detail"]
                self.context_manager.add_message("tool_response", f"{error_status} - {detail}")
                logging.error(f"❌ Error in tool call id {id}: {error_status} - {detail}")
            else:
                call_record = call_records.get(id)
                # tool calling results
                if call_record["command"] == "list":
                    self.context_manager.register_summary(json.loads(call_resp["result"]))
                if call_record["command"] == "activate":
                    if call_resp["result"]:
                        self.context_manager.register_tool(call_record["server_name"], call_resp["result"])
                    self.context_manager.inject_tool(call_record["server_name"])
                elif call_record["command"] == "stop":
                    self.context_manager.prune_tool(call_record["server_name"])
                elif call_record["command"] == "execute":
                    result = call_resp["result"]
                    self.context_manager.add_message("tool_response", result)
                
                logging.info(f"✅ Success in tool call id {id}")
                logging.info(f"ℹ️  Info: {call_resp.get('info', 'No info available')}")

    async def mock_agent_loop(self, llm_output):
        print("--- Agent received code suggestion ---")
        await self.tool_callings(llm_output)



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

def generate_execute_commands(server_names, languages, codes):
    calls = []
    for i in range(len(server_names)):
        calls.append({
            "command": "execute",
            "server_name": server_names[i],
            "language": languages[i],
            "code": codes[i]
        })
    return json.dumps(calls)

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
    cli_client = CLIClient()

    # Simulated LLM output
    commands = [
        generate_list_command(),
        generate_activate_command("weather"),
        generate_execute_commands(["weather", "weather"], ["python", "python"], [
            "print('Hello from the AI Agent!')",
            "import asyncio; print(asyncio.run(get_forecast(34.0522, -118.2437)))"
        ]),
        generate_stop_command("weather"),
        generate_activate_command("weather"),
        generate_execute_commands(["weather"], ["python"], [
            "import asyncio; print(asyncio.run(get_alerts('MA')))"
        ])
    ]
    
    for command in commands:
        asyncio.run(cli_client.mock_agent_loop(command))

    context = cli_client.context_manager.get_full_context()
    print("\n--- Final Agent Context ---")
    for message in context:
        print(f"{message['role'].upper()}:\n{message['content']}\n")

    asyncio.run(cli_client.mock_agent_loop(generate_stop_command("weather")))
    
    
