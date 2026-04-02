import json
import httpx
import asyncio
from fastapi import FastAPI, HTTPException, Response, status

import re
import logging

from typing import List, Dict, Any

from managers import ToolManualManager
from model import LanguageModel, MockLanguageModel

LOG_FORMAT = "\033[32m%(levelname)s\033[0m:    %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger("cli_client")

SERVER_URL = "http://127.0.0.1:8000/orchestrate"

# def clean_json_string(raw_string):
#     """Cleans Markdown artifacts and ensures valid JSON (Object or Array) is parsed."""
#     # 1. Remove markdown code block markers
#     cleaned = re.sub(r"```[a-z]*\s?|```", "", raw_string).strip()
    
#     # 2. Find the bounds of the JSON structure (either { } or [ ])
#     # We find the first occurrence of either { or [
#     match_start = re.search(r'[\[\{]', cleaned)
#     # We find the last occurrence of either } or ]
#     match_end = re.search(r'[\}\]](?=[^\}\]]*$)', cleaned)

#     if match_start and match_end:
#         start_idx = match_start.start()
#         end_idx = match_end.start()
#         return cleaned[start_idx:end_idx + 1]
    
#     return cleaned

class CLIClient:
    def __init__(self, model_type="google", model_name=None, mock_flag=False, test_flag=False):
        self.mock_flag = mock_flag
        self.test_flag = test_flag
        self.tool_manager = ToolManualManager()
        if mock_flag:
            logger.info("Initializing CLIClient in MOCK MODE with MockLanguageModel.")
            self.language_model = MockLanguageModel(model_type=model_type, model_name=model_name)
        else:
            self.language_model = LanguageModel(model_type=model_type, model_name=model_name)

    async def agent_loop(self):
        if self.mock_flag:
            logger.warning("Running agent loop in MOCK MODE.")
            print("--- User Input ---\nEnter your message for the agent (or 'quit' to exit):\n")
            user_input = "What is the weather like in xxx?"
            print(user_input)
            self.language_model.add_user_message(user_input)
            print(f"\n--- Agent Response ---")
            for i in range(5):
                llm_output = self.language_model.parse_response(self.language_model.generate_response(self.tool_manager.get_tool_info(), with_tool_calls=i))
                content = llm_output.get("content", "")
                if content != "":
                    print(f"{content}\n")
                tool_calls = llm_output.get("tool_calls", None)
                await self.tool_callings(tool_calls)
            print("--- User Input ---")
            user_input = "quit"
            print("Full conversation history:")
            print(f"System Instructions:\n{self.language_model.system_instruction}")
            print("\nMessage History:")
            for msg in self.language_model.history:
                print(f"{msg['role']}: {msg.get('content', msg.get('parts'))}\n")
            print("Tool Instructions:\n")
            tool_info = self.tool_manager.get_tool_info()
            if tool_info["tool_summaries"] == "":
                print("No tool summaries registered.")
            else:
                print(f"Available tools summaries:\n{tool_info['tool_summaries']}")
                if tool_info["tool_manuals"] != "":
                    print(f"Active tools full manuals:\n{tool_info['tool_manuals']}")
            logger.warning("Exiting agent loop in MOCK MODE.")
            
            return
                
        while True:
            user_input = input("--- User Input ---\nEnter your message for the agent (or 'quit' to exit):\n")
            if user_input.lower() == "quit":
                break
            self.language_model.add_user_message(user_input)

            print(f"\n--- Agent Response ---")
            if self.test_flag:
                llm_output = self.language_model.parse_response(self.language_model.generate_response(self.tool_manager.get_tool_info()))
                print(f"LLM Output (with tool calls):\n{llm_output}\n")
            else:
                llm_output = self.language_model.parse_response(self.language_model.generate_response(self.tool_manager.get_tool_info()))
            content = llm_output.get("content", "")
            if content != "":
                print(f"{content}\n")
            tool_calls = llm_output.get("tool_calls", [])
            while len(tool_calls) > 0:
                await self.tool_callings(tool_calls)
                llm_output = self.language_model.parse_response(self.language_model.generate_response(self.tool_manager.get_tool_info()))
                content = llm_output.get("content", "")
                if content != "":
                    print(f"{content}\n")
                tool_calls = llm_output.get("tool_calls", None)

    async def tool_callings(self, tool_calls: List[Dict[str, Any]]):
        logger.info(f"Processing {len(tool_calls)} tool calls...")
        command_records = {}
        calls = []
        # Tool calls preprocess
        for tool_call in tool_calls:
            if "command" not in tool_call or "id" not in tool_call:
                continue
            command_records.update({
                tool_call["id"]: [tool_call["command"], tool_call["arguments"].get("server_name", None)]
            })
            call = tool_call["arguments"]
            call["id"] = tool_call["id"]
            call["command"] = tool_call["command"]
            if call["command"] == "activate":
                call["fetch_manual"] = not self.tool_manager.check_tool(call["server_name"])
            calls.append(call)
            logger.info(f"Calling command '{call['command']}' with arguments {call}")

        async with httpx.AsyncClient() as client:
            call_requests = [client.post(SERVER_URL, json=call) for call in calls]
            call_responses = await asyncio.gather(*call_requests)

        # Tool calls reponse postprocess
        for call_resp in call_responses:
            call_resp = json.loads(call_resp.text)
            id = call_resp.get("id")
            command = command_records[id][0]
            server_name = command_records[id][1]
            if call_resp["status"] != status.HTTP_200_OK:
                error_status = call_resp["status"]
                detail = call_resp["detail"]
                self.language_model.add_tool_response(
                    content=f"{error_status} - {detail}",
                    command=command,
                    tool_call_id=id
                )
                logger.error(f"❌ Error in tool call id {id}: {error_status} - {detail}\n")
            else:
                # tool calling results
                if command == "list":
                    self.tool_manager.register_summary(json.loads(call_resp["result"]))
                    result = "Tool summaries have been registered."
                if command == "activate":
                    if call_resp["result"]:
                        self.tool_manager.register_tool(server_name, call_resp["result"])
                    self.tool_manager.inject_tool(server_name)
                    result = f"Tool '{server_name}' has been activated."
                elif command == "stop":
                    self.tool_manager.prune_tool(server_name)
                    result = f"Tool '{server_name}' has been stopped."
                elif command == "execute":
                    result = call_resp["result"]

                self.language_model.add_tool_response(
                    content=result,
                    command=command,
                    tool_call_id=id
                )
                
                logger.info(f"✅ Success in tool call id {id}")
                logger.info(f"ℹ️  Info: {call_resp.get('info', 'No info available')}\n")



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
    cli_client = CLIClient(mock_flag=False, test_flag=True)
    asyncio.run(cli_client.agent_loop())


    # cli_client = CLIClient()

    # # Simulated LLM output
    # commands = [
    #     generate_list_command(),
    #     generate_activate_command("weather"),
    #     generate_execute_commands(["weather", "weather"], ["python", "python"], [
    #         "print('Hello from the AI Agent!')",
    #         "import asyncio; print(asyncio.run(get_forecast(34.0522, -118.2437)))"
    #     ]),
    #     generate_stop_command("weather"),
    #     generate_activate_command("weather"),
    #     generate_execute_commands(["weather"], ["python"], [
    #         "import asyncio; print(asyncio.run(get_alerts('MA')))"
    #     ])
    # ]
    
    # for command in commands:
    #     asyncio.run(cli_client.mock_agent_loop(command))

    
    # asyncio.run(cli_client.mock_agent_loop(generate_stop_command("weather")))
    
    
