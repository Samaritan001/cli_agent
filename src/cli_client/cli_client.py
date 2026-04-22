import json
import httpx
import asyncio
from fastapi import FastAPI, HTTPException, Response, status

import re
import logging

from typing import List, Dict, Any

from managers import ToolManualManager
from model import ClientLanguageModel, LanguageModelConfig
from memory_engine import MemoryEngine, MemoryConfig

LOG_FORMAT = "\033[32m%(levelname)s\033[0m:    %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger("cli_client")

ORCHESTRATOR_URL = "http://127.0.0.1:8000/orchestrate"
MEMORY_DIR = "./memory_docs"

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


# TODO: Class Use Agent Sessions to control multiple agent rounds, save memory at the end of memory_session


class CLIClient:
    def __init__(self, model_type="google", model_name=None, test_flag=False):
        self.test_flag = test_flag
        self.tool_manager = ToolManualManager()
        self.language_model = ClientLanguageModel(config=LanguageModelConfig(model_type=model_type, model_name=model_name))
        self.memory_engine = MemoryEngine(config=MemoryConfig(memory_dir=MEMORY_DIR), model_config=self.language_model.config)
        self.memory_buffer_start_id = 0
        self.history_window = 10  # Number of recent turns to keep in direct history context
        
        asyncio.run(self._list_available_servers())
    
    async def _list_available_servers(self):
        async with httpx.AsyncClient() as client:
            response = await client.post(ORCHESTRATOR_URL, json={"id": "auto-listing", "command": "list_available_servers"})
            if response.status_code == status.HTTP_200_OK:
                self.tool_manager.register_summary(json.loads(response["result"]))
                logger.info("Successfully retrieved and registered available servers.")
            else:
                logger.error(f"Error listing servers: {response.status_code} - {response.text}")

    async def agent_loop(self):                
        try:
            await self.memory_engine.aload_memory()
            while True:
                user_input = await asyncio.to_thread(
                    input,
                    "--- User Input ---\nEnter your message for the agent (or 'quit' to exit):\n",
                )
                if user_input.lower() == "quit":
                    await self._flush_memory_buffer()
                    break
                message_id = self.language_model.add_user_message(user_input)

                print(f"\n--- Agent Response ---")
                earliest_message_id = max(0, message_id - self.history_window)
                latest_stored_message_id, recalled_nodes = await self.memory_engine.arecall(
                    query=user_input,
                    earliest_message_id=earliest_message_id
                )
                temp_history_window = message_id - min(earliest_message_id, latest_stored_message_id+1)
                llm_output = self.language_model.parse_response(
                    await self.language_model.agenerate_response(
                        self.tool_manager.get_tool_info(),
                        memory_nodes=recalled_nodes,
                        history_window=temp_history_window
                    )
                )
                if self.test_flag:
                    print(f"LLM Output (with tool calls):\n{llm_output}\n")
                content = llm_output.get("content", "")
                if content != "":
                    print(f"{content}\n")
                tool_calls = llm_output.get("tool_calls", [])
                had_tool_calls = len(tool_calls) > 0
                while len(tool_calls) > 0:
                    await self.tool_callings(tool_calls)
                    temp_history_window = message_id - min(earliest_message_id, latest_stored_message_id+1)
                    llm_output = self.language_model.parse_response(
                        await self.language_model.agenerate_response(
                            self.tool_manager.get_tool_info(),
                            memory_nodes=recalled_nodes,
                            history_window=temp_history_window
                        )
                    )
                    if self.test_flag:
                        print(f"LLM Output (with tool calls):\n{llm_output}\n")
                    content = llm_output.get("content", "")
                    if content != "":
                        print(f"{content}\n")
                    tool_calls = llm_output.get("tool_calls", [])

                # TODO: filter out only important tool results
                # TODO: better summarization triggering algorithm
                if self.memory_buffer_start_id == 0:
                    self.memory_buffer_start_id = message_id
                if self.memory_engine.should_summarize_turn(
                    user_input=user_input,
                    assistant_output=content,
                    had_tool_calls=had_tool_calls,
                    pending_turns=message_id - self.memory_buffer_start_id + 1,
                ):
                    await self._flush_memory_buffer()
        finally:
            await self.memory_engine.asave_memory()

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
            if call["command"] == "activate_server":
                call["fetch_manual"] = not self.tool_manager.check_tool(call["server_name"])
            calls.append(call)
            logger.info(f"Calling command '{call['command']}' with arguments {call}")

        async with httpx.AsyncClient() as client:
            call_requests = [client.post(ORCHESTRATOR_URL, json=call) for call in calls]
            call_responses = await asyncio.gather(*call_requests)

        # Tool calls reponse postprocess
        for call_resp in call_responses:
            call_resp = json.loads(call_resp.text)
            id = call_resp.get("id")
            command = command_records[id][0]
            server_name = command_records[id][1]
            message_id = 0
            if call_resp["status"] != status.HTTP_200_OK:
                error_status = call_resp["status"]
                detail = call_resp["detail"]
                message_id = self.language_model.add_tool_response(
                    content=f"{error_status} - {detail}",
                    command=command,
                    tool_call_id=id
                )
                logger.error(f"❌ Error in tool call id {id}: {error_status} - {detail}\n")
            else:
                # tool calling results
                if command == "list_available_servers":
                    self.tool_manager.register_summary(json.loads(call_resp["result"]))
                    result = "Tool summaries have been registered."
                if command == "activate_server":
                    if call_resp["result"]:
                        self.tool_manager.register_tool(server_name, call_resp["result"])
                    self.tool_manager.inject_tool(server_name)
                    result = f"Tool '{server_name}' has been activated."
                elif command == "stop_server":
                    self.tool_manager.prune_tool(server_name)
                    result = f"Tool '{server_name}' has been stopped."
                elif command == "execute_server_code":
                    result = call_resp["result"]
                else:
                    result = "Unknown command response."

                message_id = self.language_model.add_tool_response(
                    content=result,
                    command=command,
                    tool_call_id=id
                )
                
                logger.info(f"✅ Success in tool call id {id}")
                logger.info(f"ℹ️  Info: {call_resp.get('info', 'No info available')}\n")
        
        return message_id
    
    # Function: summarize context into memory content
    async def summarize_memory(self, max_memory_tokens: int = 512):
        self.language_model.add_user_message(self._summary_system_prompt(max_memory_tokens))
        response = await self.language_model.agenerate_response(
            self.tool_manager.get_tool_info(),
            memory_nodes=[],
            max_tokens=max_memory_tokens,
        )
        return self.language_model.parse_response(response)

    async def _flush_memory_buffer(self, end_message_id: int):
        if self.memory_buffer_start_id == 0:
            return
        history_window = end_message_id - self.memory_buffer_start_id + 1
        summary = self.memory_engine.summarize_memory_buffer(self.language_model.get_history(window=history_window))
        if summary.strip():
            await self.memory_engine.aremember(context=summary, message_id_range=[self.memory_buffer_start_id, end_message_id])
        self.memory_buffer_start_id = 0


def generate_list_command():
    return """
    ```json
    {
        "command": "list_available_servers"
    }
    ```
    """

def generate_activate_command(server_name):
    return f"""
    ```json
    {{
        "command": "activate_server",
        "server_name": "{server_name}"
    }}
    ```
    """

def generate_execute_commands(server_names, languages, codes):
    calls = []
    for i in range(len(server_names)):
        calls.append({
            "command": "execute_server_code",
            "server_name": server_names[i],
            "language": languages[i],
            "code": codes[i]
        })
    return json.dumps(calls)

def generate_stop_command(server_name):
    return f"""
    ```json
    {{
        "command": "stop_server",
        "server_name": "{server_name}"
    }}
    ```
    """

# Example usage
if __name__ == "__main__":
    cli_client = CLIClient(model_type="openai", test_flag=True)
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
    
    
