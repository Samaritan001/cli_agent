import os
from dotenv import load_dotenv

import json

import logging

LOG_FORMAT = "\033[32m%(levelname)s\033[0m:    %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger("model")

from openai import OpenAI          # v2.x+ (2026)
from anthropic import Anthropic    # v1.x+ (2026)
from google import genai           # New Google GenAI SDK (2026)

load_dotenv()

API_KEY_VARS = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "google": "GOOGLE_API_KEY"
}

DEFAULT_MODELS = {
    "openai": "gpt-5.4-nano",
    "anthropic": "claude-4-6-opus",
    "google": "gemini-3.1-flash-lite-preview"
}

CLIENT_TYPES = {
    "openai": OpenAI,
    "anthropic": Anthropic,
    "google": genai.Client
}



class LanguageModelBase:
    def __init__(self, model_type="google", model_name=None):
        """
        Initializes the model wrapper.
        :param model_type: 'openai', 'anthropic', or 'google'
        :param api_key: Your provider API key
        :param model_name: Optional specific model name (defaults to flagship 2026 models)
        """

        self.history = []
        self.system_instruction = ""
        
        self.model_type = model_type.lower()
        if self.model_type == "gpt":
            self.model_type = "openai"
        elif self.model_type == "claude":
            self.model_type = "anthropic"
        elif self.model_type == "gemini":
            self.model_type = "google"
        if self.model_type not in API_KEY_VARS:
            raise ValueError(f"Unsupported model type: {model_type}")

        try:
            self.api_key = os.getenv(API_KEY_VARS.get(self.model_type, ""))
        except Exception as e:
            raise ValueError(f"Error occurred while fetching API key for {self.model_type}: {e}")

        # TODO: Need to verify model names and matching with model types
        self.model_name = model_name if model_name else DEFAULT_MODELS.get(self.model_type, DEFAULT_MODELS["openai"])
        self.client = CLIENT_TYPES.get(self.model_type, OpenAI)(api_key=self.api_key)


    def update_system_instruction(self, instruction: str):
        """Allows dynamic updating of system instructions."""
        self.system_instruction = instruction

    def add_user_message(self, content: str):
        """Adds a standard message to the history."""
        if self.model_type == "openai":
            self.history.append({"role": "user", "content": content})
        elif self.model_type == "anthropic":
            self.history.append({"role": "user", "content": [{"type": "text", "text": content}]})
        elif self.model_type == "google":
            self.history.append({"role": "user", "parts": [{"text": content}]})


    def generate_response(self, max_tokens=1000):
        """
        Sends messages to the model via the respective API client.
        """
        
        # Need to convert history and tool info into model-specific formats
        # The inner roles of history messages are "system", "user", "assistant", "tool"

        if self.model_type == "openai":
            messages = self.to_openai()
            # logger.info(f"System instruction:\n{messages[0]['content']}\n")
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_completion_tokens=max_tokens
            )
            self.history.append(response.choices[0].message.to_dict()) # Record in history
            return response
            
        elif self.model_type == "anthropic":
            system_prompt, anthropic_msgs = self.to_anthropic()
            response = self.client.messages.create(
                model=self.model_name,
                max_tokens=max_tokens,
                system=system_prompt,
                messages=anthropic_msgs
            )
            self.history.append({"role": "assistant", "content": response.content}) # Record in history
            return response
            
        elif self.model_type == "google":
            system_instruction, contents = self.to_google()
            # logger.info(f"System instruction:\n{system_instruction}")
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=contents,
                config=genai.types.GenerateContentConfig(
                    system_instruction=system_instruction,
                    max_output_tokens=max_tokens
                )
            )
            self.history.append(response.candidates[0].content) # Record in history
            return response
    
    def to_openai(self, tool_info=None):
        """
        OpenAI format: system, user, assistant, tool.
        Note: 'tool' role requires a 'tool_call_id'. 
        If missing, this script adds a dummy ID for schema compliance.
        """
        openai_msgs = [{"role": "system", "content": self.system_instruction}]
        openai_msgs.extend(self.history)

        if tool_info is None:
            return openai_msgs
        
        # Add tool information
        if tool_info["tool_summaries"] == "":
            openai_msgs.append({"role": "system", "content": "No tools currently available."})
        else:
            openai_msgs.append({"role": "system", "content": f"Available tools summaries:\n{tool_info['tool_summaries']}"})
            if tool_info["tool_manuals"] != "":
                openai_msgs.append({"role": "system", "content": f"Active tools full manuals:\n{tool_info['tool_manuals']}"})
        
        return openai_msgs

    def to_anthropic(self, tool_info=None):
        """
        Anthropic format: user, assistant. 
        System messages must be passed separately.
        Tool results are sent as 'user' role with a specific content block structure.
        """
        system_prompt = self.system_instruction + "\n"
        anthropic_msgs = []
        anthropic_msgs.extend(self.history)

        if tool_info is None:
            return system_prompt, anthropic_msgs
        
        # Add tool information as system prompt
        if tool_info["tool_summaries"] == "":
            system_prompt += "No tools currently available.\n"
        else:
            system_prompt += f"Available tools summaries:\n{tool_info['tool_summaries']}\n"
            if tool_info["tool_manuals"] != "":
                system_prompt += f"Active tools full manuals:\n{tool_info['tool_manuals']}\n"

        return system_prompt, anthropic_msgs

    def to_google(self, tool_info=None):
        """
        Google Gemini format: user, model.
        System instructions are separate.
        Tool results use the 'function_response' part.
        """
        system_instruction = self.system_instruction + "\n"
        google_msgs = []
        google_msgs.extend(self.history)

        if tool_info is None:
            return system_instruction, google_msgs
        
        # Add tool information as system instruction
        if tool_info["tool_summaries"] == "":
            # logger.info("No tools currently available.")
            system_instruction += "No tools currently available.\n"
        else:
            # logger.info(f"Available tools summaries:\n{tool_info['tool_summaries']}")
            system_instruction += f"Available tools summaries:\n{tool_info['tool_summaries']}\n"
            if tool_info["tool_manuals"] != "":
                # logger.info(f"Active tools full manuals:\n{tool_info['tool_manuals']}")
                system_instruction += f"Active tools full manuals:\n{tool_info['tool_manuals']}\n"
        
        return system_instruction, google_msgs


    def parse_response(self, response):
        """
        Parses response into a standard dict including tool_calls and IDs.
        Format: {'content': '...', 'tool_calls': [Dict[id, name, arguments]]}
        """

        result = {"content": "", "tool_calls": []}
        try:
            if self.model_type == "openai":
                result = self.from_openai(response)
            elif self.model_type == "anthropic":
                result = self.from_anthropic(response)
            elif self.model_type == "google":
                result = self.from_google(response)
        except Exception as e:
            result["content"] = f"Parsing Error: {str(e)}"

        return result

    def from_openai(self, response):
        msg = response.choices[0].message
        result = {"content": "", "tool_calls": []}
        result["content"] = msg.content or ""
        if msg.tool_calls:
            for tc in msg.tool_calls:
                result["tool_calls"].append({
                    "id": tc.id,
                    "command": tc.function.name,
                    "arguments": json.loads(tc.function.arguments)
                })
        return result
    
    def from_anthropic(self, response):
        # Anthropic content blocks can be 'text' or 'tool_use'
        result = {"content": "", "tool_calls": []}
        for block in response.content:
            if block.type == "text":
                result["content"] += block.text
            elif block.type == "tool_use":
                result["tool_calls"].append({
                    "id": block.id,
                    "command": block.name,
                    "arguments": block.input
                })
        return result

    def from_google(self, response):
        # Gemini 3 returns 'parts' which may contain 'function_call'
        candidate = response.candidates[0]
        result = {"content": "", "tool_calls": []}
        for part in candidate.content.parts:
            if part.text:
                result["content"] += part.text
            if part.function_call:
                fc = part.function_call
                # Gemini 3+ includes a 'call_id' in the function_call object
                result["tool_calls"].append({
                    "id": getattr(fc, 'id', fc.name), # Fallback to name if ID missing
                    "command": fc.name,
                    "arguments": dict(fc.args)
                })
        return result
    
    


class ClientLanguageModel(LanguageModelBase):
    def __init__(self, model_type="google", model_name=None):
        super().__init__(model_type, model_name)

        tool_definition_path = f"tool_definitions_{self.model_type}.json"
        with open(tool_definition_path, "r") as f:
            self.tool_definitions = json.load(f)
        with open("system_instruction.md", "r") as f:
            self.system_instruction = f.read()
#         self.system_instruction = """
# You are a helpful assistant that can call tools to get information or perform actions.
# You have several commands to interact with servers with tools: "list", "activate", "stop", "execute". The command must be in only one of the four names.
# - list: Get summaries of all available servers. Arguments: None.
# - activate: Activate a server, and load the server's manual into the context for use. Arguments: "server_name".
# - stop: Deactivate a server, and remove its manual from the context. Arguments: "server_name".
# - execute: Execute a command on an active server, specifically executing the code you write utilizing APIs of the server introduced in its manual. Arguments: "server_name", "language", "code".
# """

    
    def add_tool_response(self, content: str, command: str, tool_call_id: str):
        """Adds a standard message to the history."""
        if self.model_type == "openai":
            self.history.append({
                "role": "tool",
                "content": content,
                "tool_call_id": tool_call_id
            })
        elif self.model_type == "anthropic":
            self.history.append({
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": tool_call_id,
                        "content": content,
                    }
                ]
            })
        elif self.model_type == "google":
            self.history.append({
                "role": "user", 
                "parts": [{
                    "function_response": {
                        "name": command,
                        "response": {"result": content}
                    }
                }]
            })

    def generate_response(self, tool_info, max_tokens=1000):
        """
        Sends messages to the model via the respective API client.
        """
        
        # Need to convert history and tool info into model-specific formats
        # The inner roles of history messages are "system", "user", "assistant", "tool"

        if self.model_type == "openai":
            messages = self.to_openai(tool_info)
            # logger.info(f"System instruction:\n{messages[0]['content']}\n")
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                tools=self.tool_definitions,
                max_completion_tokens=max_tokens
            )
            self.history.append(response.choices[0].message.to_dict()) # Record in history
            return response
            
        elif self.model_type == "anthropic":
            system_prompt, anthropic_msgs = self.to_anthropic(tool_info)
            response = self.client.messages.create(
                model=self.model_name,
                max_tokens=max_tokens,
                system=system_prompt,
                messages=anthropic_msgs
            )
            self.history.append({"role": "assistant", "content": response.content}) # Record in history
            return response
            
        elif self.model_type == "google":
            system_instruction, contents = self.to_google(tool_info)
            # logger.info(f"System instruction:\n{system_instruction}")
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=contents,
                config=genai.types.GenerateContentConfig(
                    system_instruction=system_instruction,
                    max_output_tokens=max_tokens,
                    tools=self.tool_definitions
                )
            )
            self.history.append(response.candidates[0].content) # Record in history
            return response

    # parse_response method is inherited from LanguageModelBase
    
    
class MemoryLanguageModel(LanguageModelBase):
    def __init__(self, model_type="google", model_name=None):
        super().__init__(model_type, model_name)
    
    def extract_entities(self, context: str, max_entities: int, max_tokens: int = 256):
        self.system_instruction = self._entities_system_prompt(max_entities)
        self.add_user_message(context)
        response = self.generate_response(max_tokens=max_tokens)
        return self.parse_response(response)

    def _entities_system_prompt(max_entities: int) -> str:
    return (
        "You extract named entities for memory indexing. "
        "Return ONLY valid JSON with this exact shape: "
        '{"entities": ["Entity1", "Entity2"]}. '
        f"Include at most {max_entities} entities. "
        "Prefer people, organizations, locations, and key proper nouns; "
        "use short surface forms as they appear in the text; "
        "no duplicate meanings; use an empty array if there are none."
    )



