import os
from dotenv import load_dotenv
import asyncio

from typing import Dict, List, Optional, TYPE_CHECKING
import json
import logging
from dataclasses import dataclass

LOG_FORMAT = "\033[32m%(levelname)s\033[0m:    %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger("model")

from openai import OpenAI          # v2.x+ (2026)
from anthropic import Anthropic    # v1.x+ (2026)
from google import genai           # New Google GenAI SDK (2026)

if TYPE_CHECKING:
    from memory_config import MemoryNode


load_dotenv()

NAME_MAP = {
    **dict.fromkeys(["openai", "gpt"], "openai"),
    **dict.fromkeys(["anthropic", "claude"], "anthropic"),
    **dict.fromkeys(["google", "gemini"], "google")
}


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

DEFAULT_SESSION_MEMORY_TEMPLATE = """
# Session Title
_A short and distinctive 5-10 word descriptive title for the session. Super info dense, no filler_

# Current State
_What is actively being worked on right now? Pending tasks not yet completed. Immediate next steps._

# Task specification
_What did the user ask to build? Any design decisions or other explanatory context_

# Files and Functions
_What are the important files? In short, what do they contain and why are they relevant?_

# Workflow
_What bash commands are usually run and in what order? How to interpret their output if not obvious?_

# Errors & Corrections
_Errors encountered and how they were fixed. What did the user correct? What approaches failed and should not be tried again?_

# Codebase and System Documentation
_What are the important system components? How do they work/fit together?_

# Learnings
_What has worked well? What has not? What to avoid? Do not duplicate items from other sections_

# Key results
_If the user asked a specific output such as an answer to a question, a table, or other document, repeat the exact result here_

# Worklog
_Step by step, what was attempted, done? Very terse summary for each step_
"""

@dataclass
class LanguageModelConfig:
    model_type: str = "google"
    model_name: Optional[str] = None
    max_tokens: int = 512

class BaseLanguageModel:
    def __init__(self, config: Optional[LanguageModelConfig] = None):
        """
        Initializes the model wrapper.
        :param model_type: 'openai', 'anthropic', or 'google'
        :param api_key: Your provider API key
        :param model_name: Optional specific model name (defaults to flagship 2026 models)
        """

        self.history = []
        self.system_instruction = ""
        
        self.config = config or LanguageModelConfig()
        
        self.config.model_type = NAME_MAP.get(self.config.model_type.lower(), self.config.model_type)
        if self.config.model_type not in API_KEY_VARS:
            raise ValueError(f"Unsupported model type: {self.config.model_type}")

        try:
            self.api_key = os.getenv(API_KEY_VARS.get(self.config.model_type, ""))
        except Exception as e:
            raise ValueError(f"Error occurred while fetching API key for {self.config.model_type}: {e}")

        # TODO: Need to verify model names and matching with model types
        if self.config.model_name is None:
            self.config.model_name = DEFAULT_MODELS.get(self.config.model_type)

        if self.config.max_tokens > 2048:
            logger.warning(f"Max tokens {self.config.max_tokens} may exceed limits for some models. Consider reducing to 2048 or less.")
            self.config.max_tokens = 2048

        self.client = CLIENT_TYPES.get(self.config.model_type, OpenAI)(api_key=self.api_key)


    def update_system_instruction(self, instruction: str):
        """Allows dynamic updating of system instructions."""
        self.system_instruction = instruction

    def add_user_message(self, content: str):
        """Adds a standard message to the history."""
        if self.config.model_type == "openai":
            self.history.append({"role": "user", "content": content})
        elif self.config.model_type == "anthropic":
            self.history.append({"role": "user", "content": [{"type": "text", "text": content}]})
        elif self.config.model_type == "google":
            self.history.append({"role": "user", "parts": [{"text": content}]})


    def generate_response(self, memory_nodes: Optional[List["MemoryNode"]] = None, max_tokens: Optional[int] = None):
        """
        Sends messages to the model via the respective API client.
        """
        
        # Need to convert history and tool info into model-specific formats
        # The inner roles of history messages are "system", "user", "assistant", "tool"

        if max_tokens is None:
            max_tokens = self.config.max_tokens
        
        memory = ""
        if memory_nodes is not None:
            for node in memory_nodes:
                memory += f"# Memory Type\n{node.fact_type}\n\n# Timestamp\n{node.timestamp.isoformat()}\n\n{node.text}\n\n{'-' * 10}\n\n"

        if self.config.model_type == "openai":
            messages = self.to_openai(memory=memory)
            # logger.info(f"System instruction:\n{messages[0]['content']}\n")
            response = self.client.chat.completions.create(
                model=self.config.model_name,
                messages=messages,
                max_completion_tokens=max_tokens
            )
            self.history.append(response.choices[0].message.to_dict()) # Record in history
            return response
            
        elif self.config.model_type == "anthropic":
            system_prompt, anthropic_msgs = self.to_anthropic(memory=memory)
            response = self.client.messages.create(
                model=self.config.model_name,
                max_tokens=max_tokens,
                system=system_prompt,
                messages=anthropic_msgs
            )
            self.history.append({"role": "assistant", "content": response.content}) # Record in history
            return response
            
        elif self.config.model_type == "google":
            system_instruction, contents = self.to_google(memory=memory)
            # logger.info(f"System instruction:\n{system_instruction}")
            response = self.client.models.generate_content(
                model=self.config.model_name,
                contents=contents,
                config=genai.types.GenerateContentConfig(
                    system_instruction=system_instruction,
                    max_output_tokens=max_tokens
                )
            )
            self.history.append(response.candidates[0].content) # Record in history
            return response

    async def agenerate_response(self, memory_nodes: Optional[List["MemoryNode"]] = None, max_tokens: Optional[int] = None):
        return await asyncio.to_thread(self.generate_response, memory_nodes, max_tokens)
    
    def to_openai(self, tool_info=None, memory: Optional[str] = "") -> List[Dict]:
        """
        OpenAI format: system, user, assistant, tool.
        Note: 'tool' role requires a 'tool_call_id'. 
        If missing, this script adds a dummy ID for schema compliance.
        """
        openai_msgs = [{"role": "system", "content": self.system_instruction}]
        openai_msgs.extend(self.history)

        if memory != "":
            openai_msgs.append({"role": "system", "content": f"Relevant memory for past sessions:\n{memory}"})

        if tool_info is not None:
            # Add tool information
            if tool_info["tool_summaries"] == "":
                openai_msgs.append({"role": "system", "content": "No tools currently available."})
            else:
                openai_msgs.append({"role": "system", "content": f"Available tools summaries:\n{tool_info['tool_summaries']}"})
                if tool_info["tool_manuals"] != "":
                    openai_msgs.append({"role": "system", "content": f"Active tools full manuals:\n{tool_info['tool_manuals']}"})
        
        return openai_msgs

    def to_anthropic(self, tool_info=None, memory: Optional[str] = "") -> (str, List[Dict]):
        """
        Anthropic format: user, assistant. 
        System messages must be passed separately.
        Tool results are sent as 'user' role with a specific content block structure.
        """
        system_prompt = self.system_instruction + "\n"
        anthropic_msgs = []
        anthropic_msgs.extend(self.history)

        if memory != "":
            system_prompt += f"\nRelevant memory for past sessions:\n{memory}\n"

        if tool_info is not None:
            # Add tool information as system prompt
            if tool_info["tool_summaries"] == "":
                system_prompt += "No tools currently available.\n"
            else:
                system_prompt += f"Available tools summaries:\n{tool_info['tool_summaries']}\n"
                if tool_info["tool_manuals"] != "":
                    system_prompt += f"Active tools full manuals:\n{tool_info['tool_manuals']}\n"

        return system_prompt, anthropic_msgs

    def to_google(self, tool_info=None, memory: Optional[str] = "") -> (str, List[Dict]):
        """
        Google Gemini format: user, model.
        System instructions are separate.
        Tool results use the 'function_response' part.
        """
        system_instruction = self.system_instruction + "\n"
        google_msgs = []
        google_msgs.extend(self.history)

        if memory != "":
            system_instruction += f"\nRelevant memory for past sessions:\n{memory}\n"

        if tool_info is not None:
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
            if self.config.model_type == "openai":
                result = self.from_openai(response)
            elif self.config.model_type == "anthropic":
                result = self.from_anthropic(response)
            elif self.config.model_type == "google":
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
    
    


class ClientLanguageModel(BaseLanguageModel):
    def __init__(self, config: Optional[LanguageModelConfig] = None):
        super().__init__(config=config)

        tool_definition_path = f"tool_definitions_{self.config.model_type}.json"
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
        if self.config.model_type == "openai":
            self.history.append({
                "role": "tool",
                "content": content,
                "tool_call_id": tool_call_id
            })
        elif self.config.model_type == "anthropic":
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
        elif self.config.model_type == "google":
            self.history.append({
                "role": "user", 
                "parts": [{
                    "function_response": {
                        "name": command,
                        "response": {"result": content}
                    }
                }]
            })

    def generate_response(self, tool_info, memory_nodes: Optional[List["MemoryNode"]] = None, max_tokens=None):
        """
        Sends messages to the model via the respective API client.
        """
        
        # Need to convert history and tool info into model-specific formats
        # The inner roles of history messages are "system", "user", "assistant", "tool"

        if max_tokens is None:
            max_tokens = self.config.max_tokens

        memory = ""
        if memory_nodes is not None:
            for node in memory_nodes:
                memory += f"# Memory Type\n{node.fact_type}\n\n# Timestamp\n{node.timestamp.isoformat()}\n\n{node.text}\n\n{'-' * 10}\n\n"

        if self.config.model_type == "openai":
            messages = self.to_openai(tool_info, memory=memory)
            # logger.info(f"System instruction:\n{messages[0]['content']}\n")
            response = self.client.chat.completions.create(
                model=self.config.model_name,
                messages=messages,
                tools=self.tool_definitions,
                max_completion_tokens=max_tokens
            )
            self.history.append(response.choices[0].message.to_dict()) # Record in history
            return response
            
        elif self.config.model_type == "anthropic":
            system_prompt, anthropic_msgs = self.to_anthropic(tool_info, memory=memory)
            response = self.client.messages.create(
                model=self.config.model_name,
                max_tokens=max_tokens,
                system=system_prompt,
                messages=anthropic_msgs
            )
            self.history.append({"role": "assistant", "content": response.content}) # Record in history
            return response
            
        elif self.config.model_type == "google":
            system_instruction, contents = self.to_google(tool_info, memory=memory)
            # logger.info(f"System instruction:\n{system_instruction}")
            response = self.client.models.generate_content(
                model=self.config.model_name,
                contents=contents,
                config=genai.types.GenerateContentConfig(
                    system_instruction=system_instruction,
                    max_output_tokens=max_tokens,
                    tools=self.tool_definitions
                )
            )
            self.history.append(response.candidates[0].content) # Record in history
            return response

    async def agenerate_response(self, tool_info, memory_nodes: Optional[List["MemoryNode"]] = None, max_tokens=None):
        return await asyncio.to_thread(self.generate_response, tool_info, memory_nodes, max_tokens)

    # parse_response method is inherited from BaseLanguageModel

    def get_history(self) -> List[Dict]:
        return self.history

    
class MemoryLanguageModel(BaseLanguageModel):
    def __init__(self, config: Optional[LanguageModelConfig] = None):
        super().__init__(config=config)
    
    def extract_entities(self, context: str, max_entities: int, max_tokens: int = 256):
        self.system_instruction = self._entities_system_prompt(max_entities)
        self.add_user_message(context)
        response = self.generate_response(max_tokens=max_tokens)
        return self.parse_response(response)

    async def aextract_entities(self, context: str, max_entities: int, max_tokens: int = 256):
        self.system_instruction = self._entities_system_prompt(max_entities)
        self.add_user_message(context)
        response = await self.agenerate_response(max_tokens=max_tokens)
        return self.parse_response(response)

    @staticmethod
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

    def identify_causes(self, context: str, max_causes: int, max_tokens: int = 256):
        self.system_instruction = self._causes_system_prompt(max_causes)
        self.add_user_message(context)
        response = self.generate_response(max_tokens=max_tokens)
        return self.parse_response(response)

    async def aidentify_causes(self, context: str, max_causes: int, max_tokens: int = 256):
        self.system_instruction = self._causes_system_prompt(max_causes)
        self.add_user_message(context)
        response = await self.agenerate_response(max_tokens=max_tokens)
        return self.parse_response(response)

    @staticmethod
    def _causes_system_prompt(max_causes: int) -> str:
        return (
            f"You are a Causal Logic Engine. Your task is to analyze the causal relationship "
            f"between {max_causes} previous 'Source Memories' and one 'Current Memory'.\n\n"
            "### CAUSALITY SCALE:\n"
            "0: NO RELATION - The memories are independent or share only surface-level topics/entities.\n"
            "1: WEAK/INDIRECT - The Source provides helpful background context but is not necessary for the Current memory.\n"
            "2: STRONG/DIRECT - The Source is a clear precursor or contributor to the events in the Current memory.\n"
            "3: CRITICAL/NECESSARY - The Current memory would not exist or cannot be understood without the Source.\n\n"
            "### CONSTRAINTS:\n"
            "- Ignore 'Entity Matching': Do not assign a level > 0 just because both memories mention the same person or place.\n"
            "- Focus on 'Logical Flow': Does the Source memory explain *why* or *how* the Current memory occurred?\n"
            f"- Output exactly {max_causes} entries in the JSON array.\n\n"
            "### OUTPUT FORMAT:\n"
            "Return ONLY valid JSON in this shape:\n"
            '{"causalities": [{"memory_id": integer, "level": integer}]}'
        )

    def neural_rerank(self, query: str, num_memories: int, max_tokens: int = 256):
        self.system_instruction = self._rerank_system_prompt(num_memories)
        self.add_user_message(query)
        response = self.generate_response(max_tokens=max_tokens)
        return self.parse_response(response)

    async def aneural_rerank(self, query: str, num_memories: int, max_tokens: int = 256):
        self.system_instruction = self._rerank_system_prompt(num_memories)
        self.add_user_message(query)
        response = await self.agenerate_response(max_tokens=max_tokens)
        return self.parse_response(response)

    @staticmethod
    def _rerank_system_prompt(num_memories: int) -> str:
        return (
            "You are a Semantic Relevance Auditor. Your task is to rank a set of memory nodes "
            f"based on their utility in answering the User's next query.\n\n"
            "### RANKING CRITERIA:\n"
            "1. DIRECT ANSWER: Does the memory contain the specific information requested?\n"
            "2. CONTEXTUAL SUPPORT: Does the memory provide necessary background or 'why' for the query?\n"
            "3. TEMPORAL RELEVANCE: If the query implies a sequence, is this memory a logical part of that timeline?\n"
            "4. NOISE REDUCTION: If a memory is unrelated or only shares generic keywords, rank it lowest.\n\n"
            "### CONSTRAINTS:\n"
            f"- You must rank exactly {num_memories} memory nodes.\n"
            f"- Assign a unique integer 'rank' from 1 to {num_memories}, where 1 is the MOST relevant and "
            f"{num_memories} is the LEAST relevant.\n"
            "- Do not allow ties; every memory must have a distinct rank.\n\n"
            "### OUTPUT FORMAT:\n"
            "Return ONLY valid JSON with this exact structure:\n"
            '{"ranks": [{"memory_id": integer, "rank": integer}]}'
        )


# Stateless single-time LLM request
def llm_side_request(query: str, system_instruction: str = "", config: Optional[LanguageModelConfig] = None):
    config = config or LanguageModelConfig()
    llm = BaseLanguageModel(config)
    llm.update_system_instruction(system_instruction)
    llm.add_user_message(query)
    response = llm.generate_response(max_tokens=config.max_tokens)
    result = llm.parse_response(response)
    return result


async def llm_side_request_async(query: str, system_instruction: str = "", config: Optional[LanguageModelConfig] = None):
    config = config or LanguageModelConfig()
    llm = BaseLanguageModel(config)
    llm.update_system_instruction(system_instruction)
    llm.add_user_message(query)
    response = await llm.agenerate_response(max_tokens=config.max_tokens)
    result = llm.parse_response(response)
    return result


