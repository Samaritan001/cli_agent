import os
from dotenv import load_dotenv
import asyncio

from typing import Dict, List, Optional, TYPE_CHECKING, Union
import json
from dataclasses import dataclass
import time
from pathlib import Path

from cli_client.logging_config import get_logger

logger = get_logger("model")

from openai import OpenAI          # v2.x+ (2026)
from anthropic import Anthropic    # v1.x+ (2026)
from google import genai           # New Google GenAI SDK (2026)

from cli_client.model_format import FormatContext, get_model_format

if TYPE_CHECKING:
    from cli_client.memory_config import MemoryNode


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

_PACKAGE_DIR = Path(__file__).resolve().parent


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

        self._history: List[Dict] = []
        self._message_ids: List[int] = []  # Tracks IDs parallel to history
        self.session_prefix = int(time.time() * 1000)  # Unique session ID
        self.msg_seq = 0  # Sequential message counter, first id is session_prefix << 16 + 1

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

        self._client = CLIENT_TYPES.get(self.config.model_type, OpenAI)(api_key=self.api_key)
        self._format = get_model_format(self.config.model_type)

    def get_history(self, window: Optional[int] = None) -> List[Dict]:
        """Return a copy of conversation history, optionally truncated to the last ``window`` turns."""
        if window is not None:
            return list(self._history[-window:])
        return list(self._history)

    def set_history(self, history: List[Dict]) -> None:
        """Replace conversation history (used by stateless side requests)."""
        self._history = list(history)

    def get_latest_message_id(self) -> Optional[int]:
        """Return the most recently assigned message id, or ``None`` if no messages yet."""
        if not self._message_ids:
            return None
        return self._message_ids[-1]

    def update_system_instruction(self, instruction: str):
        """Allows dynamic updating of system instructions."""
        self.system_instruction = instruction

    def _generate_msg_id(self) -> int:
        self.msg_seq += 1
        return (self.session_prefix << 16) + self.msg_seq

    def _build_format_context(
        self,
        tool_info=None,
        memory: str = "",
        history_window: Optional[int] = None,
    ) -> FormatContext:
        return FormatContext(
            system_instruction=self.system_instruction,
            history=self._history,
            tool_info=tool_info,
            memory=memory,
            history_window=history_window,
        )

    def to_messages(
        self,
        tool_info=None,
        memory: str = "",
        history_window: Optional[int] = None,
    ):
        """Build provider-specific request payload via the model-format factory."""
        return self._format.to_messages(
            self._build_format_context(tool_info, memory, history_window)
        )

    def from_response(self, response):
        """Parse a provider response into the normalized dict format."""
        return self._format.from_response(response)

    def add_user_message(self, content: str):
        """Adds a standard message to the history."""
        message_id = self._generate_msg_id()
        self._message_ids.append(message_id)
        self._history.append(self._format.format_user_message(content))
        return message_id

    def generate_response(self, max_tokens: Optional[int] = None):
        """Sends messages to the model via the respective API client."""
        if max_tokens is None:
            max_tokens = self.config.max_tokens

        ctx = self._build_format_context()
        response = self._format.generate(
            self._client,
            self.config.model_name,
            ctx,
            max_tokens,
        )
        self._format.record_assistant_turn(self._history, response)
        return response

    async def agenerate_response(self, max_tokens: Optional[int] = None):
        return await asyncio.to_thread(self.generate_response, max_tokens=max_tokens)

    def parse_response(self, response):
        """
        Parses response into a standard dict including tool_calls and IDs.
        Format: {'content': '...', 'tool_calls': [Dict[id, name, arguments]]}
        """
        result = {"content": "", "tool_calls": []}
        try:
            result = self.from_response(response)
        except Exception as e:
            result["content"] = f"Parsing Error: {str(e)}"
        return result


class ClientLanguageModel(BaseLanguageModel):
    def __init__(
        self,
        config: Optional[LanguageModelConfig] = None,
        *,
        tool_definitions_path: Optional[Union[str, Path]] = None,
        system_instruction_path: Optional[Union[str, Path]] = None,
        tool_definitions: Optional[List[Dict]] = None,
        system_instruction: Optional[str] = None,
    ):
        super().__init__(config=config)
        self._tool_definitions_path = (
            Path(tool_definitions_path) if tool_definitions_path is not None else None
        )
        self._system_instruction_path = (
            Path(system_instruction_path) if system_instruction_path is not None else None
        )
        self._tool_definitions = tool_definitions
        self._assets_loaded = tool_definitions is not None and system_instruction is not None
        if system_instruction is not None:
            self.system_instruction = system_instruction

    def _ensure_assets_loaded(self) -> None:
        if self._assets_loaded:
            return
        tool_path = self._tool_definitions_path or (
            _PACKAGE_DIR / f"tool_definitions_{self.config.model_type}.json"
        )
        system_path = self._system_instruction_path or (_PACKAGE_DIR / "system_instruction.md")
        with open(tool_path, "r", encoding="utf-8") as f:
            self._tool_definitions = json.load(f)
        with open(system_path, "r", encoding="utf-8") as f:
            self.system_instruction = f.read()
        self._assets_loaded = True

    @property
    def tool_definitions(self) -> List[Dict]:
        self._ensure_assets_loaded()
        return self._tool_definitions

    def add_tool_response(self, content: str, command: str, tool_call_id: str):
        """Adds a tool result message to the history."""
        message_id = self._generate_msg_id()
        self._message_ids.append(message_id)
        self._history.append(self._format.format_tool_response(content, command, tool_call_id))
        return message_id

    def generate_response(
        self,
        tool_info,
        *,
        max_tokens: Optional[int] = None,
        memory_nodes: Optional[List["MemoryNode"]] = None,
        history_window: Optional[int] = None,
    ):
        """Sends messages to the model with tools, memory, and truncated history."""
        if max_tokens is None:
            max_tokens = self.config.max_tokens

        memory = ""
        if memory_nodes is not None:
            for node in memory_nodes:
                memory += (
                    f"# Memory Type\n{node.fact_type}\n\n"
                    f"# Timestamp\n{node.timestamp.isoformat()}\n\n"
                    f"{node.text}\n\n{'-' * 10}\n\n"
                )

        ctx = self._build_format_context(tool_info, memory, history_window)
        response = self._format.generate(
            self._client,
            self.config.model_name,
            ctx,
            max_tokens,
            tools=self.tool_definitions,
        )
        self._format.record_assistant_turn(self._history, response)
        return response

    async def agenerate_response(
        self,
        tool_info,
        *,
        max_tokens: Optional[int] = None,
        memory_nodes: Optional[List["MemoryNode"]] = None,
        history_window: Optional[int] = None,
    ):
        return await asyncio.to_thread(
            self.generate_response,
            tool_info,
            max_tokens=max_tokens,
            memory_nodes=memory_nodes,
            history_window=history_window,
        )

# Stateless single-time LLM request
def llm_side_request(query: str, system_instruction: str = "", config: Optional[LanguageModelConfig] = None, history: Optional[List[Dict]] = None):
    config = config or LanguageModelConfig()
    llm = BaseLanguageModel(config)
    llm.update_system_instruction(system_instruction)
    llm.add_user_message(query)

    if history is not None:
        llm.set_history(history)

    response = llm.generate_response(max_tokens=config.max_tokens)
    result = llm.parse_response(response)
    return result


async def llm_side_request_async(query: str, system_instruction: str = "", config: Optional[LanguageModelConfig] = None, history: Optional[List[Dict]] = None):
    config = config or LanguageModelConfig()
    llm = BaseLanguageModel(config)
    llm.update_system_instruction(system_instruction)
    llm.add_user_message(query)

    if history is not None:
        llm.set_history(history)

    response = await llm.agenerate_response(max_tokens=config.max_tokens)
    result = llm.parse_response(response)
    return result
