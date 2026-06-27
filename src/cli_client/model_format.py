"""
Provider-specific message formatting and response parsing for language models.

Use get_model_format(model_type) to obtain the adapter for the client's chosen provider.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable

from google import genai


@dataclass
class FormatContext:
    """Inputs shared by all providers when building a request payload."""

    system_instruction: str
    history: List[Any]
    tool_info: Optional[Dict[str, str]] = None
    memory: str = ""
    history_window: Optional[int] = None

    def sliced_history(self) -> List[Any]:
        if self.history_window is not None:
            return self.history[-self.history_window :]
        return self.history


def _append_tool_info(system: str, tool_info: Optional[Dict[str, str]]) -> str:
    if tool_info is None:
        return system
    if tool_info.get("tool_summaries", "") == "":
        return system + "No tools currently available.\n"
    system += f"Available tools summaries:\n{tool_info['tool_summaries']}\n"
    if tool_info.get("tool_manuals", "") != "":
        system += f"Active tools full manuals:\n{tool_info['tool_manuals']}\n"
    return system


def _openai_tool_messages(tool_info: Optional[Dict[str, str]]) -> List[Dict[str, str]]:
    if tool_info is None:
        return []
    if tool_info.get("tool_summaries", "") == "":
        return [{"role": "system", "content": "No tools currently available."}]
    msgs = [
        {
            "role": "system",
            "content": f"Available tools summaries:\n{tool_info['tool_summaries']}",
        }
    ]
    if tool_info.get("tool_manuals", "") != "":
        msgs.append(
            {
                "role": "system",
                "content": f"Active tools full manuals:\n{tool_info['tool_manuals']}",
            }
        )
    return msgs


@runtime_checkable
class ModelFormatAdapter(Protocol):
    def format_user_message(self, content: str) -> Dict[str, Any]:
        ...

    def format_tool_response(self, content: str, command: str, tool_call_id: str) -> Dict[str, Any]:
        ...

    def to_messages(self, ctx: FormatContext) -> Any:
        ...

    def from_response(self, response: Any) -> Dict[str, Any]:
        ...

    def generate(
        self,
        client: Any,
        model_name: str,
        ctx: FormatContext,
        max_tokens: int,
        *,
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> Any:
        ...

    def record_assistant_turn(self, history: List[Any], response: Any) -> None:
        ...


class OpenAIFormatAdapter:
    def format_user_message(self, content: str) -> Dict[str, Any]:
        return {"role": "user", "content": content}

    def format_tool_response(self, content: str, command: str, tool_call_id: str) -> Dict[str, Any]:
        return {"role": "tool", "content": content, "tool_call_id": tool_call_id}

    def to_messages(self, ctx: FormatContext) -> List[Dict[str, Any]]:
        messages: List[Dict[str, Any]] = [{"role": "system", "content": ctx.system_instruction}]
        messages.extend(ctx.sliced_history())
        if ctx.memory:
            messages.append(
                {"role": "system", "content": f"Relevant memory for past sessions:\n{ctx.memory}"}
            )
        messages.extend(_openai_tool_messages(ctx.tool_info))
        return messages

    def from_response(self, response: Any) -> Dict[str, Any]:
        msg = response.choices[0].message
        result: Dict[str, Any] = {"content": msg.content or "", "tool_calls": []}
        if msg.tool_calls:
            for tc in msg.tool_calls:
                result["tool_calls"].append(
                    {
                        "id": tc.id,
                        "command": tc.function.name,
                        "arguments": json.loads(tc.function.arguments),
                    }
                )
        return result

    def generate(
        self,
        client: Any,
        model_name: str,
        ctx: FormatContext,
        max_tokens: int,
        *,
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> Any:
        kwargs: Dict[str, Any] = {
            "model": model_name,
            "messages": self.to_messages(ctx),
            "max_completion_tokens": max_tokens,
        }
        if tools is not None:
            kwargs["tools"] = tools
        return client.chat.completions.create(**kwargs)

    def record_assistant_turn(self, history: List[Any], response: Any) -> None:
        history.append(response.choices[0].message.to_dict())


class AnthropicFormatAdapter:
    def format_user_message(self, content: str) -> Dict[str, Any]:
        return {"role": "user", "content": [{"type": "text", "text": content}]}

    def format_tool_response(self, content: str, command: str, tool_call_id: str) -> Dict[str, Any]:
        return {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": tool_call_id,
                    "content": content,
                }
            ],
        }

    def to_messages(self, ctx: FormatContext) -> tuple[str, List[Dict[str, Any]]]:
        system_prompt = ctx.system_instruction + "\n"
        if ctx.memory:
            system_prompt += f"\nRelevant memory for past sessions:\n{ctx.memory}\n"
        system_prompt = _append_tool_info(system_prompt, ctx.tool_info)
        return system_prompt, list(ctx.sliced_history())

    def from_response(self, response: Any) -> Dict[str, Any]:
        result: Dict[str, Any] = {"content": "", "tool_calls": []}
        for block in response.content:
            if block.type == "text":
                result["content"] += block.text
            elif block.type == "tool_use":
                result["tool_calls"].append(
                    {"id": block.id, "command": block.name, "arguments": block.input}
                )
        return result

    def generate(
        self,
        client: Any,
        model_name: str,
        ctx: FormatContext,
        max_tokens: int,
        *,
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> Any:
        system_prompt, messages = self.to_messages(ctx)
        return client.messages.create(
            model=model_name,
            max_tokens=max_tokens,
            system=system_prompt,
            messages=messages,
        )

    def record_assistant_turn(self, history: List[Any], response: Any) -> None:
        history.append({"role": "assistant", "content": response.content})


class GoogleFormatAdapter:
    def format_user_message(self, content: str) -> Dict[str, Any]:
        return {"role": "user", "parts": [{"text": content}]}

    def format_tool_response(self, content: str, command: str, tool_call_id: str) -> Dict[str, Any]:
        return {
            "role": "user",
            "parts": [
                {
                    "function_response": {
                        "name": command,
                        "response": {"result": content},
                    }
                }
            ],
        }

    def to_messages(self, ctx: FormatContext) -> tuple[str, List[Any]]:
        system_instruction = ctx.system_instruction + "\n"
        if ctx.memory:
            system_instruction += f"\nRelevant memory for past sessions:\n{ctx.memory}\n"
        system_instruction = _append_tool_info(system_instruction, ctx.tool_info)
        return system_instruction, list(ctx.sliced_history())

    def from_response(self, response: Any) -> Dict[str, Any]:
        candidate = response.candidates[0]
        result: Dict[str, Any] = {"content": "", "tool_calls": []}
        for part in candidate.content.parts:
            if part.text:
                result["content"] += part.text
            if part.function_call:
                fc = part.function_call
                result["tool_calls"].append(
                    {
                        "id": getattr(fc, "id", fc.name),
                        "command": fc.name,
                        "arguments": dict(fc.args),
                    }
                )
        return result

    def generate(
        self,
        client: Any,
        model_name: str,
        ctx: FormatContext,
        max_tokens: int,
        *,
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> Any:
        system_instruction, contents = self.to_messages(ctx)
        config_kwargs: Dict[str, Any] = {
            "system_instruction": system_instruction,
            "max_output_tokens": max_tokens,
        }
        if tools is not None:
            config_kwargs["tools"] = tools
        return client.models.generate_content(
            model=model_name,
            contents=contents,
            config=genai.types.GenerateContentConfig(**config_kwargs),
        )

    def record_assistant_turn(self, history: List[Any], response: Any) -> None:
        history.append(response.candidates[0].content)



def get_model_format(model_type: str) -> ModelFormatAdapter:
    if model_type == "openai":
        return OpenAIFormatAdapter()
    elif model_type == "anthropic":
        return AnthropicFormatAdapter()
    elif model_type == "google":
        return GoogleFormatAdapter()
    else:
        raise ValueError(f"Unsupported model type: {model_type}")