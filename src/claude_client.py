"""Thin wrapper around the Anthropic Python SDK for tool_use conversations."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any, Callable

import anthropic

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "claude-opus-4-6"


@dataclass
class ChatResult:
    """Result from a Claude chat() call, including intermediate tool messages."""
    text: str
    tool_messages: list[dict[str, Any]] = field(default_factory=list)


def _serialize_block(block: Any) -> dict[str, Any]:
    """Serialize a Claude SDK ContentBlock to a plain dict for persistence."""
    if block.type == "text":
        return {"type": "text", "text": block.text}
    elif block.type == "tool_use":
        return {"type": "tool_use", "id": block.id, "name": block.name, "input": block.input}
    if hasattr(block, "model_dump"):
        return block.model_dump()
    return {"type": block.type}


class ClaudeClient:
    """Manages Claude API calls with tool_use support."""

    def __init__(self, model: str = DEFAULT_MODEL):
        self.model = model
        self.client = anthropic.AsyncAnthropic(
            api_key=os.environ.get("ANTHROPIC_API_KEY", ""),
        )

    async def chat(
        self,
        *,
        system: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        tool_handler: Callable[[str, dict[str, Any]], Any],
    ) -> ChatResult:
        """Send a conversation to Claude and process any tool calls.

        Args:
            system: System prompt.
            messages: Conversation messages in Claude API format.
            tools: Tool definitions in Claude API format.
            tool_handler: Callable(tool_name, tool_input) -> result dict.
                Called for each tool_use block. Should return a JSON-serializable result.

        Returns:
            ChatResult with final text and intermediate tool messages for history.
        """
        working_messages = list(messages)
        new_messages: list[dict[str, Any]] = []

        while True:
            response = await self.client.messages.create(
                model=self.model,
                max_tokens=1024,
                system=system,
                messages=working_messages,
                tools=tools,
            )

            # Collect text and tool_use blocks
            text_parts: list[str] = []
            tool_uses: list[dict[str, Any]] = []

            for block in response.content:
                if block.type == "text":
                    text_parts.append(block.text)
                elif block.type == "tool_use":
                    tool_uses.append({
                        "id": block.id,
                        "name": block.name,
                        "input": block.input,
                    })

            # If no tool calls, return the text
            if not tool_uses:
                return ChatResult(text="\n".join(text_parts), tool_messages=new_messages)

            # Serialize response content blocks to plain dicts for persistence
            serialized_content = [_serialize_block(b) for b in response.content]

            # Append the full assistant response (with tool_use blocks)
            working_messages.append({"role": "assistant", "content": response.content})
            new_messages.append({"role": "assistant", "content": serialized_content})

            # Execute each tool and build tool_result blocks
            tool_results = []
            for tool_use in tool_uses:
                try:
                    result = await tool_handler(tool_use["name"], tool_use["input"])
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": tool_use["id"],
                        "content": str(result),
                    })
                except Exception as e:
                    logger.exception("Tool %s failed", tool_use["name"])
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": tool_use["id"],
                        "content": f"Error: {e}",
                        "is_error": True,
                    })

            # Feed tool results back
            working_messages.append({"role": "user", "content": tool_results})
            new_messages.append({"role": "user", "content": tool_results})
