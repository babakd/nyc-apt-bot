"""Tests for ClaudeClient.chat() orchestrator loop.

Tests the tool_use loop directly by mocking anthropic.AsyncAnthropic.
These verify text accumulation, tool execution, error handling, iteration
limits, and message serialization — NOT tool logic (that's in test_conversation.py).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.claude_client import ChatResult, ClaudeClient, _serialize_block


# ---------------------------------------------------------------------------
# Helpers to build mock Claude API responses
# ---------------------------------------------------------------------------

@dataclass
class FakeTextBlock:
    type: str = "text"
    text: str = ""


@dataclass
class FakeToolUseBlock:
    type: str = "tool_use"
    id: str = "tool_1"
    name: str = "some_tool"
    input: dict = field(default_factory=dict)


@dataclass
class FakeResponse:
    """Mimics anthropic.types.Message with a content list."""
    content: list[Any] = field(default_factory=list)
    stop_reason: str = "end_turn"


def _text(text: str) -> FakeTextBlock:
    return FakeTextBlock(text=text)


def _tool(name: str, input_data: dict | None = None, tool_id: str = "tool_1") -> FakeToolUseBlock:
    return FakeToolUseBlock(id=tool_id, name=name, input=input_data or {})


def _response(*blocks: Any) -> FakeResponse:
    return FakeResponse(content=list(blocks))


def _make_client(responses: list[FakeResponse]) -> ClaudeClient:
    """Create a ClaudeClient with a mock that returns responses in order."""
    client = ClaudeClient.__new__(ClaudeClient)
    client.model = "test-model"
    client.client = MagicMock()
    client.client.messages = MagicMock()
    client.client.messages.create = AsyncMock(side_effect=responses)
    return client


async def _noop_handler(name: str, input_data: dict) -> str:
    return f"ok:{name}"


CHAT_DEFAULTS = dict(
    system="test system",
    messages=[{"role": "user", "content": "hello"}],
    tools=[],
)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestTextOnlyNoTools:
    """#1 — Text only response, no tool calls."""

    @pytest.mark.asyncio
    async def test_returns_text_immediately(self):
        client = _make_client([_response(_text("Hello there!"))])
        result = await client.chat(**CHAT_DEFAULTS, tool_handler=_noop_handler)
        assert result.text == "Hello there!"
        assert result.tool_messages == []

    @pytest.mark.asyncio
    async def test_multi_text_blocks_joined(self):
        """Multiple text blocks in a single response are joined."""
        client = _make_client([
            _response(_text("Part one."), _text("Part two.")),
        ])
        result = await client.chat(**CHAT_DEFAULTS, tool_handler=_noop_handler)
        assert "Part one." in result.text
        assert "Part two." in result.text


class TestTextPlusToolUse:
    """#2 — THE BUG: text + tool_use in the same response."""

    @pytest.mark.asyncio
    async def test_text_plus_tool_use_preserves_text(self):
        """Text emitted alongside tool_use must NOT be dropped."""
        client = _make_client([
            # Iteration 1: text + tool_use
            _response(
                _text("Here are your prefs"),
                _tool("update_preferences", {"move_in_date": "April 2026"}),
            ),
            # Iteration 2: final text
            _response(_text("Move-in date updated!")),
        ])
        result = await client.chat(**CHAT_DEFAULTS, tool_handler=_noop_handler)

        # Both text segments must be present
        assert "Here are your prefs" in result.text
        assert "Move-in date updated!" in result.text

    @pytest.mark.asyncio
    async def test_text_order_preserved(self):
        """Text from earlier iterations appears before later text."""
        client = _make_client([
            _response(_text("First"), _tool("t1", tool_id="t1")),
            _response(_text("Second")),
        ])
        result = await client.chat(**CHAT_DEFAULTS, tool_handler=_noop_handler)
        assert result.text.index("First") < result.text.index("Second")


class TestToolUseOnlyThenText:
    """#3 — tool_use only (no text in that response), then text."""

    @pytest.mark.asyncio
    async def test_tool_only_then_text(self):
        client = _make_client([
            _response(_tool("search_apartments")),
            _response(_text("Found 5 listings!")),
        ])
        result = await client.chat(**CHAT_DEFAULTS, tool_handler=_noop_handler)
        assert result.text == "Found 5 listings!"
        assert len(result.tool_messages) == 2  # assistant + user tool_result


class TestMultiIterationMixed:
    """#4 — tool_use only, then text+tool_use, then text."""

    @pytest.mark.asyncio
    async def test_three_iterations(self):
        client = _make_client([
            # Iter 1: tool only
            _response(_tool("mark_ready", tool_id="t1")),
            # Iter 2: text + tool
            _response(_text("Preferences confirmed!"), _tool("search_apartments", tool_id="t2")),
            # Iter 3: final text
            _response(_text("Here are your results.")),
        ])
        result = await client.chat(**CHAT_DEFAULTS, tool_handler=_noop_handler)
        assert "Preferences confirmed!" in result.text
        assert "Here are your results." in result.text
        # 4 tool messages: assistant+result for iter 1, assistant+result for iter 2
        assert len(result.tool_messages) == 4


class TestParallelToolCalls:
    """#5 — Multiple tool_use blocks in a single response."""

    @pytest.mark.asyncio
    async def test_parallel_tools_all_executed(self):
        calls = []

        async def tracking_handler(name: str, input_data: dict) -> str:
            calls.append(name)
            return f"done:{name}"

        client = _make_client([
            _response(
                _tool("update_preferences", {"budget_max": 4000}, tool_id="t1"),
                _tool("mark_ready", tool_id="t2"),
            ),
            _response(_text("All set!")),
        ])
        result = await client.chat(**CHAT_DEFAULTS, tool_handler=tracking_handler)
        assert set(calls) == {"update_preferences", "mark_ready"}
        assert result.text == "All set!"

    @pytest.mark.asyncio
    async def test_parallel_tools_results_fed_back(self):
        """Both tool results should be in the user message fed back to Claude."""
        client = _make_client([
            _response(
                _tool("t1", tool_id="id1"),
                _tool("t2", tool_id="id2"),
            ),
            _response(_text("Done")),
        ])
        result = await client.chat(**CHAT_DEFAULTS, tool_handler=_noop_handler)
        # The tool result message should contain results for both tools
        tool_result_msg = result.tool_messages[1]  # user message with tool_results
        assert tool_result_msg["role"] == "user"
        result_ids = {r["tool_use_id"] for r in tool_result_msg["content"]}
        assert result_ids == {"id1", "id2"}


class TestToolException:
    """#6 — Tool handler raises an exception."""

    @pytest.mark.asyncio
    async def test_tool_error_sets_is_error(self):
        async def failing_handler(name: str, input_data: dict) -> str:
            raise ValueError("Database connection failed")

        client = _make_client([
            _response(_tool("broken_tool")),
            _response(_text("Sorry, that didn't work.")),
        ])
        result = await client.chat(**CHAT_DEFAULTS, tool_handler=failing_handler)

        # Loop should continue — Claude sees the error and responds
        assert "Sorry" in result.text

        # The tool result should have is_error: True
        tool_result_msg = result.tool_messages[1]
        error_result = tool_result_msg["content"][0]
        assert error_result["is_error"] is True
        assert "Database connection failed" in error_result["content"]


class TestMultiToolChain:
    """#7 — Multiple sequential tool calls: tool → tool → tool → text."""

    @pytest.mark.asyncio
    async def test_three_sequential_tools(self):
        calls = []

        async def tracking_handler(name: str, input_data: dict) -> str:
            calls.append(name)
            return f"done:{name}"

        client = _make_client([
            _response(_tool("step1", tool_id="s1")),
            _response(_tool("step2", tool_id="s2")),
            _response(_tool("step3", tool_id="s3")),
            _response(_text("All three steps complete.")),
        ])
        result = await client.chat(**CHAT_DEFAULTS, tool_handler=tracking_handler)
        assert calls == ["step1", "step2", "step3"]
        assert result.text == "All three steps complete."
        # 6 tool messages: 3 × (assistant + user)
        assert len(result.tool_messages) == 6


class TestTextEveryIteration:
    """#8 — Text in every iteration of a multi-tool chain."""

    @pytest.mark.asyncio
    async def test_all_text_segments_accumulated(self):
        client = _make_client([
            _response(_text("Starting step 1..."), _tool("s1", tool_id="s1")),
            _response(_text("Step 1 done, now step 2..."), _tool("s2", tool_id="s2")),
            _response(_text("All done!")),
        ])
        result = await client.chat(**CHAT_DEFAULTS, tool_handler=_noop_handler)
        assert "Starting step 1..." in result.text
        assert "Step 1 done, now step 2..." in result.text
        assert "All done!" in result.text

    @pytest.mark.asyncio
    async def test_text_joined_with_separator(self):
        """Text segments from different iterations are separated by double newline."""
        client = _make_client([
            _response(_text("Segment A"), _tool("s1", tool_id="s1")),
            _response(_text("Segment B")),
        ])
        result = await client.chat(**CHAT_DEFAULTS, tool_handler=_noop_handler)
        assert result.text == "Segment A\n\nSegment B"


class TestEmptyTextBlock:
    """#9 — Empty text block alongside tool_use."""

    @pytest.mark.asyncio
    async def test_empty_text_not_included(self):
        client = _make_client([
            _response(_text(""), _tool("some_tool")),
            _response(_text("Final answer")),
        ])
        result = await client.chat(**CHAT_DEFAULTS, tool_handler=_noop_handler)
        # Empty string should not appear (no leading separator)
        assert result.text == "Final answer"
        assert not result.text.startswith("\n")


class TestToolHandlerNonString:
    """#10 — Tool handler returns a non-string value."""

    @pytest.mark.asyncio
    async def test_dict_return_converted_to_str(self):
        async def dict_handler(name: str, input_data: dict) -> dict:
            return {"status": "ok", "count": 42}

        client = _make_client([
            _response(_tool("data_tool")),
            _response(_text("Got the data")),
        ])
        result = await client.chat(**CHAT_DEFAULTS, tool_handler=dict_handler)
        assert result.text == "Got the data"
        # Verify the tool result was str()-ified
        tool_result_msg = result.tool_messages[1]
        content = tool_result_msg["content"][0]["content"]
        assert "status" in content
        assert "42" in content

    @pytest.mark.asyncio
    async def test_int_return_converted_to_str(self):
        async def int_handler(name: str, input_data: dict) -> int:
            return 42

        client = _make_client([
            _response(_tool("count_tool")),
            _response(_text("Count is 42")),
        ])
        result = await client.chat(**CHAT_DEFAULTS, tool_handler=int_handler)
        tool_result_msg = result.tool_messages[1]
        assert tool_result_msg["content"][0]["content"] == "42"


class TestMaxIterations:
    """#11 — Max iterations exceeded."""

    @pytest.mark.asyncio
    async def test_max_iterations_terminates_loop(self, caplog):
        """Loop must terminate after MAX_TOOL_ITERATIONS, not run forever."""
        from src.claude_client import MAX_TOOL_ITERATIONS

        # Create more responses than MAX_TOOL_ITERATIONS — all with tool calls
        responses = [
            _response(_text(f"Iter {i}"), _tool(f"tool_{i}", tool_id=f"t{i}"))
            for i in range(MAX_TOOL_ITERATIONS + 5)
        ]
        client = _make_client(responses)

        with caplog.at_level(logging.WARNING):
            result = await client.chat(**CHAT_DEFAULTS, tool_handler=_noop_handler)

        # Should have called the API exactly MAX_TOOL_ITERATIONS times
        assert client.client.messages.create.call_count == MAX_TOOL_ITERATIONS

        # Should log a warning
        assert any("max iterations" in r.message.lower() for r in caplog.records)

        # Should still return accumulated text
        assert "Iter 0" in result.text

    @pytest.mark.asyncio
    async def test_early_exit_doesnt_warn(self, caplog):
        """Normal completion (text-only response) should not log max iterations warning."""
        client = _make_client([
            _response(_tool("t1", tool_id="t1")),
            _response(_text("Done")),
        ])
        with caplog.at_level(logging.WARNING):
            await client.chat(**CHAT_DEFAULTS, tool_handler=_noop_handler)
        assert not any("max iterations" in r.message.lower() for r in caplog.records)


class TestResponseSerialization:
    """#12 — new_messages contains properly serialized tool_use and tool_result blocks."""

    @pytest.mark.asyncio
    async def test_tool_messages_serialized_as_dicts(self):
        client = _make_client([
            _response(_tool("update_preferences", {"budget_max": 5000}, tool_id="tu_123")),
            _response(_text("Budget updated!")),
        ])
        result = await client.chat(**CHAT_DEFAULTS, tool_handler=_noop_handler)

        # Assistant message should have serialized tool_use
        assistant_msg = result.tool_messages[0]
        assert assistant_msg["role"] == "assistant"
        tool_block = assistant_msg["content"][0]
        assert tool_block["type"] == "tool_use"
        assert tool_block["id"] == "tu_123"
        assert tool_block["name"] == "update_preferences"
        assert tool_block["input"] == {"budget_max": 5000}

        # User message should have tool_result
        user_msg = result.tool_messages[1]
        assert user_msg["role"] == "user"
        result_block = user_msg["content"][0]
        assert result_block["type"] == "tool_result"
        assert result_block["tool_use_id"] == "tu_123"


class TestWorkingVsNewMessages:
    """#13 — working_messages uses SDK objects, new_messages uses plain dicts."""

    @pytest.mark.asyncio
    async def test_working_messages_use_sdk_objects(self):
        """working_messages should pass SDK content objects to the API."""
        client = _make_client([
            _response(_tool("t1", tool_id="t1")),
            _response(_text("Done")),
        ])
        await client.chat(**CHAT_DEFAULTS, tool_handler=_noop_handler)

        # The second API call should have the original content objects (not dicts)
        second_call_messages = client.client.messages.create.call_args_list[1]
        messages_arg = second_call_messages.kwargs["messages"]
        # The assistant message should have the raw content (FakeResponse objects)
        assistant_msg = messages_arg[1]  # index 0 is original user msg, 1 is assistant
        assert assistant_msg["role"] == "assistant"
        # Content should be the raw list from FakeResponse, not serialized dicts
        assert isinstance(assistant_msg["content"][0], FakeToolUseBlock)


class TestNoToolsDefined:
    """#14 — No tools defined, text-only response."""

    @pytest.mark.asyncio
    async def test_no_tools_returns_text_immediately(self):
        client = _make_client([_response(_text("Just a plain response"))])
        result = await client.chat(
            system="test",
            messages=[{"role": "user", "content": "hi"}],
            tools=[],
            tool_handler=_noop_handler,
        )
        assert result.text == "Just a plain response"
        assert result.tool_messages == []
        assert client.client.messages.create.call_count == 1


class TestSerializeBlock:
    """Test the _serialize_block helper."""

    def test_text_block(self):
        block = FakeTextBlock(text="hello")
        assert _serialize_block(block) == {"type": "text", "text": "hello"}

    def test_tool_use_block(self):
        block = FakeToolUseBlock(id="abc", name="my_tool", input={"key": "val"})
        result = _serialize_block(block)
        assert result == {
            "type": "tool_use",
            "id": "abc",
            "name": "my_tool",
            "input": {"key": "val"},
        }

    def test_unknown_block_with_model_dump(self):
        """Blocks with model_dump() (Pydantic) should use that."""
        block = MagicMock()
        block.type = "custom"
        block.model_dump.return_value = {"type": "custom", "data": 123}
        assert _serialize_block(block) == {"type": "custom", "data": 123}

    def test_unknown_block_fallback(self):
        """Blocks without model_dump fall back to {type: ...}."""
        block = FakeTextBlock.__new__(FakeTextBlock)
        block.type = "unknown"
        # FakeTextBlock has no model_dump, and type != "text" if we override
        # Use a plain object
        class Bare:
            type = "mystery"
        assert _serialize_block(Bare()) == {"type": "mystery"}
