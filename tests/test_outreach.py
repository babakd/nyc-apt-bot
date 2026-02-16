"""Tests for outreach draft creation and revision."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from src.claude_client import ChatResult
from src.models import ChatState, Draft, Listing, Preferences
from src.outreach import DEFAULT_TEMPLATE, create_draft, revise_draft


def _make_listing(**kwargs) -> Listing:
    defaults = dict(
        listing_id="100",
        url="https://streeteasy.com/rental/100",
        address="100 Main St #2A",
        neighborhood="Chelsea",
        price=3500,
        bedrooms=2,
        bathrooms=1.0,
    )
    defaults.update(kwargs)
    return Listing(**defaults)


@pytest.fixture
def mock_bot():
    bot = AsyncMock()
    bot.send_text = AsyncMock(return_value={"ok": True})
    return bot


@pytest.fixture
def mock_state():
    state = ChatState(chat_id=123)
    state.preferences = Preferences(budget_max=4000, must_haves=["dishwasher"])
    return state


class TestCreateDraft:
    @pytest.mark.asyncio
    async def test_extracts_text_from_chat_result(self, mock_bot, mock_state):
        """create_draft extracts .text from ChatResult returned by claude.chat()."""
        listing = _make_listing()
        mock_claude = AsyncMock()
        mock_claude.chat = AsyncMock(
            return_value=ChatResult(text="Hi, I'm interested in the apartment.")
        )

        with patch("src.outreach.load_state", return_value=mock_state), \
             patch("src.outreach.save_state"):
            await create_draft(mock_bot, 123, listing, claude=mock_claude)

        # The draft message sent should contain the Claude-generated text
        mock_bot.send_text.assert_called_once()
        sent_text = mock_bot.send_text.call_args[0][1]
        assert "interested in the apartment" in sent_text

    @pytest.mark.asyncio
    async def test_fallback_on_empty_text(self, mock_bot, mock_state):
        """When Claude returns empty text, the default template is used."""
        listing = _make_listing()
        mock_claude = AsyncMock()
        mock_claude.chat = AsyncMock(return_value=ChatResult(text=""))

        with patch("src.outreach.load_state", return_value=mock_state), \
             patch("src.outreach.save_state"):
            await create_draft(mock_bot, 123, listing, claude=mock_claude)

        mock_bot.send_text.assert_called_once()
        sent_text = mock_bot.send_text.call_args[0][1]
        # Should contain the template text
        assert "schedule a viewing" in sent_text

    @pytest.mark.asyncio
    async def test_fallback_on_api_exception(self, mock_bot, mock_state):
        """When Claude API raises, the default template is used."""
        listing = _make_listing()
        mock_claude = AsyncMock()
        mock_claude.chat = AsyncMock(side_effect=Exception("API error"))

        with patch("src.outreach.load_state", return_value=mock_state), \
             patch("src.outreach.save_state"):
            await create_draft(mock_bot, 123, listing, claude=mock_claude)

        mock_bot.send_text.assert_called_once()
        sent_text = mock_bot.send_text.call_args[0][1]
        assert "schedule a viewing" in sent_text


class TestReviseDraft:
    @pytest.mark.asyncio
    async def test_extracts_text_from_chat_result(self, mock_bot):
        """revise_draft extracts .text from ChatResult returned by claude.chat()."""
        state = ChatState(chat_id=123)
        draft = Draft(
            draft_id="d1",
            listing_id="100",
            message_text="Original message",
        )
        state.active_drafts["d1"] = draft
        listing = _make_listing()
        state.recent_listings["100"] = listing

        mock_claude = AsyncMock()
        mock_claude.chat = AsyncMock(
            return_value=ChatResult(text="Revised message with changes.")
        )

        with patch("src.outreach.load_state", return_value=state), \
             patch("src.outreach.save_state"):
            await revise_draft(mock_bot, 123, "d1", "make it shorter", claude=mock_claude)

        mock_bot.send_text.assert_called_once()
        sent_text = mock_bot.send_text.call_args[0][1]
        assert "Revised message with changes" in sent_text

    @pytest.mark.asyncio
    async def test_draft_not_found(self, mock_bot):
        """revise_draft sends error when draft doesn't exist."""
        state = ChatState(chat_id=123)

        with patch("src.outreach.load_state", return_value=state), \
             patch("src.outreach.save_state"):
            await revise_draft(mock_bot, 123, "nonexistent", "feedback")

        mock_bot.send_text.assert_called_once()
        sent_text = mock_bot.send_text.call_args[0][1]
        assert "not found" in sent_text.lower()

    @pytest.mark.asyncio
    async def test_api_failure_sends_error(self, mock_bot):
        """revise_draft sends error message when Claude API fails."""
        state = ChatState(chat_id=123)
        draft = Draft(
            draft_id="d1",
            listing_id="100",
            message_text="Original message",
        )
        state.active_drafts["d1"] = draft

        mock_claude = AsyncMock()
        mock_claude.chat = AsyncMock(side_effect=Exception("API error"))

        with patch("src.outreach.load_state", return_value=state), \
             patch("src.outreach.save_state"):
            await revise_draft(mock_bot, 123, "d1", "feedback", claude=mock_claude)

        mock_bot.send_text.assert_called_once()
        sent_text = mock_bot.send_text.call_args[0][1]
        assert "trouble revising" in sent_text.lower()
