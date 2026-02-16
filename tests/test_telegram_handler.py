"""Tests for Telegram handler group chat support."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from src.models import ChatState, Draft
from src.telegram_handler import TelegramBot


@pytest.fixture
def bot():
    return TelegramBot(token="test-token")


class TestExtractSenderName:
    def test_private_chat_returns_none(self):
        message = {
            "chat": {"id": 123, "type": "private"},
            "from": {"first_name": "Alice", "last_name": "Smith"},
        }
        assert TelegramBot._extract_sender_name(message) is None

    def test_group_chat_full_name(self):
        message = {
            "chat": {"id": -100123, "type": "group"},
            "from": {"first_name": "Alice", "last_name": "Smith"},
        }
        assert TelegramBot._extract_sender_name(message) == "Alice Smith"

    def test_group_chat_first_name_only(self):
        message = {
            "chat": {"id": -100123, "type": "group"},
            "from": {"first_name": "Bob"},
        }
        assert TelegramBot._extract_sender_name(message) == "Bob"

    def test_group_chat_username_fallback(self):
        message = {
            "chat": {"id": -100123, "type": "supergroup"},
            "from": {"username": "alice_nyc"},
        }
        assert TelegramBot._extract_sender_name(message) == "alice_nyc"

    def test_group_chat_someone_fallback(self):
        message = {
            "chat": {"id": -100123, "type": "group"},
            "from": {},
        }
        assert TelegramBot._extract_sender_name(message) == "Someone"

    def test_supergroup_returns_name(self):
        message = {
            "chat": {"id": -100123, "type": "supergroup"},
            "from": {"first_name": "Charlie", "last_name": "D"},
        }
        assert TelegramBot._extract_sender_name(message) == "Charlie D"


class TestIsGroupChat:
    def test_private(self, bot):
        msg = {"chat": {"id": 123, "type": "private"}}
        assert bot._is_group_chat(msg) is False

    def test_group(self, bot):
        msg = {"chat": {"id": -100, "type": "group"}}
        assert bot._is_group_chat(msg) is True

    def test_supergroup(self, bot):
        msg = {"chat": {"id": -100, "type": "supergroup"}}
        assert bot._is_group_chat(msg) is True


class TestGroupMessageFiltering:
    @pytest.mark.asyncio
    async def test_group_message_without_mention_ignored(self, bot):
        """Group messages not directed at the bot are ignored."""
        bot._bot_username = "test_bot"

        message = {
            "chat": {"id": -100, "type": "group"},
            "from": {"first_name": "Alice"},
            "text": "Hey everyone, what's up?",
        }

        with patch("src.telegram_handler.load_state") as mock_load, \
             patch("src.telegram_handler.save_state"):
            await bot._handle_message(message)
            # load_state should NOT be called — message was ignored
            mock_load.assert_not_called()

    @pytest.mark.asyncio
    async def test_group_message_with_mention_processed(self, bot):
        """Group messages with @botname are processed."""
        bot._bot_username = "test_bot"

        message = {
            "chat": {"id": -100, "type": "group"},
            "from": {"first_name": "Alice"},
            "text": "@test_bot looking for 2BR in Chelsea",
        }

        mock_state = AsyncMock()
        mock_state.is_group = False
        mock_state.pending_draft_edit = None

        with patch("src.telegram_handler.load_state") as mock_load, \
             patch("src.telegram_handler.save_state"), \
             patch("src.telegram_handler.ConversationEngine") as mock_engine_cls:
            mock_load.return_value = mock_state
            mock_result = AsyncMock()
            mock_result.responses = []
            mock_result.trigger_search = False
            mock_result.trigger_draft_listing_id = None
            mock_engine_cls.return_value.handle_message = AsyncMock(return_value=mock_result)

            await bot._handle_message(message)

            # Verify handle_message was called with mention stripped and sender_name
            mock_engine_cls.return_value.handle_message.assert_called_once_with(
                "looking for 2BR in Chelsea", sender_name="Alice"
            )

    @pytest.mark.asyncio
    async def test_group_reply_to_bot_processed(self, bot):
        """Replies to the bot's messages in groups are processed."""
        bot._bot_username = "test_bot"

        message = {
            "chat": {"id": -100, "type": "group"},
            "from": {"first_name": "Bob"},
            "text": "under $4000",
            "reply_to_message": {
                "from": {"username": "test_bot"},
            },
        }

        mock_state = AsyncMock()
        mock_state.is_group = True
        mock_state.pending_draft_edit = None

        with patch("src.telegram_handler.load_state") as mock_load, \
             patch("src.telegram_handler.save_state"), \
             patch("src.telegram_handler.ConversationEngine") as mock_engine_cls:
            mock_load.return_value = mock_state
            mock_result = AsyncMock()
            mock_result.responses = []
            mock_result.trigger_search = False
            mock_result.trigger_draft_listing_id = None
            mock_engine_cls.return_value.handle_message = AsyncMock(return_value=mock_result)

            await bot._handle_message(message)

            mock_engine_cls.return_value.handle_message.assert_called_once_with(
                "under $4000", sender_name="Bob"
            )

    @pytest.mark.asyncio
    async def test_private_message_no_sender_name(self, bot):
        """Private chat messages don't pass sender_name."""
        message = {
            "chat": {"id": 123, "type": "private"},
            "from": {"first_name": "Alice"},
            "text": "looking for apartments",
        }

        mock_state = AsyncMock()
        mock_state.is_group = False
        mock_state.pending_draft_edit = None

        with patch("src.telegram_handler.load_state") as mock_load, \
             patch("src.telegram_handler.save_state"), \
             patch("src.telegram_handler.ConversationEngine") as mock_engine_cls:
            mock_load.return_value = mock_state
            mock_result = AsyncMock()
            mock_result.responses = []
            mock_result.trigger_search = False
            mock_result.trigger_draft_listing_id = None
            mock_engine_cls.return_value.handle_message = AsyncMock(return_value=mock_result)

            await bot._handle_message(message)

            mock_engine_cls.return_value.handle_message.assert_called_once_with(
                "looking for apartments", sender_name=None
            )

    @pytest.mark.asyncio
    async def test_mention_only_message_ignored(self, bot):
        """A message that's just '@botname' with no other text is ignored."""
        bot._bot_username = "test_bot"

        message = {
            "chat": {"id": -100, "type": "group"},
            "from": {"first_name": "Alice"},
            "text": "@test_bot",
        }

        with patch("src.telegram_handler.load_state") as mock_load:
            await bot._handle_message(message)
            mock_load.assert_not_called()


class TestBotAddedToGroup:
    @pytest.mark.asyncio
    async def test_welcome_message_on_join(self, bot):
        """Bot sends welcome message and sets is_group when added to group."""
        bot._bot_username = "test_bot"

        message = {
            "chat": {"id": -100, "type": "group"},
            "new_chat_members": [{"username": "test_bot", "id": 999}],
        }

        mock_state = AsyncMock()
        mock_state.is_group = False

        with patch("src.telegram_handler.load_state") as mock_load, \
             patch("src.telegram_handler.save_state") as mock_save:
            mock_load.return_value = mock_state
            bot.send_text = AsyncMock(return_value={"ok": True})

            await bot._handle_message(message)

            assert mock_state.is_group is True
            mock_save.assert_called_once_with(mock_state)
            bot.send_text.assert_called_once()
            welcome_text = bot.send_text.call_args[0][1]
            assert "apartment hunting assistant" in welcome_text
            assert "@test_bot" in welcome_text


class TestDraftEditCallback:
    @pytest.mark.asyncio
    async def test_edit_callback_stores_pending_draft_edit(self, bot):
        """Edit callback stores pending_draft_edit and saves state."""
        state = ChatState(chat_id=123)
        draft = Draft(draft_id="d1", listing_id="100", message_text="Hello")
        state.active_drafts["d1"] = draft

        callback = {
            "id": "cb1",
            "message": {"chat": {"id": 123}, "message_id": 1},
            "data": "draft_edit:d1",
        }

        with patch("src.telegram_handler.load_state", return_value=state), \
             patch("src.telegram_handler.save_state") as mock_save:
            bot.send_text = AsyncMock(return_value={"ok": True})
            bot._answer_callback = AsyncMock()

            await bot._handle_callback_query(callback)

            assert state.pending_draft_edit == "d1"
            mock_save.assert_called_once_with(state)
            bot.send_text.assert_called_once()
            sent_text = bot.send_text.call_args[0][1]
            assert "What would you like to change" in sent_text

    @pytest.mark.asyncio
    async def test_edit_callback_nonexistent_draft(self, bot):
        """Edit callback on non-existent draft shows error."""
        state = ChatState(chat_id=123)

        callback = {
            "id": "cb1",
            "message": {"chat": {"id": 123}, "message_id": 1},
            "data": "draft_edit:nonexistent",
        }

        with patch("src.telegram_handler.load_state", return_value=state), \
             patch("src.telegram_handler.save_state"):
            bot.send_text = AsyncMock(return_value={"ok": True})
            bot._answer_callback = AsyncMock()

            await bot._handle_callback_query(callback)

            assert state.pending_draft_edit is None
            bot.send_text.assert_called_once()
            sent_text = bot.send_text.call_args[0][1]
            assert "no longer available" in sent_text.lower()

    @pytest.mark.asyncio
    async def test_edit_callback_non_pending_draft(self, bot):
        """Edit callback on sent draft shows error."""
        state = ChatState(chat_id=123)
        draft = Draft(draft_id="d1", listing_id="100", message_text="Hello", status="sent")
        state.active_drafts["d1"] = draft

        callback = {
            "id": "cb1",
            "message": {"chat": {"id": 123}, "message_id": 1},
            "data": "draft_edit:d1",
        }

        with patch("src.telegram_handler.load_state", return_value=state), \
             patch("src.telegram_handler.save_state"):
            bot.send_text = AsyncMock(return_value={"ok": True})
            bot._answer_callback = AsyncMock()

            await bot._handle_callback_query(callback)

            assert state.pending_draft_edit is None
            bot.send_text.assert_called_once()
            sent_text = bot.send_text.call_args[0][1]
            assert "no longer available" in sent_text.lower()


class TestDraftEditMessageRouting:
    @pytest.mark.asyncio
    async def test_next_message_routes_to_revise_draft(self, bot):
        """Next message after Edit callback routes to revise_draft and clears field."""
        state = ChatState(chat_id=123)
        state.pending_draft_edit = "d1"

        message = {
            "chat": {"id": 123, "type": "private"},
            "from": {"first_name": "Alice"},
            "text": "make it more casual",
        }

        with patch("src.telegram_handler.load_state", return_value=state) as mock_load, \
             patch("src.telegram_handler.save_state") as mock_save, \
             patch("src.outreach.revise_draft", new_callable=AsyncMock) as mock_revise:
            await bot._handle_message(message)

            # pending_draft_edit should be cleared
            assert state.pending_draft_edit is None
            mock_save.assert_called_once_with(state)
            mock_revise.assert_called_once_with(bot, 123, "d1", "make it more casual")

    @pytest.mark.asyncio
    async def test_revise_draft_error_sends_message(self, bot):
        """If revise_draft raises, error message is sent and field is cleared."""
        state = ChatState(chat_id=123)
        state.pending_draft_edit = "d1"

        message = {
            "chat": {"id": 123, "type": "private"},
            "from": {"first_name": "Alice"},
            "text": "make it shorter",
        }

        with patch("src.telegram_handler.load_state", return_value=state), \
             patch("src.telegram_handler.save_state"), \
             patch("src.outreach.revise_draft", new_callable=AsyncMock, side_effect=Exception("fail")):
            bot.send_text = AsyncMock(return_value={"ok": True})

            await bot._handle_message(message)

            assert state.pending_draft_edit is None
            bot.send_text.assert_called_once()
            sent_text = bot.send_text.call_args[0][1]
            assert "Failed to revise" in sent_text
