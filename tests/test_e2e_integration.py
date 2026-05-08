"""End-to-end integration tests with mocked external services.

These tests exercise full code paths that span multiple modules.
Marked with @pytest.mark.e2e — run separately with:
    python3 -m pytest tests/test_e2e_integration.py -v -m e2e

They do NOT require API credentials; all external calls are mocked.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.apify_scraper import AmenityEnrichmentResult
from src.conversation import ConversationEngine
from src.claude_client import ChatResult, ClaudeClient
from src.formatter import format_listing_card, format_scan_header, listing_keyboard
from src.models import ChatState, Draft, Listing, Preferences
from src.scanner import (
    NEIGHBORHOOD_ALIASES,
    _apply_pre_filters,
    _neighborhood_pre_filter,
    _normalize_hood,
    scan_for_chat,
)
from src.storage import save_state, load_state
from tests.conftest import make_listing


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mock_claude_chat_result(text: str, tool_calls: list | None = None) -> ChatResult:
    """Build a ChatResult from mocked Claude response."""
    return ChatResult(text=text, tool_messages=tool_calls or [])


class TrackingBot:
    """Mock Telegram bot that records sent messages."""

    def __init__(self):
        self.sent: list[tuple[str, str, dict | None]] = []  # (type, text, kwargs)

    async def send_text(self, chat_id, text, **kwargs):
        self.sent.append(("text", text, kwargs))
        return {"ok": True}

    async def send_listing_photo(self, chat_id, **kwargs):
        self.sent.append(("photo", kwargs.get("caption", ""), kwargs))
        return {"ok": True}

    async def send_photo(self, chat_id, photo_url, caption=None, **kwargs):
        self.sent.append(("photo", caption or "", kwargs))
        return {"ok": True}

    async def close(self):
        pass


# ---------------------------------------------------------------------------
# E2E Tests
# ---------------------------------------------------------------------------


@pytest.mark.e2e
class TestConversationE2E:
    """Full conversation flow with mocked Claude."""

    @pytest.mark.asyncio
    async def test_greeting_to_search_flow(self):
        """User greets -> sets prefs -> searches. Tools are dispatched by the engine."""
        state = ChatState(chat_id=1)

        # The ClaudeClient.chat method is the one that calls tools.
        # We mock it to simulate Claude calling tools, but the actual tool
        # dispatch happens inside ConversationEngine._execute_tool which IS
        # called by the real ClaudeClient.chat loop. For this test, we mock
        # the chat method to return results as if tools were already called.

        # Mock: Claude calls update_preferences, then returns text
        async def mock_chat_with_tools(*, system, messages, tools, tool_handler):
            call_count = mock_chat_with_tools.call_count
            mock_chat_with_tools.call_count += 1

            if call_count == 0:
                # Greeting — no tools
                return ChatResult(text="Hey! What are you looking for?")
            elif call_count == 1:
                # Set preferences — call the real tool handler
                await tool_handler("update_preferences", {
                    "budget_max": 4000, "bedrooms": [1],
                    "neighborhoods": ["East Village"],
                    "must_haves": ["Dishwasher"],
                })
                return ChatResult(text="Got it! 1BR in East Village under $4k.")
            elif call_count == 2:
                # Search — call the real tool handler
                await tool_handler("search_apartments", {})
                return ChatResult(text="Searching now!")
            return ChatResult(text="ok")

        mock_chat_with_tools.call_count = 0

        mock_claude = MagicMock(spec=ClaudeClient)
        mock_claude.chat = AsyncMock(side_effect=mock_chat_with_tools)

        engine = ConversationEngine(state, mock_claude)

        # Turn 1: Greeting
        result = await engine.handle_message("Hello!")
        assert not result.trigger_search

        # Turn 2: Set preferences — tool handler updates state
        result = await engine.handle_message("1BR in East Village under 4k, must have dishwasher")
        assert state.preferences.budget_max == 4000
        assert state.preferences.neighborhoods == ["East Village"]
        assert state.preferences.must_haves == ["Dishwasher"]

        # Turn 3: Search
        result = await engine.handle_message("Search for apartments")
        assert result.trigger_search
        assert state.preferences_ready  # Auto-enabled


@pytest.mark.e2e
class TestScanPipelineE2E:
    """Full scan pipeline with mock Apify + mocked LLM scoring."""

    @pytest.mark.asyncio
    async def test_full_scan_filters_and_scores(self):
        """Scan -> pre-filter -> enrich -> LLM score -> send results."""
        state = ChatState(
            chat_id=1,
            preferences=Preferences(
                budget_max=4000, bedrooms=[1],
                neighborhoods=["East Village"],
            ),
            preferences_ready=True,
        )

        raw_listings = [
            {"listing_id": "1", "url": "https://streeteasy.com/rental/1",
             "address": "100 E 7th St", "neighborhood": "East Village",
             "price": 3500, "bedrooms": 1, "bathrooms": 1.0, "photos": [], "amenities": []},
            {"listing_id": "2", "url": "https://streeteasy.com/rental/2",
             "address": "200 Broadway", "neighborhood": "Financial District",
             "price": 3000, "bedrooms": 1, "bathrooms": 1.0, "photos": [], "amenities": []},
            {"listing_id": "3", "url": "https://streeteasy.com/rental/3",
             "address": "300 E 10th St", "neighborhood": "East Village",
             "price": 5000, "bedrooms": 1, "bathrooms": 1.0, "photos": [], "amenities": []},
        ]

        mock_scraper = AsyncMock()
        mock_scraper.search_with_retry = AsyncMock(return_value=raw_listings)

        scores = [
            {"id": "1", "include": True, "score": 85, "pros": ["Under budget"], "cons": ["Small"]},
        ]

        mock_response = MagicMock()
        mock_response.content = [MagicMock(text=json.dumps(scores))]
        mock_client = AsyncMock()
        mock_client.messages.create = AsyncMock(return_value=mock_response)

        bot = TrackingBot()

        with patch("src.scanner.anthropic.AsyncAnthropic", return_value=mock_client), \
             patch("src.scanner.save_state"), \
             patch("src.scanner._pick_hero_photos", return_value={}):
            await scan_for_chat(mock_scraper, bot, state, is_daily=False)

        # Should have sent header + 1 listing (FiDi filtered, over-budget filtered)
        text_messages = [t for t in bot.sent if t[0] == "text"]
        assert any("Search Results" in msg[1] for msg in text_messages)
        assert "1" in state.seen_listing_ids

    @pytest.mark.asyncio
    async def test_full_scan_ranks_better_evidence_first_and_shows_professional_card(self):
        """Scan output ranks deterministic fit above close model score and exposes why."""
        state = ChatState(
            chat_id=1,
            preferences=Preferences(
                budget_max=4000,
                bedrooms=[1],
                neighborhoods=["Chelsea"],
                must_haves=["Dishwasher"],
                no_fee_only=True,
            ),
            preferences_ready=True,
        )

        raw_listings = [
            {"listing_id": "weak", "url": "https://streeteasy.com/rental/weak",
             "address": "90 Ninth Ave", "neighborhood": "Chelsea",
             "price": 3900, "bedrooms": 1, "bathrooms": 1.0,
             "photos": ["https://img.example/weak.jpg"], "amenities": []},
            {"listing_id": "strong", "url": "https://streeteasy.com/rental/strong",
             "address": "120 West 20th St", "neighborhood": "Chelsea",
             "price": 3600, "bedrooms": 1, "bathrooms": 1.0, "sqft": 650,
             "photos": ["https://img.example/strong.jpg"], "amenities": []},
        ]

        mock_scraper = AsyncMock()
        mock_scraper.search_with_retry = AsyncMock(return_value=raw_listings)
        mock_scraper.enrich_with_amenities = AsyncMock(return_value=AmenityEnrichmentResult(
            data_by_listing_id={
                "weak": {"unit_features": [], "building_amenities": ["Laundry in Building"]},
                "strong": {"unit_features": ["Dishwasher"], "building_amenities": ["Elevator"]},
            },
            coverage=1.0,
            target_count=2,
            run_summaries=[],
            failed=False,
        ))

        scores = [
            {"id": "weak", "include": True, "score": 83, "pros": ["Good Chelsea block"], "cons": ["Dishwasher unverified"]},
            {"id": "strong", "include": True, "score": 80, "pros": ["Dishwasher confirmed"], "cons": []},
        ]

        mock_response = MagicMock()
        mock_response.content = [MagicMock(text=json.dumps(scores))]
        mock_client = AsyncMock()
        mock_client.messages.create = AsyncMock(return_value=mock_response)

        bot = TrackingBot()

        with patch("src.scanner.anthropic.AsyncAnthropic", return_value=mock_client), \
             patch("src.scanner.save_state"), \
             patch("src.scanner._pick_hero_photos", return_value={}):
            await scan_for_chat(mock_scraper, bot, state, is_daily=False)

        photo_cards = [msg for msg in bot.sent if msg[0] == "photo"]
        assert photo_cards[0][1].startswith("#1")
        assert "120 West 20th St" in photo_cards[0][1]
        assert "Best signals" in photo_cards[0][1]
        assert "Must-haves verified" in photo_cards[0][1]
        assert "No fee" in photo_cards[0][1]
        assert state.last_scan_listing_ids[0] == "strong"


@pytest.mark.e2e
class TestOutreachE2E:
    """Draft -> revise -> approve outreach flow."""

    @pytest.mark.asyncio
    async def test_create_revise_approve(self, tmp_path):
        """Full outreach lifecycle with mocked Claude."""
        from src.outreach import create_draft, revise_draft, send_approved_draft

        data_dir = str(tmp_path)
        state = ChatState(chat_id=1)
        listing = make_listing("test-draft")
        state.recent_listings[listing.listing_id] = listing

        bot = TrackingBot()

        with patch("src.storage.DATA_DIR", data_dir), \
             patch("src.outreach.load_state") as mock_load, \
             patch("src.outreach.save_state") as mock_save:

            mock_load.return_value = state

            # Mock Claude for draft generation
            mock_result = _mock_claude_chat_result("Hi, I'm interested in your apartment.")
            with patch.object(ClaudeClient, "chat", return_value=mock_result):
                await create_draft(bot, 1, listing)

            # Verify draft created
            assert len(state.active_drafts) == 1
            draft_id = list(state.active_drafts.keys())[0]
            assert state.active_drafts[draft_id].status == "pending"

            # Revise
            revised_result = _mock_claude_chat_result("Hey! Super interested in this place.")
            with patch.object(ClaudeClient, "chat", return_value=revised_result):
                await revise_draft(bot, 1, draft_id, "Make it more casual")

            assert "Super interested" in state.active_drafts[draft_id].message_text

            # Approve
            await send_approved_draft(bot, 1, draft_id)
            assert state.active_drafts[draft_id].status == "sent"
            assert any("Message ready" in msg[1] or "message" in msg[1].lower() for msg in bot.sent)


@pytest.mark.e2e
class TestNeighborhoodAliasesE2E:
    """Verify all alias pairs work bidirectionally with realistic listing data."""

    def test_all_aliases_bidirectional_with_realistic_data(self):
        """For every alias A->B, user with A in prefs matches listing from B."""
        for alias_from, alias_to in NEIGHBORHOOD_ALIASES.items():
            prefs = Preferences(neighborhoods=[alias_from.title()])
            listing = make_listing("x", neighborhood=alias_to.title())
            result = _neighborhood_pre_filter([listing], prefs)
            assert len(result) == 1, (
                f"FAIL: pref='{alias_from.title()}' listing='{alias_to.title()}'"
            )

    def test_deployed_user_gramercy_park_scenario(self):
        """Reproduce the exact deployed user bug: pref='Gramercy Park', listing='Gramercy'."""
        prefs = Preferences(
            neighborhoods=["Gramercy Park", "Flatiron", "Greenwich Village",
                           "West Village", "NoHo", "SoHo", "Tribeca"],
        )
        listings = [
            make_listing("1", neighborhood="Gramercy Park"),
            make_listing("2", neighborhood="Gramercy"),
            make_listing("3", neighborhood="Flatiron"),
            make_listing("4", neighborhood="NoMad"),
            make_listing("5", neighborhood="Noho"),
            make_listing("6", neighborhood="Chelsea"),  # not in prefs
        ]
        result = _neighborhood_pre_filter(listings, prefs)
        kept_ids = {l.listing_id for l in result}
        assert "1" in kept_ids  # Gramercy Park direct match
        assert "2" in kept_ids  # Gramercy via alias normalization
        assert "3" in kept_ids  # Flatiron direct match
        assert "4" in kept_ids  # NoMad -> Flatiron alias
        assert "5" in kept_ids  # Noho direct match (user has NoHo)
        assert "6" not in kept_ids  # Chelsea not in prefs


@pytest.mark.e2e
class TestCooldownRecoveryE2E:
    """Verify failure streak recovery persists state correctly."""

    @pytest.mark.asyncio
    async def test_cooldown_reset_persisted_on_empty_results(self):
        """After failure streak, a successful-but-empty search persists the reset."""
        state = ChatState(
            chat_id=1,
            preferences=Preferences(budget_max=4000, neighborhoods=["Chelsea"]),
            preferences_ready=True,
            search_failure_streak=5,
            search_cooldown_until=datetime.now(timezone.utc) - timedelta(seconds=1),
        )

        mock_scraper = AsyncMock()
        mock_scraper.search_with_retry = AsyncMock(return_value=[])
        bot = TrackingBot()

        with patch("src.scanner.save_state") as mock_save:
            await scan_for_chat(mock_scraper, bot, state, is_daily=False)
            # save_state must be called to persist the reset
            mock_save.assert_called()

        assert state.search_failure_streak == 0
        assert state.search_cooldown_until is None

    @pytest.mark.asyncio
    async def test_cooldown_blocks_search_when_active(self):
        """Active cooldown suppresses search and sends user message."""
        state = ChatState(
            chat_id=1,
            preferences=Preferences(budget_max=4000, neighborhoods=["Chelsea"]),
            preferences_ready=True,
            search_failure_streak=3,
            search_cooldown_until=datetime.now(timezone.utc) + timedelta(minutes=5),
        )

        mock_scraper = AsyncMock()
        bot = TrackingBot()

        with patch("src.scanner.save_state"):
            await scan_for_chat(mock_scraper, bot, state, is_daily=False)

        mock_scraper.search_with_retry.assert_not_called()
        assert any("blocking" in msg[1].lower() or "retry" in msg[1].lower()
                    for msg in bot.sent)
