"""Tests for LLM-powered conversation engine."""

from __future__ import annotations

from datetime import date
from unittest.mock import AsyncMock, patch

import pytest

from src.claude_client import ChatResult, ClaudeClient
from src.conversation import ConversationEngine, ConversationResult, Response
from src.models import ChatState, ConversationTurn, CurrentApartment, Listing


@pytest.fixture
def fresh_state():
    return ChatState(chat_id=12345)


@pytest.fixture
def state_with_prefs():
    state = ChatState(chat_id=12345)
    state.preferences.budget_max = 4000
    state.preferences.bedrooms = [1, 2]
    state.preferences.neighborhoods = ["Chelsea"]
    state.preferences_ready = True
    return state


def _make_listing(listing_id: str, **overrides) -> Listing:
    """Helper to create a test listing with sensible defaults."""
    defaults = dict(
        listing_id=listing_id,
        url=f"https://streeteasy.com/rental/{listing_id}",
        address=f"123 Test St #{listing_id}",
        neighborhood="Chelsea",
        price=3500,
        bedrooms=2,
        bathrooms=1.0,
        match_score=75,
    )
    defaults.update(overrides)
    return Listing(**defaults)


@pytest.fixture
def state_with_listings():
    """State with recent and liked listings for detail/compare/draft tests."""
    state = ChatState(chat_id=12345)
    state.preferences.budget_max = 4000
    state.preferences.neighborhoods = ["Chelsea"]

    l1 = _make_listing("100", address="100 Main St", price=3000, bedrooms=1, neighborhood="Chelsea", match_score=90)
    l2 = _make_listing("200", address="200 Broadway", price=3800, bedrooms=2, neighborhood="SoHo", match_score=70, broker_fee="Broker fee")
    l3 = _make_listing("300", address="300 Park Ave", price=4200, bedrooms=2, neighborhood="Gramercy Park", match_score=60)

    state.recent_listings = {"100": l1, "200": l2, "300": l3}
    state.liked_listing_ids = {"100", "200"}
    state.liked_listings = {"100": l1, "200": l2}
    return state


class TestConversationEngine:
    @pytest.mark.asyncio
    async def test_handle_message_adds_to_history(self, fresh_state):
        """Messages are added to conversation history."""
        mock_claude = AsyncMock(spec=ClaudeClient)
        mock_claude.chat = AsyncMock(return_value=ChatResult(
            text="Hello! I can help you find an apartment.", tool_messages=[]
        ))

        engine = ConversationEngine(fresh_state, claude=mock_claude)
        result = await engine.handle_message("Hi there!")

        assert len(fresh_state.conversation_history) == 2
        assert fresh_state.conversation_history[0].role == "user"
        assert fresh_state.conversation_history[0].content == "Hi there!"
        assert fresh_state.conversation_history[1].role == "assistant"

    @pytest.mark.asyncio
    async def test_handle_message_returns_response(self, fresh_state):
        """Handle message returns a ConversationResult with responses."""
        mock_claude = AsyncMock(spec=ClaudeClient)
        mock_claude.chat = AsyncMock(return_value=ChatResult(
            text="Welcome! What's your budget?", tool_messages=[]
        ))

        engine = ConversationEngine(fresh_state, claude=mock_claude)
        result = await engine.handle_message("Hi!")

        assert isinstance(result, ConversationResult)
        assert len(result.responses) == 1
        assert "Welcome" in result.responses[0].text

    @pytest.mark.asyncio
    async def test_handle_message_claude_api_failure(self, fresh_state):
        """Graceful handling when Claude API fails."""
        mock_claude = AsyncMock(spec=ClaudeClient)
        mock_claude.chat = AsyncMock(side_effect=Exception("API error"))

        engine = ConversationEngine(fresh_state, claude=mock_claude)
        result = await engine.handle_message("Hello")

        assert len(result.responses) == 1
        assert "trouble" in result.responses[0].text.lower()

    @pytest.mark.asyncio
    async def test_history_trimming(self, fresh_state):
        """History is trimmed when it exceeds MAX_HISTORY_TURNS."""
        mock_claude = AsyncMock(spec=ClaudeClient)
        mock_claude.chat = AsyncMock(return_value=ChatResult(text="OK", tool_messages=[]))

        # Pre-fill history with many turns
        for i in range(40):
            fresh_state.conversation_history.append(
                ConversationTurn(role="user" if i % 2 == 0 else "assistant", content=f"msg {i}")
            )

        engine = ConversationEngine(fresh_state, claude=mock_claude)
        await engine.handle_message("one more")

        # Should be trimmed to MAX_HISTORY_TURNS (30)
        assert len(fresh_state.conversation_history) <= 30


class TestToolExecution:
    @pytest.mark.asyncio
    async def test_update_preferences_budget(self, fresh_state):
        """update_preferences tool updates budget correctly."""
        engine = ConversationEngine(fresh_state)

        result = ConversationResult()
        output = engine._execute_tool("update_preferences", {"budget_max": 3000}, result)

        assert fresh_state.preferences.budget_max == 3000
        assert "budget_max" in output

    @pytest.mark.asyncio
    async def test_update_preferences_bedrooms(self, fresh_state):
        """update_preferences tool updates bedrooms correctly."""
        engine = ConversationEngine(fresh_state)

        result = ConversationResult()
        engine._execute_tool("update_preferences", {"bedrooms": [0, 1]}, result)

        assert fresh_state.preferences.bedrooms == [0, 1]

    @pytest.mark.asyncio
    async def test_update_preferences_neighborhoods(self, fresh_state):
        """update_preferences tool updates neighborhoods correctly."""
        engine = ConversationEngine(fresh_state)

        result = ConversationResult()
        engine._execute_tool(
            "update_preferences",
            {"neighborhoods": ["East Village", "West Village"]},
            result,
        )

        assert fresh_state.preferences.neighborhoods == ["East Village", "West Village"]

    @pytest.mark.asyncio
    async def test_update_preferences_multiple_fields(self, fresh_state):
        """update_preferences tool handles multiple fields at once."""
        engine = ConversationEngine(fresh_state)

        result = ConversationResult()
        engine._execute_tool(
            "update_preferences",
            {
                "budget_max": 4000,
                "bedrooms": [1, 2],
                "neighborhoods": ["Chelsea"],
                "must_haves": ["Dishwasher", "Laundry in Unit"],
            },
            result,
        )

        assert fresh_state.preferences.budget_max == 4000
        assert fresh_state.preferences.bedrooms == [1, 2]
        assert fresh_state.preferences.neighborhoods == ["Chelsea"]
        assert fresh_state.preferences.must_haves == ["Dishwasher", "Laundry in Unit"]

    @pytest.mark.asyncio
    async def test_update_preferences_min_bathrooms(self, fresh_state):
        """update_preferences tool updates min_bathrooms correctly."""
        engine = ConversationEngine(fresh_state)

        result = ConversationResult()
        output = engine._execute_tool("update_preferences", {"min_bathrooms": 2}, result)

        assert fresh_state.preferences.min_bathrooms == 2
        assert "min_bathrooms" in output

    @pytest.mark.asyncio
    async def test_update_preferences_nice_to_haves(self, fresh_state):
        """update_preferences tool updates nice_to_haves correctly."""
        engine = ConversationEngine(fresh_state)

        result = ConversationResult()
        output = engine._execute_tool(
            "update_preferences",
            {"nice_to_haves": ["Gym", "Roof Deck"]},
            result,
        )

        assert fresh_state.preferences.nice_to_haves == ["Gym", "Roof Deck"]
        assert "nice_to_haves" in output

    @pytest.mark.asyncio
    async def test_update_preferences_no_fee(self, fresh_state):
        """update_preferences tool updates no_fee_only correctly."""
        engine = ConversationEngine(fresh_state)

        result = ConversationResult()
        engine._execute_tool("update_preferences", {"no_fee_only": True}, result)

        assert fresh_state.preferences.no_fee_only is True

    @pytest.mark.asyncio
    async def test_show_preferences(self, state_with_prefs):
        """show_preferences tool returns formatted summary."""
        engine = ConversationEngine(state_with_prefs)

        result = ConversationResult()
        output = engine._execute_tool("show_preferences", {}, result)

        assert "Preferences" in output
        assert "Chelsea" in output

    @pytest.mark.asyncio
    async def test_search_apartments_triggers_search(self, state_with_prefs):
        """search_apartments tool sets trigger_search flag."""
        engine = ConversationEngine(state_with_prefs)

        result = ConversationResult()
        engine._execute_tool("search_apartments", {}, result)

        assert result.trigger_search is True

    @pytest.mark.asyncio
    async def test_search_apartments_requires_prefs(self, fresh_state):
        """search_apartments tool fails without minimum preferences."""
        engine = ConversationEngine(fresh_state)

        result = ConversationResult()
        output = engine._execute_tool("search_apartments", {}, result)

        assert "Cannot search" in output
        assert result.trigger_search is False

    @pytest.mark.asyncio
    async def test_mark_ready(self, fresh_state):
        """mark_ready tool sets preferences_ready flag."""
        engine = ConversationEngine(fresh_state)

        result = ConversationResult()
        engine._execute_tool("mark_ready", {}, result)

        assert fresh_state.preferences_ready is True

    @pytest.mark.asyncio
    async def test_clear_search_history(self, fresh_state):
        """clear_search_history tool clears seen listing IDs."""
        fresh_state.seen_listing_ids = {"100", "200", "300"}
        engine = ConversationEngine(fresh_state)

        result = ConversationResult()
        output = engine._execute_tool("clear_search_history", {}, result)

        assert len(fresh_state.seen_listing_ids) == 0
        assert "3" in output

    @pytest.mark.asyncio
    async def test_clear_search_history_empty(self, fresh_state):
        """clear_search_history on empty set works fine."""
        engine = ConversationEngine(fresh_state)

        result = ConversationResult()
        output = engine._execute_tool("clear_search_history", {}, result)

        assert len(fresh_state.seen_listing_ids) == 0
        assert "0" in output

    @pytest.mark.asyncio
    async def test_pause_daily_scans(self, state_with_prefs):
        """pause_daily_scans tool unsets preferences_ready."""
        assert state_with_prefs.preferences_ready is True
        engine = ConversationEngine(state_with_prefs)

        result = ConversationResult()
        output = engine._execute_tool("pause_daily_scans", {}, result)

        assert state_with_prefs.preferences_ready is False
        assert "paused" in output.lower()

    @pytest.mark.asyncio
    async def test_pause_daily_scans_already_paused(self, fresh_state):
        """pause_daily_scans when already paused returns appropriate message."""
        engine = ConversationEngine(fresh_state)

        result = ConversationResult()
        output = engine._execute_tool("pause_daily_scans", {}, result)

        assert "already paused" in output.lower()

    @pytest.mark.asyncio
    async def test_get_liked_listings(self, fresh_state):
        """get_liked_listings returns liked listing info."""
        fresh_state.liked_listing_ids = {"4961650", "4940118"}
        engine = ConversationEngine(fresh_state)

        result = ConversationResult()
        output = engine._execute_tool("get_liked_listings", {}, result)

        assert "2 liked" in output
        assert "4961650" in output
        assert "4940118" in output
        assert "streeteasy.com" in output

    @pytest.mark.asyncio
    async def test_get_liked_listings_empty(self, fresh_state):
        """get_liked_listings with no likes returns appropriate message."""
        engine = ConversationEngine(fresh_state)

        result = ConversationResult()
        output = engine._execute_tool("get_liked_listings", {}, result)

        assert "No liked" in output

    @pytest.mark.asyncio
    async def test_remove_liked_listing(self, fresh_state):
        """remove_liked_listing removes a listing from liked set."""
        fresh_state.liked_listing_ids = {"100", "200", "300"}
        engine = ConversationEngine(fresh_state)

        result = ConversationResult()
        output = engine._execute_tool("remove_liked_listing", {"listing_id": "200"}, result)

        assert "200" not in fresh_state.liked_listing_ids
        assert "100" in fresh_state.liked_listing_ids
        assert "300" in fresh_state.liked_listing_ids
        assert "Removed" in output

    @pytest.mark.asyncio
    async def test_remove_liked_listing_not_found(self, fresh_state):
        """remove_liked_listing with unknown ID returns not found."""
        fresh_state.liked_listing_ids = {"100"}
        engine = ConversationEngine(fresh_state)

        result = ConversationResult()
        output = engine._execute_tool("remove_liked_listing", {"listing_id": "999"}, result)

        assert "not in" in output
        assert len(fresh_state.liked_listing_ids) == 1

    @pytest.mark.asyncio
    async def test_reset_preferences(self, state_with_prefs):
        """reset_preferences clears all preferences and pauses scans."""
        engine = ConversationEngine(state_with_prefs)

        result = ConversationResult()
        output = engine._execute_tool("reset_preferences", {}, result)

        assert state_with_prefs.preferences.budget_max is None
        assert state_with_prefs.preferences.bedrooms == []
        assert state_with_prefs.preferences.neighborhoods == []
        assert state_with_prefs.preferences_ready is False
        assert "reset" in output.lower()

    @pytest.mark.asyncio
    async def test_remove_neighborhoods(self, state_with_prefs):
        """remove_neighborhoods removes specific neighborhoods."""
        state_with_prefs.preferences.neighborhoods = ["Chelsea", "SoHo", "NoHo"]
        engine = ConversationEngine(state_with_prefs)

        result = ConversationResult()
        output = engine._execute_tool(
            "remove_neighborhoods", {"neighborhoods": ["SoHo"]}, result
        )

        assert "SoHo" not in state_with_prefs.preferences.neighborhoods
        assert "Chelsea" in state_with_prefs.preferences.neighborhoods
        assert "NoHo" in state_with_prefs.preferences.neighborhoods
        assert "Removed" in output

    @pytest.mark.asyncio
    async def test_remove_neighborhoods_case_insensitive(self, state_with_prefs):
        """remove_neighborhoods is case-insensitive."""
        state_with_prefs.preferences.neighborhoods = ["Chelsea", "SoHo"]
        engine = ConversationEngine(state_with_prefs)

        result = ConversationResult()
        engine._execute_tool(
            "remove_neighborhoods", {"neighborhoods": ["soho"]}, result
        )

        assert state_with_prefs.preferences.neighborhoods == ["Chelsea"]

    @pytest.mark.asyncio
    async def test_remove_neighborhoods_not_found(self, fresh_state):
        """remove_neighborhoods with unknown neighborhoods returns not found."""
        fresh_state.preferences.neighborhoods = ["Chelsea"]
        engine = ConversationEngine(fresh_state)

        result = ConversationResult()
        output = engine._execute_tool(
            "remove_neighborhoods", {"neighborhoods": ["Brooklyn Heights"]}, result
        )

        assert "None of" in output
        assert fresh_state.preferences.neighborhoods == ["Chelsea"]

    # --- Listing detail tools ---

    @pytest.mark.asyncio
    async def test_get_listing_details_from_recent(self, state_with_listings):
        """get_listing_details returns full details for a recent listing."""
        engine = ConversationEngine(state_with_listings)

        result = ConversationResult()
        output = engine._execute_tool("get_listing_details", {"listing_id": "100"}, result)

        assert "100 Main St" in output
        assert "$3,000" in output
        assert "Chelsea" in output
        assert "90/100" in output

    @pytest.mark.asyncio
    async def test_get_listing_details_from_liked(self, state_with_listings):
        """get_listing_details falls back to liked listings."""
        # Remove from recent, keep in liked
        del state_with_listings.recent_listings["200"]
        engine = ConversationEngine(state_with_listings)

        result = ConversationResult()
        output = engine._execute_tool("get_listing_details", {"listing_id": "200"}, result)

        assert "200 Broadway" in output

    @pytest.mark.asyncio
    async def test_get_listing_details_not_found(self, fresh_state):
        """get_listing_details returns link when listing not found."""
        engine = ConversationEngine(fresh_state)

        result = ConversationResult()
        output = engine._execute_tool("get_listing_details", {"listing_id": "999"}, result)

        assert "not found" in output.lower()
        assert "streeteasy.com" in output

    @pytest.mark.asyncio
    async def test_compare_listings(self, state_with_listings):
        """compare_listings returns comparison of multiple listings."""
        engine = ConversationEngine(state_with_listings)

        result = ConversationResult()
        output = engine._execute_tool(
            "compare_listings", {"listing_ids": ["100", "200"]}, result
        )

        assert "100 Main St" in output
        assert "200 Broadway" in output
        assert "$3,000" in output
        assert "$3,800" in output

    @pytest.mark.asyncio
    async def test_compare_listings_partial(self, state_with_listings):
        """compare_listings with some missing listings reports which are missing."""
        engine = ConversationEngine(state_with_listings)

        result = ConversationResult()
        output = engine._execute_tool(
            "compare_listings", {"listing_ids": ["100", "200", "999"]}, result
        )

        assert "100 Main St" in output
        assert "999" in output

    @pytest.mark.asyncio
    async def test_compare_listings_too_few(self, fresh_state):
        """compare_listings needs at least 2 IDs."""
        engine = ConversationEngine(fresh_state)

        result = ConversationResult()
        output = engine._execute_tool(
            "compare_listings", {"listing_ids": ["100"]}, result
        )

        assert "at least 2" in output.lower()

    @pytest.mark.asyncio
    async def test_draft_outreach(self, state_with_listings):
        """draft_outreach sets trigger flag for a known listing."""
        engine = ConversationEngine(state_with_listings)

        result = ConversationResult()
        output = engine._execute_tool(
            "draft_outreach", {"listing_id": "100"}, result
        )

        assert result.trigger_draft_listing_id == "100"
        assert "draft" in output.lower()

    @pytest.mark.asyncio
    async def test_draft_outreach_not_found(self, fresh_state):
        """draft_outreach fails for unknown listing."""
        engine = ConversationEngine(fresh_state)

        result = ConversationResult()
        output = engine._execute_tool(
            "draft_outreach", {"listing_id": "999"}, result
        )

        assert result.trigger_draft_listing_id is None
        assert "not found" in output.lower()

    # --- Current apartment tools ---

    @pytest.mark.asyncio
    async def test_update_current_apartment(self, fresh_state):
        """update_current_apartment stores apartment info."""
        engine = ConversationEngine(fresh_state)

        result = ConversationResult()
        output = engine._execute_tool(
            "update_current_apartment",
            {
                "address": "456 Oak St",
                "neighborhood": "Williamsburg",
                "price": 2800,
                "bedrooms": 1,
                "pros": ["great light", "quiet"],
                "cons": ["no dishwasher"],
            },
            result,
        )

        apt = fresh_state.current_apartment
        assert apt is not None
        assert apt.address == "456 Oak St"
        assert apt.neighborhood == "Williamsburg"
        assert apt.price == 2800
        assert apt.bedrooms == 1
        assert apt.pros == ["great light", "quiet"]
        assert apt.cons == ["no dishwasher"]
        assert "Updated" in output

    @pytest.mark.asyncio
    async def test_update_current_apartment_incremental(self, fresh_state):
        """update_current_apartment adds to existing info."""
        fresh_state.current_apartment = CurrentApartment(
            address="456 Oak St", price=2800
        )
        engine = ConversationEngine(fresh_state)

        result = ConversationResult()
        engine._execute_tool(
            "update_current_apartment",
            {"cons": ["noisy neighbors"]},
            result,
        )

        assert fresh_state.current_apartment.address == "456 Oak St"
        assert fresh_state.current_apartment.price == 2800
        assert fresh_state.current_apartment.cons == ["noisy neighbors"]

    @pytest.mark.asyncio
    async def test_show_current_apartment(self, fresh_state):
        """show_current_apartment displays saved info."""
        fresh_state.current_apartment = CurrentApartment(
            address="456 Oak St",
            neighborhood="Williamsburg",
            price=2800,
            bedrooms=1,
            pros=["great light"],
            cons=["no dishwasher"],
        )
        engine = ConversationEngine(fresh_state)

        result = ConversationResult()
        output = engine._execute_tool("show_current_apartment", {}, result)

        assert "456 Oak St" in output
        assert "Williamsburg" in output
        assert "$2,800" in output
        assert "great light" in output
        assert "no dishwasher" in output

    @pytest.mark.asyncio
    async def test_show_current_apartment_empty(self, fresh_state):
        """show_current_apartment with no data returns appropriate message."""
        engine = ConversationEngine(fresh_state)

        result = ConversationResult()
        output = engine._execute_tool("show_current_apartment", {}, result)

        assert "No current apartment" in output

    # --- Updated liked listing tools with full data ---

    @pytest.mark.asyncio
    async def test_get_liked_listings_with_data(self, state_with_listings):
        """get_liked_listings returns full listing details when available."""
        engine = ConversationEngine(state_with_listings)

        result = ConversationResult()
        output = engine._execute_tool("get_liked_listings", {}, result)

        assert "2 liked" in output
        assert "100 Main St" in output
        assert "200 Broadway" in output
        assert "$3,000" in output

    @pytest.mark.asyncio
    async def test_remove_liked_listing_cleans_both(self, state_with_listings):
        """remove_liked_listing removes from both liked_listing_ids and liked_listings."""
        engine = ConversationEngine(state_with_listings)

        result = ConversationResult()
        engine._execute_tool("remove_liked_listing", {"listing_id": "100"}, result)

        assert "100" not in state_with_listings.liked_listing_ids
        assert "100" not in state_with_listings.liked_listings
        assert "200" in state_with_listings.liked_listing_ids
        assert "200" in state_with_listings.liked_listings

    @pytest.mark.asyncio
    async def test_update_preferences_constraint_context(self, fresh_state):
        """update_preferences with constraint_context stores it on preferences."""
        engine = ConversationEngine(fresh_state)
        result = ConversationResult()
        output = engine._execute_tool(
            "update_preferences",
            {
                "budget_max": 3500,
                "constraint_context": "Budget $3,500 is a firm max. Dishwasher non-negotiable.",
            },
            result,
        )

        assert fresh_state.preferences.budget_max == 3500
        assert fresh_state.preferences.constraint_context == "Budget $3,500 is a firm max. Dishwasher non-negotiable."
        assert "constraint_context" in output

    @pytest.mark.asyncio
    async def test_update_preferences_without_constraint_context_preserves(self, fresh_state):
        """update_preferences without constraint_context preserves existing value."""
        fresh_state.preferences.constraint_context = "Budget is firm."
        engine = ConversationEngine(fresh_state)

        result = ConversationResult()
        engine._execute_tool("update_preferences", {"budget_max": 4000}, result)

        assert fresh_state.preferences.budget_max == 4000
        assert fresh_state.preferences.constraint_context == "Budget is firm."

    @pytest.mark.asyncio
    async def test_unknown_tool(self, fresh_state):
        """Unknown tool returns error message."""
        engine = ConversationEngine(fresh_state)

        result = ConversationResult()
        output = engine._execute_tool("nonexistent_tool", {}, result)

        assert "Unknown" in output


class TestSystemPrompt:
    def test_system_prompt_includes_empty_prefs(self, fresh_state):
        """System prompt indicates no preferences set."""
        engine = ConversationEngine(fresh_state)
        prompt = engine._build_system_prompt()

        assert "No preferences set yet" in prompt

    def test_system_prompt_includes_prefs(self, state_with_prefs):
        """System prompt includes current preferences."""
        engine = ConversationEngine(state_with_prefs)
        prompt = engine._build_system_prompt()

        assert "Chelsea" in prompt
        assert "$4,000" in prompt

    def test_system_prompt_includes_ready_status(self, state_with_prefs):
        """System prompt includes preferences_ready status."""
        engine = ConversationEngine(state_with_prefs)
        prompt = engine._build_system_prompt()

        assert "True" in prompt

    def test_system_prompt_includes_tool_guardrails(self, fresh_state):
        """System prompt includes tool usage rules."""
        engine = ConversationEngine(fresh_state)
        prompt = engine._build_system_prompt()

        assert "MUST use the search_apartments tool" in prompt
        assert "Never simulate tool output" in prompt

    def test_system_prompt_instructs_tool_results_not_visible(self, fresh_state):
        """System prompt tells Claude that tool results are not visible to the user."""
        engine = ConversationEngine(fresh_state)
        prompt = engine._build_system_prompt()

        assert "CANNOT see tool results" in prompt
        assert "MUST include any relevant data from tool results" in prompt

    def test_system_prompt_includes_constraint_context(self, state_with_prefs):
        """System prompt includes constraint_context when present."""
        state_with_prefs.preferences.constraint_context = (
            "Budget $4k is firm. 2BR needed for kids."
        )
        engine = ConversationEngine(state_with_prefs)
        prompt = engine._build_system_prompt()

        assert "Budget $4k is firm. 2BR needed for kids." in prompt
        assert "Constraint context" in prompt

    def test_system_prompt_omits_constraint_context_when_none(self, fresh_state):
        """System prompt omits constraint section when constraint_context is None."""
        engine = ConversationEngine(fresh_state)
        prompt = engine._build_system_prompt()

        assert "\nConstraint context: " not in prompt

    def test_system_prompt_includes_current_date(self, fresh_state):
        """System prompt includes today's date."""
        engine = ConversationEngine(fresh_state)
        prompt = engine._build_system_prompt()

        assert f"Today's date is {date.today().isoformat()}" in prompt

    def test_system_prompt_shows_prefs_with_budget_min_only(self, fresh_state):
        """System prompt shows preferences (not 'No preferences') when only budget_min is set."""
        fresh_state.preferences.budget_min = 2000
        engine = ConversationEngine(fresh_state)
        prompt = engine._build_system_prompt()

        assert "No preferences set yet" not in prompt

    def test_system_prompt_shows_prefs_with_no_fee_only(self, fresh_state):
        """System prompt shows preferences when only no_fee_only is set."""
        fresh_state.preferences.no_fee_only = True
        engine = ConversationEngine(fresh_state)
        prompt = engine._build_system_prompt()

        assert "No preferences set yet" not in prompt

    def test_system_prompt_shows_prefs_with_nice_to_haves_only(self, fresh_state):
        """System prompt shows preferences when only nice_to_haves is set."""
        fresh_state.preferences.nice_to_haves = ["Gym", "Roof Deck"]
        engine = ConversationEngine(fresh_state)
        prompt = engine._build_system_prompt()

        assert "No preferences set yet" not in prompt

    def test_system_prompt_shows_prefs_with_move_in_date_only(self, fresh_state):
        """System prompt shows preferences when only move_in_date is set."""
        fresh_state.preferences.move_in_date = "2026-04-01"
        engine = ConversationEngine(fresh_state)
        prompt = engine._build_system_prompt()

        assert "No preferences set yet" not in prompt

    def test_system_prompt_shows_prefs_with_commute_only(self, fresh_state):
        """System prompt shows preferences when only commute_address is set."""
        fresh_state.preferences.commute_address = "Times Square"
        engine = ConversationEngine(fresh_state)
        prompt = engine._build_system_prompt()

        assert "No preferences set yet" not in prompt


class TestStructuredHistory:
    @pytest.mark.asyncio
    async def test_tool_messages_stored_in_history(self, fresh_state):
        """Tool use/result turns are stored in conversation history."""
        mock_claude = AsyncMock(spec=ClaudeClient)
        tool_messages = [
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "Let me search for you."},
                    {"type": "tool_use", "id": "toolu_123", "name": "search_apartments", "input": {}},
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "toolu_123", "content": "Search triggered."},
                ],
            },
        ]
        mock_claude.chat = AsyncMock(return_value=ChatResult(
            text="I've started a search for you!", tool_messages=tool_messages
        ))

        engine = ConversationEngine(fresh_state, claude=mock_claude)
        await engine.handle_message("search for apartments")

        # user text + assistant tool_use + user tool_result + assistant final text = 4 turns
        assert len(fresh_state.conversation_history) == 4
        assert fresh_state.conversation_history[0].role == "user"
        assert fresh_state.conversation_history[0].content == "search for apartments"
        assert fresh_state.conversation_history[1].role == "assistant"
        assert isinstance(fresh_state.conversation_history[1].content, list)
        assert fresh_state.conversation_history[2].role == "user"
        assert isinstance(fresh_state.conversation_history[2].content, list)
        assert fresh_state.conversation_history[3].role == "assistant"
        assert fresh_state.conversation_history[3].content == "I've started a search for you!"

    @pytest.mark.asyncio
    async def test_trim_respects_tool_boundaries(self, fresh_state):
        """Trimming doesn't orphan tool_result messages at the start."""
        mock_claude = AsyncMock(spec=ClaudeClient)
        mock_claude.chat = AsyncMock(return_value=ChatResult(text="OK", tool_messages=[]))

        # Build history: 28 user/assistant text turns, then a tool exchange, then more text
        for i in range(28):
            fresh_state.conversation_history.append(
                ConversationTurn(role="user" if i % 2 == 0 else "assistant", content=f"msg {i}")
            )
        # Add a tool exchange that will land at the boundary after trimming
        fresh_state.conversation_history.append(
            ConversationTurn(role="assistant", content=[
                {"type": "tool_use", "id": "toolu_x", "name": "show_preferences", "input": {}},
            ])
        )
        fresh_state.conversation_history.append(
            ConversationTurn(role="user", content=[
                {"type": "tool_result", "tool_use_id": "toolu_x", "content": "prefs here"},
            ])
        )
        # A few more text turns
        fresh_state.conversation_history.append(
            ConversationTurn(role="assistant", content="Here are your prefs.")
        )

        engine = ConversationEngine(fresh_state, claude=mock_claude)
        await engine.handle_message("one more")

        history = fresh_state.conversation_history
        # First turn should be a user text message, not an orphaned tool_result
        assert history[0].role == "user"
        assert isinstance(history[0].content, str)

    @pytest.mark.asyncio
    async def test_backward_compatible_history_loading(self):
        """Old string-only history loads correctly with the new Union type."""
        state = ChatState(chat_id=12345)
        state.conversation_history = [
            ConversationTurn(role="user", content="Hi"),
            ConversationTurn(role="assistant", content="Hello!"),
        ]

        # JSON roundtrip
        data = state.model_dump_json()
        restored = ChatState.model_validate_json(data)

        assert len(restored.conversation_history) == 2
        assert restored.conversation_history[0].content == "Hi"
        assert isinstance(restored.conversation_history[0].content, str)

    @pytest.mark.asyncio
    async def test_history_json_roundtrip_with_structured_content(self):
        """Structured content (tool_use/tool_result) survives JSON serialization."""
        state = ChatState(chat_id=12345)
        state.conversation_history = [
            ConversationTurn(role="user", content="search apartments"),
            ConversationTurn(role="assistant", content=[
                {"type": "text", "text": "Searching..."},
                {"type": "tool_use", "id": "toolu_abc", "name": "search_apartments", "input": {}},
            ]),
            ConversationTurn(role="user", content=[
                {"type": "tool_result", "tool_use_id": "toolu_abc", "content": "Search triggered."},
            ]),
            ConversationTurn(role="assistant", content="Found some listings!"),
        ]

        data = state.model_dump_json()
        restored = ChatState.model_validate_json(data)

        assert len(restored.conversation_history) == 4
        assert isinstance(restored.conversation_history[1].content, list)
        assert restored.conversation_history[1].content[1]["name"] == "search_apartments"
        assert isinstance(restored.conversation_history[2].content, list)
        assert restored.conversation_history[2].content[0]["type"] == "tool_result"
        assert isinstance(restored.conversation_history[3].content, str)


class TestGroupChatConversation:
    @pytest.mark.asyncio
    async def test_handle_message_with_sender_name(self, fresh_state):
        """Messages with sender_name are prefixed with [Name] in history."""
        mock_claude = AsyncMock(spec=ClaudeClient)
        mock_claude.chat = AsyncMock(return_value=ChatResult(
            text="Got it, John!", tool_messages=[]
        ))

        engine = ConversationEngine(fresh_state, claude=mock_claude)
        await engine.handle_message("looking for 2BR", sender_name="John")

        assert fresh_state.conversation_history[0].content == "[John]: looking for 2BR"
        assert fresh_state.conversation_history[0].sender_name == "John"

    @pytest.mark.asyncio
    async def test_handle_message_without_sender_name(self, fresh_state):
        """Messages without sender_name store text unchanged (backward compatible)."""
        mock_claude = AsyncMock(spec=ClaudeClient)
        mock_claude.chat = AsyncMock(return_value=ChatResult(
            text="Hello!", tool_messages=[]
        ))

        engine = ConversationEngine(fresh_state, claude=mock_claude)
        await engine.handle_message("looking for 2BR")

        assert fresh_state.conversation_history[0].content == "looking for 2BR"
        assert fresh_state.conversation_history[0].sender_name is None

    def test_system_prompt_includes_group_context(self, fresh_state):
        """System prompt includes group chat context when is_group is True."""
        fresh_state.is_group = True
        engine = ConversationEngine(fresh_state)
        prompt = engine._build_system_prompt()

        assert "GROUP CHAT CONTEXT" in prompt
        assert "prefixed with [Name]" in prompt
        assert "consensus" in prompt

    def test_system_prompt_omits_group_context(self, fresh_state):
        """System prompt omits group context when is_group is False."""
        engine = ConversationEngine(fresh_state)
        prompt = engine._build_system_prompt()

        assert "GROUP CHAT CONTEXT" not in prompt


class TestResponseEscaping:
    @pytest.mark.asyncio
    async def test_claude_response_html_escaped(self, fresh_state):
        """Claude response with < and & characters is escaped in Response."""
        mock_claude = AsyncMock(spec=ClaudeClient)
        mock_claude.chat = AsyncMock(return_value=ChatResult(
            text="Try searching for price < $3000 & location = 'East Village'",
            tool_messages=[],
        ))

        engine = ConversationEngine(fresh_state, claude=mock_claude)
        result = await engine.handle_message("What should I search?")

        assert "&lt;" in result.responses[0].text
        assert "&amp;" in result.responses[0].text
        assert "<" not in result.responses[0].text.replace("&lt;", "").replace("&amp;", "")

    @pytest.mark.asyncio
    async def test_error_response_html_escaped(self, fresh_state):
        """Error fallback response is also HTML-escaped."""
        mock_claude = AsyncMock(spec=ClaudeClient)
        mock_claude.chat = AsyncMock(side_effect=Exception("API error"))

        engine = ConversationEngine(fresh_state, claude=mock_claude)
        result = await engine.handle_message("Hello")

        # Error message doesn't contain special chars, but should still be escaped
        assert len(result.responses) == 1
        # Verify the escape function was applied (no raw < or & in error message)
        text = result.responses[0].text
        assert "trouble" in text.lower()


class TestToolDispatchDict:
    def test_all_tool_names_registered(self, fresh_state):
        """Every tool in TOOLS must have a handler in _tool_dispatch."""
        from src.conversation import TOOLS

        engine = ConversationEngine(fresh_state)
        tool_names = {t["name"] for t in TOOLS}
        dispatch_names = set(engine._tool_dispatch.keys())

        assert tool_names == dispatch_names, (
            f"Mismatch — in TOOLS but not dispatch: {tool_names - dispatch_names}; "
            f"in dispatch but not TOOLS: {dispatch_names - tool_names}"
        )

    def test_unknown_tool_returns_error(self, fresh_state):
        """Unknown tool names return an error string."""
        engine = ConversationEngine(fresh_state)
        result = ConversationResult()
        msg = engine._execute_tool("nonexistent_tool", {}, result)
        assert "Unknown tool" in msg
