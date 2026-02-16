"""Tests for the smart filtering + scoring pipeline in scanner.py."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.models import ChatState, CurrentApartment, Listing, Preferences
from src.scanner import (
    NEIGHBORHOOD_ALIASES,
    SCORE_FLOOR,
    ScoringResult,
    _llm_score_listings,
    _neighborhood_pre_filter,
    scan_for_chat,
)


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
        match_score=None,
        scraped_at=datetime.now(timezone.utc),
    )
    defaults.update(overrides)
    return Listing(**defaults)


def _raw_listing(listing_id: str, **overrides) -> dict:
    """Helper to create a raw listing dict as returned by ApifyScraper."""
    defaults = dict(
        listing_id=listing_id,
        url=f"https://streeteasy.com/rental/{listing_id}",
        address=f"123 Test St #{listing_id}",
        neighborhood="Chelsea",
        price=3500,
        bedrooms=2,
        bathrooms=1.0,
        photos=[],
        amenities=[],
    )
    defaults.update(overrides)
    return defaults


# ---------------------------------------------------------------------------
# A) Neighborhood pre-filter tests
# ---------------------------------------------------------------------------


class TestNeighborhoodPreFilter:
    def test_exact_match(self):
        """Listings matching preferred neighborhoods are kept; others dropped."""
        prefs = Preferences(neighborhoods=["Chelsea", "SoHo"])
        listings = [
            _make_listing("1", neighborhood="Chelsea"),
            _make_listing("2", neighborhood="SoHo"),
            _make_listing("3", neighborhood="Harlem"),
        ]
        result = _neighborhood_pre_filter(listings, prefs)
        assert len(result) == 2
        assert {l.listing_id for l in result} == {"1", "2"}

    def test_alias_match(self):
        """Listing with alias neighborhood ('West Chelsea') kept when prefs has 'Chelsea'."""
        prefs = Preferences(neighborhoods=["Chelsea"])
        listings = [
            _make_listing("1", neighborhood="West Chelsea"),
            _make_listing("2", neighborhood="Midtown"),
        ]
        result = _neighborhood_pre_filter(listings, prefs)
        assert len(result) == 1
        assert result[0].listing_id == "1"

    def test_case_insensitive(self):
        """Matching is case-insensitive ('east village' matches 'East Village')."""
        prefs = Preferences(neighborhoods=["east village"])
        listings = [
            _make_listing("1", neighborhood="East Village"),
            _make_listing("2", neighborhood="EAST VILLAGE"),
        ]
        result = _neighborhood_pre_filter(listings, prefs)
        assert len(result) == 2

    def test_no_neighborhoods_passes_all(self):
        """When prefs.neighborhoods is empty, all listings pass through."""
        prefs = Preferences(neighborhoods=[])
        listings = [
            _make_listing("1", neighborhood="Chelsea"),
            _make_listing("2", neighborhood="Harlem"),
            _make_listing("3", neighborhood="Bushwick"),
        ]
        result = _neighborhood_pre_filter(listings, prefs)
        assert len(result) == 3

    def test_empty_listings_returns_empty(self):
        """Empty input returns empty output."""
        prefs = Preferences(neighborhoods=["Chelsea"])
        result = _neighborhood_pre_filter([], prefs)
        assert result == []

    def test_all_filtered_out(self):
        """All listings from non-matching neighborhoods returns empty list."""
        prefs = Preferences(neighborhoods=["Chelsea"])
        listings = [
            _make_listing("1", neighborhood="Harlem"),
            _make_listing("2", neighborhood="Bushwick"),
            _make_listing("3", neighborhood="Astoria"),
        ]
        result = _neighborhood_pre_filter(listings, prefs)
        assert result == []

    def test_mixed_match_and_nonmatch(self):
        """Only matching listings survive; non-matching are dropped."""
        prefs = Preferences(neighborhoods=["Chelsea", "Williamsburg"])
        listings = [
            _make_listing("1", neighborhood="Chelsea"),
            _make_listing("2", neighborhood="Harlem"),
            _make_listing("3", neighborhood="Williamsburg"),
            _make_listing("4", neighborhood="Financial District"),
            _make_listing("5", neighborhood="North Williamsburg"),  # alias
        ]
        result = _neighborhood_pre_filter(listings, prefs)
        ids = {l.listing_id for l in result}
        assert ids == {"1", "3", "5"}

    def test_multiple_aliases_to_same_canonical(self):
        """Multiple alias neighborhoods all map to the same canonical name."""
        prefs = Preferences(neighborhoods=["Upper East Side"])
        listings = [
            _make_listing("1", neighborhood="Yorkville"),
            _make_listing("2", neighborhood="Lenox Hill"),
            _make_listing("3", neighborhood="Carnegie Hill"),
            _make_listing("4", neighborhood="Upper East Side"),
            _make_listing("5", neighborhood="SoHo"),
        ]
        result = _neighborhood_pre_filter(listings, prefs)
        ids = {l.listing_id for l in result}
        assert ids == {"1", "2", "3", "4"}


# ---------------------------------------------------------------------------
# B) LLM filter + score integration tests
# ---------------------------------------------------------------------------


def _mock_llm_response(scores_json: list[dict]) -> MagicMock:
    """Build a mock Anthropic response with the given scores JSON."""
    mock_response = MagicMock()
    mock_response.content = [MagicMock(text=json.dumps(scores_json))]
    return mock_response


def _mock_anthropic_client(response: MagicMock) -> AsyncMock:
    """Build a mock AsyncAnthropic client returning the given response."""
    mock_client = AsyncMock()
    mock_client.messages.create = AsyncMock(return_value=response)
    return mock_client


class TestLLMScoring:
    @pytest.mark.asyncio
    async def test_hard_constraint_exclude(self):
        """Listings with include=false are excluded from results."""
        listings = [
            _make_listing("1", price=3000),
            _make_listing("2", price=5000),
        ]
        prefs = Preferences(budget_max=4000)

        scores = [
            {"id": "1", "include": True, "score": 80, "pros": ["great price"], "cons": []},
            {"id": "2", "include": False, "score": 30, "pros": [], "cons": ["over budget"]},
        ]
        mock_client = _mock_anthropic_client(_mock_llm_response(scores))

        with patch("src.scanner.anthropic.AsyncAnthropic", return_value=mock_client):
            scoring_result = await _llm_score_listings(listings, prefs)
            result = scoring_result.listings

        assert len(result) == 1
        assert result[0].listing_id == "1"
        assert scoring_result.is_fallback is False

    @pytest.mark.asyncio
    async def test_soft_constraint_scoring(self):
        """Included listings are sorted by score (highest first)."""
        listings = [
            _make_listing("1", price=3000),
            _make_listing("2", price=3200),
            _make_listing("3", price=3500),
        ]
        prefs = Preferences(budget_max=4000)

        scores = [
            {"id": "1", "include": True, "score": 60, "pros": ["ok"], "cons": []},
            {"id": "2", "include": True, "score": 90, "pros": ["best"], "cons": []},
            {"id": "3", "include": True, "score": 75, "pros": ["good"], "cons": []},
        ]
        mock_client = _mock_anthropic_client(_mock_llm_response(scores))

        with patch("src.scanner.anthropic.AsyncAnthropic", return_value=mock_client):
            scoring_result = await _llm_score_listings(listings, prefs)
            result = scoring_result.listings

        assert len(result) == 3
        # Verify scores assigned correctly
        score_by_id = {l.listing_id: l.match_score for l in result}
        assert score_by_id["1"] == 60
        assert score_by_id["2"] == 90
        assert score_by_id["3"] == 75

    @pytest.mark.asyncio
    async def test_score_floor(self):
        """Listings with include=true but score < SCORE_FLOOR are excluded."""
        listings = [
            _make_listing("1", price=3000),
            _make_listing("2", price=3500),
        ]
        prefs = Preferences(budget_max=4000)

        scores = [
            {"id": "1", "include": True, "score": 80, "pros": ["good"], "cons": []},
            {"id": "2", "include": True, "score": 15, "pros": ["cheap"], "cons": ["bad"]},
        ]
        mock_client = _mock_anthropic_client(_mock_llm_response(scores))

        with patch("src.scanner.anthropic.AsyncAnthropic", return_value=mock_client):
            scoring_result = await _llm_score_listings(listings, prefs)
            result = scoring_result.listings

        assert len(result) == 1
        assert result[0].listing_id == "1"
        # Confirm SCORE_FLOOR is 25
        assert SCORE_FLOOR == 25

    @pytest.mark.asyncio
    async def test_score_exactly_at_floor(self):
        """Listing with score exactly at SCORE_FLOOR is included."""
        listings = [_make_listing("1", price=3000)]
        prefs = Preferences(budget_max=4000)

        scores = [
            {"id": "1", "include": True, "score": SCORE_FLOOR, "pros": ["ok"], "cons": []},
        ]
        mock_client = _mock_anthropic_client(_mock_llm_response(scores))

        with patch("src.scanner.anthropic.AsyncAnthropic", return_value=mock_client):
            scoring_result = await _llm_score_listings(listings, prefs)
            result = scoring_result.listings

        assert len(result) == 1
        assert result[0].match_score == SCORE_FLOOR

    @pytest.mark.asyncio
    async def test_constraint_context_in_prompt(self):
        """When constraint_context is set, its text appears in the prompt sent to Claude."""
        listings = [_make_listing("1")]
        prefs = Preferences(
            budget_max=4000,
            constraint_context="Budget is firm. Neighborhood is flexible.",
        )

        scores = [
            {"id": "1", "include": True, "score": 70, "pros": ["ok"], "cons": []},
        ]
        mock_client = _mock_anthropic_client(_mock_llm_response(scores))

        with patch("src.scanner.anthropic.AsyncAnthropic", return_value=mock_client):
            scoring_result = await _llm_score_listings(listings, prefs)
            result = scoring_result.listings

            # Verify the prompt sent to Claude contains constraint_context
            call_kwargs = mock_client.messages.create.call_args
            messages = call_kwargs.kwargs.get("messages") or call_kwargs[1].get("messages")
            prompt_text = messages[0]["content"]
            assert "Budget is firm. Neighborhood is flexible." in prompt_text

    @pytest.mark.asyncio
    async def test_no_constraint_context(self):
        """Scoring works correctly when constraint_context is None."""
        listings = [_make_listing("1"), _make_listing("2")]
        prefs = Preferences(budget_max=4000, constraint_context=None)

        scores = [
            {"id": "1", "include": True, "score": 80, "pros": ["good"], "cons": []},
            {"id": "2", "include": True, "score": 60, "pros": ["ok"], "cons": []},
        ]
        mock_client = _mock_anthropic_client(_mock_llm_response(scores))

        with patch("src.scanner.anthropic.AsyncAnthropic", return_value=mock_client):
            scoring_result = await _llm_score_listings(listings, prefs)
            result = scoring_result.listings

        assert len(result) == 2
        # Verify constraint_context text is NOT in the prompt
        call_kwargs = mock_client.messages.create.call_args
        messages = call_kwargs.kwargs.get("messages") or call_kwargs[1].get("messages")
        prompt_text = messages[0]["content"]
        assert "Constraint context" not in prompt_text

    @pytest.mark.asyncio
    async def test_llm_omits_listing(self):
        """Listing not in LLM response is included with default score 50."""
        listings = [
            _make_listing("1"),
            _make_listing("2"),
            _make_listing("3"),
        ]
        prefs = Preferences(budget_max=4000)

        # LLM only returns scores for listings 1 and 3, omitting 2
        scores = [
            {"id": "1", "include": True, "score": 80, "pros": ["great"], "cons": []},
            {"id": "3", "include": True, "score": 60, "pros": ["ok"], "cons": []},
        ]
        mock_client = _mock_anthropic_client(_mock_llm_response(scores))

        with patch("src.scanner.anthropic.AsyncAnthropic", return_value=mock_client):
            scoring_result = await _llm_score_listings(listings, prefs)
            result = scoring_result.listings

        assert len(result) == 3
        omitted = [l for l in result if l.listing_id == "2"][0]
        assert omitted.match_score == 50

    @pytest.mark.asyncio
    async def test_llm_api_failure(self):
        """API error returns all listings unscored, sorted by price ascending."""
        listings = [
            _make_listing("1", price=4000),
            _make_listing("2", price=2500),
            _make_listing("3", price=3200),
        ]
        prefs = Preferences(budget_max=5000)

        mock_client = AsyncMock()
        mock_client.messages.create = AsyncMock(
            side_effect=Exception("API connection error")
        )

        with patch("src.scanner.anthropic.AsyncAnthropic", return_value=mock_client):
            scoring_result = await _llm_score_listings(listings, prefs)
            result = scoring_result.listings

        # All listings returned
        assert len(result) == 3
        # Sorted by price ascending
        prices = [l.price for l in result]
        assert prices == [2500, 3200, 4000]

    @pytest.mark.asyncio
    async def test_json_parse_failure(self):
        """Invalid JSON from LLM returns all listings unscored, sorted by price."""
        listings = [
            _make_listing("1", price=3800),
            _make_listing("2", price=2900),
        ]
        prefs = Preferences(budget_max=4000)

        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="This is not valid JSON at all!!!")]
        mock_client = _mock_anthropic_client(mock_response)

        with patch("src.scanner.anthropic.AsyncAnthropic", return_value=mock_client):
            scoring_result = await _llm_score_listings(listings, prefs)
            result = scoring_result.listings

        assert len(result) == 2
        prices = [l.price for l in result]
        assert prices == [2900, 3800]

    @pytest.mark.asyncio
    async def test_all_excluded_fallback(self):
        """When all listings are excluded, top 3 by score returned as fallback."""
        listings = [
            _make_listing("1", price=5000),
            _make_listing("2", price=6000),
            _make_listing("3", price=7000),
            _make_listing("4", price=8000),
            _make_listing("5", price=9000),
        ]
        prefs = Preferences(budget_max=4000)

        scores = [
            {"id": "1", "include": False, "score": 40, "pros": [], "cons": ["over budget"]},
            {"id": "2", "include": False, "score": 35, "pros": [], "cons": ["over budget"]},
            {"id": "3", "include": False, "score": 50, "pros": [], "cons": ["way over"]},
            {"id": "4", "include": False, "score": 20, "pros": [], "cons": ["way over"]},
            {"id": "5", "include": False, "score": 10, "pros": [], "cons": ["way over"]},
        ]
        mock_client = _mock_anthropic_client(_mock_llm_response(scores))

        with patch("src.scanner.anthropic.AsyncAnthropic", return_value=mock_client):
            scoring_result = await _llm_score_listings(listings, prefs)
            result = scoring_result.listings

        # Exactly 3 returned (top 3 by score)
        assert len(result) == 3
        result_ids = {l.listing_id for l in result}
        # Top 3 scores: listing 3 (50), listing 1 (40), listing 2 (35)
        assert result_ids == {"1", "2", "3"}
        # Fallback flag must be set
        assert scoring_result.is_fallback is True

    @pytest.mark.asyncio
    async def test_pros_cons_assigned(self):
        """Pros and cons from LLM response are assigned to listings."""
        listings = [_make_listing("1")]
        prefs = Preferences(budget_max=4000)

        scores = [
            {
                "id": "1",
                "include": True,
                "score": 85,
                "pros": ["great location", "no fee", "laundry"],
                "cons": ["small kitchen", "no elevator"],
            },
        ]
        mock_client = _mock_anthropic_client(_mock_llm_response(scores))

        with patch("src.scanner.anthropic.AsyncAnthropic", return_value=mock_client):
            scoring_result = await _llm_score_listings(listings, prefs)
            result = scoring_result.listings

        assert result[0].pros == ["great location", "no fee", "laundry"]
        # Cons are capped at 2
        assert result[0].cons == ["small kitchen", "no elevator"]

    @pytest.mark.asyncio
    async def test_score_clamped_to_range(self):
        """Scores outside 0-100 are clamped."""
        listings = [_make_listing("1"), _make_listing("2")]
        prefs = Preferences(budget_max=4000)

        scores = [
            {"id": "1", "include": True, "score": 150, "pros": [], "cons": []},
            {"id": "2", "include": True, "score": -10, "pros": [], "cons": []},
        ]
        mock_client = _mock_anthropic_client(_mock_llm_response(scores))

        with patch("src.scanner.anthropic.AsyncAnthropic", return_value=mock_client):
            scoring_result = await _llm_score_listings(listings, prefs)
            result = scoring_result.listings

        score_by_id = {l.listing_id: l.match_score for l in result}
        assert score_by_id["1"] == 100
        # score -10 clamped to 0, which is < SCORE_FLOOR, so excluded from normal
        # results. But since listing "2" is included by LLM, it just won't pass floor.
        # Listing "1" with score 100 is the only one that passes.
        # Check: listing "2" has score 0, which is < 25 (SCORE_FLOOR).
        assert len(result) == 1
        assert result[0].listing_id == "1"

    @pytest.mark.asyncio
    async def test_markdown_code_fence_stripped(self):
        """LLM response wrapped in markdown code fences is handled correctly."""
        listings = [_make_listing("1")]
        prefs = Preferences(budget_max=4000)

        json_text = '[{"id":"1","include":true,"score":75,"pros":["nice"],"cons":[]}]'
        fenced_text = f"```json\n{json_text}\n```"

        mock_response = MagicMock()
        mock_response.content = [MagicMock(text=fenced_text)]
        mock_client = _mock_anthropic_client(mock_response)

        with patch("src.scanner.anthropic.AsyncAnthropic", return_value=mock_client):
            scoring_result = await _llm_score_listings(listings, prefs)
            result = scoring_result.listings

        assert len(result) == 1
        assert result[0].match_score == 75

    @pytest.mark.asyncio
    async def test_empty_listings_returns_empty(self):
        """Empty listing input returns empty without calling LLM."""
        prefs = Preferences(budget_max=4000)
        scoring_result = await _llm_score_listings([], prefs)
        assert scoring_result.listings == []
        assert scoring_result.is_fallback is False

    @pytest.mark.asyncio
    async def test_current_apartment_in_prompt(self):
        """Current apartment context is included in the LLM prompt when provided."""
        listings = [_make_listing("1")]
        prefs = Preferences(budget_max=4000)
        current_apt = CurrentApartment(
            price=3000,
            neighborhood="East Village",
            pros=["great light"],
            cons=["noisy street"],
        )

        scores = [
            {"id": "1", "include": True, "score": 70, "pros": ["ok"], "cons": []},
        ]
        mock_client = _mock_anthropic_client(_mock_llm_response(scores))

        with patch("src.scanner.anthropic.AsyncAnthropic", return_value=mock_client):
            scoring_result = await _llm_score_listings(listings, prefs, current_apt)
            result = scoring_result.listings

            call_kwargs = mock_client.messages.create.call_args
            messages = call_kwargs.kwargs.get("messages") or call_kwargs[1].get("messages")
            prompt_text = messages[0]["content"]
            assert "$3,000/mo" in prompt_text
            assert "East Village" in prompt_text
            assert "great light" in prompt_text
            assert "noisy street" in prompt_text


# ---------------------------------------------------------------------------
# C) End-to-end scan_for_chat tests
# ---------------------------------------------------------------------------


class TestScanForChat:
    def _make_state(self, **overrides) -> ChatState:
        """Build a ChatState ready for scanning."""
        state = ChatState(chat_id=12345)
        state.preferences.budget_max = 4000
        state.preferences.neighborhoods = ["Chelsea", "SoHo"]
        state.preferences_ready = True
        for key, val in overrides.items():
            setattr(state, key, val)
        return state

    @pytest.mark.asyncio
    async def test_neighborhood_filter_applied(self):
        """Raw listings from wrong neighborhoods are filtered before LLM scoring."""
        state = self._make_state()

        raw_listings = [
            _raw_listing("1", neighborhood="Chelsea"),
            _raw_listing("2", neighborhood="Harlem"),
            _raw_listing("3", neighborhood="SoHo"),
        ]

        mock_scraper = AsyncMock()
        mock_scraper.search_streeteasy = AsyncMock(return_value=raw_listings)

        mock_bot = AsyncMock()
        mock_bot.send_text = AsyncMock()
        mock_bot.send_listing_photo = AsyncMock()

        scored_listings = [
            _make_listing("1", neighborhood="Chelsea", match_score=80, photos=[]),
            _make_listing("3", neighborhood="SoHo", match_score=70, photos=[]),
        ]

        with (
            patch("src.scanner.save_state"),
            patch(
                "src.scanner._llm_score_listings",
                new_callable=AsyncMock,
                return_value=ScoringResult(listings=scored_listings),
            ) as mock_llm,
        ):
            await scan_for_chat(mock_scraper, mock_bot, state)

            # Verify _llm_score_listings was called with only the filtered listings
            call_args = mock_llm.call_args
            filtered_input = call_args[0][0]
            neighborhoods = {l.neighborhood for l in filtered_input}
            assert "Harlem" not in neighborhoods
            assert "Chelsea" in neighborhoods
            assert "SoHo" in neighborhoods

    @pytest.mark.asyncio
    async def test_zero_results_after_prefilter(self):
        """All listings filtered by neighborhood pre-filter triggers appropriate message."""
        state = self._make_state()

        raw_listings = [
            _raw_listing("1", neighborhood="Harlem"),
            _raw_listing("2", neighborhood="Bushwick"),
        ]

        mock_scraper = AsyncMock()
        mock_scraper.search_streeteasy = AsyncMock(return_value=raw_listings)

        mock_bot = AsyncMock()
        mock_bot.send_text = AsyncMock()

        with patch("src.scanner.save_state"):
            await scan_for_chat(mock_scraper, mock_bot, state)

        # Verify the "no listings in your neighborhoods" message was sent
        sent_texts = [
            call.args[1] if len(call.args) > 1 else call.kwargs.get("text", "")
            for call in mock_bot.send_text.call_args_list
        ]
        assert any("No listings found in your neighborhoods" in t for t in sent_texts)

    @pytest.mark.asyncio
    async def test_scored_listings_sent(self):
        """Scored listings are sent to telegram as listing cards."""
        state = self._make_state()

        raw_listings = [
            _raw_listing("1", neighborhood="Chelsea"),
            _raw_listing("2", neighborhood="SoHo"),
        ]

        mock_scraper = AsyncMock()
        mock_scraper.search_streeteasy = AsyncMock(return_value=raw_listings)

        mock_bot = AsyncMock()
        mock_bot.send_text = AsyncMock()
        mock_bot.send_listing_photo = AsyncMock()

        scored_listings = [
            _make_listing("1", neighborhood="Chelsea", match_score=80, photos=["http://img1.jpg"]),
            _make_listing("2", neighborhood="SoHo", match_score=70, photos=["http://img2.jpg"]),
        ]

        with (
            patch("src.scanner.save_state"),
            patch(
                "src.scanner._llm_score_listings",
                new_callable=AsyncMock,
                return_value=ScoringResult(listings=scored_listings),
            ),
        ):
            await scan_for_chat(mock_scraper, mock_bot, state)

        # scan header + 2 listing photos
        assert mock_bot.send_text.call_count >= 1  # at least the header
        assert mock_bot.send_listing_photo.call_count == 2

    @pytest.mark.asyncio
    async def test_results_stored_in_recent(self):
        """Scored listings are stored in state.recent_listings."""
        state = self._make_state()

        raw_listings = [_raw_listing("1", neighborhood="Chelsea")]

        mock_scraper = AsyncMock()
        mock_scraper.search_streeteasy = AsyncMock(return_value=raw_listings)

        mock_bot = AsyncMock()
        mock_bot.send_text = AsyncMock()
        mock_bot.send_listing_photo = AsyncMock()

        scored_listings = [
            _make_listing("1", neighborhood="Chelsea", match_score=85),
        ]

        with (
            patch("src.scanner.save_state"),
            patch(
                "src.scanner._llm_score_listings",
                new_callable=AsyncMock,
                return_value=ScoringResult(listings=scored_listings),
            ),
        ):
            await scan_for_chat(mock_scraper, mock_bot, state)

        assert "1" in state.recent_listings
        assert state.recent_listings["1"].match_score == 85

    @pytest.mark.asyncio
    async def test_no_raw_results(self):
        """Empty Apify results sends scan header with 0 count."""
        state = self._make_state()

        mock_scraper = AsyncMock()
        mock_scraper.search_streeteasy = AsyncMock(return_value=[])

        mock_bot = AsyncMock()
        mock_bot.send_text = AsyncMock()

        with patch("src.scanner.save_state"):
            await scan_for_chat(mock_scraper, mock_bot, state)

        # Should have sent the "0 results" header
        mock_bot.send_text.assert_called_once()
        sent_text = mock_bot.send_text.call_args[0][1]
        assert "No new listings" in sent_text or "0" in sent_text

    @pytest.mark.asyncio
    async def test_deduplication(self):
        """Listings already in seen_listing_ids are deduplicated."""
        state = self._make_state()
        state.seen_listing_ids = {"1"}  # listing 1 already seen

        raw_listings = [
            _raw_listing("1", neighborhood="Chelsea"),
            _raw_listing("2", neighborhood="Chelsea"),
        ]

        mock_scraper = AsyncMock()
        mock_scraper.search_streeteasy = AsyncMock(return_value=raw_listings)

        mock_bot = AsyncMock()
        mock_bot.send_text = AsyncMock()
        mock_bot.send_listing_photo = AsyncMock()

        scored_listings = [
            _make_listing("2", neighborhood="Chelsea", match_score=75),
        ]

        with (
            patch("src.scanner.save_state"),
            patch(
                "src.scanner._llm_score_listings",
                new_callable=AsyncMock,
                return_value=ScoringResult(listings=scored_listings),
            ) as mock_llm,
        ):
            await scan_for_chat(mock_scraper, mock_bot, state)

            # Only listing "2" should have been passed to LLM scoring
            call_args = mock_llm.call_args
            filtered_input = call_args[0][0]
            assert len(filtered_input) == 1
            assert filtered_input[0].listing_id == "2"

        # Both IDs should now be in seen_listing_ids
        assert "1" in state.seen_listing_ids
        assert "2" in state.seen_listing_ids

    @pytest.mark.asyncio
    async def test_listings_without_photos_sent_as_text(self):
        """Listings with no photos are sent as text messages, not photos."""
        state = self._make_state()

        raw_listings = [_raw_listing("1", neighborhood="Chelsea")]

        mock_scraper = AsyncMock()
        mock_scraper.search_streeteasy = AsyncMock(return_value=raw_listings)

        mock_bot = AsyncMock()
        mock_bot.send_text = AsyncMock()
        mock_bot.send_listing_photo = AsyncMock()

        scored_listings = [
            _make_listing("1", neighborhood="Chelsea", match_score=80, photos=[]),
        ]

        with (
            patch("src.scanner.save_state"),
            patch(
                "src.scanner._llm_score_listings",
                new_callable=AsyncMock,
                return_value=ScoringResult(listings=scored_listings),
            ),
        ):
            await scan_for_chat(mock_scraper, mock_bot, state)

        # Photo not sent (no photos); text used instead
        assert mock_bot.send_listing_photo.call_count == 0
        # Header + listing card = 2 text sends
        assert mock_bot.send_text.call_count == 2

    @pytest.mark.asyncio
    async def test_search_error_sends_error_message(self):
        """StreetEasy search error sends user-friendly error message."""
        state = self._make_state()

        mock_scraper = AsyncMock()
        mock_scraper.search_streeteasy = AsyncMock(
            side_effect=Exception("Apify timeout")
        )

        mock_bot = AsyncMock()
        mock_bot.send_text = AsyncMock()

        await scan_for_chat(mock_scraper, mock_bot, state)

        mock_bot.send_text.assert_called_once()
        sent_text = mock_bot.send_text.call_args[0][1]
        assert "trouble searching StreetEasy" in sent_text

    @pytest.mark.asyncio
    async def test_fallback_caveat_sent(self):
        """When scoring result is_fallback=True, caveat message is sent."""
        state = self._make_state()

        raw_listings = [_raw_listing("1", neighborhood="Chelsea")]

        mock_scraper = AsyncMock()
        mock_scraper.search_streeteasy = AsyncMock(return_value=raw_listings)

        mock_bot = AsyncMock()
        mock_bot.send_text = AsyncMock()
        mock_bot.send_listing_photo = AsyncMock()

        scored_listings = [
            _make_listing("1", neighborhood="Chelsea", match_score=40, photos=[]),
        ]

        with (
            patch("src.scanner.save_state"),
            patch(
                "src.scanner._llm_score_listings",
                new_callable=AsyncMock,
                return_value=ScoringResult(listings=scored_listings, is_fallback=True),
            ),
        ):
            await scan_for_chat(mock_scraper, mock_bot, state)

        sent_texts = [
            call.args[1] if len(call.args) > 1 else call.kwargs.get("text", "")
            for call in mock_bot.send_text.call_args_list
        ]
        assert any("None of these perfectly matched" in t for t in sent_texts)

    @pytest.mark.asyncio
    async def test_no_caveat_when_not_fallback(self):
        """When scoring result is_fallback=False, no caveat message is sent."""
        state = self._make_state()

        raw_listings = [_raw_listing("1", neighborhood="Chelsea")]

        mock_scraper = AsyncMock()
        mock_scraper.search_streeteasy = AsyncMock(return_value=raw_listings)

        mock_bot = AsyncMock()
        mock_bot.send_text = AsyncMock()
        mock_bot.send_listing_photo = AsyncMock()

        scored_listings = [
            _make_listing("1", neighborhood="Chelsea", match_score=80, photos=[]),
        ]

        with (
            patch("src.scanner.save_state"),
            patch(
                "src.scanner._llm_score_listings",
                new_callable=AsyncMock,
                return_value=ScoringResult(listings=scored_listings, is_fallback=False),
            ),
        ):
            await scan_for_chat(mock_scraper, mock_bot, state)

        sent_texts = [
            call.args[1] if len(call.args) > 1 else call.kwargs.get("text", "")
            for call in mock_bot.send_text.call_args_list
        ]
        assert not any("None of these perfectly matched" in t for t in sent_texts)
