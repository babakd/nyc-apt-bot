"""Tests for the smart filtering + scoring pipeline in scanner.py."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.models import ChatState, CurrentApartment, Listing, Preferences
from src.apify_scraper import ApifyScraperError
from src.scanner import (
    CACHE_MAX_AGE_HOURS,
    NEIGHBORHOOD_ALIASES,
    SCORE_FLOOR,
    ScoringResult,
    _enrich_listings,
    _extract_listing_detail,
    _get_cached_listings,
    _has_cached_scan,
    _llm_score_listings,
    _neighborhood_pre_filter,
    _parse_listing,
    _pick_hero_photos,
    _sample_photo_keys,
    _vision_pick_heroes,
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
        mock_scraper.search_with_retry = AsyncMock(return_value=raw_listings)

        mock_bot = AsyncMock()
        mock_bot.send_text = AsyncMock()
        mock_bot.send_listing_photo = AsyncMock()

        scored_listings = [
            _make_listing("1", neighborhood="Chelsea", match_score=80, photos=[]),
            _make_listing("3", neighborhood="SoHo", match_score=70, photos=[]),
        ]

        with (
            patch("src.scanner.save_state"),
            patch("src.scanner._pick_hero_photos", new_callable=AsyncMock, return_value={}),
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
        mock_scraper.search_with_retry = AsyncMock(return_value=raw_listings)

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
        mock_scraper.search_with_retry = AsyncMock(return_value=raw_listings)

        mock_bot = AsyncMock()
        mock_bot.send_text = AsyncMock()
        mock_bot.send_listing_photo = AsyncMock()

        scored_listings = [
            _make_listing("1", neighborhood="Chelsea", match_score=80, photos=["http://img1.jpg"]),
            _make_listing("2", neighborhood="SoHo", match_score=70, photos=["http://img2.jpg"]),
        ]

        with (
            patch("src.scanner.save_state"),
            patch("src.scanner._pick_hero_photos", new_callable=AsyncMock, return_value={}),
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
        mock_scraper.search_with_retry = AsyncMock(return_value=raw_listings)

        mock_bot = AsyncMock()
        mock_bot.send_text = AsyncMock()
        mock_bot.send_listing_photo = AsyncMock()

        scored_listings = [
            _make_listing("1", neighborhood="Chelsea", match_score=85),
        ]

        with (
            patch("src.scanner.save_state"),
            patch("src.scanner._pick_hero_photos", new_callable=AsyncMock, return_value={}),
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
        mock_scraper.search_with_retry = AsyncMock(return_value=[])

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
        mock_scraper.search_with_retry = AsyncMock(return_value=raw_listings)

        mock_bot = AsyncMock()
        mock_bot.send_text = AsyncMock()
        mock_bot.send_listing_photo = AsyncMock()

        scored_listings = [
            _make_listing("2", neighborhood="Chelsea", match_score=75),
        ]

        with (
            patch("src.scanner.save_state"),
            patch("src.scanner._pick_hero_photos", new_callable=AsyncMock, return_value={}),
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
        mock_scraper.search_with_retry = AsyncMock(return_value=raw_listings)

        mock_bot = AsyncMock()
        mock_bot.send_text = AsyncMock()
        mock_bot.send_listing_photo = AsyncMock()

        scored_listings = [
            _make_listing("1", neighborhood="Chelsea", match_score=80, photos=[]),
        ]

        with (
            patch("src.scanner.save_state"),
            patch("src.scanner._pick_hero_photos", new_callable=AsyncMock, return_value={}),
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
        mock_scraper.search_with_retry = AsyncMock(
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
        mock_scraper.search_with_retry = AsyncMock(return_value=raw_listings)

        mock_bot = AsyncMock()
        mock_bot.send_text = AsyncMock()
        mock_bot.send_listing_photo = AsyncMock()

        scored_listings = [
            _make_listing("1", neighborhood="Chelsea", match_score=40, photos=[]),
        ]

        with (
            patch("src.scanner.save_state"),
            patch("src.scanner._pick_hero_photos", new_callable=AsyncMock, return_value={}),
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
        mock_scraper.search_with_retry = AsyncMock(return_value=raw_listings)

        mock_bot = AsyncMock()
        mock_bot.send_text = AsyncMock()
        mock_bot.send_listing_photo = AsyncMock()

        scored_listings = [
            _make_listing("1", neighborhood="Chelsea", match_score=80, photos=[]),
        ]

        with (
            patch("src.scanner.save_state"),
            patch("src.scanner._pick_hero_photos", new_callable=AsyncMock, return_value={}),
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

    @pytest.mark.asyncio
    async def test_hero_photo_used_when_available(self):
        """When hero picker returns a URL, it's used instead of photos[0]."""
        state = self._make_state()

        raw_listings = [_raw_listing("1", neighborhood="Chelsea")]

        mock_scraper = AsyncMock()
        mock_scraper.search_with_retry = AsyncMock(return_value=raw_listings)

        mock_bot = AsyncMock()
        mock_bot.send_text = AsyncMock()
        mock_bot.send_listing_photo = AsyncMock()

        scored_listings = [
            _make_listing(
                "1",
                neighborhood="Chelsea",
                match_score=80,
                photos=["http://original.jpg"],
            ),
        ]

        hero_url = "http://hero-picked.jpg"

        with (
            patch("src.scanner.save_state"),
            patch(
                "src.scanner._pick_hero_photos",
                new_callable=AsyncMock,
                return_value={"1": hero_url},
            ),
            patch(
                "src.scanner._llm_score_listings",
                new_callable=AsyncMock,
                return_value=ScoringResult(listings=scored_listings),
            ),
        ):
            await scan_for_chat(mock_scraper, mock_bot, state)

        # Verify the hero photo URL was used
        assert mock_bot.send_listing_photo.call_count == 1
        call_kwargs = mock_bot.send_listing_photo.call_args
        assert call_kwargs.kwargs.get("photo_url") or call_kwargs[1].get("photo_url") == hero_url


# ---------------------------------------------------------------------------
# D) Hero photo picker tests
# ---------------------------------------------------------------------------


class TestSamplePhotoKeys:
    def test_fewer_than_max(self):
        """Keys fewer than max returned as-is."""
        keys = ["a", "b", "c"]
        assert _sample_photo_keys(keys, max_count=8) == ["a", "b", "c"]

    def test_exactly_max(self):
        """Keys exactly at max returned as-is."""
        keys = list("abcdefgh")
        assert _sample_photo_keys(keys, max_count=8) == list("abcdefgh")

    def test_more_than_max(self):
        """More than max keys are sampled: first 3 + evenly spaced."""
        keys = [str(i) for i in range(20)]
        sampled = _sample_photo_keys(keys, max_count=8)
        assert len(sampled) == 8
        # First 3 are always the first 3
        assert sampled[:3] == ["0", "1", "2"]

    def test_empty(self):
        assert _sample_photo_keys([]) == []


class TestHeroPhotoPicker:
    @pytest.mark.asyncio
    async def test_successful_pick(self):
        """Vision model picks are mapped back to full-size URLs."""
        listings = [
            _make_listing("1", photo_keys=["keyA", "keyB", "keyC"]),
        ]

        mock_response = MagicMock()
        mock_response.content = [MagicMock(text='{"1": "B"}')]
        mock_client = AsyncMock()
        mock_client.messages.create = AsyncMock(return_value=mock_response)

        # Mock httpx downloads
        async def fake_get(url, **kwargs):
            resp = MagicMock()
            resp.status_code = 200
            resp.content = b"\xff\xd8\xff\xe0fake-jpeg-data"
            return resp

        mock_http = AsyncMock()
        mock_http.get = fake_get
        mock_http.__aenter__ = AsyncMock(return_value=mock_http)
        mock_http.__aexit__ = AsyncMock(return_value=False)

        with (
            patch("src.scanner.anthropic.AsyncAnthropic", return_value=mock_client),
            patch("src.scanner.httpx.AsyncClient", return_value=mock_http),
        ):
            result = await _pick_hero_photos(listings)

        assert "1" in result
        assert "keyB" in result["1"]

    @pytest.mark.asyncio
    async def test_api_failure_returns_empty(self):
        """API failure returns empty dict (graceful fallback)."""
        listings = [
            _make_listing("1", photo_keys=["keyA", "keyB"]),
        ]

        mock_client = AsyncMock()
        mock_client.messages.create = AsyncMock(side_effect=Exception("API error"))

        async def fake_get(url, **kwargs):
            resp = MagicMock()
            resp.status_code = 200
            resp.content = b"\xff\xd8\xff\xe0fake"
            return resp

        mock_http = AsyncMock()
        mock_http.get = fake_get
        mock_http.__aenter__ = AsyncMock(return_value=mock_http)
        mock_http.__aexit__ = AsyncMock(return_value=False)

        with (
            patch("src.scanner.anthropic.AsyncAnthropic", return_value=mock_client),
            patch("src.scanner.httpx.AsyncClient", return_value=mock_http),
        ):
            result = await _pick_hero_photos(listings)

        assert result == {}

    @pytest.mark.asyncio
    async def test_single_photo_skipped(self):
        """Listings with fewer than 2 photo_keys are skipped."""
        listings = [
            _make_listing("1", photo_keys=["onlyOne"]),
        ]
        result = await _pick_hero_photos(listings)
        assert result == {}

    @pytest.mark.asyncio
    async def test_no_keys_skipped(self):
        """Listings with no photo_keys are skipped."""
        listings = [
            _make_listing("1", photo_keys=[]),
        ]
        result = await _pick_hero_photos(listings)
        assert result == {}

    @pytest.mark.asyncio
    async def test_invalid_letter_omitted(self):
        """Invalid letter in vision response omits that listing from result."""
        listings = [
            _make_listing("1", photo_keys=["keyA", "keyB", "keyC"]),
        ]

        # Model returns letter "Z" which doesn't exist
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text='{"1": "Z"}')]
        mock_client = AsyncMock()
        mock_client.messages.create = AsyncMock(return_value=mock_response)

        async def fake_get(url, **kwargs):
            resp = MagicMock()
            resp.status_code = 200
            resp.content = b"\xff\xd8\xff\xe0fake"
            return resp

        mock_http = AsyncMock()
        mock_http.get = fake_get
        mock_http.__aenter__ = AsyncMock(return_value=mock_http)
        mock_http.__aexit__ = AsyncMock(return_value=False)

        with (
            patch("src.scanner.anthropic.AsyncAnthropic", return_value=mock_client),
            patch("src.scanner.httpx.AsyncClient", return_value=mock_http),
        ):
            result = await _pick_hero_photos(listings)

        # Letter "Z" doesn't map to any key, so listing "1" should be omitted
        assert "1" not in result

    @pytest.mark.asyncio
    async def test_large_batch_split(self):
        """15 listings are split into 2 API calls (12 + 3)."""
        listings = [
            _make_listing(str(i), photo_keys=[f"k{i}a", f"k{i}b", f"k{i}c"])
            for i in range(15)
        ]

        # Vision model picks photo "A" for every listing
        picks_batch1 = {str(i): "A" for i in range(12)}
        picks_batch2 = {str(i): "A" for i in range(12, 15)}

        call_count = 0

        def make_response(picks):
            resp = MagicMock()
            resp.content = [MagicMock(text=json.dumps(picks))]
            return resp

        async def mock_create(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return make_response(picks_batch1)
            return make_response(picks_batch2)

        mock_client = AsyncMock()
        mock_client.messages.create = AsyncMock(side_effect=mock_create)

        async def fake_get(url, **kwargs):
            resp = MagicMock()
            resp.status_code = 200
            resp.content = b"\xff\xd8\xff\xe0fake"
            return resp

        mock_http = AsyncMock()
        mock_http.get = fake_get
        mock_http.__aenter__ = AsyncMock(return_value=mock_http)
        mock_http.__aexit__ = AsyncMock(return_value=False)

        with (
            patch("src.scanner.anthropic.AsyncAnthropic", return_value=mock_client),
            patch("src.scanner.httpx.AsyncClient", return_value=mock_http),
        ):
            result = await _pick_hero_photos(listings)

        # Should have made 2 API calls (12 + 3 listings)
        assert call_count == 2
        # Should have results for all 15 listings
        assert len(result) == 15

    @pytest.mark.asyncio
    async def test_at_batch_limit(self):
        """12 listings (exactly at batch limit) → 1 API call."""
        listings = [
            _make_listing(str(i), photo_keys=[f"k{i}a", f"k{i}b"])
            for i in range(12)
        ]

        picks = {str(i): "A" for i in range(12)}
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text=json.dumps(picks))]
        mock_client = AsyncMock()
        mock_client.messages.create = AsyncMock(return_value=mock_response)

        async def fake_get(url, **kwargs):
            resp = MagicMock()
            resp.status_code = 200
            resp.content = b"\xff\xd8\xff\xe0fake"
            return resp

        mock_http = AsyncMock()
        mock_http.get = fake_get
        mock_http.__aenter__ = AsyncMock(return_value=mock_http)
        mock_http.__aexit__ = AsyncMock(return_value=False)

        with (
            patch("src.scanner.anthropic.AsyncAnthropic", return_value=mock_client),
            patch("src.scanner.httpx.AsyncClient", return_value=mock_http),
        ):
            result = await _pick_hero_photos(listings)

        # Should have made exactly 1 API call
        assert mock_client.messages.create.call_count == 1
        assert len(result) == 12


# ---------------------------------------------------------------------------
# E) Concession data in _parse_listing tests
# ---------------------------------------------------------------------------


class TestParseListingConcessions:
    def test_concession_data_flows_through(self):
        """net_effective_price and months_free are passed through _parse_listing."""
        raw = _raw_listing(
            "1",
            net_effective_price=15833,
            months_free=2.0,
        )
        listing = _parse_listing(raw)
        assert listing.net_effective_price == 15833
        assert listing.months_free == 2.0

    def test_no_concession_data(self):
        """Missing concession fields default to None."""
        raw = _raw_listing("1")
        listing = _parse_listing(raw)
        assert listing.net_effective_price is None
        assert listing.months_free is None


# ---------------------------------------------------------------------------
# F) Enrichment tests
# ---------------------------------------------------------------------------


class TestExtractListingDetail:
    def test_next_data_extraction(self):
        """Extracts amenities and description from __NEXT_DATA__ JSON."""
        html = '''
        <html><body>
        <script id="__NEXT_DATA__" type="application/json">
        {"props":{"pageProps":{"listingData":{
            "amenities":["Dishwasher","Elevator","Doorman"],
            "description":"Spacious 2BR in Chelsea with great views."
        }}}}
        </script>
        </body></html>
        '''
        amenities, description = _extract_listing_detail(html)
        assert amenities == ["Dishwasher", "Elevator", "Doorman"]
        assert description == "Spacious 2BR in Chelsea with great views."

    def test_json_ld_fallback(self):
        """Falls back to JSON-LD when __NEXT_DATA__ is absent."""
        html = '''
        <html><body>
        <script type="application/ld+json">
        {"@type":"Apartment","amenityFeature":[{"name":"Laundry"},{"name":"Gym"}],
         "description":"Great apartment in SoHo."}
        </script>
        </body></html>
        '''
        amenities, description = _extract_listing_detail(html)
        assert amenities == ["Laundry", "Gym"]
        assert description == "Great apartment in SoHo."

    def test_no_data_returns_empty(self):
        """Returns empty when no structured data is found."""
        html = "<html><body><p>Nothing here</p></body></html>"
        amenities, description = _extract_listing_detail(html)
        assert amenities == []
        assert description is None

    def test_malformed_json_returns_empty(self):
        """Handles malformed JSON gracefully."""
        html = '<html><body><script id="__NEXT_DATA__" type="application/json">{broken</script></body></html>'
        amenities, description = _extract_listing_detail(html)
        assert amenities == []
        assert description is None

    def test_dict_amenities_with_name_key(self):
        """Handles amenities as list of dicts with 'name' key."""
        html = '''
        <html><body>
        <script id="__NEXT_DATA__" type="application/json">
        {"props":{"pageProps":{"listingData":{
            "amenities":[{"name":"Pool"},{"name":"Concierge"}]
        }}}}
        </script>
        </body></html>
        '''
        amenities, description = _extract_listing_detail(html)
        assert amenities == ["Pool", "Concierge"]


class TestEnrichListings:
    @pytest.mark.asyncio
    async def test_enrichment_populates_data(self):
        """Successful enrichment populates amenities and description."""
        listings = [
            _make_listing("1", amenities=[], description=None),
        ]
        listings[0].url = "https://streeteasy.com/rental/1"

        html = '''
        <script id="__NEXT_DATA__" type="application/json">
        {"props":{"pageProps":{"listingData":{
            "amenities":["Dishwasher","Elevator"],
            "description":"Nice place"
        }}}}
        </script>
        '''

        mock_run = {"status": "SUCCEEDED", "defaultDatasetId": "ds123"}
        mock_dataset = AsyncMock()
        mock_dataset.list_items = AsyncMock(return_value=MagicMock(
            items=[{"url": "https://streeteasy.com/rental/1", "html": html}]
        ))

        mock_actor = AsyncMock()
        mock_actor.call = AsyncMock(return_value=mock_run)

        mock_client = AsyncMock()
        mock_client.actor = MagicMock(return_value=mock_actor)
        mock_client.dataset = MagicMock(return_value=mock_dataset)

        with (
            patch("src.scanner.ApifyClientAsync", return_value=mock_client),
            patch.dict("os.environ", {"APIFY_API_TOKEN": "test-token"}),
        ):
            result = await _enrich_listings(listings)

        assert result[0].amenities == ["Dishwasher", "Elevator"]
        assert result[0].description == "Nice place"

    @pytest.mark.asyncio
    async def test_enrichment_failure_graceful(self):
        """Apify failure returns listings unchanged (graceful degradation)."""
        listings = [
            _make_listing("1", amenities=[], description=None),
        ]
        listings[0].url = "https://streeteasy.com/rental/1"

        with (
            patch("src.scanner.ApifyClientAsync", side_effect=Exception("Apify down")),
            patch.dict("os.environ", {"APIFY_API_TOKEN": "test-token"}),
        ):
            result = await _enrich_listings(listings)

        assert len(result) == 1
        assert result[0].amenities == []
        assert result[0].description is None

    @pytest.mark.asyncio
    async def test_enrichment_no_token_skips(self):
        """Missing APIFY_API_TOKEN skips enrichment."""
        listings = [_make_listing("1")]
        with patch.dict("os.environ", {}, clear=True):
            result = await _enrich_listings(listings)
        assert result is listings

    @pytest.mark.asyncio
    async def test_enrichment_preserves_existing_data(self):
        """Enrichment doesn't overwrite existing amenities/description."""
        listings = [
            _make_listing("1", amenities=["Existing"], description="Existing desc"),
        ]
        listings[0].url = "https://streeteasy.com/rental/1"

        html = '''
        <script id="__NEXT_DATA__" type="application/json">
        {"props":{"pageProps":{"listingData":{
            "amenities":["New"],
            "description":"New desc"
        }}}}
        </script>
        '''

        mock_run = {"status": "SUCCEEDED", "defaultDatasetId": "ds123"}
        mock_dataset = AsyncMock()
        mock_dataset.list_items = AsyncMock(return_value=MagicMock(
            items=[{"url": "https://streeteasy.com/rental/1", "html": html}]
        ))

        mock_actor = AsyncMock()
        mock_actor.call = AsyncMock(return_value=mock_run)

        mock_client = AsyncMock()
        mock_client.actor = MagicMock(return_value=mock_actor)
        mock_client.dataset = MagicMock(return_value=mock_dataset)

        with (
            patch("src.scanner.ApifyClientAsync", return_value=mock_client),
            patch.dict("os.environ", {"APIFY_API_TOKEN": "test-token"}),
        ):
            result = await _enrich_listings(listings)

        # Should NOT overwrite existing data
        assert result[0].amenities == ["Existing"]
        assert result[0].description == "Existing desc"


# ---------------------------------------------------------------------------
# G) LLM scoring payload tests (enriched data, concessions, canonical hoods)
# ---------------------------------------------------------------------------


class TestLLMScoringPayload:
    @pytest.mark.asyncio
    async def test_description_in_prompt(self):
        """Description appears in LLM scoring payload (truncated to 300 chars)."""
        long_desc = "A" * 500
        listings = [_make_listing("1", description=long_desc)]
        prefs = Preferences(budget_max=4000)

        scores = [{"id": "1", "include": True, "score": 70, "pros": [], "cons": []}]
        mock_client = _mock_anthropic_client(_mock_llm_response(scores))

        with patch("src.scanner.anthropic.AsyncAnthropic", return_value=mock_client):
            await _llm_score_listings(listings, prefs)

            call_kwargs = mock_client.messages.create.call_args
            messages = call_kwargs.kwargs.get("messages") or call_kwargs[1].get("messages")
            prompt_text = messages[0]["content"]
            # Truncated to 300 chars
            assert "A" * 300 in prompt_text
            assert "A" * 301 not in prompt_text

    @pytest.mark.asyncio
    async def test_concessions_in_prompt(self):
        """Net effective price and months free appear in LLM scoring payload."""
        listings = [_make_listing("1", price=19000, net_effective_price=15833, months_free=2.0)]
        prefs = Preferences(budget_max=16000)

        scores = [{"id": "1", "include": True, "score": 70, "pros": [], "cons": []}]
        mock_client = _mock_anthropic_client(_mock_llm_response(scores))

        with patch("src.scanner.anthropic.AsyncAnthropic", return_value=mock_client):
            await _llm_score_listings(listings, prefs)

            call_kwargs = mock_client.messages.create.call_args
            messages = call_kwargs.kwargs.get("messages") or call_kwargs[1].get("messages")
            prompt_text = messages[0]["content"]
            assert "15833" in prompt_text
            assert "net_effective" in prompt_text

    @pytest.mark.asyncio
    async def test_concessions_not_in_prompt_when_same_price(self):
        """No net effective in listing data when it equals gross price."""
        listings = [_make_listing("1", price=3500, net_effective_price=3500)]
        prefs = Preferences(budget_max=4000)

        scores = [{"id": "1", "include": True, "score": 70, "pros": [], "cons": []}]
        mock_client = _mock_anthropic_client(_mock_llm_response(scores))

        with patch("src.scanner.anthropic.AsyncAnthropic", return_value=mock_client):
            await _llm_score_listings(listings, prefs)

            call_kwargs = mock_client.messages.create.call_args
            messages = call_kwargs.kwargs.get("messages") or call_kwargs[1].get("messages")
            prompt_text = messages[0]["content"]
            # The listing data JSON should not contain "net_effective":3500
            # (prompt instructions will mention "net_effective" but that's expected)
            assert '"net_effective":3500' not in prompt_text

    @pytest.mark.asyncio
    async def test_canonical_neighborhood_in_prompt(self):
        """Canonical neighborhood name appears in LLM scoring payload for alias matches."""
        listings = [_make_listing("1", neighborhood="Lincoln Square")]
        prefs = Preferences(budget_max=4000, neighborhoods=["Upper West Side"])

        scores = [{"id": "1", "include": True, "score": 70, "pros": [], "cons": []}]
        mock_client = _mock_anthropic_client(_mock_llm_response(scores))

        with patch("src.scanner.anthropic.AsyncAnthropic", return_value=mock_client):
            await _llm_score_listings(listings, prefs)

            call_kwargs = mock_client.messages.create.call_args
            messages = call_kwargs.kwargs.get("messages") or call_kwargs[1].get("messages")
            prompt_text = messages[0]["content"]
            assert "hood_canonical" in prompt_text
            assert "Upper West Side" in prompt_text

    @pytest.mark.asyncio
    async def test_no_canonical_for_direct_match(self):
        """No hood_canonical value in listing data when neighborhood directly matches."""
        listings = [_make_listing("1", neighborhood="Chelsea")]
        prefs = Preferences(budget_max=4000, neighborhoods=["Chelsea"])

        scores = [{"id": "1", "include": True, "score": 70, "pros": [], "cons": []}]
        mock_client = _mock_anthropic_client(_mock_llm_response(scores))

        with patch("src.scanner.anthropic.AsyncAnthropic", return_value=mock_client):
            await _llm_score_listings(listings, prefs)

            call_kwargs = mock_client.messages.create.call_args
            messages = call_kwargs.kwargs.get("messages") or call_kwargs[1].get("messages")
            prompt_text = messages[0]["content"]
            # Chelsea is not in NEIGHBORHOOD_ALIASES, so no hood_canonical in the listing data
            # The prompt instructions mention hood_canonical, but that's expected
            listings_json_part = prompt_text.split("Listings:\n")[1].split("\n\nTwo-step")[0]
            assert "hood_canonical" not in listings_json_part

    @pytest.mark.asyncio
    async def test_temperature_zero(self):
        """temperature=0 is set in the LLM scoring API call."""
        listings = [_make_listing("1")]
        prefs = Preferences(budget_max=4000)

        scores = [{"id": "1", "include": True, "score": 70, "pros": [], "cons": []}]
        mock_client = _mock_anthropic_client(_mock_llm_response(scores))

        with patch("src.scanner.anthropic.AsyncAnthropic", return_value=mock_client):
            await _llm_score_listings(listings, prefs)

            call_kwargs = mock_client.messages.create.call_args
            kwargs = call_kwargs.kwargs if call_kwargs.kwargs else {}
            assert kwargs.get("temperature") == 0

    @pytest.mark.asyncio
    async def test_enrichment_wired_in_pipeline(self):
        """Enrichment is called between pre-filter and LLM scoring in scan_for_chat."""
        state = ChatState(chat_id=12345)
        state.preferences.budget_max = 4000
        state.preferences.neighborhoods = ["Chelsea"]
        state.preferences_ready = True

        raw_listings = [_raw_listing("1", neighborhood="Chelsea")]

        mock_scraper = AsyncMock()
        mock_scraper.search_with_retry = AsyncMock(return_value=raw_listings)

        mock_bot = AsyncMock()
        mock_bot.send_text = AsyncMock()
        mock_bot.send_listing_photo = AsyncMock()

        scored_listings = [
            _make_listing("1", neighborhood="Chelsea", match_score=80, photos=[]),
        ]

        call_order = []

        async def mock_enrich(listings):
            call_order.append("enrich")
            return listings

        async def mock_llm_score(listings, prefs, current_apt=None):
            call_order.append("llm_score")
            return ScoringResult(listings=scored_listings)

        with (
            patch("src.scanner.save_state"),
            patch("src.scanner._pick_hero_photos", new_callable=AsyncMock, return_value={}),
            patch("src.scanner._enrich_listings", side_effect=mock_enrich) as mock_enrich_fn,
            patch("src.scanner._llm_score_listings", side_effect=mock_llm_score),
        ):
            await scan_for_chat(mock_scraper, mock_bot, state)

        assert call_order == ["enrich", "llm_score"]
        mock_enrich_fn.assert_called_once()


# ---------------------------------------------------------------------------
# H) Scan cache fallback tests
# ---------------------------------------------------------------------------


class TestScanCacheFallback:
    def _make_state(self, **overrides) -> ChatState:
        state = ChatState(chat_id=12345)
        state.preferences.budget_max = 4000
        state.preferences.neighborhoods = ["Chelsea"]
        state.preferences_ready = True
        for key, val in overrides.items():
            setattr(state, key, val)
        return state

    @pytest.mark.asyncio
    async def test_cache_fallback_on_scraper_failure(self):
        """Cached listings re-sent when scraper fails."""
        listing = _make_listing("1", neighborhood="Chelsea", match_score=80, photos=["http://img.jpg"])
        state = self._make_state()
        state.recent_listings["1"] = listing
        state.last_scan_listing_ids = ["1"]
        state.last_scan_at = datetime.now(timezone.utc) - timedelta(hours=1)

        mock_scraper = AsyncMock()
        mock_scraper.search_with_retry = AsyncMock(
            side_effect=ApifyScraperError("WAF block")
        )

        mock_bot = AsyncMock()
        mock_bot.send_text = AsyncMock()
        mock_bot.send_listing_photo = AsyncMock()

        await scan_for_chat(mock_scraper, mock_bot, state)

        # Should send the "temporarily unavailable" message + cached listing
        sent_texts = [
            call.args[1] if len(call.args) > 1 else call.kwargs.get("text", "")
            for call in mock_bot.send_text.call_args_list
        ]
        assert any("temporarily unavailable" in t for t in sent_texts)
        assert mock_bot.send_listing_photo.call_count == 1

    @pytest.mark.asyncio
    async def test_cache_expired_shows_error(self):
        """Stale cache (>48h) falls through to error message."""
        listing = _make_listing("1", neighborhood="Chelsea", match_score=80)
        state = self._make_state()
        state.recent_listings["1"] = listing
        state.last_scan_listing_ids = ["1"]
        state.last_scan_at = datetime.now(timezone.utc) - timedelta(hours=CACHE_MAX_AGE_HOURS + 1)

        mock_scraper = AsyncMock()
        mock_scraper.search_with_retry = AsyncMock(
            side_effect=ApifyScraperError("WAF block")
        )

        mock_bot = AsyncMock()
        mock_bot.send_text = AsyncMock()

        await scan_for_chat(mock_scraper, mock_bot, state)

        sent_texts = [
            call.args[1] if len(call.args) > 1 else call.kwargs.get("text", "")
            for call in mock_bot.send_text.call_args_list
        ]
        assert any("trouble searching StreetEasy" in t for t in sent_texts)
        assert not any("temporarily unavailable" in t for t in sent_texts)

    @pytest.mark.asyncio
    async def test_cache_evicted_listings_skipped(self):
        """Graceful when some cached IDs are missing from recent_listings."""
        listing = _make_listing("2", neighborhood="Chelsea", match_score=70, photos=[])
        state = self._make_state()
        state.recent_listings["2"] = listing
        state.last_scan_listing_ids = ["1", "2", "3"]  # 1 and 3 are evicted
        state.last_scan_at = datetime.now(timezone.utc) - timedelta(hours=1)

        mock_scraper = AsyncMock()
        mock_scraper.search_with_retry = AsyncMock(
            side_effect=ApifyScraperError("WAF block")
        )

        mock_bot = AsyncMock()
        mock_bot.send_text = AsyncMock()
        mock_bot.send_listing_photo = AsyncMock()

        await scan_for_chat(mock_scraper, mock_bot, state)

        # Should still send available cached listing (text, since no photos)
        sent_texts = [
            call.args[1] if len(call.args) > 1 else call.kwargs.get("text", "")
            for call in mock_bot.send_text.call_args_list
        ]
        assert any("temporarily unavailable" in t for t in sent_texts)

    @pytest.mark.asyncio
    async def test_no_cache_shows_error(self):
        """No cache at all falls through to error message."""
        state = self._make_state()

        mock_scraper = AsyncMock()
        mock_scraper.search_with_retry = AsyncMock(
            side_effect=ApifyScraperError("WAF block")
        )

        mock_bot = AsyncMock()
        mock_bot.send_text = AsyncMock()

        await scan_for_chat(mock_scraper, mock_bot, state)

        sent_texts = [
            call.args[1] if len(call.args) > 1 else call.kwargs.get("text", "")
            for call in mock_bot.send_text.call_args_list
        ]
        assert any("trouble searching StreetEasy" in t for t in sent_texts)


class TestEnrichmentConfig:
    @pytest.mark.asyncio
    async def test_enrichment_uses_reduced_concurrency(self):
        """Verify maxConcurrency=3, maxRequestRetries=2, US proxy."""
        listings = [_make_listing("1", amenities=[], description=None)]
        listings[0].url = "https://streeteasy.com/rental/1"

        mock_run = {"status": "SUCCEEDED", "defaultDatasetId": "ds123"}
        mock_dataset = AsyncMock()
        mock_dataset.list_items = AsyncMock(return_value=MagicMock(items=[]))

        mock_actor = AsyncMock()
        mock_actor.call = AsyncMock(return_value=mock_run)

        mock_client = AsyncMock()
        mock_client.actor = MagicMock(return_value=mock_actor)
        mock_client.dataset = MagicMock(return_value=mock_dataset)

        with (
            patch("src.scanner.ApifyClientAsync", return_value=mock_client),
            patch.dict("os.environ", {"APIFY_API_TOKEN": "test-token"}),
        ):
            await _enrich_listings(listings)

        call_kwargs = mock_actor.call.call_args
        run_input = call_kwargs.kwargs.get("run_input") or call_kwargs[1].get("run_input")
        assert run_input["maxConcurrency"] == 3
        assert run_input["maxRequestRetries"] == 2
        assert run_input["proxy"]["countryCode"] == "US"


# ---------------------------------------------------------------------------
# I) seen_listing_ids timing tests (Fix 2)
# ---------------------------------------------------------------------------


class TestSeenListingIdsTiming:
    def _make_state(self, **overrides) -> ChatState:
        state = ChatState(chat_id=12345)
        state.preferences.budget_max = 4000
        state.preferences.neighborhoods = ["Chelsea"]
        state.preferences_ready = True
        for key, val in overrides.items():
            setattr(state, key, val)
        return state

    @pytest.mark.asyncio
    async def test_filtered_listing_not_marked_seen(self):
        """Listings excluded by LLM scoring are NOT marked as seen."""
        state = self._make_state()

        raw_listings = [
            _raw_listing("1", neighborhood="Chelsea"),
            _raw_listing("2", neighborhood="Chelsea"),
        ]

        mock_scraper = AsyncMock()
        mock_scraper.search_with_retry = AsyncMock(return_value=raw_listings)

        mock_bot = AsyncMock()
        mock_bot.send_text = AsyncMock()
        mock_bot.send_listing_photo = AsyncMock()

        # LLM only includes listing "1"; listing "2" is excluded
        scored_listings = [
            _make_listing("1", neighborhood="Chelsea", match_score=80, photos=[]),
        ]

        with (
            patch("src.scanner.save_state"),
            patch("src.scanner._pick_hero_photos", new_callable=AsyncMock, return_value={}),
            patch(
                "src.scanner._llm_score_listings",
                new_callable=AsyncMock,
                return_value=ScoringResult(listings=scored_listings),
            ),
        ):
            await scan_for_chat(mock_scraper, mock_bot, state)

        # Listing "1" was scored and sent → should be marked seen
        assert "1" in state.seen_listing_ids
        # Listing "2" was excluded by scoring → should NOT be marked seen
        assert "2" not in state.seen_listing_ids

    @pytest.mark.asyncio
    async def test_parse_failure_not_marked_seen(self):
        """Listings that fail parsing are NOT marked as seen."""
        state = self._make_state()

        raw_listings = [
            _raw_listing("1", neighborhood="Chelsea"),
            {"listing_id": "2", "neighborhood": "Chelsea", "price": "not-a-number"},
        ]

        mock_scraper = AsyncMock()
        mock_scraper.search_with_retry = AsyncMock(return_value=raw_listings)

        mock_bot = AsyncMock()
        mock_bot.send_text = AsyncMock()
        mock_bot.send_listing_photo = AsyncMock()

        scored_listings = [
            _make_listing("1", neighborhood="Chelsea", match_score=80, photos=[]),
        ]

        with (
            patch("src.scanner.save_state"),
            patch("src.scanner._pick_hero_photos", new_callable=AsyncMock, return_value={}),
            patch(
                "src.scanner._llm_score_listings",
                new_callable=AsyncMock,
                return_value=ScoringResult(listings=scored_listings),
            ),
        ):
            await scan_for_chat(mock_scraper, mock_bot, state)

        assert "1" in state.seen_listing_ids
        # Listing "2" failed to parse → should NOT be marked seen
        assert "2" not in state.seen_listing_ids


# ---------------------------------------------------------------------------
# J) Scoring prompt date injection tests (Fix 4)
# ---------------------------------------------------------------------------


class TestScoringPromptDate:
    @pytest.mark.asyncio
    async def test_scoring_prompt_includes_date(self):
        """The scoring prompt sent to Claude includes today's date."""
        from datetime import date

        listings = [_make_listing("1")]
        prefs = Preferences(budget_max=4000)

        scores = [
            {"id": "1", "include": True, "score": 70, "pros": ["ok"], "cons": []},
        ]
        mock_client = _mock_anthropic_client(_mock_llm_response(scores))

        with patch("src.scanner.anthropic.AsyncAnthropic", return_value=mock_client):
            await _llm_score_listings(listings, prefs)

            call_kwargs = mock_client.messages.create.call_args
            messages = call_kwargs.kwargs.get("messages") or call_kwargs[1].get("messages")
            prompt_text = messages[0]["content"]
            assert f"Today's date is {date.today().isoformat()}" in prompt_text


# ---------------------------------------------------------------------------
# K) is_daily parameter tests (Fix 7)
# ---------------------------------------------------------------------------


class TestIsDailyParam:
    def _make_state(self, **overrides) -> ChatState:
        state = ChatState(chat_id=12345)
        state.preferences.budget_max = 4000
        state.preferences.neighborhoods = ["Chelsea"]
        state.preferences_ready = True
        for key, val in overrides.items():
            setattr(state, key, val)
        return state

    @pytest.mark.asyncio
    async def test_manual_scan_header(self):
        """Manual search (is_daily=False) uses 'Search Results' header."""
        state = self._make_state()

        raw_listings = [_raw_listing("1", neighborhood="Chelsea")]

        mock_scraper = AsyncMock()
        mock_scraper.search_with_retry = AsyncMock(return_value=raw_listings)

        mock_bot = AsyncMock()
        mock_bot.send_text = AsyncMock()
        mock_bot.send_listing_photo = AsyncMock()

        scored_listings = [
            _make_listing("1", neighborhood="Chelsea", match_score=80, photos=[]),
        ]

        with (
            patch("src.scanner.save_state"),
            patch("src.scanner._pick_hero_photos", new_callable=AsyncMock, return_value={}),
            patch(
                "src.scanner._llm_score_listings",
                new_callable=AsyncMock,
                return_value=ScoringResult(listings=scored_listings),
            ),
        ):
            await scan_for_chat(mock_scraper, mock_bot, state, is_daily=False)

        sent_texts = [
            call.args[1] if len(call.args) > 1 else call.kwargs.get("text", "")
            for call in mock_bot.send_text.call_args_list
        ]
        assert any("Search Results" in t for t in sent_texts)
        assert not any("Daily Scan" in t for t in sent_texts)

    @pytest.mark.asyncio
    async def test_daily_scan_header(self):
        """Daily scan (is_daily=True, default) uses 'Daily Scan Complete' header."""
        state = self._make_state()

        raw_listings = [_raw_listing("1", neighborhood="Chelsea")]

        mock_scraper = AsyncMock()
        mock_scraper.search_with_retry = AsyncMock(return_value=raw_listings)

        mock_bot = AsyncMock()
        mock_bot.send_text = AsyncMock()
        mock_bot.send_listing_photo = AsyncMock()

        scored_listings = [
            _make_listing("1", neighborhood="Chelsea", match_score=80, photos=[]),
        ]

        with (
            patch("src.scanner.save_state"),
            patch("src.scanner._pick_hero_photos", new_callable=AsyncMock, return_value={}),
            patch(
                "src.scanner._llm_score_listings",
                new_callable=AsyncMock,
                return_value=ScoringResult(listings=scored_listings),
            ),
        ):
            await scan_for_chat(mock_scraper, mock_bot, state)

        sent_texts = [
            call.args[1] if len(call.args) > 1 else call.kwargs.get("text", "")
            for call in mock_bot.send_text.call_args_list
        ]
        assert any("Daily Scan Complete" in t for t in sent_texts)

    @pytest.mark.asyncio
    async def test_manual_scan_no_results_no_tomorrow(self):
        """Manual search with 0 results omits 'tomorrow' phrasing."""
        state = self._make_state()

        mock_scraper = AsyncMock()
        mock_scraper.search_with_retry = AsyncMock(return_value=[])

        mock_bot = AsyncMock()
        mock_bot.send_text = AsyncMock()

        with patch("src.scanner.save_state"):
            await scan_for_chat(mock_scraper, mock_bot, state, is_daily=False)

        sent_text = mock_bot.send_text.call_args[0][1]
        assert "tomorrow" not in sent_text
        assert "Search Results" in sent_text
