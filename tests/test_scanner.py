"""Tests for the smart filtering + scoring pipeline in scanner.py."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.models import ChatState, CurrentApartment, Listing, Preferences
from src.apify_scraper import AmenityEnrichmentResult, ApifyScraperError
from tests.conftest import make_listing
from src.scanner import (
    CACHE_MAX_AGE_HOURS,
    MAX_PER_BUILDING,
    MAX_PER_NEIGHBORHOOD,
    NEIGHBORHOOD_ALIASES,
    SCORE_FLOOR,
    STALE_LISTING_MONTHS,
    ScoringResult,
    _apply_pre_filters,
    _cap_per_building,
    _cap_per_neighborhood,
    _canonical_neighborhood,
    _extract_building_address,
    _filter_broker_fee,
    _filter_over_budget,
    _filter_stale_listings,
    _filter_under_budget,
    _filter_wrong_bathrooms,
    _parse_scoring_response,
    _filter_wrong_bedrooms,
    _get_cached_listings,
    _has_cached_scan,
    _interleave_by_neighborhood,
    _llm_score_listings,
    _neighborhood_pre_filter,
    _normalize_hood,
    _parse_listing,
    _pick_hero_photos,
    _sample_photo_keys,
    _vision_pick_heroes,
    scan_for_chat,
)


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


def _enrichment_result(
    data_by_listing_id: dict[str, dict],
    *,
    coverage: float,
    target_count: int,
    failed: bool = False,
    failure_reason: str | None = None,
) -> AmenityEnrichmentResult:
    """Build a typed enrichment result for scanner tests."""
    return AmenityEnrichmentResult(
        data_by_listing_id=data_by_listing_id,
        coverage=coverage,
        target_count=target_count,
        run_summaries=[],
        failed=failed,
        failure_reason=failure_reason,
    )


# ---------------------------------------------------------------------------
# A) Neighborhood pre-filter tests
# ---------------------------------------------------------------------------


class TestNeighborhoodPreFilter:
    def test_exact_match(self):
        """Listings matching preferred neighborhoods are kept; others dropped."""
        prefs = Preferences(neighborhoods=["Chelsea", "SoHo"])
        listings = [
            make_listing("1", neighborhood="Chelsea"),
            make_listing("2", neighborhood="SoHo"),
            make_listing("3", neighborhood="Harlem"),
        ]
        result = _neighborhood_pre_filter(listings, prefs)
        assert len(result) == 2
        assert {l.listing_id for l in result} == {"1", "2"}

    def test_alias_match(self):
        """Listing with alias neighborhood ('West Chelsea') kept when prefs has 'Chelsea'."""
        prefs = Preferences(neighborhoods=["Chelsea"])
        listings = [
            make_listing("1", neighborhood="West Chelsea"),
            make_listing("2", neighborhood="Midtown"),
        ]
        result = _neighborhood_pre_filter(listings, prefs)
        assert len(result) == 1
        assert result[0].listing_id == "1"

    def test_case_insensitive(self):
        """Matching is case-insensitive ('east village' matches 'East Village')."""
        prefs = Preferences(neighborhoods=["east village"])
        listings = [
            make_listing("1", neighborhood="East Village"),
            make_listing("2", neighborhood="EAST VILLAGE"),
        ]
        result = _neighborhood_pre_filter(listings, prefs)
        assert len(result) == 2

    def test_no_neighborhoods_passes_all(self):
        """When prefs.neighborhoods is empty, all listings pass through."""
        prefs = Preferences(neighborhoods=[])
        listings = [
            make_listing("1", neighborhood="Chelsea"),
            make_listing("2", neighborhood="Harlem"),
            make_listing("3", neighborhood="Bushwick"),
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
            make_listing("1", neighborhood="Harlem"),
            make_listing("2", neighborhood="Bushwick"),
            make_listing("3", neighborhood="Astoria"),
        ]
        result = _neighborhood_pre_filter(listings, prefs)
        assert result == []

    def test_mixed_match_and_nonmatch(self):
        """Only matching listings survive; non-matching are dropped."""
        prefs = Preferences(neighborhoods=["Chelsea", "Williamsburg"])
        listings = [
            make_listing("1", neighborhood="Chelsea"),
            make_listing("2", neighborhood="Harlem"),
            make_listing("3", neighborhood="Williamsburg"),
            make_listing("4", neighborhood="Financial District"),
            make_listing("5", neighborhood="North Williamsburg"),  # alias
        ]
        result = _neighborhood_pre_filter(listings, prefs)
        ids = {l.listing_id for l in result}
        assert ids == {"1", "3", "5"}

    def test_multiple_aliases_to_same_canonical(self):
        """Multiple alias neighborhoods all map to the same canonical name."""
        prefs = Preferences(neighborhoods=["Upper East Side"])
        listings = [
            make_listing("1", neighborhood="Yorkville"),
            make_listing("2", neighborhood="Lenox Hill"),
            make_listing("3", neighborhood="Carnegie Hill"),
            make_listing("4", neighborhood="Upper East Side"),
            make_listing("5", neighborhood="SoHo"),
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
            make_listing("1", price=3000),
            make_listing("2", price=5000),
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
            make_listing("1", price=3000),
            make_listing("2", price=3200),
            make_listing("3", price=3500),
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
            make_listing("1", price=3000),
            make_listing("2", price=3500),
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
        listings = [make_listing("1", price=3000)]
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
        listings = [make_listing("1")]
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
        listings = [make_listing("1"), make_listing("2")]
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
            make_listing("1"),
            make_listing("2"),
            make_listing("3"),
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
            make_listing("1", price=4000),
            make_listing("2", price=2500),
            make_listing("3", price=3200),
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
            make_listing("1", price=3800),
            make_listing("2", price=2900),
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
            make_listing("1", price=5000),
            make_listing("2", price=6000),
            make_listing("3", price=7000),
            make_listing("4", price=8000),
            make_listing("5", price=9000),
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
        listings = [make_listing("1")]
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
        listings = [make_listing("1"), make_listing("2")]
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
        listings = [make_listing("1")]
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
        listings = [make_listing("1")]
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
            make_listing("1", neighborhood="Chelsea", match_score=80, photos=[]),
            make_listing("3", neighborhood="SoHo", match_score=70, photos=[]),
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
            make_listing("1", neighborhood="Chelsea", match_score=80, photos=["http://img1.jpg"]),
            make_listing("2", neighborhood="SoHo", match_score=70, photos=["http://img2.jpg"]),
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
            make_listing("1", neighborhood="Chelsea", match_score=85),
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
            make_listing("2", neighborhood="Chelsea", match_score=75),
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
            make_listing("1", neighborhood="Chelsea", match_score=80, photos=[]),
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

        with patch("src.scanner.save_state"):
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
            make_listing("1", neighborhood="Chelsea", match_score=40, photos=[]),
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
            make_listing("1", neighborhood="Chelsea", match_score=80, photos=[]),
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
            make_listing(
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
            make_listing("1", photo_keys=["keyA", "keyB", "keyC"]),
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
            make_listing("1", photo_keys=["keyA", "keyB"]),
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
            make_listing("1", photo_keys=["onlyOne"]),
        ]
        result = await _pick_hero_photos(listings)
        assert result == {}

    @pytest.mark.asyncio
    async def test_no_keys_skipped(self):
        """Listings with no photo_keys are skipped."""
        listings = [
            make_listing("1", photo_keys=[]),
        ]
        result = await _pick_hero_photos(listings)
        assert result == {}

    @pytest.mark.asyncio
    async def test_invalid_letter_omitted(self):
        """Invalid letter in vision response omits that listing from result."""
        listings = [
            make_listing("1", photo_keys=["keyA", "keyB", "keyC"]),
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
            make_listing(str(i), photo_keys=[f"k{i}a", f"k{i}b", f"k{i}c"])
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
            make_listing(str(i), photo_keys=[f"k{i}a", f"k{i}b"])
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
# F) Amenity annotation tests
# ---------------------------------------------------------------------------


class TestParseListingAmenities:
    def test_amenity_fields_flow_through(self):
        """matched_amenities and missing_amenities pass through _parse_listing."""
        raw = _raw_listing(
            "1",
            matched_amenities=["Dishwasher", "Elevator"],
            missing_amenities=["Gym"],
        )
        listing = _parse_listing(raw)
        assert listing.matched_amenities == ["Dishwasher", "Elevator"]
        assert listing.missing_amenities == ["Gym"]

    def test_no_amenity_data_defaults_empty(self):
        """Missing amenity fields default to empty list."""
        raw = _raw_listing("1")
        listing = _parse_listing(raw)
        assert listing.matched_amenities == []
        assert listing.missing_amenities == []


# ---------------------------------------------------------------------------
# F2) Stale listing filter tests
# ---------------------------------------------------------------------------


class TestFilterStaleListings:
    def test_stale_listing_dropped(self):
        """Listing with available_date 4 months ago is dropped."""
        from datetime import date, timedelta
        old_date = (date.today() - timedelta(days=4 * 30)).isoformat()
        listings = [make_listing("1", available_date=old_date)]
        result = _filter_stale_listings(listings)
        assert len(result) == 0

    def test_recent_listing_kept(self):
        """Listing with available_date 2 months ago is kept."""
        from datetime import date, timedelta
        recent_date = (date.today() - timedelta(days=2 * 30)).isoformat()
        listings = [make_listing("1", available_date=recent_date)]
        result = _filter_stale_listings(listings)
        assert len(result) == 1

    def test_future_date_kept(self):
        """Listing with future available_date is kept."""
        from datetime import date, timedelta
        future_date = (date.today() + timedelta(days=30)).isoformat()
        listings = [make_listing("1", available_date=future_date)]
        result = _filter_stale_listings(listings)
        assert len(result) == 1

    def test_no_available_date_kept(self):
        """Listing with no available_date passes through."""
        listings = [make_listing("1", available_date=None)]
        result = _filter_stale_listings(listings)
        assert len(result) == 1

    def test_unparseable_date_kept(self):
        """Listing with unparseable date passes through."""
        listings = [make_listing("1", available_date="not-a-date")]
        result = _filter_stale_listings(listings)
        assert len(result) == 1

    def test_empty_list_returns_empty(self):
        """Empty input returns empty output."""
        result = _filter_stale_listings([])
        assert result == []

    def test_mixed_stale_and_fresh(self):
        """Mix of stale and fresh listings filters correctly."""
        from datetime import date, timedelta
        old_date = (date.today() - timedelta(days=4 * 30)).isoformat()
        recent_date = (date.today() - timedelta(days=1 * 30)).isoformat()
        listings = [
            make_listing("1", available_date=old_date),
            make_listing("2", available_date=recent_date),
            make_listing("3", available_date=None),
        ]
        result = _filter_stale_listings(listings)
        assert len(result) == 2
        assert {l.listing_id for l in result} == {"2", "3"}


class TestBuildingDedup:
    def test_extract_building_address(self):
        """Extracts building address from full address."""
        assert _extract_building_address("123 Main St #4A") == "123 main st"
        assert _extract_building_address("400 West 61st St #12B") == "400 west 61st st"
        assert _extract_building_address("100 Broadway") == "100 broadway"

    def test_cap_same_building(self):
        """5 listings from same building → top 2 kept."""
        listings = [
            make_listing("1", address="400 West 61st St #12B", match_score=90),
            make_listing("2", address="400 West 61st St #3A", match_score=85),
            make_listing("3", address="400 West 61st St #5C", match_score=80),
            make_listing("4", address="400 West 61st St #8D", match_score=75),
            make_listing("5", address="400 West 61st St #1E", match_score=70),
        ]
        result = _cap_per_building(listings)
        assert len(result) == MAX_PER_BUILDING
        assert result[0].listing_id == "1"
        assert result[1].listing_id == "2"

    def test_mixed_buildings(self):
        """3 from building A + 2 from B + 1 from C → 2+2+1."""
        listings = [
            make_listing("1", address="100 Main St #1A", match_score=90),
            make_listing("2", address="200 Oak Ave #2A", match_score=85),
            make_listing("3", address="100 Main St #3B", match_score=80),
            make_listing("4", address="200 Oak Ave #4B", match_score=75),
            make_listing("5", address="100 Main St #5C", match_score=70),
            make_listing("6", address="300 Elm St #6A", match_score=65),
        ]
        result = _cap_per_building(listings)
        assert len(result) == 5
        ids = [l.listing_id for l in result]
        assert ids == ["1", "2", "3", "4", "6"]

    def test_different_buildings_all_pass(self):
        """Listings from different buildings all pass through."""
        listings = [
            make_listing("1", address="100 Main St #1A", match_score=90),
            make_listing("2", address="200 Oak Ave #2A", match_score=85),
            make_listing("3", address="300 Elm St #3A", match_score=80),
        ]
        result = _cap_per_building(listings)
        assert len(result) == 3

    def test_empty_list(self):
        """Empty input returns empty output."""
        result = _cap_per_building([])
        assert result == []


# ---------------------------------------------------------------------------
# G) LLM scoring payload tests (enriched data, concessions, canonical hoods)
# ---------------------------------------------------------------------------


class TestLLMScoringPayload:
    @pytest.mark.asyncio
    async def test_description_in_prompt(self):
        """Description appears in LLM scoring payload (truncated to 300 chars)."""
        long_desc = "A" * 500
        listings = [make_listing("1", description=long_desc)]
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
        listings = [make_listing("1", price=19000, net_effective_price=15833, months_free=2.0)]
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
        listings = [make_listing("1", price=3500, net_effective_price=3500)]
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
        listings = [make_listing("1", neighborhood="Lincoln Square")]
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
        listings = [make_listing("1", neighborhood="Chelsea")]
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
    async def test_temperature_not_set_for_opus_4_7(self):
        """Opus 4.7 deprecated the `temperature` parameter — scoring must not send it.

        Previously we sent temperature=0 for determinism. The 2026-04-18
        model bump to claude-opus-4-7 means passing that parameter triggers
        an API error, so the call site drops it entirely.
        """
        listings = [make_listing("1")]
        prefs = Preferences(budget_max=4000)

        scores = [{"id": "1", "include": True, "score": 70, "pros": [], "cons": []}]
        mock_client = _mock_anthropic_client(_mock_llm_response(scores))

        with patch("src.scanner.anthropic.AsyncAnthropic", return_value=mock_client):
            await _llm_score_listings(listings, prefs)

            call_kwargs = mock_client.messages.create.call_args
            kwargs = call_kwargs.kwargs if call_kwargs.kwargs else {}
            assert "temperature" not in kwargs, (
                "temperature must not be sent to Opus 4.7; it was deprecated"
            )

    @pytest.mark.asyncio
    async def test_amenity_fields_in_prompt(self):
        """matched_amenities and missing_amenities appear in LLM scoring payload."""
        listings = [make_listing(
            "1",
            matched_amenities=["Dishwasher", "Elevator"],
            missing_amenities=["Gym"],
        )]
        prefs = Preferences(budget_max=4000, must_haves=["Dishwasher"])

        scores = [{"id": "1", "include": True, "score": 80, "pros": [], "cons": []}]
        mock_client = _mock_anthropic_client(_mock_llm_response(scores))

        with patch("src.scanner.anthropic.AsyncAnthropic", return_value=mock_client):
            await _llm_score_listings(listings, prefs)

            call_kwargs = mock_client.messages.create.call_args
            messages = call_kwargs.kwargs.get("messages") or call_kwargs[1].get("messages")
            prompt_text = messages[0]["content"]
            assert "has_amenities" in prompt_text
            assert "Dishwasher" in prompt_text
            assert "missing_amenities" in prompt_text
            assert "Gym" in prompt_text

    @pytest.mark.asyncio
    async def test_amenity_fields_excluded_when_empty(self):
        """has_amenities and missing_amenities not in prompt when empty."""
        listings = [make_listing("1")]  # no matched/missing amenities
        prefs = Preferences(budget_max=4000)

        scores = [{"id": "1", "include": True, "score": 70, "pros": [], "cons": []}]
        mock_client = _mock_anthropic_client(_mock_llm_response(scores))

        with patch("src.scanner.anthropic.AsyncAnthropic", return_value=mock_client):
            await _llm_score_listings(listings, prefs)

            call_kwargs = mock_client.messages.create.call_args
            messages = call_kwargs.kwargs.get("messages") or call_kwargs[1].get("messages")
            prompt_text = messages[0]["content"]
            listings_json_part = prompt_text.split("Listings:\n")[1].split("\n\nTwo-step")[0]
            assert "has_amenities" not in listings_json_part
            assert '"missing_amenities"' not in listings_json_part


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
        listing = make_listing("1", neighborhood="Chelsea", match_score=80, photos=["http://img.jpg"])
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

        with patch("src.scanner.save_state"):
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
        listing = make_listing("1", neighborhood="Chelsea", match_score=80)
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

        with patch("src.scanner.save_state"):
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
        listing = make_listing("2", neighborhood="Chelsea", match_score=70, photos=[])
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

        with patch("src.scanner.save_state"):
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

        with patch("src.scanner.save_state"):
            await scan_for_chat(mock_scraper, mock_bot, state)

        sent_texts = [
            call.args[1] if len(call.args) > 1 else call.kwargs.get("text", "")
            for call in mock_bot.send_text.call_args_list
        ]
        assert any("trouble searching StreetEasy" in t for t in sent_texts)

    @pytest.mark.asyncio
    async def test_search_failures_enter_cooldown(self):
        """Repeated scraper failures activate cooldown tracking."""
        state = self._make_state()

        mock_scraper = AsyncMock()
        mock_scraper.search_with_retry = AsyncMock(
            side_effect=ApifyScraperError("WAF block")
        )

        mock_bot = AsyncMock()
        mock_bot.send_text = AsyncMock()

        with patch("src.scanner.save_state"):
            await scan_for_chat(mock_scraper, mock_bot, state, is_daily=False)
        assert state.search_failure_streak == 1
        assert state.search_cooldown_until is None

        with patch("src.scanner.save_state"):
            await scan_for_chat(mock_scraper, mock_bot, state, is_daily=False)
        assert state.search_failure_streak == 2
        assert state.search_cooldown_until is not None

    @pytest.mark.asyncio
    async def test_active_cooldown_skips_search(self):
        """Cooldown suppresses new scraper calls and sends retry-later message."""
        state = self._make_state()
        state.search_failure_streak = 3
        state.search_cooldown_until = datetime.now(timezone.utc) + timedelta(minutes=2)

        mock_scraper = AsyncMock()
        mock_scraper.search_with_retry = AsyncMock()

        mock_bot = AsyncMock()
        mock_bot.send_text = AsyncMock()

        with patch("src.scanner.save_state"):
            await scan_for_chat(mock_scraper, mock_bot, state, is_daily=False)

        mock_scraper.search_with_retry.assert_not_called()
        mock_bot.send_text.assert_called_once()
        assert "temporarily blocking requests" in mock_bot.send_text.call_args[0][1]


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
            make_listing("1", neighborhood="Chelsea", match_score=80, photos=[]),
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
            make_listing("1", neighborhood="Chelsea", match_score=80, photos=[]),
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

        listings = [make_listing("1")]
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
            make_listing("1", neighborhood="Chelsea", match_score=80, photos=[]),
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
            make_listing("1", neighborhood="Chelsea", match_score=80, photos=[]),
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


# ---------------------------------------------------------------------------
# L) Bathroom filter tests (Phase 0B)
# ---------------------------------------------------------------------------


class TestBathroomFilter:
    def test_exact_match_passes(self):
        """Listing with exactly min_bathrooms passes."""
        prefs = Preferences(min_bathrooms=2)
        listings = [make_listing("1", bathrooms=2.0)]
        result = _filter_wrong_bathrooms(listings, prefs)
        assert len(result) == 1

    def test_above_threshold_passes(self):
        """Listing with more than min_bathrooms passes."""
        prefs = Preferences(min_bathrooms=1)
        listings = [make_listing("1", bathrooms=2.0)]
        result = _filter_wrong_bathrooms(listings, prefs)
        assert len(result) == 1

    def test_below_threshold_dropped(self):
        """Listing with fewer than min_bathrooms is dropped."""
        prefs = Preferences(min_bathrooms=2)
        listings = [make_listing("1", bathrooms=1.0)]
        result = _filter_wrong_bathrooms(listings, prefs)
        assert len(result) == 0

    def test_no_pref_passes_all(self):
        """When min_bathrooms is not set, all listings pass."""
        prefs = Preferences()
        listings = [
            make_listing("1", bathrooms=1.0),
            make_listing("2", bathrooms=0.5),
        ]
        result = _filter_wrong_bathrooms(listings, prefs)
        assert len(result) == 2

    def test_half_bath_edge_case(self):
        """1.5 bathrooms meets min of 1 but not min of 2."""
        prefs_1 = Preferences(min_bathrooms=1)
        prefs_2 = Preferences(min_bathrooms=2)
        listings = [make_listing("1", bathrooms=1.5)]
        assert len(_filter_wrong_bathrooms(listings, prefs_1)) == 1
        assert len(_filter_wrong_bathrooms(listings, prefs_2)) == 0

    def test_mixed_pass_and_fail(self):
        """Mix of listings with various bathroom counts."""
        prefs = Preferences(min_bathrooms=2)
        listings = [
            make_listing("1", bathrooms=1.0),
            make_listing("2", bathrooms=2.0),
            make_listing("3", bathrooms=2.5),
            make_listing("4", bathrooms=1.5),
        ]
        result = _filter_wrong_bathrooms(listings, prefs)
        assert {l.listing_id for l in result} == {"2", "3"}

    def test_empty_list(self):
        """Empty input returns empty output."""
        prefs = Preferences(min_bathrooms=2)
        result = _filter_wrong_bathrooms([], prefs)
        assert result == []


# ---------------------------------------------------------------------------
# M) Broker fee filter tests (Phase 2A)
# ---------------------------------------------------------------------------


class TestBrokerFeeFilter:
    def test_no_fee_only_drops_fee_listings(self):
        """When no_fee_only=True, listings with broker_fee are dropped."""
        prefs = Preferences(no_fee_only=True)
        listings = [
            make_listing("1", broker_fee=None),      # no fee
            make_listing("2", broker_fee="Broker fee"),  # has fee
            make_listing("3", broker_fee=None),      # no fee
        ]
        result = _filter_broker_fee(listings, prefs)
        assert {l.listing_id for l in result} == {"1", "3"}

    def test_no_fee_pref_false_passes_all(self):
        """When no_fee_only=False, all listings pass."""
        prefs = Preferences(no_fee_only=False)
        listings = [
            make_listing("1", broker_fee=None),
            make_listing("2", broker_fee="Broker fee"),
        ]
        result = _filter_broker_fee(listings, prefs)
        assert len(result) == 2

    def test_all_have_fees(self):
        """All listings have fees and no_fee_only=True → empty result."""
        prefs = Preferences(no_fee_only=True)
        listings = [
            make_listing("1", broker_fee="Broker fee"),
            make_listing("2", broker_fee="Broker fee"),
        ]
        result = _filter_broker_fee(listings, prefs)
        assert result == []

    def test_empty_list(self):
        """Empty input returns empty output."""
        prefs = Preferences(no_fee_only=True)
        result = _filter_broker_fee([], prefs)
        assert result == []


# ---------------------------------------------------------------------------
# N) Budget overage filter tests (Phase 2B)
# ---------------------------------------------------------------------------


class TestBudgetOverageFilter:
    def test_under_budget_passes(self):
        """Listing under budget passes."""
        prefs = Preferences(budget_max=4000)
        listings = [make_listing("1", price=3500)]
        result = _filter_over_budget(listings, prefs)
        assert len(result) == 1

    def test_at_budget_passes(self):
        """Listing at exactly budget_max passes."""
        prefs = Preferences(budget_max=4000)
        listings = [make_listing("1", price=4000)]
        result = _filter_over_budget(listings, prefs)
        assert len(result) == 1

    def test_over_budget_dropped(self):
        """Listing over budget is dropped."""
        prefs = Preferences(budget_max=4000)
        listings = [make_listing("1", price=4500)]
        result = _filter_over_budget(listings, prefs)
        assert len(result) == 0

    def test_net_effective_under_budget_passes(self):
        """Gross over but net effective under budget passes."""
        prefs = Preferences(budget_max=4000)
        listings = [make_listing("1", price=4500, net_effective_price=3750)]
        result = _filter_over_budget(listings, prefs)
        assert len(result) == 1

    def test_both_over_budget_dropped(self):
        """Both gross and net effective over budget is dropped."""
        prefs = Preferences(budget_max=4000)
        listings = [make_listing("1", price=5000, net_effective_price=4500)]
        result = _filter_over_budget(listings, prefs)
        assert len(result) == 0

    def test_no_budget_max_passes_all(self):
        """When budget_max is None, all listings pass."""
        prefs = Preferences()
        listings = [
            make_listing("1", price=10000),
            make_listing("2", price=50000),
        ]
        result = _filter_over_budget(listings, prefs)
        assert len(result) == 2

    def test_mixed_budget(self):
        """Mix of under, over, and net-effective-under listings."""
        prefs = Preferences(budget_max=4000)
        listings = [
            make_listing("1", price=3500),                          # under
            make_listing("2", price=5000),                          # over
            make_listing("3", price=4500, net_effective_price=3800), # net under
            make_listing("4", price=4000),                          # at budget
        ]
        result = _filter_over_budget(listings, prefs)
        assert {l.listing_id for l in result} == {"1", "3", "4"}

    def test_empty_list(self):
        """Empty input returns empty output."""
        prefs = Preferences(budget_max=4000)
        result = _filter_over_budget([], prefs)
        assert result == []


# ---------------------------------------------------------------------------
# O) Neighborhood cap tests (Phase 3A)
# ---------------------------------------------------------------------------


class TestNeighborhoodCap:
    def test_over_cap_trimmed(self):
        """7 listings from same neighborhood → top 5 kept."""
        listings = [
            make_listing(str(i), neighborhood="Chelsea", match_score=90 - i * 5)
            for i in range(7)
        ]
        result = _cap_per_neighborhood(listings)
        assert len(result) == MAX_PER_NEIGHBORHOOD
        # First 5 (highest scores) kept
        assert [l.listing_id for l in result] == [str(i) for i in range(5)]

    def test_under_cap_passes(self):
        """3 listings from same neighborhood all pass."""
        listings = [
            make_listing(str(i), neighborhood="Chelsea", match_score=90 - i * 5)
            for i in range(3)
        ]
        result = _cap_per_neighborhood(listings)
        assert len(result) == 3

    def test_aliases_grouped(self):
        """Lincoln Square and Upper West Side count as same neighborhood."""
        listings = [
            make_listing("1", neighborhood="Upper West Side", match_score=90),
            make_listing("2", neighborhood="Lincoln Square", match_score=85),
            make_listing("3", neighborhood="Upper West Side", match_score=80),
            make_listing("4", neighborhood="Lincoln Square", match_score=75),
            make_listing("5", neighborhood="Upper West Side", match_score=70),
            make_listing("6", neighborhood="Lincoln Square", match_score=65),
        ]
        result = _cap_per_neighborhood(listings)
        assert len(result) == MAX_PER_NEIGHBORHOOD

    def test_multiple_neighborhoods(self):
        """Different neighborhoods each get their own cap."""
        listings = []
        for hood in ["Chelsea", "SoHo", "Tribeca"]:
            for i in range(6):
                listings.append(make_listing(
                    f"{hood}-{i}", neighborhood=hood, match_score=90 - i * 5
                ))
        # Sort by score desc (as pipeline does)
        listings.sort(key=lambda l: l.match_score or 0, reverse=True)
        result = _cap_per_neighborhood(listings)
        # Each neighborhood: 5 max, 3 neighborhoods = 15 max
        assert len(result) == 15

    def test_empty_list(self):
        """Empty input returns empty output."""
        assert _cap_per_neighborhood([]) == []


# ---------------------------------------------------------------------------
# P) Neighborhood interleaving tests (Phase 3B)
# ---------------------------------------------------------------------------


class TestNeighborhoodInterleaving:
    def test_interleave_two_neighborhoods(self):
        """Two neighborhoods interleaved in round-robin."""
        listings = [
            make_listing("c1", neighborhood="Chelsea", match_score=90),
            make_listing("c2", neighborhood="Chelsea", match_score=80),
            make_listing("s1", neighborhood="SoHo", match_score=85),
            make_listing("s2", neighborhood="SoHo", match_score=75),
        ]
        result = _interleave_by_neighborhood(listings)
        # Chelsea has best score (90), goes first. Then SoHo, then Chelsea, SoHo.
        hoods = [l.neighborhood for l in result]
        assert hoods == ["Chelsea", "SoHo", "Chelsea", "SoHo"]

    def test_interleave_three_neighborhoods(self):
        """Three neighborhoods interleaved."""
        listings = [
            make_listing("a1", neighborhood="Chelsea", match_score=95),
            make_listing("b1", neighborhood="SoHo", match_score=90),
            make_listing("c1", neighborhood="Tribeca", match_score=85),
            make_listing("a2", neighborhood="Chelsea", match_score=80),
            make_listing("b2", neighborhood="SoHo", match_score=75),
        ]
        result = _interleave_by_neighborhood(listings)
        hoods = [l.neighborhood for l in result]
        # Round 1: Chelsea, SoHo, Tribeca. Round 2: Chelsea, SoHo.
        assert hoods == ["Chelsea", "SoHo", "Tribeca", "Chelsea", "SoHo"]

    def test_single_neighborhood(self):
        """Single neighborhood preserves order."""
        listings = [
            make_listing("1", neighborhood="Chelsea", match_score=90),
            make_listing("2", neighborhood="Chelsea", match_score=80),
        ]
        result = _interleave_by_neighborhood(listings)
        assert [l.listing_id for l in result] == ["1", "2"]

    def test_preserves_all_listings(self):
        """All listings are present in output."""
        listings = [
            make_listing("1", neighborhood="Chelsea", match_score=90),
            make_listing("2", neighborhood="SoHo", match_score=85),
            make_listing("3", neighborhood="Chelsea", match_score=80),
        ]
        result = _interleave_by_neighborhood(listings)
        assert len(result) == 3
        assert {l.listing_id for l in result} == {"1", "2", "3"}

    def test_aliases_grouped_for_interleave(self):
        """Alias neighborhoods are grouped together."""
        listings = [
            make_listing("1", neighborhood="Lincoln Square", match_score=90),
            make_listing("2", neighborhood="Chelsea", match_score=85),
            make_listing("3", neighborhood="Upper West Side", match_score=80),
        ]
        result = _interleave_by_neighborhood(listings)
        # Lincoln Square and Upper West Side are both "upper west side" canonical
        # Chelsea is "chelsea" canonical
        # UWS group has best score 90, Chelsea has 85
        hoods = [_canonical_neighborhood(l.neighborhood) for l in result]
        # Round 1: UWS, Chelsea. Round 2: UWS.
        assert hoods == ["upper west side", "chelsea", "upper west side"]

    def test_empty_list(self):
        """Empty input returns empty output."""
        assert _interleave_by_neighborhood([]) == []

    def test_neighborhood_order_by_best_score(self):
        """Neighborhoods ordered by their best listing score."""
        listings = [
            make_listing("s1", neighborhood="SoHo", match_score=95),
            make_listing("c1", neighborhood="Chelsea", match_score=90),
            make_listing("s2", neighborhood="SoHo", match_score=70),
            make_listing("c2", neighborhood="Chelsea", match_score=60),
        ]
        result = _interleave_by_neighborhood(listings)
        # SoHo has best=95, Chelsea has best=90
        assert result[0].neighborhood == "SoHo"
        assert result[1].neighborhood == "Chelsea"


# ---------------------------------------------------------------------------
# Bedroom filter tests
# ---------------------------------------------------------------------------


class TestBedroomFilter:
    def test_matching_bedrooms_kept(self):
        """Listings with matching bedroom counts are kept."""
        prefs = Preferences(bedrooms=[1, 2])
        listings = [
            make_listing("1", bedrooms=1),
            make_listing("2", bedrooms=2),
            make_listing("3", bedrooms=3),
        ]
        result = _filter_wrong_bedrooms(listings, prefs)
        assert len(result) == 2
        assert {l.listing_id for l in result} == {"1", "2"}

    def test_no_bedrooms_pref_passes_all(self):
        """When no bedrooms preference, all listings pass."""
        prefs = Preferences(bedrooms=[])
        listings = [
            make_listing("1", bedrooms=0),
            make_listing("2", bedrooms=1),
            make_listing("3", bedrooms=3),
        ]
        result = _filter_wrong_bedrooms(listings, prefs)
        assert len(result) == 3

    def test_studios_handled(self):
        """Studios (bedrooms=0) are correctly filtered."""
        prefs = Preferences(bedrooms=[0])
        listings = [
            make_listing("1", bedrooms=0),
            make_listing("2", bedrooms=1),
        ]
        result = _filter_wrong_bedrooms(listings, prefs)
        assert len(result) == 1
        assert result[0].listing_id == "1"

    def test_all_filtered_out(self):
        """All listings filtered when none match bedrooms."""
        prefs = Preferences(bedrooms=[3])
        listings = [
            make_listing("1", bedrooms=1),
            make_listing("2", bedrooms=2),
        ]
        result = _filter_wrong_bedrooms(listings, prefs)
        assert result == []

    def test_empty_listings(self):
        """Empty input returns empty output."""
        prefs = Preferences(bedrooms=[1, 2])
        result = _filter_wrong_bedrooms([], prefs)
        assert result == []


# ---------------------------------------------------------------------------
# Enrichment integration tests
# ---------------------------------------------------------------------------


class TestEnrichmentIntegration:
    @pytest.mark.asyncio
    async def test_enrichment_skipped_no_amenity_prefs(self):
        """No must_haves/nice_to_haves → enrichment not called."""
        prefs = Preferences(budget_max=5000, neighborhoods=["Chelsea"])
        state = ChatState(chat_id=12345, preferences=prefs, preferences_ready=True)

        mock_scraper = AsyncMock()
        mock_scraper.search_with_retry = AsyncMock(return_value=[
            _raw_listing("1", neighborhood="Chelsea"),
        ])
        # enrich_with_amenities should NOT be called
        mock_scraper.enrich_with_amenities = AsyncMock()

        mock_bot = AsyncMock()
        mock_bot.send_text = AsyncMock()
        mock_bot.send_listing_photo = AsyncMock()

        scores = [{"id": "1", "include": True, "score": 80, "pros": ["Nice"], "cons": []}]
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text=json.dumps(scores))]
        mock_client = AsyncMock()
        mock_client.messages.create = AsyncMock(return_value=mock_response)

        with patch("src.scanner.anthropic.AsyncAnthropic", return_value=mock_client), \
             patch("src.scanner.save_state"), \
             patch("src.scanner._pick_hero_photos", return_value={}):
            await scan_for_chat(mock_scraper, mock_bot, state, is_daily=False)

        mock_scraper.enrich_with_amenities.assert_not_called()

    @pytest.mark.asyncio
    async def test_enrichment_called_with_amenity_prefs(self):
        """must_haves present → enrichment is called."""
        prefs = Preferences(
            budget_max=5000,
            neighborhoods=["Chelsea"],
            must_haves=["Doorman"],
        )
        state = ChatState(chat_id=12345, preferences=prefs, preferences_ready=True)

        mock_scraper = AsyncMock()
        mock_scraper.search_with_retry = AsyncMock(return_value=[
            _raw_listing("1", neighborhood="Chelsea"),
        ])
        mock_scraper.enrich_with_amenities = AsyncMock(return_value=_enrichment_result(
            {
                "1": {
                    "building_amenities": ["Doorman", "Gym"],
                    "unit_features": ["Dishwasher"],
                    "description": "Great place",
                }
            },
            coverage=1.0,
            target_count=1,
        ))

        mock_bot = AsyncMock()
        mock_bot.send_text = AsyncMock()
        mock_bot.send_listing_photo = AsyncMock()

        scores = [{"id": "1", "include": True, "score": 85, "pros": ["Doorman"], "cons": []}]
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text=json.dumps(scores))]
        mock_client = AsyncMock()
        mock_client.messages.create = AsyncMock(return_value=mock_response)

        with patch("src.scanner.anthropic.AsyncAnthropic", return_value=mock_client), \
             patch("src.scanner.save_state"), \
             patch("src.scanner._pick_hero_photos", return_value={}):
            await scan_for_chat(mock_scraper, mock_bot, state, is_daily=False)

        mock_scraper.enrich_with_amenities.assert_called_once()

        # Verify the listing was enriched before LLM scoring
        # The LLM prompt should include the amenity data
        call_kwargs = mock_client.messages.create.call_args
        prompt_text = call_kwargs.kwargs.get("messages", call_kwargs[1].get("messages", []))[0]["content"]
        assert "confirmed_building_amenities" in prompt_text

    @pytest.mark.asyncio
    async def test_enrichment_failure_blocks_send(self):
        """Enrichment hard failure blocks listing send and returns failure notice."""
        prefs = Preferences(
            budget_max=5000,
            neighborhoods=["Chelsea"],
            must_haves=["Doorman"],
        )
        state = ChatState(chat_id=12345, preferences=prefs, preferences_ready=True)

        mock_scraper = AsyncMock()
        mock_scraper.search_with_retry = AsyncMock(return_value=[
            _raw_listing("1", neighborhood="Chelsea"),
        ])
        mock_scraper.enrich_with_amenities = AsyncMock(
            return_value=_enrichment_result(
                {},
                coverage=0.0,
                target_count=1,
                failed=True,
                failure_reason="actor_run_failed",
            )
        )

        mock_bot = AsyncMock()
        mock_bot.send_text = AsyncMock()
        mock_bot.send_listing_photo = AsyncMock()

        with (
            patch("src.scanner.save_state"),
            patch("src.scanner._llm_score_listings", new_callable=AsyncMock) as mock_llm,
            patch("src.scanner._pick_hero_photos", return_value={}),
        ):
            await scan_for_chat(mock_scraper, mock_bot, state, is_daily=False)

        mock_llm.assert_not_called()
        mock_bot.send_listing_photo.assert_not_called()
        mock_bot.send_text.assert_called_once()
        assert "Couldn't verify amenities reliably right now. Please retry shortly." in mock_bot.send_text.call_args[0][1]

    @pytest.mark.asyncio
    async def test_enrichment_partial_coverage_still_sends(self):
        """Partial coverage warns the user but still sends scored listings."""
        prefs = Preferences(
            budget_max=5000,
            neighborhoods=["Chelsea"],
            must_haves=["Doorman"],
        )
        state = ChatState(chat_id=12345, preferences=prefs, preferences_ready=True)

        mock_scraper = AsyncMock()
        mock_scraper.search_with_retry = AsyncMock(return_value=[
            _raw_listing("1", neighborhood="Chelsea"),
            _raw_listing("2", neighborhood="Chelsea"),
        ])
        mock_scraper.enrich_with_amenities = AsyncMock(return_value=_enrichment_result(
            {
                "1": {
                    "building_amenities": ["Doorman"],
                    "unit_features": [],
                    "description": "Only one mapped",
                }
            },
            coverage=0.5,
            target_count=2,
        ))

        mock_bot = AsyncMock()
        mock_bot.send_text = AsyncMock()
        mock_bot.send_listing_photo = AsyncMock()

        scores = [
            {"id": "1", "include": True, "score": 85, "pros": ["Doorman"], "cons": []},
            {"id": "2", "include": True, "score": 60, "pros": [], "cons": ["Amenities unverified"]},
        ]
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text=json.dumps(scores))]
        mock_client = AsyncMock()
        mock_client.messages.create = AsyncMock(return_value=mock_response)

        with (
            patch("src.scanner.anthropic.AsyncAnthropic", return_value=mock_client),
            patch("src.scanner.save_state"),
            patch("src.scanner._pick_hero_photos", return_value={}),
        ):
            await scan_for_chat(mock_scraper, mock_bot, state, is_daily=False)

        # Scoring should run and listings should be sent (no photos → send_text).
        mock_client.messages.create.assert_called_once()
        sent_texts = [call.args[1] for call in mock_bot.send_text.call_args_list]
        warn_msgs = [t for t in sent_texts if "StreetEasy blocked" in t]
        assert warn_msgs, "Expected partial-coverage warning to be sent"
        listing_cards = [t for t in sent_texts if "123 Test St" in t]
        assert len(listing_cards) == 2, f"Expected 2 listing cards, got {sent_texts!r}"

    @pytest.mark.asyncio
    async def test_enrichment_must_have_starved_blocks(self):
        """When must-haves exist and coverage is below the floor, block send."""
        prefs = Preferences(
            budget_max=5000,
            neighborhoods=["Chelsea"],
            must_haves=["Doorman"],
        )
        state = ChatState(chat_id=12345, preferences=prefs, preferences_ready=True)

        mock_scraper = AsyncMock()
        mock_scraper.search_with_retry = AsyncMock(return_value=[
            _raw_listing(str(i), neighborhood="Chelsea") for i in range(1, 11)
        ])
        mock_scraper.enrich_with_amenities = AsyncMock(return_value=_enrichment_result(
            {
                "1": {
                    "building_amenities": ["Doorman"],
                    "unit_features": [],
                    "description": "One verified out of ten",
                }
            },
            coverage=0.1,
            target_count=10,
        ))

        mock_bot = AsyncMock()
        mock_bot.send_text = AsyncMock()
        mock_bot.send_listing_photo = AsyncMock()

        with (
            patch("src.scanner.save_state"),
            patch("src.scanner._llm_score_listings", new_callable=AsyncMock) as mock_llm,
            patch("src.scanner._pick_hero_photos", return_value={}),
        ):
            await scan_for_chat(mock_scraper, mock_bot, state, is_daily=False)

        mock_llm.assert_not_called()
        mock_bot.send_listing_photo.assert_not_called()
        mock_bot.send_text.assert_called_once()
        assert "Couldn't verify amenities reliably right now. Please retry shortly." in mock_bot.send_text.call_args[0][1]

    @pytest.mark.asyncio
    async def test_daily_enrichment_failure_notice(self):
        """Daily scans use the daily-specific amenity verification failure notice."""
        prefs = Preferences(
            budget_max=5000,
            neighborhoods=["Chelsea"],
            must_haves=["Doorman"],
        )
        state = ChatState(chat_id=12345, preferences=prefs, preferences_ready=True)

        mock_scraper = AsyncMock()
        mock_scraper.search_with_retry = AsyncMock(return_value=[
            _raw_listing("1", neighborhood="Chelsea"),
        ])
        mock_scraper.enrich_with_amenities = AsyncMock(return_value=_enrichment_result(
            {},
            coverage=0.0,
            target_count=1,
            failed=True,
            failure_reason="actor_run_failed",
        ))

        mock_bot = AsyncMock()
        mock_bot.send_text = AsyncMock()
        mock_bot.send_listing_photo = AsyncMock()

        with (
            patch("src.scanner.save_state"),
            patch("src.scanner._llm_score_listings", new_callable=AsyncMock) as mock_llm,
            patch("src.scanner._pick_hero_photos", return_value={}),
        ):
            await scan_for_chat(mock_scraper, mock_bot, state, is_daily=True)

        mock_llm.assert_not_called()
        mock_bot.send_listing_photo.assert_not_called()
        mock_bot.send_text.assert_called_once()
        assert "Today's scan couldn't verify amenities reliably; I'll try again next run." in mock_bot.send_text.call_args[0][1]

    @pytest.mark.asyncio
    async def test_llm_prompt_includes_amenity_data(self):
        """Building amenities and unit features appear in LLM scoring prompt."""
        listings = [
            make_listing("1", building_amenities=["Doorman", "Gym"], unit_features=["Dishwasher"]),
        ]
        prefs = Preferences(budget_max=5000, must_haves=["Doorman"])

        scores = [{"id": "1", "include": True, "score": 85, "pros": ["Doorman"], "cons": []}]
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text=json.dumps(scores))]
        mock_client = AsyncMock()
        mock_client.messages.create = AsyncMock(return_value=mock_response)

        with patch("src.scanner.anthropic.AsyncAnthropic", return_value=mock_client):
            await _llm_score_listings(listings, prefs)

        call_kwargs = mock_client.messages.create.call_args
        prompt_text = call_kwargs.kwargs.get("messages", call_kwargs[1].get("messages", []))[0]["content"]
        assert "confirmed_building_amenities" in prompt_text
        assert "Doorman" in prompt_text
        assert "confirmed_unit_features" in prompt_text
        assert "Dishwasher" in prompt_text
        assert "amenity_signal_status" in prompt_text
        assert "amenity_text_dump" in prompt_text

    @pytest.mark.asyncio
    async def test_enrichment_uses_state_cache(self):
        """Cached amenity data is reused without calling scraper enrichment."""
        prefs = Preferences(
            budget_max=5000,
            neighborhoods=["Chelsea"],
            must_haves=["Doorman"],
        )
        state = ChatState(chat_id=12345, preferences=prefs, preferences_ready=True)
        state.amenity_cache["1"] = {
            "building_amenities": ["Doorman", "Gym"],
            "unit_features": ["Dishwasher"],
            "description": "Cached description",
        }

        mock_scraper = AsyncMock()
        mock_scraper.search_with_retry = AsyncMock(return_value=[
            _raw_listing("1", neighborhood="Chelsea"),
        ])
        mock_scraper.enrich_with_amenities = AsyncMock()

        mock_bot = AsyncMock()
        mock_bot.send_text = AsyncMock()
        mock_bot.send_listing_photo = AsyncMock()

        scores = [{"id": "1", "include": True, "score": 85, "pros": ["Doorman"], "cons": []}]
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text=json.dumps(scores))]
        mock_client = AsyncMock()
        mock_client.messages.create = AsyncMock(return_value=mock_response)

        with patch("src.scanner.anthropic.AsyncAnthropic", return_value=mock_client), \
             patch("src.scanner.save_state"), \
             patch("src.scanner._pick_hero_photos", return_value={}):
            await scan_for_chat(mock_scraper, mock_bot, state, is_daily=False)

        mock_scraper.enrich_with_amenities.assert_not_called()
        call_kwargs = mock_client.messages.create.call_args
        prompt_text = call_kwargs.kwargs.get("messages", call_kwargs[1].get("messages", []))[0]["content"]
        assert "confirmed_building_amenities" in prompt_text
        assert "Cached description" in prompt_text

    @pytest.mark.asyncio
    async def test_scan_dedupes_listing_ids_within_run_before_enrichment(self):
        """Duplicate listing ids from actor output are collapsed before enrichment."""
        prefs = Preferences(
            budget_max=5000,
            neighborhoods=["Chelsea"],
            must_haves=["Doorman"],
        )
        state = ChatState(chat_id=12345, preferences=prefs, preferences_ready=True)

        mock_scraper = AsyncMock()
        mock_scraper.search_with_retry = AsyncMock(return_value=[
            _raw_listing("1", neighborhood="Chelsea", url="https://streeteasy.com/building/a/1"),
            _raw_listing("1", neighborhood="Chelsea", url="https://streeteasy.com/building/a/1"),
            _raw_listing("2", neighborhood="Chelsea", url="https://streeteasy.com/building/b/2"),
        ])
        mock_scraper.enrich_with_amenities = AsyncMock(return_value=_enrichment_result(
            {
                "1": {"building_amenities": ["Doorman"], "unit_features": [], "description": ""},
                "2": {"building_amenities": ["Doorman"], "unit_features": [], "description": ""},
            },
            coverage=1.0,
            target_count=2,
        ))

        mock_bot = AsyncMock()
        mock_bot.send_text = AsyncMock()
        mock_bot.send_listing_photo = AsyncMock()

        scores = [
            {"id": "1", "include": True, "score": 80, "pros": [], "cons": []},
            {"id": "2", "include": True, "score": 70, "pros": [], "cons": []},
        ]
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text=json.dumps(scores))]
        mock_client = AsyncMock()
        mock_client.messages.create = AsyncMock(return_value=mock_response)

        with patch("src.scanner.anthropic.AsyncAnthropic", return_value=mock_client), \
             patch("src.scanner.save_state"), \
             patch("src.scanner._pick_hero_photos", return_value={}):
            await scan_for_chat(mock_scraper, mock_bot, state, is_daily=False)

        mock_scraper.enrich_with_amenities.assert_called_once()
        urls_arg = mock_scraper.enrich_with_amenities.call_args[0][0]
        assert sorted(urls_arg) == sorted([
            "https://streeteasy.com/building/a/1",
            "https://streeteasy.com/building/b/2",
        ])


class TestBidirectionalNeighborhoodAliases:
    """Bug 2: Neighborhood alias matching should work in both directions."""

    def test_user_pref_gramercy_park_listing_gramercy(self):
        """User says 'Gramercy Park', listing says 'Gramercy' -> should match."""
        prefs = Preferences(neighborhoods=["Gramercy Park"])
        listings = [make_listing("1", neighborhood="Gramercy")]
        result = _neighborhood_pre_filter(listings, prefs)
        assert len(result) == 1

    def test_user_pref_gramercy_listing_gramercy_park(self):
        """User says 'Gramercy', listing says 'Gramercy Park' -> should match."""
        prefs = Preferences(neighborhoods=["Gramercy"])
        listings = [make_listing("1", neighborhood="Gramercy Park")]
        result = _neighborhood_pre_filter(listings, prefs)
        assert len(result) == 1

    def test_user_pref_yorkville_listing_upper_east_side(self):
        """User says 'Yorkville', listing says 'Upper East Side' -> should match."""
        prefs = Preferences(neighborhoods=["Yorkville"])
        listings = [make_listing("1", neighborhood="Upper East Side")]
        result = _neighborhood_pre_filter(listings, prefs)
        assert len(result) == 1

    def test_user_pref_nomad_listing_flatiron(self):
        """User says 'NoMad', listing says 'Flatiron' -> should match."""
        prefs = Preferences(neighborhoods=["NoMad"])
        listings = [make_listing("1", neighborhood="Flatiron")]
        result = _neighborhood_pre_filter(listings, prefs)
        assert len(result) == 1

    def test_user_pref_lincoln_square_listing_upper_west_side(self):
        """User says 'Lincoln Square', listing says 'Upper West Side' -> should match."""
        prefs = Preferences(neighborhoods=["Lincoln Square"])
        listings = [make_listing("1", neighborhood="Upper West Side")]
        result = _neighborhood_pre_filter(listings, prefs)
        assert len(result) == 1

    def test_normalize_hood_idempotent(self):
        """Normalizing a canonical name returns itself."""
        assert _normalize_hood("Chelsea") == "chelsea"
        assert _normalize_hood("chelsea") == "chelsea"

    def test_normalize_hood_alias(self):
        """Normalizing an alias returns the canonical name."""
        assert _normalize_hood("Gramercy Park") == "gramercy"
        assert _normalize_hood("West Chelsea") == "chelsea"
        assert _normalize_hood("NoMad") == "flatiron"

    def test_all_aliases_bidirectional(self):
        """Every alias A->B should result in listings from B matching prefs with A."""
        for alias_from, alias_to in NEIGHBORHOOD_ALIASES.items():
            prefs = Preferences(neighborhoods=[alias_from.title()])
            listings = [make_listing("x", neighborhood=alias_to.title())]
            result = _neighborhood_pre_filter(listings, prefs)
            assert len(result) == 1, (
                f"Alias '{alias_from}' -> '{alias_to}' not bidirectional: "
                f"user pref='{alias_from.title()}', listing='{alias_to.title()}' should match"
            )


class TestFilterOverBudgetFix:
    """Bug 4: _filter_over_budget net_effective logic edge cases."""

    def test_gross_under_net_over_kept(self):
        """Gross=$3800 under budget, net_effective=$4200 over -> kept (gross is under)."""
        prefs = Preferences(budget_max=4000)
        listing = make_listing("1", price=3800, net_effective_price=4200)
        result = _filter_over_budget([listing], prefs)
        assert len(result) == 1

    def test_gross_over_net_under_kept(self):
        """Gross=$4500 over budget, net_effective=$3800 under -> kept (net is under)."""
        prefs = Preferences(budget_max=4000)
        listing = make_listing("1", price=4500, net_effective_price=3800)
        result = _filter_over_budget([listing], prefs)
        assert len(result) == 1

    def test_both_over_excluded(self):
        """Gross=$4500 and net=$4200, both over budget -> excluded."""
        prefs = Preferences(budget_max=4000)
        listing = make_listing("1", price=4500, net_effective_price=4200)
        result = _filter_over_budget([listing], prefs)
        assert len(result) == 0

    def test_no_net_gross_under_kept(self):
        """No net_effective, gross under budget -> kept."""
        prefs = Preferences(budget_max=4000)
        listing = make_listing("1", price=3500)
        result = _filter_over_budget([listing], prefs)
        assert len(result) == 1

    def test_no_net_gross_over_excluded(self):
        """No net_effective, gross over budget -> excluded."""
        prefs = Preferences(budget_max=4000)
        listing = make_listing("1", price=4500)
        result = _filter_over_budget([listing], prefs)
        assert len(result) == 0


class TestFilterUnderBudget:
    """Bug 8: Filter listings below budget_min."""

    def test_under_budget_min_excluded(self):
        """Listings priced below budget_min are dropped."""
        prefs = Preferences(budget_min=3000)
        listings = [
            make_listing("1", price=500),
            make_listing("2", price=2999),
            make_listing("3", price=3000),
            make_listing("4", price=5000),
        ]
        result = _filter_under_budget(listings, prefs)
        assert len(result) == 2
        assert {l.listing_id for l in result} == {"3", "4"}

    def test_no_budget_min_passes_all(self):
        """No budget_min set -> all listings pass."""
        prefs = Preferences()
        listings = [make_listing("1", price=100)]
        result = _filter_under_budget(listings, prefs)
        assert len(result) == 1

    def test_under_budget_in_pre_filter_chain(self):
        """_filter_under_budget is part of _apply_pre_filters."""
        prefs = Preferences(budget_min=3000, neighborhoods=["Chelsea"])
        listings = [
            make_listing("1", price=500, neighborhood="Chelsea"),
            make_listing("2", price=3500, neighborhood="Chelsea"),
        ]
        result = _apply_pre_filters(listings, prefs)
        assert len(result.listings) == 1
        assert result.listings[0].listing_id == "2"
        assert result.after_min_budget == 1


class TestStateSaveOnEmptyRawListings:
    """Bug 3: State must be saved when raw_listings is empty after cooldown reset."""

    @pytest.mark.asyncio
    async def test_save_state_called_on_empty_raw_listings(self):
        """save_state is called even when search returns 0 raw listings."""
        state = ChatState(
            chat_id=1,
            preferences=Preferences(budget_max=4000, neighborhoods=["Chelsea"]),
            preferences_ready=True,
            search_failure_streak=3,
            search_cooldown_until=datetime.now(timezone.utc) - timedelta(seconds=1),
        )

        mock_scraper = AsyncMock()
        mock_scraper.search_with_retry = AsyncMock(return_value=[])
        mock_bot = AsyncMock()

        with patch("src.scanner.save_state") as mock_save:
            await scan_for_chat(mock_scraper, mock_bot, state, is_daily=False)
            mock_save.assert_called()

        # Failure streak should be reset
        assert state.search_failure_streak == 0
        assert state.search_cooldown_until is None


class TestParseScoringResponse:
    """Bug 11: JSON parse robustness for LLM scoring responses."""

    def test_clean_json(self):
        listings = [make_listing("1", match_score=None)]
        result = _parse_scoring_response(
            '[{"id":"1","include":true,"score":80,"pros":["good"],"cons":[]}]',
            listings,
        )
        assert len(result.listings) == 1
        assert result.listings[0].match_score == 80

    def test_markdown_code_fences(self):
        listings = [make_listing("1", match_score=None)]
        result = _parse_scoring_response(
            '```json\n[{"id":"1","include":true,"score":85,"pros":[],"cons":[]}]\n```',
            listings,
        )
        assert len(result.listings) == 1
        assert result.listings[0].match_score == 85

    def test_trailing_text_after_json(self):
        """Claude sometimes appends commentary after the JSON array."""
        listings = [make_listing("1", match_score=None), make_listing("2", match_score=None)]
        result = _parse_scoring_response(
            '[{"id":"1","include":true,"score":85,"pros":[],"cons":[]},{"id":"2","include":true,"score":70,"pros":[],"cons":[]}]\n\nNote: These scores are based on the criteria.',
            listings,
        )
        assert len(result.listings) == 2

    def test_leading_text_before_json(self):
        """Claude sometimes prepends text before the JSON array."""
        listings = [make_listing("1", match_score=None)]
        result = _parse_scoring_response(
            'Here are the scores:\n[{"id":"1","include":true,"score":85,"pros":[],"cons":[]}]',
            listings,
        )
        assert len(result.listings) == 1

    def test_brackets_inside_strings(self):
        """Brackets inside JSON string values don't confuse the parser."""
        listings = [make_listing("1", match_score=None)]
        result = _parse_scoring_response(
            '[{"id":"1","include":true,"score":85,"pros":["Has [nice] amenities"],"cons":[]}]',
            listings,
        )
        assert result.listings[0].pros == ["Has [nice] amenities"]

    def test_no_json_array_raises(self):
        listings = [make_listing("1", match_score=None)]
        with pytest.raises(ValueError, match="No JSON array"):
            _parse_scoring_response("No JSON here", listings)
