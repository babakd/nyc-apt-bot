"""Tests for Apify scraper polling, abort, retry, and config."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.apify_scraper import ApifyScraperError, ApifyScraper


def _make_scraper():
    """Create an ApifyScraper with mocked env."""
    with patch.dict("os.environ", {"APIFY_API_TOKEN": "test-token"}):
        return ApifyScraper()


def _mock_prefs():
    """Minimal preferences for testing."""
    from src.models import Preferences
    return Preferences(budget_max=4000, neighborhoods=["Chelsea"])


class TestPollingAndAbort:
    @pytest.mark.asyncio
    async def test_early_abort_on_zero_items(self):
        """Run is aborted after ABORT_AFTER_SECS_NO_ITEMS with 0 items."""
        scraper = _make_scraper()

        # Mock the actor start
        mock_run_info = {"id": "run123", "defaultDatasetId": "ds123"}

        mock_actor = AsyncMock()
        mock_actor.start = AsyncMock(return_value=mock_run_info)
        scraper._client.actor = MagicMock(return_value=mock_actor)

        # Mock run client — always RUNNING
        mock_run_client = AsyncMock()
        mock_run_client.get = AsyncMock(return_value={"status": "RUNNING"})
        mock_run_client.abort = AsyncMock()
        scraper._client.run = MagicMock(return_value=mock_run_client)

        # Mock dataset client — always 0 items
        mock_ds_client = AsyncMock()
        mock_ds_client.get = AsyncMock(return_value={"itemCount": 0})
        scraper._client.dataset = MagicMock(return_value=mock_ds_client)

        with patch("src.apify_scraper.asyncio.sleep", new_callable=AsyncMock):
            with pytest.raises(ApifyScraperError, match="aborted"):
                await scraper.search_streeteasy(_mock_prefs())

        mock_run_client.abort.assert_called_once()

    @pytest.mark.asyncio
    async def test_successful_run_completes(self):
        """Run that succeeds on 2nd poll returns listings."""
        scraper = _make_scraper()

        mock_run_info = {"id": "run123", "defaultDatasetId": "ds123"}

        mock_actor = AsyncMock()
        mock_actor.start = AsyncMock(return_value=mock_run_info)
        scraper._client.actor = MagicMock(return_value=mock_actor)

        # First poll: RUNNING, second poll: SUCCEEDED
        poll_count = 0

        async def mock_run_get():
            nonlocal poll_count
            poll_count += 1
            if poll_count == 1:
                return {"status": "RUNNING"}
            return {"status": "SUCCEEDED"}

        mock_run_client = AsyncMock()
        mock_run_client.get = AsyncMock(side_effect=mock_run_get)
        scraper._client.run = MagicMock(return_value=mock_run_client)

        # Dataset — first poll shows items, then list_items returns data
        mock_ds_client = AsyncMock()
        mock_ds_client.get = AsyncMock(return_value={"itemCount": 5})
        mock_ds_client.list_items = AsyncMock(return_value=MagicMock(items=[
            {"node": {"id": "1", "areaName": "Chelsea", "price": 3000,
                      "bedroomCount": 1, "fullBathroomCount": 1, "halfBathroomCount": 0,
                      "street": "100 Main", "unit": "1A", "urlPath": "/rental/1",
                      "photos": [], "noFee": True}},
        ]))
        scraper._client.dataset = MagicMock(return_value=mock_ds_client)

        with patch("src.apify_scraper.asyncio.sleep", new_callable=AsyncMock):
            results = await scraper.search_streeteasy(_mock_prefs())

        assert len(results) == 1
        assert results[0]["listing_id"] == "1"


class TestRetry:
    @pytest.mark.asyncio
    async def test_retry_succeeds_on_second_attempt(self):
        """First call raises, second succeeds — returns results."""
        scraper = _make_scraper()

        call_count = 0

        async def mock_search(prefs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise ApifyScraperError("WAF block")
            return [{"listing_id": "1", "url": "", "address": "Test", "neighborhood": "Chelsea",
                      "price": 3000, "bedrooms": 1, "bathrooms": 1}]

        scraper.search_streeteasy = mock_search

        with patch("src.apify_scraper.asyncio.sleep", new_callable=AsyncMock):
            results = await scraper.search_with_retry(_mock_prefs())

        assert len(results) == 1
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_retry_exhausted_raises(self):
        """All retries fail — error propagated."""
        scraper = _make_scraper()

        async def mock_search(prefs):
            raise ApifyScraperError("WAF block")

        scraper.search_streeteasy = mock_search

        with patch("src.apify_scraper.asyncio.sleep", new_callable=AsyncMock):
            with pytest.raises(ApifyScraperError, match="WAF block"):
                await scraper.search_with_retry(_mock_prefs())

    @pytest.mark.asyncio
    async def test_search_with_retry_returns_empty_without_retry(self):
        """Empty results from a successful run are returned immediately — no retry."""
        scraper = _make_scraper()

        call_count = 0

        async def mock_search(prefs):
            nonlocal call_count
            call_count += 1
            return []

        scraper.search_streeteasy = mock_search

        with patch("src.apify_scraper.asyncio.sleep", new_callable=AsyncMock):
            results = await scraper.search_with_retry(_mock_prefs())

        assert results == []
        assert call_count == 1  # Only one attempt — no retries for empty results


class TestConfig:
    @pytest.mark.asyncio
    async def test_us_country_code_in_proxy(self):
        """Proxy config includes countryCode: US."""
        scraper = _make_scraper()

        mock_run_info = {"id": "run123", "defaultDatasetId": "ds123"}
        mock_actor = AsyncMock()
        mock_actor.start = AsyncMock(return_value=mock_run_info)
        scraper._client.actor = MagicMock(return_value=mock_actor)

        # Make run succeed immediately
        mock_run_client = AsyncMock()
        mock_run_client.get = AsyncMock(return_value={"status": "SUCCEEDED"})
        scraper._client.run = MagicMock(return_value=mock_run_client)

        mock_ds_client = AsyncMock()
        mock_ds_client.list_items = AsyncMock(return_value=MagicMock(items=[]))
        scraper._client.dataset = MagicMock(return_value=mock_ds_client)

        with patch("src.apify_scraper.asyncio.sleep", new_callable=AsyncMock):
            await scraper.search_streeteasy(_mock_prefs())

        call_kwargs = mock_actor.start.call_args
        run_input = call_kwargs.kwargs.get("run_input") or call_kwargs[1].get("run_input")
        assert run_input["proxy"]["countryCode"] == "US"

    @pytest.mark.asyncio
    async def test_max_request_retries_in_input(self):
        """run_input includes maxRequestRetries=15."""
        scraper = _make_scraper()

        mock_run_info = {"id": "run123", "defaultDatasetId": "ds123"}
        mock_actor = AsyncMock()
        mock_actor.start = AsyncMock(return_value=mock_run_info)
        scraper._client.actor = MagicMock(return_value=mock_actor)

        mock_run_client = AsyncMock()
        mock_run_client.get = AsyncMock(return_value={"status": "SUCCEEDED"})
        scraper._client.run = MagicMock(return_value=mock_run_client)

        mock_ds_client = AsyncMock()
        mock_ds_client.list_items = AsyncMock(return_value=MagicMock(items=[]))
        scraper._client.dataset = MagicMock(return_value=mock_ds_client)

        with patch("src.apify_scraper.asyncio.sleep", new_callable=AsyncMock):
            await scraper.search_streeteasy(_mock_prefs())

        call_kwargs = mock_actor.start.call_args
        run_input = call_kwargs.kwargs.get("run_input") or call_kwargs[1].get("run_input")
        assert run_input["maxRequestRetries"] == 15
