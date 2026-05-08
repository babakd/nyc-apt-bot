"""Tests for Apify scraper polling, abort, retry, and config."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from apify_client._errors import ApifyApiError

from src.apify_scraper import (
    ActorRunResult,
    AMENITY_DISPLAY_NAMES,
    ApifyScraperError,
    ApifyScraper,
    _is_unhealthy_empty_run,
    _build_path_urls,
    _map_detail_amenities,
    _normalize_amenity,
)


@pytest.fixture(autouse=True)
def _isolate_actor_storage_paths(tmp_path, monkeypatch):
    """Isolate actor pin/canary storage paths for every scraper test."""
    monkeypatch.setattr(
        "src.storage.APIFY_ACTOR_BUILD_PIN_FILE",
        str(tmp_path / "apify_actor_build_pin.json"),
    )
    monkeypatch.setattr(
        "src.storage.APIFY_ACTOR_CANARY_FILE",
        str(tmp_path / "apify_actor_canary_status.json"),
    )
    monkeypatch.setattr(
        "src.storage.APIFY_ACTOR_CANARY_URLS_FILE",
        str(tmp_path / "apify_actor_canary_urls.json"),
    )


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
    async def test_partial_stop_after_item_target(self):
        """Interactive searches can stop once enough candidate rows are available."""
        scraper = _make_scraper()

        mock_run_info = {"id": "run123", "defaultDatasetId": "ds123"}
        mock_actor = AsyncMock()
        mock_actor.start = AsyncMock(return_value=mock_run_info)
        scraper._client.actor = MagicMock(return_value=mock_actor)

        mock_run_client = AsyncMock()
        mock_run_client.get = AsyncMock(return_value={"status": "RUNNING"})
        mock_run_client.abort = AsyncMock()
        scraper._client.run = MagicMock(return_value=mock_run_client)

        mock_ds_client = AsyncMock()
        mock_ds_client.get = AsyncMock(return_value={"itemCount": 2})
        mock_ds_client.list_items = AsyncMock(return_value=MagicMock(items=[{"id": "1"}, {"id": "2"}]))
        scraper._client.dataset = MagicMock(return_value=mock_ds_client)

        with patch("src.apify_scraper.asyncio.sleep", new_callable=AsyncMock):
            result = await scraper._run_actor(
                start_urls=[{"url": "https://streeteasy.com/for-rent/chelsea"}],
                max_items=2,
                allow_partial_after_items=True,
            )

        assert result.status == "PARTIAL"
        assert len(result.items) == 2
        assert "Stopped after collecting" in result.status_message
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
        """First attempt fails both paths, second attempt succeeds — returns results."""
        scraper = _make_scraper()

        attempt_count = 0

        async def mock_path_search(prefs, max_items=2000):
            nonlocal attempt_count
            attempt_count += 1
            if attempt_count == 1:
                raise ApifyScraperError("WAF block")
            return [{"listing_id": "1", "url": "", "address": "Test", "neighborhood": "Chelsea",
                      "price": 3000, "bedrooms": 1, "bathrooms": 1}]

        async def mock_pipe_search(prefs):
            raise ApifyScraperError("WAF block")

        scraper.search_by_neighborhoods = mock_path_search
        scraper.search_streeteasy = mock_pipe_search

        with patch("src.apify_scraper.asyncio.sleep", new_callable=AsyncMock):
            results = await scraper.search_with_retry(_mock_prefs())

        assert len(results) == 1
        assert attempt_count == 2

    @pytest.mark.asyncio
    async def test_retry_exhausted_raises(self):
        """All retries fail — error propagated."""
        scraper = _make_scraper()

        async def mock_path_search(prefs, max_items=2000):
            raise ApifyScraperError("WAF block")

        async def mock_pipe_search(prefs):
            raise ApifyScraperError("WAF block")

        scraper.search_by_neighborhoods = mock_path_search
        scraper.search_streeteasy = mock_pipe_search

        with patch("src.apify_scraper.asyncio.sleep", new_callable=AsyncMock):
            with pytest.raises(ApifyScraperError, match="WAF block"):
                await scraper.search_with_retry(_mock_prefs())

    @pytest.mark.asyncio
    async def test_search_with_retry_returns_empty_without_retry(self):
        """Empty results from a successful run are returned immediately — no retry."""
        scraper = _make_scraper()

        call_count = 0

        async def mock_path_search(prefs, max_items=2000):
            nonlocal call_count
            call_count += 1
            return []

        scraper.search_by_neighborhoods = mock_path_search

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


class TestBuildPathUrls:
    def test_single_neighborhood(self):
        """Generates one path URL for a single neighborhood."""
        from src.models import Preferences
        prefs = Preferences(neighborhoods=["Chelsea"])
        urls = _build_path_urls(prefs)
        assert len(urls) == 1
        assert urls[0] == {"url": "https://streeteasy.com/for-rent/chelsea"}

    def test_multiple_neighborhoods(self):
        """Generates one path URL per neighborhood."""
        from src.models import Preferences
        prefs = Preferences(neighborhoods=["Chelsea", "West Village", "SoHo"])
        urls = _build_path_urls(prefs)
        assert len(urls) == 3
        assert urls[0] == {"url": "https://streeteasy.com/for-rent/chelsea"}
        assert urls[1] == {"url": "https://streeteasy.com/for-rent/west-village"}
        assert urls[2] == {"url": "https://streeteasy.com/for-rent/soho"}

    def test_empty_neighborhoods(self):
        """Empty neighborhoods returns empty list."""
        from src.models import Preferences
        prefs = Preferences(neighborhoods=[])
        urls = _build_path_urls(prefs)
        assert urls == []

    def test_unknown_neighborhood_slug_fallback(self):
        """Unknown neighborhoods fall back to lowered-hyphenated slug."""
        from src.models import Preferences
        prefs = Preferences(neighborhoods=["Some New Place"])
        urls = _build_path_urls(prefs)
        assert len(urls) == 1
        assert urls[0] == {"url": "https://streeteasy.com/for-rent/some-new-place"}


class TestSearchByNeighborhoods:
    @pytest.mark.asyncio
    async def test_passes_multiple_start_urls(self):
        """Actor is called with one startUrl per neighborhood."""
        scraper = _make_scraper()

        mock_run_info = {"id": "run123", "defaultDatasetId": "ds123"}
        mock_actor = AsyncMock()
        mock_actor.start = AsyncMock(return_value=mock_run_info)
        scraper._client.actor = MagicMock(return_value=mock_actor)

        mock_run_client = AsyncMock()
        mock_run_client.get = AsyncMock(return_value={"status": "SUCCEEDED"})
        scraper._client.run = MagicMock(return_value=mock_run_client)

        mock_ds_client = AsyncMock()
        mock_ds_client.list_items = AsyncMock(return_value=MagicMock(items=[
            {"node": {"id": "1", "areaName": "Chelsea", "price": 3000,
                      "bedroomCount": 1, "fullBathroomCount": 1, "halfBathroomCount": 0,
                      "street": "100 Main", "unit": "1A", "urlPath": "/rental/1",
                      "photos": [], "noFee": True}},
            {"node": {"id": "2", "areaName": "West Village", "price": 4000,
                      "bedroomCount": 2, "fullBathroomCount": 1, "halfBathroomCount": 0,
                      "street": "200 Bleecker", "unit": "2B", "urlPath": "/rental/2",
                      "photos": [], "noFee": False}},
        ]))
        scraper._client.dataset = MagicMock(return_value=mock_ds_client)

        from src.models import Preferences
        prefs = Preferences(neighborhoods=["Chelsea", "West Village"])

        with patch("src.apify_scraper.asyncio.sleep", new_callable=AsyncMock):
            results = await scraper.search_by_neighborhoods(prefs)

        assert len(results) == 2

        # Verify startUrls had both neighborhood URLs
        call_kwargs = mock_actor.start.call_args
        run_input = call_kwargs.kwargs.get("run_input") or call_kwargs[1].get("run_input")
        start_urls = run_input["startUrls"]
        assert len(start_urls) == 2
        assert any("chelsea" in u["url"] for u in start_urls)
        assert any("west-village" in u["url"] for u in start_urls)

    @pytest.mark.asyncio
    async def test_empty_neighborhoods_raises(self):
        """Raises ApifyScraperError when no neighborhoods specified."""
        scraper = _make_scraper()
        from src.models import Preferences
        prefs = Preferences(neighborhoods=[])

        with pytest.raises(ApifyScraperError, match="No neighborhood URLs"):
            await scraper.search_by_neighborhoods(prefs)


class TestEnrichWithAmenities:
    @pytest.mark.asyncio
    async def test_enriches_listings(self):
        """Returns typed amenity enrichment data mapped to listing IDs."""
        scraper = _make_scraper()

        mock_run_info = {"id": "run123", "defaultDatasetId": "ds123"}
        mock_actor = AsyncMock()
        mock_actor.start = AsyncMock(return_value=mock_run_info)
        scraper._client.actor = MagicMock(return_value=mock_actor)

        mock_run_client = AsyncMock()
        mock_run_client.get = AsyncMock(return_value={"status": "SUCCEEDED"})
        scraper._client.run = MagicMock(return_value=mock_run_client)

        mock_ds_client = AsyncMock()
        mock_ds_client.list_items = AsyncMock(return_value=MagicMock(items=[
            {
                "originalUrl": "https://streeteasy.com/building/test/1a",
                "basicInfo": {"id": 123},
                "federatedData": {
                    "rentalByListingId": {
                        "description": "Beautiful apartment",
                        "propertyDetails": {
                            "amenities": {"list": ["DOORMAN", "GYM", "ELEVATOR"]},
                            "features": {"list": ["DISHWASHER", "CENTRAL_AC"]},
                        },
                    },
                },
            },
        ]))
        scraper._client.dataset = MagicMock(return_value=mock_ds_client)

        url_to_id = {"https://streeteasy.com/building/test/1a": "123"}

        with patch("src.apify_scraper.asyncio.sleep", new_callable=AsyncMock):
            result = await scraper.enrich_with_amenities(
                ["https://streeteasy.com/building/test/1a"],
                url_to_id,
            )

        assert "123" in result.data_by_listing_id
        data = result.data_by_listing_id["123"]
        assert "Doorman" in data["building_amenities"]
        assert "Gym" in data["building_amenities"]
        assert "Dishwasher" in data["unit_features"]
        assert data["description"] == "Beautiful apartment"
        assert result.coverage == 1.0
        assert result.target_count == 1
        assert result.failed is False
        assert len(result.run_summaries) == 1
        assert result.run_summaries[0].mapped_count == 1

    @pytest.mark.asyncio
    async def test_empty_urls_returns_empty(self):
        """Empty URL list returns an empty typed enrichment result."""
        scraper = _make_scraper()
        result = await scraper.enrich_with_amenities([], {})
        assert result.data_by_listing_id == {}
        assert result.coverage == 1.0
        assert result.target_count == 0
        assert result.failed is False
        assert result.run_summaries == []

    @pytest.mark.asyncio
    async def test_partial_enrichment(self):
        """Partial maps still return diagnostics with accurate coverage."""
        scraper = _make_scraper()

        mock_run_info = {"id": "run123", "defaultDatasetId": "ds123"}
        mock_actor = AsyncMock()
        mock_actor.start = AsyncMock(return_value=mock_run_info)
        scraper._client.actor = MagicMock(return_value=mock_actor)

        mock_run_client = AsyncMock()
        mock_run_client.get = AsyncMock(return_value={"status": "SUCCEEDED"})
        scraper._client.run = MagicMock(return_value=mock_run_client)

        mock_ds_client = AsyncMock()
        mock_ds_client.list_items = AsyncMock(return_value=MagicMock(items=[
            {
                "originalUrl": "https://streeteasy.com/building/test/1a",
                "federatedData": {
                    "rentalByListingId": {
                        "id": 123,
                        "propertyDetails": {
                            "amenities": {"list": ["DOORMAN"]},
                            "features": {"list": []},
                        },
                    },
                },
            },
            # Item with URL not in url_to_id (unmapped)
            {
                "originalUrl": "https://streeteasy.com/building/unknown/2b",
                "federatedData": {},
            },
        ]))
        scraper._client.dataset = MagicMock(return_value=mock_ds_client)

        url_to_id = {"https://streeteasy.com/building/test/1a": "123"}

        with patch("src.apify_scraper.asyncio.sleep", new_callable=AsyncMock):
            result = await scraper.enrich_with_amenities(
                [
                    "https://streeteasy.com/building/test/1a",
                    "https://streeteasy.com/building/test/2b",
                ],
                {
                    "https://streeteasy.com/building/test/1a": "123",
                    "https://streeteasy.com/building/test/2b": "456",
                },
            )

        assert len(result.data_by_listing_id) == 1
        assert "123" in result.data_by_listing_id
        assert result.target_count == 2
        assert result.coverage == 0.5
        assert result.failed is False
        assert len(result.run_summaries) >= 1
        assert result.run_summaries[0].mapped_count == 1

    @pytest.mark.asyncio
    async def test_enrichment_maps_by_listing_id_when_url_missing(self):
        """If URL mapping is missing, listing id fallback still maps the row."""
        scraper = _make_scraper()

        mock_run_info = {"id": "run123", "defaultDatasetId": "ds123"}
        mock_actor = AsyncMock()
        mock_actor.start = AsyncMock(return_value=mock_run_info)
        scraper._client.actor = MagicMock(return_value=mock_actor)

        mock_run_client = AsyncMock()
        mock_run_client.get = AsyncMock(return_value={"status": "SUCCEEDED"})
        scraper._client.run = MagicMock(return_value=mock_run_client)

        mock_ds_client = AsyncMock()
        mock_ds_client.list_items = AsyncMock(return_value=MagicMock(items=[
            {
                "federatedData": {
                    "rentalByListingId": {
                        "id": 555,
                        "propertyDetails": {
                            "amenities": {"list": ["GYM"]},
                            "features": {"list": []},
                        },
                    },
                },
            },
        ]))
        scraper._client.dataset = MagicMock(return_value=mock_ds_client)

        with patch("src.apify_scraper.asyncio.sleep", new_callable=AsyncMock):
            result = await scraper.enrich_with_amenities(
                ["https://streeteasy.com/building/test/1a"],
                {"https://streeteasy.com/building/test/1a": "555"},
            )

        assert "555" in result.data_by_listing_id
        assert result.data_by_listing_id["555"]["building_amenities"] == ["Gym"]
        assert result.coverage == 1.0


class TestBuildHealthAndPinning:
    def test_unhealthy_empty_run_signature_detected(self):
        run_result = ActorRunResult(
            items=[],
            run_id="r1",
            dataset_id="d1",
            build_id="b1",
            status="SUCCEEDED",
            status_message="Total 2 requests: 0 succeeded, 2 failed.",
            requests_succeeded=0,
            requests_failed=2,
        )
        assert _is_unhealthy_empty_run(run_result) is True

    @pytest.mark.asyncio
    async def test_enrichment_treats_unhealthy_empty_as_failure(self):
        """SUCCEEDED+empty+all-failed status is treated as enrichment failure."""
        scraper = _make_scraper()

        mock_run_info = {"id": "run123", "defaultDatasetId": "ds123"}
        mock_actor = AsyncMock()
        mock_actor.start = AsyncMock(return_value=mock_run_info)
        scraper._client.actor = MagicMock(return_value=mock_actor)

        mock_run_client = AsyncMock()
        mock_run_client.get = AsyncMock(
            return_value={
                "status": "SUCCEEDED",
                "statusMessage": "Total 1 requests: 0 succeeded, 1 failed.",
            }
        )
        scraper._client.run = MagicMock(return_value=mock_run_client)

        mock_ds_client = AsyncMock()
        mock_ds_client.list_items = AsyncMock(return_value=MagicMock(items=[]))
        scraper._client.dataset = MagicMock(return_value=mock_ds_client)

        with patch("src.apify_scraper.asyncio.sleep", new_callable=AsyncMock):
            result = await scraper.enrich_with_amenities(
                ["https://streeteasy.com/building/test/1a"],
                {"https://streeteasy.com/building/test/1a": "123"},
            )

        assert result.failed is True
        assert result.coverage == 0.0
        assert result.data_by_listing_id == {}
        assert "all requests failed" in (result.failure_reason or "").lower()

    @pytest.mark.asyncio
    async def test_pinned_build_failure_falls_back_and_promotes(self):
        """Pinned build failure falls back to latest and updates build pin."""
        scraper = _make_scraper()

        success = ActorRunResult(
            items=[{"node": {"id": "1"}}],
            run_id="latest-run",
            dataset_id="ds-latest",
            build_id="latest-build-id",
            status="SUCCEEDED",
            status_message="Total 1 requests: 1 succeeded, 0 failed.",
            requests_succeeded=1,
            requests_failed=0,
        )
        scraper._run_actor = AsyncMock(
            side_effect=[ApifyScraperError("pinned failed"), success]
        )

        with (
            patch.object(ApifyScraper, "_effective_pin", return_value="pinned-build"),
            patch.object(ApifyScraper, "_can_persist_pin", return_value=True),
            patch.object(
                ApifyScraper,
                "_build_number_for_id",
                new_callable=AsyncMock,
                return_value="0.0.95",
            ),
            patch("src.apify_scraper.save_actor_build_pin") as save_pin,
        ):
            run_result = await scraper._run_actor_with_build_policy(
                start_urls=[{"url": "https://streeteasy.com/for-rent/chelsea"}],
                max_items=1,
                abort_after_secs_no_items=60,
                max_wait_secs=60,
                poll_context="test",
                fail_on_unhealthy_empty=True,
                force_build=None,
                allow_build_fallback=True,
                auto_promote_on_fallback=True,
            )

        assert run_result.run_id == "latest-run"
        assert run_result.build_id == "latest-build-id"
        assert scraper._run_actor.call_count == 2
        assert scraper._run_actor.call_args_list[0].kwargs["build"] == "pinned-build"
        assert scraper._run_actor.call_args_list[1].kwargs["build"] == "latest"
        save_pin.assert_called_once_with("0.0.95")

    @pytest.mark.asyncio
    async def test_apify_api_error_falls_back_to_latest(self):
        """ApifyApiError (e.g. deleted build) triggers fallback to latest build."""
        scraper = _make_scraper()

        success = ActorRunResult(
            items=[{"node": {"id": "1"}}],
            run_id="latest-run",
            dataset_id="ds-latest",
            build_id="latest-build-id",
            status="SUCCEEDED",
            status_message="Total 1 requests: 1 succeeded, 0 failed.",
            requests_succeeded=1,
            requests_failed=0,
        )
        # ApifyApiError is what the apify-client raises for a deleted/missing build
        api_error = ApifyApiError.__new__(ApifyApiError)
        api_error.args = ('Build with tag "stale-pin" was not found.',)
        api_error.message = 'Build with tag "stale-pin" was not found.'
        api_error.type = "record-not-found"
        api_error.status_code = 404

        scraper._run_actor = AsyncMock(side_effect=[api_error, success])

        with (
            patch.object(ApifyScraper, "_effective_pin", return_value="stale-pin"),
            patch.object(ApifyScraper, "_can_persist_pin", return_value=True),
            patch.object(
                ApifyScraper,
                "_build_number_for_id",
                new_callable=AsyncMock,
                return_value="0.0.95",
            ),
            patch("src.apify_scraper.save_actor_build_pin") as save_pin,
        ):
            run_result = await scraper._run_actor_with_build_policy(
                start_urls=[{"url": "https://streeteasy.com/for-rent/chelsea"}],
                max_items=1,
                abort_after_secs_no_items=60,
                max_wait_secs=60,
                poll_context="test",
                fail_on_unhealthy_empty=True,
                force_build=None,
                allow_build_fallback=True,
                auto_promote_on_fallback=True,
            )

        assert run_result.run_id == "latest-run"
        assert scraper._run_actor.call_count == 2
        assert scraper._run_actor.call_args_list[0].kwargs["build"] == "stale-pin"
        assert scraper._run_actor.call_args_list[1].kwargs["build"] == "latest"
        save_pin.assert_called_once_with("0.0.95")


class TestMapDetailAmenities:
    def test_full_data(self):
        """Extracts amenities and features from complete detail page data."""
        item = {
            "url": "https://streeteasy.com/building/test/1a",
            "basicInfo": {"id": 123},
            "federatedData": {
                "rentalByListingId": {
                    "id": 123,
                    "description": "Lovely apartment",
                    "propertyDetails": {
                        "amenities": {"list": ["DOORMAN", "GYM"]},
                        "features": {"list": ["DISHWASHER", "WASHER_DRYER"]},
                    },
                },
            },
        }
        url, listing_id, data = _map_detail_amenities(item)
        assert url == "https://streeteasy.com/building/test/1a"
        assert listing_id == "123"
        assert data["building_amenities"] == ["Doorman", "Gym"]
        assert data["unit_features"] == ["Dishwasher", "In-unit Laundry"]
        assert data["description"] == "Lovely apartment"

    def test_missing_federated_data(self):
        """Returns empty lists when federatedData is missing."""
        item = {"originalUrl": "https://streeteasy.com/building/test/1a"}
        url, listing_id, data = _map_detail_amenities(item)
        assert url == "https://streeteasy.com/building/test/1a"
        assert listing_id == ""
        assert data["building_amenities"] == []
        assert data["unit_features"] == []
        assert data["description"] == ""

    def test_empty_amenity_lists(self):
        """Returns empty lists when amenity/feature lists are empty."""
        item = {
            "url": "https://streeteasy.com/building/test/1a",
            "federatedData": {
                "rentalByListingId": {
                    "propertyDetails": {
                        "amenities": {"list": []},
                        "features": {"list": []},
                    },
                },
            },
        }
        _, _, data = _map_detail_amenities(item)
        assert data["building_amenities"] == []
        assert data["unit_features"] == []

    def test_no_url(self):
        """Returns empty string URL when url is missing."""
        item = {"federatedData": {}}
        url, listing_id, _ = _map_detail_amenities(item)
        assert url == ""
        assert listing_id == ""


class TestNormalizeAmenity:
    def test_known_enum(self):
        """Known enum values get display names."""
        assert _normalize_amenity("DOORMAN") == "Doorman"
        assert _normalize_amenity("WASHER_DRYER") == "In-unit Laundry"
        assert _normalize_amenity("CENTRAL_AC") == "Central AC"

    def test_unknown_enum_fallback(self):
        """Unknown enum values get title-cased with underscores replaced."""
        assert _normalize_amenity("SOME_UNKNOWN_AMENITY") == "Some Unknown Amenity"

    def test_all_display_names_defined(self):
        """All values in AMENITY_DISPLAY_NAMES are non-empty strings."""
        for key, value in AMENITY_DISPLAY_NAMES.items():
            assert isinstance(value, str)
            assert len(value) > 0


class TestSearchWithRetryFallbackChain:
    @pytest.mark.asyncio
    async def test_path_fails_pipe_succeeds(self):
        """When path-based search fails, falls back to pipe URL."""
        scraper = _make_scraper()

        async def mock_path_search(prefs, max_items=2000):
            raise ApifyScraperError("Path search failed")

        async def mock_pipe_search(prefs):
            return [{"listing_id": "1", "url": "", "address": "Test",
                      "neighborhood": "Chelsea", "price": 3000, "bedrooms": 1, "bathrooms": 1}]

        scraper.search_by_neighborhoods = mock_path_search
        scraper.search_streeteasy = mock_pipe_search

        with patch("src.apify_scraper.asyncio.sleep", new_callable=AsyncMock):
            results = await scraper.search_with_retry(_mock_prefs())

        assert len(results) == 1

    @pytest.mark.asyncio
    async def test_no_neighborhoods_uses_pipe_directly(self):
        """When no neighborhoods, uses pipe URL without trying path search."""
        scraper = _make_scraper()

        path_called = False

        async def mock_path_search(prefs, max_items=2000):
            nonlocal path_called
            path_called = True
            return []

        async def mock_pipe_search(prefs):
            return [{"listing_id": "1", "url": "", "address": "Test",
                      "neighborhood": "Chelsea", "price": 3000, "bedrooms": 1, "bathrooms": 1}]

        scraper.search_by_neighborhoods = mock_path_search
        scraper.search_streeteasy = mock_pipe_search

        from src.models import Preferences
        prefs = Preferences(budget_max=4000)  # No neighborhoods

        with patch("src.apify_scraper.asyncio.sleep", new_callable=AsyncMock):
            results = await scraper.search_with_retry(prefs)

        assert len(results) == 1
        assert not path_called  # Path search was not attempted

    @pytest.mark.asyncio
    async def test_apify_api_error_path_falls_back_to_pipe(self):
        """ApifyApiError from path search falls back to pipe URL (not swallowed)."""
        scraper = _make_scraper()

        api_error = ApifyApiError.__new__(ApifyApiError)
        api_error.args = ('Build not found',)
        api_error.message = 'Build not found'
        api_error.type = "record-not-found"
        api_error.status_code = 404

        async def mock_path_search(prefs, max_items=2000):
            raise api_error

        async def mock_pipe_search(prefs):
            return [{"listing_id": "1", "url": "", "address": "Test",
                      "neighborhood": "Chelsea", "price": 3000, "bedrooms": 1, "bathrooms": 1}]

        scraper.search_by_neighborhoods = mock_path_search
        scraper.search_streeteasy = mock_pipe_search

        with patch("src.apify_scraper.asyncio.sleep", new_callable=AsyncMock):
            results = await scraper.search_with_retry(_mock_prefs())

        assert len(results) == 1

    @pytest.mark.asyncio
    async def test_apify_api_error_triggers_retry(self):
        """ApifyApiError from both paths triggers retry loop (not swallowed)."""
        scraper = _make_scraper()

        api_error = ApifyApiError.__new__(ApifyApiError)
        api_error.args = ('Build not found',)
        api_error.message = 'Build not found'
        api_error.type = "record-not-found"
        api_error.status_code = 404

        attempt_count = 0

        async def mock_path_search(prefs, max_items=2000):
            nonlocal attempt_count
            attempt_count += 1
            if attempt_count == 1:
                raise api_error
            return [{"listing_id": "1", "url": "", "address": "Test",
                      "neighborhood": "Chelsea", "price": 3000, "bedrooms": 1, "bathrooms": 1}]

        async def mock_pipe_search(prefs):
            raise api_error

        scraper.search_by_neighborhoods = mock_path_search
        scraper.search_streeteasy = mock_pipe_search

        with patch("src.apify_scraper.asyncio.sleep", new_callable=AsyncMock):
            results = await scraper.search_with_retry(_mock_prefs())

        assert len(results) == 1
        assert attempt_count == 2  # Retried after first failure
