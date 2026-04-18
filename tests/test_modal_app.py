"""Tests for Modal canary helper logic."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from modal_app import (
    _acquire_scan_lock,
    _claim_update,
    _release_scan_lock,
    _run_actor_build_canary_once,
)
from src.apify_scraper import AmenityEnrichmentResult, AmenityRunSummary


class _FakeDict:
    def __init__(self):
        self._data = {}

    def get(self, key):
        return self._data.get(key)

    def put(self, key, value):
        self._data[key] = value

    def delete(self, key):
        self._data.pop(key, None)


def _canary_result(
    *,
    coverage: float = 1.0,
    target_count: int = 2,
    requests_succeeded: int | None = 2,
    requests_failed: int | None = 0,
    build_id: str = "new-build",
    failed: bool = False,
    failure_reason: str | None = None,
) -> AmenityEnrichmentResult:
    run_summary = AmenityRunSummary(
        run_id="run-1",
        dataset_id="ds-1",
        build_id=build_id,
        status="SUCCEEDED",
        status_message="status",
        rows_returned=target_count,
        mapped_count=target_count,
        requests_succeeded=requests_succeeded,
        requests_failed=requests_failed,
    )
    data = {f"canary-{i}": {"building_amenities": ["Doorman"]} for i in range(1, target_count + 1)}
    return AmenityEnrichmentResult(
        data_by_listing_id=data,
        coverage=coverage,
        target_count=target_count,
        run_summaries=[run_summary],
        failed=failed,
        failure_reason=failure_reason,
    )


class TestActorBuildCanary:
    @pytest.mark.asyncio
    async def test_skips_when_no_canary_urls(self):
        with (
            patch("src.storage.load_actor_build_pin", return_value="pin-1"),
            patch("src.storage.load_canary_urls", return_value=[]),
            patch("src.storage.save_canary_status") as save_status,
        ):
            payload = await _run_actor_build_canary_once()

        assert payload["ok"] is False
        assert payload["skipped"] is True
        assert payload["failure_reason"] == "no_canary_urls"
        save_status.assert_called_once()

    @pytest.mark.asyncio
    async def test_promotes_pin_when_latest_canary_is_healthy(self, monkeypatch):
        monkeypatch.delenv("APIFY_ACTOR_BUILD_PIN", raising=False)

        mock_scraper = MagicMock()
        mock_scraper.enrich_with_amenities = AsyncMock(return_value=_canary_result())
        mock_scraper._latest_build_id = AsyncMock(return_value="new-build")
        mock_scraper._latest_build_number = AsyncMock(return_value="0.0.42")
        mock_scraper._build_number_for_id = AsyncMock(return_value="0.0.42")

        with (
            patch("src.storage.load_actor_build_pin", return_value="0.0.40"),
            patch("src.storage.load_canary_urls", return_value=["u1", "u2"]),
            patch("src.storage.save_actor_build_pin") as save_pin,
            patch("src.storage.save_canary_status") as save_status,
            patch("src.apify_scraper.ApifyScraper", return_value=mock_scraper),
        ):
            payload = await _run_actor_build_canary_once()

        assert payload["ok"] is True
        assert payload["promotion_applied"] is True
        assert payload["pin_before"] == "0.0.40"
        assert payload["pin_after"] == "0.0.42"
        save_pin.assert_called_once_with("0.0.42")
        save_status.assert_called_once()

    @pytest.mark.asyncio
    async def test_env_pin_override_blocks_promotion(self, monkeypatch):
        monkeypatch.setenv("APIFY_ACTOR_BUILD_PIN", "forced-build")

        mock_scraper = MagicMock()
        mock_scraper.enrich_with_amenities = AsyncMock(return_value=_canary_result())
        mock_scraper._latest_build_id = AsyncMock(return_value="new-build")
        mock_scraper._latest_build_number = AsyncMock(return_value="0.0.42")
        mock_scraper._build_number_for_id = AsyncMock(return_value="0.0.42")

        with (
            patch("src.storage.load_actor_build_pin", return_value="0.0.40"),
            patch("src.storage.load_canary_urls", return_value=["u1", "u2"]),
            patch("src.storage.save_actor_build_pin") as save_pin,
            patch("src.storage.save_canary_status"),
            patch("src.apify_scraper.ApifyScraper", return_value=mock_scraper),
        ):
            payload = await _run_actor_build_canary_once()

        assert payload["ok"] is True
        assert payload["promotion_applied"] is False
        assert payload["pin_before"] == "0.0.40"
        assert payload["pin_after"] == "0.0.40"
        save_pin.assert_not_called()


class TestWebhookHelpers:
    def test_claim_update_marks_duplicate(self):
        fake = _FakeDict()
        with patch("modal_app.update_claims", fake):
            claimed, token = _claim_update(123)
            assert claimed is True
            assert token

            claimed2, reason = _claim_update(123)
            assert claimed2 is False
            assert reason in {"queued", "processing", "done", "existing"}

    def test_scan_lock_single_owner(self):
        fake = _FakeDict()
        with patch("modal_app.scan_locks", fake):
            assert _acquire_scan_lock(1, "owner-a") is True
            assert _acquire_scan_lock(1, "owner-b") is False
            _release_scan_lock(1, "owner-a")
            assert _acquire_scan_lock(1, "owner-b") is True
