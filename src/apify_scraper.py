"""StreetEasy apartment search via Apify actor (replaces Playwright browser layer)."""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Any

from apify_client import ApifyClientAsync

from src.config import NEIGHBORHOODS, STREETEASY_RENTALS
from src.models import Preferences

logger = logging.getLogger(__name__)

POLL_INTERVAL_SECS = 10
ABORT_AFTER_SECS_NO_ITEMS = 60


class ApifyScraperError(Exception):
    """Error from Apify actor run."""


class ApifyScraper:
    """Searches StreetEasy via the memo23/apify-streeteasy-cheerio Apify actor."""

    def __init__(self) -> None:
        token = os.environ["APIFY_API_TOKEN"]
        self._client = ApifyClientAsync(token=token)
        self._actor_id = "memo23/apify-streeteasy-cheerio"

    async def search_streeteasy(self, prefs: Preferences) -> list[dict[str, Any]]:
        """Call Apify actor with a StreetEasy search URL, return listing dicts.

        Uses start + poll + abort pattern instead of blocking .call():
        - Polls every POLL_INTERVAL_SECS for status/items
        - Aborts if ABORT_AFTER_SECS_NO_ITEMS elapsed with 0 items
        - Uses maxRequestRetries=15 (down from actor default of 100)
        """
        url = _build_streeteasy_url(prefs)
        logger.info("Searching StreetEasy via Apify: %s", url)

        run_input = {
            "startUrls": [{"url": url}],
            "maxItems": 1000,
            "maxRequestRetries": 15,
            "proxy": {
                "useApifyProxy": True,
                "apifyProxyGroups": ["RESIDENTIAL"],
                "countryCode": "US",
            },
        }

        # Start the actor run (non-blocking)
        run_info = await self._client.actor(self._actor_id).start(
            run_input=run_input,
            memory_mbytes=512,
        )

        run_id = run_info["id"]
        dataset_id = run_info["defaultDatasetId"]
        run_client = self._client.run(run_id)
        dataset_client = self._client.dataset(dataset_id)
        elapsed = 0

        # Poll until terminal status
        while True:
            await asyncio.sleep(POLL_INTERVAL_SECS)
            elapsed += POLL_INTERVAL_SECS

            run_data = await run_client.get()
            status = (run_data or {}).get("status", "")

            if status in ("SUCCEEDED", "FAILED", "ABORTED", "TIMED-OUT"):
                break

            # Check item count while still running
            ds_info = await dataset_client.get()
            item_count = (ds_info or {}).get("itemCount", 0)
            logger.info(
                "Apify poll: %ds elapsed, status=%s, items=%d",
                elapsed, status, item_count,
            )

            if elapsed >= ABORT_AFTER_SECS_NO_ITEMS and item_count == 0:
                logger.warning(
                    "Aborting Apify run after %ds with 0 items (likely WAF block)",
                    elapsed,
                )
                await run_client.abort()
                raise ApifyScraperError(
                    f"Actor run aborted after {elapsed}s with 0 items"
                )

        # Check final status
        if status != "SUCCEEDED":
            raise ApifyScraperError(f"Actor run failed with status: {status}")

        result = await dataset_client.list_items()

        listings = [_map_apify_item(item) for item in result.items]
        listings = [l for l in listings if l is not None]
        logger.info("Fetched %d listings from Apify", len(listings))
        return listings

    async def search_with_retry(
        self,
        prefs: Preferences,
        max_retries: int = 2,
        retry_delays: tuple[int, ...] = (30, 60),
    ) -> list[dict[str, Any]]:
        """Wrap search_streeteasy with exponential backoff retries.

        Retries only on ApifyScraperError (infrastructure failures).
        Empty results from a successful run are returned immediately —
        they indicate a legitimate "no matches" rather than a failure.
        """
        last_error: Exception | None = None

        for attempt in range(1 + max_retries):
            try:
                logger.info("search_with_retry: attempt %d/%d", attempt + 1, 1 + max_retries)
                results = await self.search_streeteasy(prefs)
                return results
            except ApifyScraperError as e:
                logger.warning("search_with_retry: attempt %d failed: %s", attempt + 1, e)
                last_error = e

            # Delay before next retry (if we have retries left)
            if attempt < max_retries:
                delay = retry_delays[attempt] if attempt < len(retry_delays) else retry_delays[-1]
                logger.info("search_with_retry: waiting %ds before retry", delay)
                await asyncio.sleep(delay)

        raise last_error or ApifyScraperError("All retries exhausted")


def _build_streeteasy_url(prefs: Preferences) -> str:
    """Build a StreetEasy search URL from user preferences.

    Only includes structural search params (price, beds, areas, no-fee).
    Amenities, bathrooms, and other preference-gradient filters are left to
    the LLM scorer so near-misses aren't silently dropped.
    """
    url = STREETEASY_RENTALS

    parts = []

    # Neighborhoods (hint — StreetEasy's RECOMMENDED sort often ignores this)
    if prefs.neighborhoods:
        slugs = []
        for name in prefs.neighborhoods:
            slug = NEIGHBORHOODS.get(name)
            if slug:
                slugs.append(slug)
            else:
                slugs.append(name.lower().replace(" ", "-"))
        if slugs:
            parts.append("area:" + ",".join(slugs))

    # Price
    if prefs.budget_min and prefs.budget_max:
        parts.append(f"price:{prefs.budget_min}-{prefs.budget_max}")
    elif prefs.budget_max:
        parts.append(f"price:-{prefs.budget_max}")
    elif prefs.budget_min:
        parts.append(f"price:{prefs.budget_min}-")

    # Bedrooms
    if prefs.bedrooms:
        beds_str = ",".join(str(b) for b in prefs.bedrooms)
        parts.append(f"beds:{beds_str}")

    # No fee
    if prefs.no_fee_only:
        parts.append("no_fee:1")

    if parts:
        url += "/" + "|".join(parts)

    return url


STREETEASY_PHOTO_URL = "https://photos.zillowstatic.com/fp/{key}-se_large_800_400.jpg"
STREETEASY_PHOTO_URL_THUMB = "https://photos.zillowstatic.com/fp/{key}-se_medium_500_250.jpg"


def _map_apify_item(item: dict[str, Any]) -> dict[str, Any] | None:
    """Map an Apify actor output item to the raw listing dict format expected by scanner.py.

    The memo23 actor returns GraphQL-shaped items with a nested "node" object:
      { "node": { "id", "areaName", "price", "bedroomCount", "street", "urlPath", ... } }
    """
    node = item.get("node") or item
    listing_id = str(node.get("id", ""))
    if not listing_id:
        return None

    # Build listing URL from urlPath (e.g. "/building/ray-harlem/15r")
    url_path = node.get("urlPath", "")
    listing_url = f"https://streeteasy.com{url_path}" if url_path else ""

    # Address from street + unit
    street = node.get("street", "")
    unit = node.get("unit", "")
    address = f"{street} #{unit}".strip(" #") if street else f"Listing {listing_id}"

    # Photos: array of {"key": "abc123"} -> full image URLs + raw keys
    photos = []
    photo_keys = []
    for photo in node.get("photos") or []:
        key = photo.get("key") if isinstance(photo, dict) else None
        if key:
            photos.append(STREETEASY_PHOTO_URL.format(key=key))
            photo_keys.append(key)

    # Bathrooms: full + half
    full_bath = int(node.get("fullBathroomCount", 0) or 0)
    half_bath = int(node.get("halfBathroomCount", 0) or 0)
    bathrooms = full_bath + 0.5 * half_bath if (full_bath or half_bath) else 1.0

    # Square footage
    living_area = node.get("livingAreaSize", 0) or 0
    sqft = int(living_area) if living_area else None

    return {
        "listing_id": listing_id,
        "url": listing_url,
        "address": address,
        "neighborhood": node.get("areaName", ""),
        "price": int(node.get("price", 0) or 0),
        "bedrooms": int(node.get("bedroomCount", 0) or 0),
        "bathrooms": bathrooms,
        "sqft": sqft,
        "photos": photos,
        "photo_keys": photo_keys,
        "available_date": node.get("availableAt"),
        "broker_fee": None if node.get("noFee", False) else "Broker fee",
        "net_effective_price": int(node.get("netEffectivePrice")) if node.get("netEffectivePrice") else None,
        "months_free": float(node.get("monthsFree")) if node.get("monthsFree") else None,
    }
