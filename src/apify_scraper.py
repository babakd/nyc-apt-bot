"""StreetEasy apartment search via Apify actor (replaces Playwright browser layer)."""

from __future__ import annotations

import logging
import os
from typing import Any

from apify_client import ApifyClientAsync

from src.config import NEIGHBORHOODS, STREETEASY_RENTALS
from src.models import Preferences

logger = logging.getLogger(__name__)


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

        The scraper is intentionally dumb — it fetches raw data and maps it to
        a flat dict format.  All intelligent filtering, scoring, and ranking is
        handled downstream by the LLM.
        """
        url = _build_streeteasy_url(prefs)
        logger.info("Searching StreetEasy via Apify: %s", url)

        run_input = {
            "startUrls": [{"url": url}],
            "maxItems": 1000,
            "proxy": {
                "useApifyProxy": True,
                "apifyProxyGroups": ["RESIDENTIAL"],
            },
        }

        run = await self._client.actor(self._actor_id).call(
            run_input=run_input,
            timeout_secs=300,
            memory_mbytes=512,
        )

        if not run or run.get("status") != "SUCCEEDED":
            raise ApifyScraperError(f"Actor run failed: {run}")

        dataset = self._client.dataset(run["defaultDatasetId"])
        result = await dataset.list_items()

        listings = [_map_apify_item(item) for item in result.items]
        listings = [l for l in listings if l is not None]
        logger.info("Fetched %d listings from Apify", len(listings))
        return listings


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


STREETEASY_PHOTO_URL = "https://streeteasy.imgix.net/image/{key}/image.jpg"


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

    # Photos: array of {"key": "abc123"} -> full image URLs
    photos = []
    for photo in node.get("photos") or []:
        key = photo.get("key") if isinstance(photo, dict) else None
        if key:
            photos.append(STREETEASY_PHOTO_URL.format(key=key))

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
        "available_date": node.get("availableAt"),
        "broker_fee": None if node.get("noFee", False) else "Broker fee",
    }
