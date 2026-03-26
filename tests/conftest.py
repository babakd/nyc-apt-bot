"""Shared test fixtures and helpers."""

from __future__ import annotations

from datetime import datetime, timezone

from src.models import Listing


def make_listing(listing_id: str = "123", **overrides) -> Listing:
    """Create a test Listing with sensible defaults.

    All fields can be overridden via keyword arguments.
    """
    defaults = dict(
        listing_id=listing_id,
        url=f"https://streeteasy.com/rental/{listing_id}",
        address=f"123 Test St #{listing_id}",
        neighborhood="East Village",
        price=3000,
        bedrooms=2,
        bathrooms=1.0,
        match_score=85,
        scraped_at=datetime.now(timezone.utc),
    )
    defaults.update(overrides)
    return Listing(**defaults)
