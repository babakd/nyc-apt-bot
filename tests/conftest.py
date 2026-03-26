"""Shared test fixtures and helpers."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest
from src.models import Listing


def pytest_configure(config):
    config.addinivalue_line("markers", "e2e: end-to-end integration tests (may be slow)")


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
