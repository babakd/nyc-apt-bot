"""Tests for Telegraph page creation with HTML escaping."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from src.models import Listing
from src.telegraph_pages import create_listing_page


def _make_listing(**overrides) -> Listing:
    """Helper to create a test listing with sensible defaults."""
    defaults = dict(
        listing_id="999",
        url="https://streeteasy.com/rental/999",
        address="123 Test St #4A",
        neighborhood="East Village",
        price=3500,
        bedrooms=2,
        bathrooms=1.0,
        match_score=80,
        photos=["https://streeteasy.imgix.net/image/abc/image.jpg"],
        pros=["Great location"],
        cons=["No elevator"],
        description="A lovely apartment.",
        amenities=["Dishwasher"],
        available_date="Mar 1, 2026",
    )
    defaults.update(overrides)
    return Listing(**defaults)


class TestTelegraphEscaping:
    @pytest.mark.asyncio
    async def test_description_html_escaped(self):
        """Description with HTML special chars is escaped."""
        listing = _make_listing(description="Price < $3000 & very spacious")

        mock_telegraph = AsyncMock()
        mock_telegraph.create_page = AsyncMock(return_value={"url": "https://telegra.ph/test"})

        with patch("src.telegraph_pages._get_telegraph", return_value=mock_telegraph):
            await create_listing_page(listing)

        html_content = mock_telegraph.create_page.call_args[1]["html_content"]
        assert "&lt;" in html_content
        assert "&amp;" in html_content
        assert "Price < $3000" not in html_content

    @pytest.mark.asyncio
    async def test_pros_cons_html_escaped(self):
        """Pros and cons with special chars are escaped."""
        listing = _make_listing(
            pros=["Size > average", "A & B"],
            cons=["Price < budget"],
        )

        mock_telegraph = AsyncMock()
        mock_telegraph.create_page = AsyncMock(return_value={"url": "https://telegra.ph/test"})

        with patch("src.telegraph_pages._get_telegraph", return_value=mock_telegraph):
            await create_listing_page(listing)

        html_content = mock_telegraph.create_page.call_args[1]["html_content"]
        assert "<li>Size &gt; average</li>" in html_content
        assert "<li>A &amp; B</li>" in html_content
        assert "<li>Price &lt; budget</li>" in html_content

    @pytest.mark.asyncio
    async def test_neighborhood_html_escaped(self):
        """Neighborhood name with special chars is escaped."""
        listing = _make_listing(neighborhood="Test <Area> & Zone")

        mock_telegraph = AsyncMock()
        mock_telegraph.create_page = AsyncMock(return_value={"url": "https://telegra.ph/test"})

        with patch("src.telegraph_pages._get_telegraph", return_value=mock_telegraph):
            await create_listing_page(listing)

        html_content = mock_telegraph.create_page.call_args[1]["html_content"]
        assert "Test &lt;Area&gt; &amp; Zone" in html_content

    @pytest.mark.asyncio
    async def test_amenities_html_escaped(self):
        """Amenities with special chars are escaped."""
        listing = _make_listing(amenities=["Washer & Dryer", "A/C <central>"])

        mock_telegraph = AsyncMock()
        mock_telegraph.create_page = AsyncMock(return_value={"url": "https://telegra.ph/test"})

        with patch("src.telegraph_pages._get_telegraph", return_value=mock_telegraph):
            await create_listing_page(listing)

        html_content = mock_telegraph.create_page.call_args[1]["html_content"]
        assert "<li>Washer &amp; Dryer</li>" in html_content
        assert "<li>A/C &lt;central&gt;</li>" in html_content

    @pytest.mark.asyncio
    async def test_photo_url_escaped(self):
        """Photo URLs with special chars are escaped in src attributes."""
        listing = _make_listing(
            photos=["https://example.com/image?a=1&b=2"]
        )

        mock_telegraph = AsyncMock()
        mock_telegraph.create_page = AsyncMock(return_value={"url": "https://telegra.ph/test"})

        with patch("src.telegraph_pages._get_telegraph", return_value=mock_telegraph):
            await create_listing_page(listing)

        html_content = mock_telegraph.create_page.call_args[1]["html_content"]
        assert 'src="https://example.com/image?a=1&amp;b=2"' in html_content

    @pytest.mark.asyncio
    async def test_broker_fee_escaped(self):
        """Broker fee string with special chars is escaped."""
        listing = _make_listing(broker_fee="Fee <special> & tax")

        mock_telegraph = AsyncMock()
        mock_telegraph.create_page = AsyncMock(return_value={"url": "https://telegra.ph/test"})

        with patch("src.telegraph_pages._get_telegraph", return_value=mock_telegraph):
            await create_listing_page(listing)

        html_content = mock_telegraph.create_page.call_args[1]["html_content"]
        assert "Fee &lt;special&gt; &amp; tax" in html_content

    @pytest.mark.asyncio
    async def test_listing_url_escaped(self):
        """StreetEasy URL with special chars is escaped in href attribute."""
        listing = _make_listing(url="https://streeteasy.com/rental/999?a=1&b=2")

        mock_telegraph = AsyncMock()
        mock_telegraph.create_page = AsyncMock(return_value={"url": "https://telegra.ph/test"})

        with patch("src.telegraph_pages._get_telegraph", return_value=mock_telegraph):
            await create_listing_page(listing)

        html_content = mock_telegraph.create_page.call_args[1]["html_content"]
        assert 'href="https://streeteasy.com/rental/999?a=1&amp;b=2"' in html_content
