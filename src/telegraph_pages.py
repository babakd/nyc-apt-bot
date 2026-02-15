"""Telegraph (Instant View) page creation for detailed listing views."""

from __future__ import annotations

import logging
from typing import Optional

from src.models import Listing

logger = logging.getLogger(__name__)

# Lazy-initialized Telegraph client
_telegraph = None


async def _get_telegraph():
    """Get or create the Telegraph client."""
    global _telegraph
    if _telegraph is None:
        try:
            from telegraph.aio import Telegraph
            _telegraph = Telegraph()
            await _telegraph.create_account(
                short_name="StreetEasyBot",
                author_name="StreetEasy Bot",
            )
        except Exception:
            logger.exception("Failed to initialize Telegraph client")
            raise
    return _telegraph


async def create_listing_page(listing: Listing) -> Optional[str]:
    """Create a Telegraph page with full listing details.

    Returns the Telegraph page URL, or None if creation fails.
    """
    try:
        telegraph = await _get_telegraph()
    except Exception:
        return None

    html_parts: list[str] = []

    # Photos (up to 8)
    for url in listing.photos[:8]:
        html_parts.append(f'<img src="{url}"/>')

    # Key details
    beds = "Studio" if listing.bedrooms == 0 else f"{listing.bedrooms} BR"
    fee = "No fee" if not listing.broker_fee else listing.broker_fee
    html_parts.append(
        f"<p><strong>${listing.price:,}/mo</strong> · {beds} · {listing.bathrooms} BA · {fee}</p>"
    )
    html_parts.append(f"<p>{listing.neighborhood}</p>")

    if listing.sqft:
        html_parts.append(f"<p>{listing.sqft:,} sqft</p>")

    if listing.available_date:
        html_parts.append(f"<p>Available: {listing.available_date}</p>")

    if listing.match_score is not None:
        html_parts.append(f"<p>Match score: {listing.match_score}/100</p>")

    # Pros/cons
    if listing.pros:
        items = "".join(f"<li>{p}</li>" for p in listing.pros)
        html_parts.append(f"<h4>Pros</h4><ul>{items}</ul>")
    if listing.cons:
        items = "".join(f"<li>{c}</li>" for c in listing.cons)
        html_parts.append(f"<h4>Cons</h4><ul>{items}</ul>")

    # Description
    if listing.description:
        html_parts.append(f"<h4>Description</h4><p>{listing.description}</p>")

    # Amenities
    if listing.amenities:
        items = "".join(f"<li>{a}</li>" for a in listing.amenities)
        html_parts.append(f"<h4>Amenities</h4><ul>{items}</ul>")

    # StreetEasy link
    if listing.url:
        html_parts.append(f'<p><a href="{listing.url}">View on StreetEasy</a></p>')

    try:
        page = await telegraph.create_page(
            title=f"{listing.address} — ${listing.price:,}/mo",
            html_content="".join(html_parts),
        )
        return page["url"]
    except Exception:
        logger.exception("Failed to create Telegraph page for listing %s", listing.listing_id)
        return None
