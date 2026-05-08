"""Telegraph (Instant View) page creation for detailed listing views."""

from __future__ import annotations

import logging
from html import escape as _esc
from typing import Optional

from src.models import Listing
from src.storage import load_telegraph_account, save_telegraph_account

logger = logging.getLogger(__name__)

# Lazy-initialized Telegraph client (per-container)
_telegraph = None


async def _get_telegraph():
    """Get or create the Telegraph client.

    Reuses a persisted access_token across containers; creates one on first use.
    """
    global _telegraph
    if _telegraph is not None:
        return _telegraph

    from telegraph.aio import Telegraph

    persisted = load_telegraph_account()
    access_token = persisted.get("access_token") if isinstance(persisted, dict) else None

    try:
        if access_token:
            _telegraph = Telegraph(access_token=access_token)
            return _telegraph
        _telegraph = Telegraph()
        info = await _telegraph.create_account(
            short_name="StreetEasyBot",
            author_name="StreetEasy Bot",
        )
        token = info.get("access_token") if isinstance(info, dict) else None
        if token:
            save_telegraph_account({
                "access_token": token,
                "short_name": "StreetEasyBot",
                "author_name": "StreetEasy Bot",
            })
        return _telegraph
    except Exception:
        logger.exception("Failed to initialize Telegraph client")
        _telegraph = None
        raise


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
        html_parts.append(f'<img src="{_esc(url, quote=True)}"/>')

    # Key details
    beds = "Studio" if listing.bedrooms == 0 else f"{listing.bedrooms} BR"
    fee = "No fee" if not listing.broker_fee else _esc(listing.broker_fee)
    facts = [
        f"<strong>${listing.price:,}/mo</strong>",
        beds,
        f"{_format_bathrooms(listing.bathrooms)} BA",
        fee,
        _esc(listing.neighborhood),
    ]
    if listing.net_effective_price and listing.net_effective_price != listing.price:
        facts.insert(1, f"${listing.net_effective_price:,} net effective")
    html_parts.append(f"<p>{' · '.join(facts)}</p>")

    if listing.sqft:
        html_parts.append(f"<p>{listing.sqft:,} sqft</p>")

    if listing.available_date:
        html_parts.append(f"<p>Available: {_esc(listing.available_date)}</p>")

    display_score = listing.rank_score if listing.rank_score is not None else listing.match_score
    if display_score is not None:
        score_text = f"Match score: {display_score}/100"
        if listing.rank_score is not None and listing.match_score is not None and listing.rank_score != listing.match_score:
            score_text += f" (Claude match {listing.match_score}/100, adjusted with local fit signals)"
        html_parts.append(f"<p><strong>{score_text}</strong></p>")

    if listing.rank_badges:
        badges = " · ".join(_esc(b) for b in listing.rank_badges[:5])
        html_parts.append(f"<p><strong>Best signals:</strong> {badges}</p>")

    # Pros/cons
    if listing.pros:
        items = "".join(f"<li>{_esc(p)}</li>" for p in listing.pros)
        html_parts.append(f"<h4>Pros</h4><ul>{items}</ul>")
    if listing.cons:
        items = "".join(f"<li>{_esc(c)}</li>" for c in listing.cons)
        html_parts.append(f"<h4>Cons</h4><ul>{items}</ul>")

    # Description
    if listing.description:
        html_parts.append(f"<h4>Description</h4><p>{_esc(listing.description)}</p>")

    # Amenities
    amenities = _combined_amenities(listing)
    if amenities:
        items = "".join(f"<li>{_esc(a)}</li>" for a in amenities)
        html_parts.append(f"<h4>Amenities</h4><ul>{items}</ul>")

    # StreetEasy link
    if listing.url:
        html_parts.append(f'<p><a href="{_esc(listing.url, quote=True)}">View on StreetEasy</a></p>')

    try:
        page = await telegraph.create_page(
            title=f"{listing.address} — ${listing.price:,}/mo",
            html_content="".join(html_parts),
        )
        return page["url"]
    except Exception:
        logger.exception("Failed to create Telegraph page for listing %s", listing.listing_id)
        return None


def _format_bathrooms(value: float) -> str:
    """Format bathroom count without a noisy .0 suffix."""
    if float(value).is_integer():
        return str(int(value))
    return str(value)


def _combined_amenities(listing: Listing) -> list[str]:
    """Return deduped amenities with verified detail-page fields first."""
    seen: set[str] = set()
    result: list[str] = []
    for value in (
        listing.confirmed_unit_features
        + listing.confirmed_building_amenities
        + listing.unit_features
        + listing.building_amenities
        + listing.matched_amenities
        + listing.amenities
    ):
        key = value.lower()
        if key in seen:
            continue
        seen.add(key)
        result.append(value)
    return result
