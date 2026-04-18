"""Telegram message formatting and inline keyboard builders."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from src.config import MAX_PROS_DISPLAYED, MAX_CONS_DISPLAYED
from src.models import Listing, Preferences

if TYPE_CHECKING:
    from src.models import CurrentApartment


# --- Inline Keyboard Builders ---

def listing_keyboard(listing_id: str, listing_url: str = "") -> list[list[dict[str, str]]]:
    """Build inline keyboard for a listing card: Like / Pass / Details / StreetEasy."""
    kb: list[list[dict[str, str]]] = [
        [
            {"text": "👍 Like", "callback_data": f"like:{listing_id}"},
            {"text": "👎 Pass", "callback_data": f"pass:{listing_id}"},
        ],
        [
            {"text": "📋 Details", "callback_data": f"details:{listing_id}"},
        ],
    ]
    if listing_url:
        kb[1].append({"text": "🔗 StreetEasy ↗", "url": listing_url})
    return kb


def draft_keyboard(draft_id: str) -> list[list[dict[str, str]]]:
    """Build inline keyboard for a draft message: Send / Edit / Cancel."""
    return [
        [
            {"text": "✅ Send", "callback_data": f"draft_send:{draft_id}"},
            {"text": "✏️ Edit", "callback_data": f"draft_edit:{draft_id}"},
            {"text": "❌ Cancel", "callback_data": f"draft_cancel:{draft_id}"},
        ],
    ]


# --- Message Formatters ---

def format_listing_card(listing: Listing, rank: int | None = None) -> str:
    """Format a listing as a Telegram HTML message."""
    parts: list[str] = []

    # Header: rank + address
    header = f"<b>{_escape_html(listing.address)}</b>"
    if rank is not None:
        header = f"#{rank}  {header}"
    parts.append(header)

    # Neighborhood · price · fee
    price_line = f"{_escape_html(listing.neighborhood)} · <b>${listing.price:,}/mo</b>"
    if listing.months_free and listing.net_effective_price:
        price_line += f" · <i>${listing.net_effective_price:,} net</i>"
    if not listing.broker_fee:
        price_line += " · NO FEE"
    parts.append(price_line)

    # Beds / baths / sqft + available date
    beds = "Studio" if listing.bedrooms == 0 else f"{listing.bedrooms} BR"
    facts = f"{beds} · {listing.bathrooms} BA"
    if listing.sqft:
        facts += f" · {listing.sqft:,} sqft"
    parts.append(f"\n{facts}")

    if listing.available_date:
        parts.append(f"Available {_escape_html(listing.available_date)}")

    # Match score
    if listing.match_score is not None:
        bar = _score_bar(listing.match_score)
        parts.append(f"\n{bar}  <b>{listing.match_score}% match</b>")

    # Pros/cons as compact bullets
    if listing.pros or listing.cons:
        bullet_lines = []
        for p in listing.pros[:MAX_PROS_DISPLAYED]:
            bullet_lines.append(f"+ {_escape_html(p)}")
        for c in listing.cons[:MAX_CONS_DISPLAYED]:
            bullet_lines.append(f"- {_escape_html(c)}")
        parts.append("\n" + "\n".join(bullet_lines))

    # Link
    parts.append(f'\n<a href="{_escape_attr(listing.url)}">View on StreetEasy \u2192</a>')

    return "\n".join(parts)


def format_listing_detail(listing: Listing) -> str:
    """Format detailed listing info (for the Details button)."""
    parts = [format_listing_card(listing)]

    if listing.description:
        desc = listing.description[:500]
        if len(listing.description) > 500:
            desc += "..."
        parts.append(f"\n<b>Description:</b>\n{_escape_html(desc)}")

    if listing.amenities:
        amenity_list = ", ".join(_escape_html(a) for a in listing.amenities)
        parts.append(f"\n<b>Amenities:</b> {amenity_list}")

    return "\n".join(parts)


def format_preferences_summary(prefs: Preferences) -> str:
    """Format current preferences for Telegram HTML display."""
    parts = ["<b>🏠 Your Search Preferences</b>\n"]

    if prefs.budget_min or prefs.budget_max:
        if prefs.budget_min and prefs.budget_max:
            budget = f"${prefs.budget_min:,} – ${prefs.budget_max:,}/mo"
        elif prefs.budget_max:
            budget = f"Up to ${prefs.budget_max:,}/mo"
        else:
            budget = f"${prefs.budget_min:,}+/mo"
        parts.append(f"💰 Budget: {budget}")

    if prefs.bedrooms:
        beds = ", ".join(
            "Studio" if b == 0 else f"{b} BR" for b in prefs.bedrooms
        )
        parts.append(f"🛏 Bedrooms: {beds}")

    if prefs.neighborhoods:
        hoods = ", ".join(_escape_html(n) for n in prefs.neighborhoods[:8])
        if len(prefs.neighborhoods) > 8:
            hoods += f" +{len(prefs.neighborhoods) - 8} more"
        parts.append(f"📍 Neighborhoods: {hoods}")

    if prefs.commute_address:
        commute = f"📍 Commute to: {_escape_html(prefs.commute_address)}"
        if prefs.commute_max_minutes:
            commute += f" (max {prefs.commute_max_minutes} min)"
        parts.append(commute)

    if prefs.min_bathrooms:
        parts.append(f"🚿 Min bathrooms: {prefs.min_bathrooms}")

    if prefs.must_haves:
        parts.append(f"✅ Must-haves: {', '.join(_escape_html(m) for m in prefs.must_haves)}")

    if prefs.nice_to_haves:
        parts.append(f"⭐ Nice-to-haves: {', '.join(_escape_html(n) for n in prefs.nice_to_haves)}")

    if prefs.no_fee_only:
        parts.append("🚫 No-fee only")

    if prefs.move_in_date:
        parts.append(f"📅 Move-in: {_escape_html(prefs.move_in_date)}")

    return "\n".join(parts)


def format_preferences_summary_plain(prefs: Preferences) -> str:
    """Format current preferences as plain text (for Claude tool results)."""
    parts = ["Your Search Preferences:\n"]

    if prefs.budget_max or prefs.budget_min:
        budget = f"${prefs.budget_min:,}" if prefs.budget_min else "$0"
        if prefs.budget_max:
            budget += f" - ${prefs.budget_max:,}/mo"
        else:
            budget += "+/mo"
        parts.append(f"Budget: {budget}")

    if prefs.bedrooms:
        beds = ", ".join(
            "Studio" if b == 0 else f"{b} BR" for b in prefs.bedrooms
        )
        parts.append(f"Bedrooms: {beds}")

    if prefs.neighborhoods:
        hoods = ", ".join(prefs.neighborhoods[:8])
        if len(prefs.neighborhoods) > 8:
            hoods += f" +{len(prefs.neighborhoods) - 8} more"
        parts.append(f"Neighborhoods: {hoods}")

    if prefs.commute_address:
        commute = f"Commute to: {prefs.commute_address}"
        if prefs.commute_max_minutes:
            commute += f" (max {prefs.commute_max_minutes} min)"
        parts.append(commute)

    if prefs.min_bathrooms:
        parts.append(f"Min bathrooms: {prefs.min_bathrooms}")

    if prefs.must_haves:
        parts.append(f"Must-haves: {', '.join(prefs.must_haves)}")

    if prefs.nice_to_haves:
        parts.append(f"Nice-to-haves: {', '.join(prefs.nice_to_haves)}")

    if prefs.no_fee_only:
        parts.append("No-fee only")

    if prefs.move_in_date:
        parts.append(f"Move-in: {prefs.move_in_date}")

    return "\n".join(parts)


def format_draft_message(draft_text: str, listing: Listing) -> str:
    """Format a draft outreach message for user review."""
    return (
        f"<b>📝 Draft message to agent for:</b>\n"
        f"{_escape_html(listing.address)}\n\n"
        f"<i>{_escape_html(draft_text)}</i>\n\n"
        f"Review and choose an action below:"
    )


def format_scan_header(count: int, is_daily: bool = True) -> str:
    """Format the header message for scan results.

    Args:
        count: Number of listings found.
        is_daily: True for daily cron scan, False for user-triggered search.
    """
    title = "Daily Scan Complete" if is_daily else "Search Results"
    if count == 0:
        if is_daily:
            return f"🔍 <b>{title}</b>\n\nNo new listings found matching your criteria today. I'll check again tomorrow!"
        return f"🔍 <b>{title}</b>\n\nNo new listings found matching your criteria."
    return (
        f"🔍 <b>{title}</b>\n\n"
        f"Found <b>{count}</b> new listing{'s' if count != 1 else ''} "
        f"matching your preferences, ranked by match score:"
    )


def format_apartment_context(apt: "CurrentApartment") -> str:
    """Format current apartment info for inclusion in prompts."""
    parts = []
    if apt.price:
        parts.append(f"Current rent: ${apt.price:,}/mo")
    if apt.neighborhood:
        parts.append(f"Current neighborhood: {apt.neighborhood}")
    if apt.bedrooms is not None:
        parts.append(f"Bedrooms: {apt.bedrooms}")
    if apt.address:
        parts.append(f"Address: {apt.address}")
    if apt.move_out_date:
        parts.append(f"Moving out: {apt.move_out_date}")
    if apt.pros:
        parts.append(f"Likes: {', '.join(apt.pros)}")
    if apt.cons:
        parts.append(f"Dislikes: {', '.join(apt.cons)}")
    if apt.notes:
        parts.append(f"Notes: {apt.notes}")
    return "\n".join(parts)


# --- Telegram API Payload Builders ---

def build_send_message_payload(
    chat_id: int,
    text: str,
    reply_markup: list[list[dict[str, str]]] | None = None,
    parse_mode: str = "HTML",
) -> dict[str, Any]:
    """Build a Telegram sendMessage API payload."""
    payload: dict[str, Any] = {
        "chat_id": chat_id,
        "text": text,
        "parse_mode": parse_mode,
        "disable_web_page_preview": True,
    }
    if reply_markup is not None:
        payload["reply_markup"] = json.dumps({"inline_keyboard": reply_markup})
    return payload


def build_send_photo_payload(
    chat_id: int,
    photo_url: str,
    caption: str | None = None,
    reply_markup: list[list[dict[str, str]]] | None = None,
) -> dict[str, Any]:
    """Build a Telegram sendPhoto API payload."""
    payload: dict[str, Any] = {
        "chat_id": chat_id,
        "photo": photo_url,
    }
    if caption:
        payload["caption"] = caption
        payload["parse_mode"] = "HTML"
    if reply_markup is not None:
        payload["reply_markup"] = json.dumps({"inline_keyboard": reply_markup})
    return payload


# --- Helpers ---

def _escape_html(text: str) -> str:
    """Escape HTML special characters for Telegram and strip surrogate characters."""
    # Strip surrogate characters that crash Telegram's API
    text = text.encode("utf-8", errors="surrogatepass").decode("utf-8", errors="ignore")
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


def _escape_attr(text: str) -> str:
    """Escape an HTML attribute value (URL or similar). Includes quote escaping."""
    return _escape_html(text).replace('"', "&quot;")


def _score_bar(score: int) -> str:
    """Generate a visual score bar."""
    filled = score // 10
    empty = 10 - filled
    return "\u2588" * filled + "\u2591" * empty
