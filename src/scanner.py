"""Daily apartment scanning: search StreetEasy, deduplicate, LLM-score, notify."""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from typing import Any

import anthropic

from src.formatter import format_listing_card, format_scan_header, listing_keyboard
from src.models import MAX_RECENT_LISTINGS, ChatState, CurrentApartment, Listing, Preferences
from src.apify_scraper import ApifyScraper
from src.storage import load_all_states, save_state

logger = logging.getLogger(__name__)

SCORING_MODEL = "claude-opus-4-6"
SCORE_FLOOR = 25

# Neighborhood alias map: StreetEasy naming variants → canonical preference name.
# Used by _neighborhood_pre_filter to match listings whose neighborhood names
# differ from what the user/Claude set in preferences.
NEIGHBORHOOD_ALIASES: dict[str, str] = {
    "west chelsea": "chelsea",
    "nomad": "flatiron",
    "noho": "greenwich village",
    "east williamsburg": "williamsburg",
    "north williamsburg": "williamsburg",
    "south williamsburg": "williamsburg",
    "yorkville": "upper east side",
    "lenox hill": "upper east side",
    "carnegie hill": "upper east side",
    "lincoln square": "upper west side",
    "manhattan valley": "upper west side",
    "two bridges": "lower east side",
    "chinatown": "lower east side",
    "little italy": "nolita",
    "hudson yards": "hells kitchen",
    "gramercy park": "gramercy",
    "stuyvesant town": "gramercy",
    "kips bay": "murray hill",
    "turtle bay": "midtown east",
    "sutton place": "midtown east",
    "beekman": "midtown east",
    "civic center": "financial district",
    "seaport": "financial district",
    "fulton": "financial district",
    "vinegar hill": "dumbo",
    "columbia street waterfront": "carroll gardens",
    "prospect park south": "flatbush",
    "ditmas park": "flatbush",
    "south slope": "park slope",
}


async def run_daily_scan(
    scraper: ApifyScraper,
    telegram_bot: "TelegramBot",
) -> None:
    """Run the daily scan for all registered chats."""
    states = load_all_states()
    if not states:
        logger.info("No registered chats, skipping scan.")
        return

    for state in states:
        if not state.preferences_ready:
            logger.info("Chat %s hasn't confirmed preferences, skipping.", state.chat_id)
            continue

        try:
            await scan_for_chat(scraper, telegram_bot, state)
        except Exception:
            logger.exception("Scan failed for chat %s", state.chat_id)
            await telegram_bot.send_text(
                state.chat_id,
                "\u26a0\ufe0f Today's scan ran into an issue. I'll try again tomorrow.",
            )


async def scan_for_chat(
    scraper: ApifyScraper,
    telegram_bot: "TelegramBot",
    state: ChatState,
) -> None:
    """Scan StreetEasy for a single chat's preferences."""
    prefs = state.preferences
    logger.info("Scanning for chat %s with prefs: %s", state.chat_id, prefs.neighborhoods)

    # Search StreetEasy via Apify actor
    try:
        raw_listings = await scraper.search_streeteasy(prefs)
    except Exception:
        logger.exception("StreetEasy search failed")
        await telegram_bot.send_text(
            state.chat_id,
            "\u26a0\ufe0f I had trouble searching StreetEasy today. I'll try again tomorrow.",
        )
        return

    if not raw_listings:
        await telegram_bot.send_text(state.chat_id, format_scan_header(0))
        return

    # Deduplicate
    new_listings = []
    for raw in raw_listings:
        lid = str(raw.get("listing_id", ""))
        if lid and lid not in state.seen_listing_ids:
            new_listings.append(raw)
            state.seen_listing_ids.add(lid)

    if not new_listings:
        await telegram_bot.send_text(state.chat_id, format_scan_header(0))
        save_state(state)
        return

    # Parse listings
    parsed: list[Listing] = []
    for raw in new_listings:
        try:
            parsed.append(_parse_listing(raw))
        except Exception:
            logger.exception("Failed to parse listing %s", raw.get("listing_id"))
            continue

    # Layer 1: Neighborhood pre-filter (data quality fix for StreetEasy's broken RECOMMENDED sort)
    filtered = _neighborhood_pre_filter(parsed, prefs)

    if not filtered and parsed:
        logger.info("All %d listings filtered out by neighborhood pre-filter", len(parsed))
        await telegram_bot.send_text(
            state.chat_id,
            "\ud83d\udd0d No listings found in your neighborhoods today. "
            "I'll check again tomorrow!",
        )
        save_state(state)
        return

    # Layer 2: LLM filter + score
    scored = await _llm_score_listings(filtered, prefs, state.current_apartment)

    # Sort by score descending
    scored.sort(key=lambda l: l.match_score or 0, reverse=True)

    # Store in recent_listings for later reference (details, compare, draft)
    for listing in scored:
        state.recent_listings[listing.listing_id] = listing
    # Cap recent_listings size
    if len(state.recent_listings) > MAX_RECENT_LISTINGS:
        ids = list(state.recent_listings.keys())
        for old_id in ids[: len(ids) - MAX_RECENT_LISTINGS]:
            del state.recent_listings[old_id]

    # Send results
    if not scored:
        await telegram_bot.send_text(state.chat_id, format_scan_header(0))
    else:
        await telegram_bot.send_text(state.chat_id, format_scan_header(len(scored)))

        for i, listing in enumerate(scored, 1):
            card_text = format_listing_card(listing, rank=i)
            keyboard = listing_keyboard(listing.listing_id, listing.url)

            if listing.photos:
                await telegram_bot.send_listing_photo(
                    state.chat_id,
                    listing_url=listing.url,
                    photo_url=listing.photos[0],
                    caption=card_text,
                    keyboard=keyboard,
                )
            else:
                await telegram_bot.send_text(
                    state.chat_id,
                    card_text,
                    keyboard=keyboard,
                )

    save_state(state)
    logger.info("Sent %d listings to chat %s", len(scored), state.chat_id)


def _parse_listing(raw: dict) -> Listing:
    """Parse a raw listing dict into a Listing model."""
    return Listing(
        listing_id=str(raw.get("listing_id", "")),
        url=raw.get("url", ""),
        address=raw.get("address", "Unknown"),
        neighborhood=raw.get("neighborhood", "Unknown"),
        price=int(raw.get("price", 0)),
        bedrooms=int(raw.get("bedrooms", 0)),
        bathrooms=float(raw.get("bathrooms", 1)),
        sqft=raw.get("sqft"),
        amenities=raw.get("amenities", []),
        photos=raw.get("photos", []),
        broker_fee=raw.get("broker_fee"),
        available_date=raw.get("available_date"),
        description=raw.get("description"),
        scraped_at=datetime.now(timezone.utc),
    )


def _neighborhood_pre_filter(
    listings: list[Listing],
    prefs: Preferences,
) -> list[Listing]:
    """Filter listings to only those in the user's preferred neighborhoods.

    This is a data quality fix, NOT preference enforcement. StreetEasy's
    RECOMMENDED sort ignores the areas filter, returning ~98% wrong-area results.
    This pre-filter compensates for that specific API bug.

    Uses NEIGHBORHOOD_ALIASES to handle StreetEasy naming variants
    (e.g. "West Chelsea" matches "Chelsea" in preferences).

    If no neighborhoods are set in preferences, all listings pass through.
    """
    if not prefs.neighborhoods:
        return listings

    # Build a set of canonical neighborhood names (lowered) from preferences
    pref_hoods_lower = {n.lower() for n in prefs.neighborhoods}

    result = []
    for listing in listings:
        hood_lower = listing.neighborhood.lower()
        # Direct match
        if hood_lower in pref_hoods_lower:
            result.append(listing)
            continue
        # Alias match: check if listing's neighborhood maps to a preferred one
        canonical = NEIGHBORHOOD_ALIASES.get(hood_lower)
        if canonical and canonical in pref_hoods_lower:
            result.append(listing)

    return result


async def _llm_score_listings(
    listings: list[Listing],
    prefs: Preferences,
    current_apartment: CurrentApartment | None = None,
) -> list[Listing]:
    """Score and filter listings using Claude.

    Two-step process:
    1. Check hard constraints → include: true/false
    2. Score soft preferences → 0-100

    Uses constraint_context from preferences to determine what's a dealbreaker
    vs a nice-to-have. When constraint_context is None, the LLM uses its own judgment.

    Returns only listings with include=true AND score >= SCORE_FLOOR.
    If all listings are excluded, returns top 3 by score as fallback.
    """
    if not listings:
        return listings

    # Format listings compactly
    listings_data = []
    for l in listings:
        entry: dict[str, Any] = {
            "id": l.listing_id,
            "address": l.address,
            "hood": l.neighborhood,
            "price": l.price,
            "beds": l.bedrooms,
            "baths": l.bathrooms,
            "fee": l.broker_fee or "No fee",
        }
        if l.sqft:
            entry["sqft"] = l.sqft
        if l.available_date:
            entry["avail"] = l.available_date
        if l.amenities:
            entry["amenities"] = l.amenities
        listings_data.append(entry)

    # Format preferences
    pref_lines = []
    if prefs.budget_min or prefs.budget_max:
        if prefs.budget_max:
            budget = f"${prefs.budget_min or 0:,}-${prefs.budget_max:,}/mo"
        else:
            budget = f"${prefs.budget_min:,}+/mo"
        pref_lines.append(f"Budget: {budget}")
    if prefs.bedrooms:
        beds = ", ".join("Studio" if b == 0 else f"{b}BR" for b in prefs.bedrooms)
        pref_lines.append(f"Bedrooms: {beds}")
    if prefs.neighborhoods:
        pref_lines.append(f"Neighborhoods: {', '.join(prefs.neighborhoods)}")
    if prefs.min_bathrooms:
        pref_lines.append(f"Min bathrooms: {prefs.min_bathrooms}")
    if prefs.must_haves:
        pref_lines.append(f"Must-haves: {', '.join(prefs.must_haves)}")
    if prefs.nice_to_haves:
        pref_lines.append(f"Nice-to-haves: {', '.join(prefs.nice_to_haves)}")
    if prefs.no_fee_only:
        pref_lines.append("No-fee only: yes")
    if prefs.commute_address:
        commute = f"Commute to: {prefs.commute_address}"
        if prefs.commute_max_minutes:
            commute += f" (max {prefs.commute_max_minutes} min)"
        pref_lines.append(commute)
    if prefs.move_in_date:
        pref_lines.append(f"Move-in: {prefs.move_in_date}")
    prefs_text = "\n".join(pref_lines) if pref_lines else "No specific preferences."

    # Constraint context
    constraint_text = ""
    if prefs.constraint_context:
        constraint_text = (
            f"\nConstraint context (what's firm vs flexible):\n"
            f"{prefs.constraint_context}\n"
        )

    # Current apartment context
    apt_text = ""
    if current_apartment:
        apt_parts = []
        if current_apartment.price:
            apt_parts.append(f"Current rent: ${current_apartment.price:,}/mo")
        if current_apartment.neighborhood:
            apt_parts.append(f"Current neighborhood: {current_apartment.neighborhood}")
        if current_apartment.pros:
            apt_parts.append(f"Likes: {', '.join(current_apartment.pros)}")
        if current_apartment.cons:
            apt_parts.append(f"Dislikes: {', '.join(current_apartment.cons)}")
        if apt_parts:
            apt_text = "\nCurrent apartment:\n" + "\n".join(apt_parts)

    prompt = (
        "Filter and score these NYC apartment listings against the user's preferences.\n"
        "Return ONLY a JSON array, no other text.\n\n"
        f"User preferences:\n{prefs_text}{constraint_text}{apt_text}\n\n"
        f"Listings:\n{json.dumps(listings_data, separators=(',', ':'))}\n\n"
        "Two-step evaluation for each listing:\n"
        "1. FILTER: Does this listing violate a hard constraint (dealbreaker)? "
        "If yes, set include=false. If no, set include=true.\n"
        "2. SCORE: Rate how well it matches soft preferences on a 0-100 scale.\n\n"
        "Refer to the constraint context to determine what the user considers a "
        "dealbreaker. If no constraint context is provided, use your best judgment "
        "based on the preferences.\n\n"
        "Return format — one object per listing:\n"
        '[{"id":"...","include":true,"score":75,"pros":["pro1","pro2"],"cons":["con1"]}]\n\n'
        "Scoring guide (for included listings):\n"
        "- 80-100: Excellent — hits core criteria (budget, beds, neighborhood)\n"
        "- 60-79: Good — meets most criteria with minor trade-offs\n"
        "- 40-59: Decent — some compromises but worth considering\n"
        "- 25-39: Weak — misses some criteria but still viable\n"
        "- 0-24: Poor match\n\n"
        "Be nuanced: a slightly over-budget listing in a perfect location can score well. "
        "Adjacent neighborhoods to preferred areas are good, not zero. "
        "Must-haves matter more than nice-to-haves. "
        "2-3 pros, 1-2 cons per listing. Return ONLY the JSON array."
    )

    try:
        client = anthropic.AsyncAnthropic(
            api_key=os.environ.get("ANTHROPIC_API_KEY", ""),
        )
        response = await client.messages.create(
            model=SCORING_MODEL,
            max_tokens=8192,
            messages=[{"role": "user", "content": prompt}],
        )

        text = response.content[0].text.strip()

        # Strip markdown code fences if present
        if text.startswith("```"):
            text = text.split("\n", 1)[1]
            text = text.rsplit("```", 1)[0].strip()

        scores = json.loads(text)
        score_map = {item["id"]: item for item in scores}

        # Apply scores and filter
        for listing in listings:
            scored = score_map.get(listing.listing_id)
            if scored:
                listing.match_score = max(0, min(100, scored.get("score", 50)))
                listing.pros = scored.get("pros", [])[:3]
                listing.cons = scored.get("cons", [])[:2]
            else:
                # LLM omitted this listing — include with default score
                listing.match_score = 50

        # Filter: only include listings where LLM said include=true AND score >= SCORE_FLOOR
        included = []
        for listing in listings:
            scored_data = score_map.get(listing.listing_id)
            # If LLM omitted it, default to include=True
            llm_include = scored_data.get("include", True) if scored_data else True
            if llm_include and (listing.match_score or 0) >= SCORE_FLOOR:
                included.append(listing)

        # Fallback: if all excluded, return top 3 by score
        if not included and listings:
            listings.sort(key=lambda l: l.match_score or 0, reverse=True)
            included = listings[:3]
            logger.info("All listings excluded by LLM filter, returning top 3 as fallback")

        return included

    except Exception:
        logger.exception("LLM scoring failed, returning listings unscored")
        # Sort by price as fallback when scoring fails
        listings.sort(key=lambda l: l.price)
        return listings
