"""Daily apartment scanning: search StreetEasy, deduplicate, LLM-score, notify."""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import re
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from typing import Any

import anthropic
import httpx

from src.claude_client import call_with_retry
from src.config import (
    SCORING_MODEL, VISION_MODEL, SCORE_FLOOR, CACHE_MAX_AGE_HOURS, AMENITY_CACHE_MAX_ITEMS,
    AMENITY_TEXT_MAX_LEN, STALE_LISTING_MONTHS, MAX_PER_BUILDING, MAX_PER_NEIGHBORHOOD,
    MAX_RESULTS_TO_SEND, MAX_LISTINGS_PER_BATCH, PHOTO_DOWNLOAD_SEMAPHORE,
    MAX_PROS_DISPLAYED, MAX_CONS_DISPLAYED,
    MAX_AMENITIES_DISPLAY, AMENITY_REQUIRED_COVERAGE, AMENITY_ENRICHMENT_MAX_WAIT_SECS,
    AMENITY_ENRICHMENT_NO_ITEMS_ABORT_SECS, SEARCH_FAILURE_STREAK_FOR_COOLDOWN,
    SEARCH_FAILURE_COOLDOWN_BASE_SECS, SEARCH_FAILURE_COOLDOWN_MAX_SECS,
)
from src.formatter import format_apartment_context, format_listing_card, format_scan_header, listing_keyboard
from src.models import MAX_RECENT_LISTINGS, ChatState, CurrentApartment, Listing, Preferences
from src.apify_scraper import ApifyScraper, STREETEASY_PHOTO_URL, STREETEASY_PHOTO_URL_THUMB
from src.storage import load_all_states, save_state

logger = logging.getLogger(__name__)


@dataclass
class ScoringResult:
    """Result from LLM scoring, with fallback flag."""
    listings: list[Listing]
    is_fallback: bool = False


@dataclass
class FilterResult:
    """Result from pre-filter chain with intermediate counts for funnel logging."""
    listings: list[Listing]
    after_hood: int = 0
    after_beds: int = 0
    after_baths: int = 0
    after_stale: int = 0
    after_budget: int = 0
    after_min_budget: int = 0
    after_fee: int = 0


def _score_key(listing: Listing) -> int:
    """Sort key returning composite rank_score, then match_score."""
    if listing.rank_score is not None:
        return listing.rank_score
    return listing.match_score or 0


_SIGNAL_WORD_RE = re.compile(r"[^a-z0-9]+")

AMENITY_SYNONYMS: dict[str, tuple[str, ...]] = {
    "laundry in unit": ("in unit laundry", "washer dryer", "washer and dryer", "w d in unit"),
    "in unit laundry": ("laundry in unit", "washer dryer", "washer and dryer", "w d in unit"),
    "laundry in building": ("building laundry", "laundry room", "laundry"),
    "dishwasher": ("dish washer",),
    "central ac": ("central air", "air conditioning", "a c"),
    "doorman": ("door man", "concierge", "attended lobby"),
    "outdoor space": ("balcony", "terrace", "roof deck", "patio", "garden"),
    "pets allowed": ("pet friendly", "cats allowed", "dogs allowed", "pets"),
    "gym": ("fitness center", "fitness room", "health club"),
}


def _normalize_signal(value: str) -> str:
    """Normalize free text for lightweight deterministic signal matching."""
    return _SIGNAL_WORD_RE.sub(" ", value.lower()).strip()


def _term_variants(term: str) -> set[str]:
    """Return normalized variants for a user preference term."""
    normalized = _normalize_signal(term)
    variants = {normalized}
    for synonym in AMENITY_SYNONYMS.get(normalized, ()):
        variants.add(_normalize_signal(synonym))
    return {v for v in variants if v}


def _contains_term(haystack: str, term: str) -> bool:
    """Check whether any normalized variant of term appears in haystack."""
    if not haystack:
        return False
    return any(variant in haystack for variant in _term_variants(term))


def _listing_signal_text(listing: Listing) -> str:
    """Combine structured and text fields into one normalized amenity signal string."""
    parts: list[str] = []
    for values in (
        listing.amenities,
        listing.matched_amenities,
        listing.building_amenities,
        listing.unit_features,
        listing.confirmed_building_amenities,
        listing.confirmed_unit_features,
    ):
        parts.extend(values)
    if listing.amenity_text_dump:
        parts.append(listing.amenity_text_dump)
    if listing.description:
        parts.append(listing.description)
    return _normalize_signal(" ".join(parts))


def _missing_signal_text(listing: Listing) -> str:
    """Return normalized negative amenity evidence from search/enrichment data."""
    return _normalize_signal(" ".join(listing.missing_amenities))


def _effective_price(listing: Listing) -> int:
    """Return the price users experience for ranking purposes."""
    if listing.net_effective_price and listing.net_effective_price > 0:
        return listing.net_effective_price
    return listing.price


def _add_badge(badges: list[str], badge: str, *, limit: int = 5) -> None:
    """Append a short rank badge once, preserving priority order."""
    if badge and badge not in badges and len(badges) < limit:
        badges.append(badge)


def _local_rank_adjustment(
    listing: Listing,
    prefs: Preferences,
    current_apartment: CurrentApartment | None = None,
) -> tuple[int, list[str]]:
    """Compute deterministic ranking adjustment and explanation badges.

    Claude remains responsible for subjective fit, but this layer makes ranking
    stable and data-aware when the model gives close scores or misses obvious
    structured signals like concessions, no-fee status, and verified amenities.
    """
    adjustment = 0
    badges: list[str] = []
    effective_price = _effective_price(listing)

    if prefs.budget_max:
        if effective_price <= prefs.budget_max:
            headroom = prefs.budget_max - effective_price
            budget_points = min(8, max(1, round((headroom / prefs.budget_max) * 32)))
            adjustment += budget_points
            if headroom >= 100:
                _add_badge(badges, f"${headroom:,} under budget")
            else:
                _add_badge(badges, "Within budget")
        else:
            overage_ratio = (effective_price - prefs.budget_max) / prefs.budget_max
            adjustment -= min(24, max(6, round(overage_ratio * 100)))

    if prefs.budget_min and effective_price < prefs.budget_min:
        adjustment -= 6

    if listing.net_effective_price and listing.net_effective_price != listing.price:
        _add_badge(badges, "Net-effective deal")

    if prefs.bedrooms:
        if listing.bedrooms in prefs.bedrooms:
            adjustment += 4
            _add_badge(badges, "Bedroom match")
        else:
            adjustment -= 18

    if prefs.min_bathrooms:
        if listing.bathrooms >= prefs.min_bathrooms:
            adjustment += 3
        else:
            adjustment -= 12

    if prefs.neighborhoods:
        pref_raw = {n.lower() for n in prefs.neighborhoods}
        pref_canonical = {_normalize_hood(n) for n in prefs.neighborhoods}
        hood_lower = listing.neighborhood.lower()
        hood_canonical = _normalize_hood(listing.neighborhood)
        if hood_lower in pref_raw:
            adjustment += 5
            _add_badge(badges, "Preferred area")
        elif hood_canonical in pref_canonical:
            adjustment += 4
            _add_badge(badges, "Preferred area")
        else:
            adjustment -= 8

    if listing.broker_fee:
        adjustment -= 5
        if prefs.no_fee_only:
            adjustment -= 20
    else:
        adjustment += 5
        _add_badge(badges, "No fee")

    signal_text = _listing_signal_text(listing)
    missing_text = _missing_signal_text(listing)
    must_hits = 0
    for must_have in prefs.must_haves:
        if _contains_term(missing_text, must_have):
            adjustment -= 16
            continue
        if _contains_term(signal_text, must_have):
            adjustment += 7
            must_hits += 1
        elif listing.amenity_signal_status == "missing":
            adjustment -= 4

    if prefs.must_haves and must_hits:
        if must_hits == len(prefs.must_haves):
            _add_badge(badges, "Must-haves verified")
        else:
            _add_badge(badges, f"{must_hits}/{len(prefs.must_haves)} must-haves")

    nice_hits = 0
    for nice_to_have in prefs.nice_to_haves:
        if _contains_term(signal_text, nice_to_have):
            adjustment += 3
            nice_hits += 1
    if nice_hits:
        _add_badge(badges, f"{nice_hits} bonus amenity{'ies' if nice_hits != 1 else ''}")

    if listing.sqft:
        sqft_targets = {0: 400, 1: 600, 2: 800, 3: 1000}
        target = sqft_targets.get(listing.bedrooms, 1000 + max(0, listing.bedrooms - 3) * 180)
        if listing.sqft >= target:
            adjustment += 3
            _add_badge(badges, "Roomier layout")

    if current_apartment:
        current_price = current_apartment.price
        if current_price:
            if effective_price <= current_price:
                adjustment += 3
                _add_badge(badges, "Rent improvement")
            elif effective_price > current_price * 1.1:
                adjustment -= 3
        if current_apartment.neighborhood:
            if _normalize_hood(current_apartment.neighborhood) == _normalize_hood(listing.neighborhood):
                adjustment += 2

    return adjustment, badges


def _apply_rank_scores(
    listings: list[Listing],
    prefs: Preferences,
    current_apartment: CurrentApartment | None = None,
) -> list[Listing]:
    """Populate composite rank scores and explanation badges on listings."""
    for listing in listings:
        base_score = listing.match_score if listing.match_score is not None else 50
        adjustment, badges = _local_rank_adjustment(listing, prefs, current_apartment)
        listing.rank_score = max(0, min(100, round(base_score + adjustment)))
        listing.rank_badges = badges
    return listings


def _rank_listings(
    listings: list[Listing],
    prefs: Preferences,
    current_apartment: CurrentApartment | None = None,
) -> list[Listing]:
    """Return listings ordered by composite fit, with deterministic tie-breakers."""
    ranked = _apply_rank_scores(listings, prefs, current_apartment)
    return sorted(
        ranked,
        key=lambda l: (
            _score_key(l),
            l.match_score or 0,
            -_effective_price(l),
            l.sqft or 0,
        ),
        reverse=True,
    )


def _cooldown_remaining_secs(state: ChatState, now: datetime) -> int:
    """Return remaining cooldown seconds (0 when no active cooldown)."""
    if not state.search_cooldown_until:
        return 0
    remaining = int((state.search_cooldown_until - now).total_seconds())
    return max(0, remaining)


def _reset_search_failure_state(state: ChatState) -> None:
    """Clear consecutive search failure tracking."""
    state.search_failure_streak = 0
    state.last_search_failure_at = None
    state.search_cooldown_until = None


def _record_search_failure(state: ChatState, now: datetime) -> tuple[int, int]:
    """Update search failure counters and cooldown window.

    Returns:
        tuple[int, int]: (new_streak, applied_cooldown_secs)
    """
    state.search_failure_streak = (state.search_failure_streak or 0) + 1
    state.last_search_failure_at = now

    cooldown_secs = 0
    if state.search_failure_streak >= SEARCH_FAILURE_STREAK_FOR_COOLDOWN:
        exponent = state.search_failure_streak - SEARCH_FAILURE_STREAK_FOR_COOLDOWN
        cooldown_secs = min(
            SEARCH_FAILURE_COOLDOWN_BASE_SECS * (2 ** exponent),
            SEARCH_FAILURE_COOLDOWN_MAX_SECS,
        )
        state.search_cooldown_until = now + timedelta(seconds=cooldown_secs)

    return state.search_failure_streak, cooldown_secs


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
    is_daily: bool = True,
) -> None:
    """Scan StreetEasy for a single chat's preferences."""
    prefs = state.preferences
    logger.info("Scanning for chat %s with prefs: %s", state.chat_id, prefs.neighborhoods)

    now = datetime.now(timezone.utc)
    cooldown_remaining = _cooldown_remaining_secs(state, now)
    if cooldown_remaining > 0:
        logger.warning(
            "Search suppressed by cooldown: chat_id=%s streak=%d remaining_secs=%d",
            state.chat_id,
            state.search_failure_streak,
            cooldown_remaining,
        )
        if is_daily:
            await telegram_bot.send_text(
                state.chat_id,
                "StreetEasy is temporarily blocking requests. I'll retry on the next scan.",
            )
        else:
            wait_minutes = max(1, cooldown_remaining // 60)
            await telegram_bot.send_text(
                state.chat_id,
                f"StreetEasy is temporarily blocking requests. Please retry in about {wait_minutes} minute(s).",
            )
        save_state(state)
        return

    # Search StreetEasy via Apify actor (with retry)
    try:
        raw_listings = await scraper.search_with_retry(prefs)
        if state.search_failure_streak or state.search_cooldown_until:
            logger.info(
                "Search recovered: chat_id=%s previous_streak=%d",
                state.chat_id,
                state.search_failure_streak,
            )
            _reset_search_failure_state(state)
    except Exception:
        failure_time = datetime.now(timezone.utc)
        streak, cooldown_secs = _record_search_failure(state, failure_time)
        logger.exception("StreetEasy search failed")
        logger.warning(
            "Search failure recorded: chat_id=%s streak=%d cooldown_secs=%d",
            state.chat_id,
            streak,
            cooldown_secs,
        )
        # Try cache fallback
        if _has_cached_scan(state):
            cached = _get_cached_listings(state)
            if cached:
                await telegram_bot.send_text(
                    state.chat_id,
                    "\u26a0\ufe0f StreetEasy is temporarily unavailable. "
                    "Here are your results from last time:",
                )
                for i, listing in enumerate(cached, 1):
                    card_text = format_listing_card(listing, rank=i)
                    keyboard = listing_keyboard(listing.listing_id, listing.url)
                    photo_url = listing.photos[0] if listing.photos else None
                    if photo_url:
                        await telegram_bot.send_listing_photo(
                            state.chat_id,
                            listing_url=listing.url,
                            photo_url=photo_url,
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
                return
        await telegram_bot.send_text(
            state.chat_id,
            "\u26a0\ufe0f I had trouble searching StreetEasy today. I'll try again tomorrow.",
        )
        save_state(state)
        return

    if not raw_listings:
        await telegram_bot.send_text(state.chat_id, format_scan_header(0, is_daily=is_daily))
        save_state(state)
        return

    # Deduplicate by listing_id within this run, and drop already-seen listings.
    new_listings = []
    seen_in_run: set[str] = set()
    for raw in raw_listings:
        lid = str(raw.get("listing_id", ""))
        if lid and lid not in state.seen_listing_ids and lid not in seen_in_run:
            seen_in_run.add(lid)
            new_listings.append(raw)

    if not new_listings:
        await telegram_bot.send_text(state.chat_id, format_scan_header(0, is_daily=is_daily))
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

    # Layer 1: Local pre-filters (compensate for path/pipe URL limitations)
    filter_result = _apply_pre_filters(parsed, prefs)
    filtered = filter_result.listings

    if not filtered and parsed:
        logger.info("All %d listings filtered out by local pre-filters", len(parsed))
        if is_daily:
            msg = "\U0001f50d No listings found in your neighborhoods today. I'll check again tomorrow!"
        else:
            msg = "\U0001f50d No listings found in your neighborhoods."
        await telegram_bot.send_text(state.chat_id, msg)
        save_state(state)
        return

    # Layer 1.5: Amenity enrichment via detail pages (skip if no amenity prefs)
    enriched_count = 0
    amenity_coverage = 1.0
    amenity_failure_reason: str | None = None
    if (prefs.must_haves or prefs.nice_to_haves) and filtered:
        cache_hits = 0
        to_fetch_urls: list[str] = []
        url_to_id: dict[str, str] = {}

        for listing in filtered:
            cached = state.amenity_cache.get(listing.listing_id)
            if cached:
                _apply_amenity_data(listing, cached)
                enriched_count += 1
                cache_hits += 1
            elif listing.url:
                to_fetch_urls.append(listing.url)
                url_to_id[listing.url] = listing.listing_id

        try:
            if to_fetch_urls:
                enrichment_result = await scraper.enrich_with_amenities(
                    to_fetch_urls,
                    url_to_id,
                    max_total_wait_secs=AMENITY_ENRICHMENT_MAX_WAIT_SECS,
                    required_coverage=AMENITY_REQUIRED_COVERAGE,
                    no_items_abort_secs=AMENITY_ENRICHMENT_NO_ITEMS_ABORT_SECS,
                )
                amenity_data = enrichment_result.data_by_listing_id
                if enrichment_result.failed:
                    amenity_failure_reason = enrichment_result.failure_reason
            else:
                amenity_data = {}
                enrichment_result = None

            for listing in filtered:
                data = amenity_data.get(listing.listing_id)
                if not data:
                    continue
                _apply_amenity_data(listing, data)
                state.amenity_cache[listing.listing_id] = data
                enriched_count += 1

            _trim_amenity_cache(state)
            amenity_coverage = enriched_count / len(filtered)

            logger.info(
                "Amenity enrichment: cache_hits=%d fetched=%d enriched=%d/%d coverage=%.1f%%",
                cache_hits,
                len(amenity_data),
                enriched_count,
                len(filtered),
                amenity_coverage * 100,
            )
            logger.info(
                "Amenity send decision: coverage=%.3f threshold=%.3f send_allowed=%s failure_reason=%s",
                amenity_coverage,
                AMENITY_REQUIRED_COVERAGE,
                amenity_coverage >= AMENITY_REQUIRED_COVERAGE and not amenity_failure_reason,
                amenity_failure_reason or "",
            )
        except Exception:
            logger.exception("Amenity enrichment failed")
            amenity_failure_reason = "enrichment_exception"

        if amenity_failure_reason or amenity_coverage < AMENITY_REQUIRED_COVERAGE:
            if is_daily:
                await telegram_bot.send_text(
                    state.chat_id,
                    "Today's scan couldn't verify amenities reliably; I'll try again next run.",
                )
            else:
                await telegram_bot.send_text(
                    state.chat_id,
                    "Couldn't verify amenities reliably right now. Please retry shortly.",
                )
            save_state(state)
            return

    # Layer 2: LLM filter + score
    scoring_result = await _llm_score_listings(filtered, prefs, state.current_apartment)
    scored = scoring_result.listings

    # Sort by composite fit, then apply diversity caps
    scored = _rank_listings(scored, prefs, state.current_apartment)
    pre_building_dedup = len(scored)
    scored = _cap_per_building(scored)
    scored = _cap_per_neighborhood(scored)
    scored = _interleave_by_neighborhood(scored)
    scored = scored[:MAX_RESULTS_TO_SEND]

    # Funnel logging
    logger.info(
        "Scan funnel for chat %s: Raw=%d | Dedup=%d | Hood=%d | Beds=%d | Baths=%d | "
        "Stale=%d | Budget=%d | MinBudget=%d | Fee=%d | Enriched=%d | LLM=%d | Bldg=%d | Sent=%d",
        state.chat_id, len(raw_listings), len(new_listings), filter_result.after_hood,
        filter_result.after_beds, filter_result.after_baths, filter_result.after_stale,
        filter_result.after_budget, filter_result.after_min_budget, filter_result.after_fee,
        enriched_count, pre_building_dedup, len(scored), len(scored),
    )

    # Pick hero photos via vision model (graceful fallback to photos[0])
    hero_photos = await _pick_hero_photos(scored)

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
        await telegram_bot.send_text(state.chat_id, format_scan_header(0, is_daily=is_daily))
    else:
        await telegram_bot.send_text(state.chat_id, format_scan_header(len(scored), is_daily=is_daily))

        if scoring_result.is_fallback:
            await telegram_bot.send_text(
                state.chat_id,
                "\u26a0\ufe0f None of these perfectly matched your criteria, but here are the closest options.",
            )

        for i, listing in enumerate(scored, 1):
            card_text = format_listing_card(listing, rank=i)
            keyboard = listing_keyboard(listing.listing_id, listing.url)

            photo_url = hero_photos.get(listing.listing_id, listing.photos[0]) if listing.photos else None
            if photo_url:
                await telegram_bot.send_listing_photo(
                    state.chat_id,
                    listing_url=listing.url,
                    photo_url=photo_url,
                    caption=card_text,
                    keyboard=keyboard,
                )
            else:
                await telegram_bot.send_text(
                    state.chat_id,
                    card_text,
                    keyboard=keyboard,
                )

    # Mark scored listings as seen (only after successful scoring/sending)
    for listing in scored:
        state.seen_listing_ids.add(listing.listing_id)

    # Cache successful scan results
    if scored:
        state.last_scan_listing_ids = [l.listing_id for l in scored]
        state.last_scan_at = datetime.now(timezone.utc)

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
        matched_amenities=raw.get("matched_amenities", []),
        missing_amenities=raw.get("missing_amenities", []),
        confirmed_building_amenities=raw.get("confirmed_building_amenities", []),
        confirmed_unit_features=raw.get("confirmed_unit_features", []),
        amenity_text_dump=raw.get("amenity_text_dump"),
        amenity_signal_status=raw.get("amenity_signal_status"),
        photos=raw.get("photos", []),
        photo_keys=raw.get("photo_keys", []),
        broker_fee=raw.get("broker_fee"),
        available_date=raw.get("available_date"),
        description=raw.get("description"),
        net_effective_price=raw.get("net_effective_price"),
        months_free=raw.get("months_free"),
        scraped_at=datetime.now(timezone.utc),
    )


def _has_cached_scan(state: ChatState) -> bool:
    """Check if a recent cached scan exists (< CACHE_MAX_AGE_HOURS old)."""
    if not state.last_scan_listing_ids or not state.last_scan_at:
        return False
    age_hours = (datetime.now(timezone.utc) - state.last_scan_at).total_seconds() / 3600
    return age_hours < CACHE_MAX_AGE_HOURS


def _get_cached_listings(state: ChatState) -> list[Listing]:
    """Look up cached scan listing IDs in recent_listings. Skips missing IDs."""
    result = []
    for lid in state.last_scan_listing_ids:
        listing = state.recent_listings.get(lid)
        if listing:
            result.append(listing)
    return result


def _build_amenity_text_dump(
    building_amenities: list[str],
    unit_features: list[str],
    description: str | None,
) -> str:
    """Build a short amenity evidence summary for LLM scoring."""
    parts = []
    if building_amenities:
        parts.append("Building: " + ", ".join(building_amenities[:MAX_AMENITIES_DISPLAY]))
    if unit_features:
        parts.append("Unit: " + ", ".join(unit_features[:MAX_AMENITIES_DISPLAY]))
    if description:
        desc = description.strip()
        if len(desc) > 140:
            desc = desc[:140]
        parts.append("Description: " + desc)
    return " | ".join(parts)[:AMENITY_TEXT_MAX_LEN]


def _derive_amenity_signal_status(
    building_amenities: list[str],
    unit_features: list[str],
    description: str | None,
) -> str:
    """Classify amenity evidence completeness."""
    has_structured = bool(building_amenities or unit_features)
    has_text = bool(description and description.strip())
    if has_structured:
        return "present"
    if has_text:
        return "partial"
    return "missing"


def _apply_amenity_data(listing: Listing, data: dict[str, Any]) -> None:
    """Apply enrichment payload to a listing and derive contract fields."""
    building = list(data.get("building_amenities", []))
    unit = list(data.get("unit_features", []))
    description = data.get("description")

    listing.building_amenities = building
    listing.unit_features = unit
    listing.confirmed_building_amenities = building
    listing.confirmed_unit_features = unit

    if description and not listing.description:
        listing.description = description

    listing.amenity_signal_status = _derive_amenity_signal_status(
        building,
        unit,
        listing.description,
    )
    listing.amenity_text_dump = _build_amenity_text_dump(
        building,
        unit,
        listing.description,
    )


def _trim_amenity_cache(state: ChatState) -> None:
    """Bound amenity cache size in state."""
    if len(state.amenity_cache) <= AMENITY_CACHE_MAX_ITEMS:
        return
    overflow = len(state.amenity_cache) - AMENITY_CACHE_MAX_ITEMS
    old_ids = list(state.amenity_cache.keys())[:overflow]
    for lid in old_ids:
        del state.amenity_cache[lid]


def _normalize_hood(name: str) -> str:
    """Normalize a neighborhood name to its canonical form (lowered).

    Resolves aliases in both directions: sub-areas map to parent neighborhoods
    AND non-canonical variants map to config canonical names.
    """
    lower = name.lower()
    return NEIGHBORHOOD_ALIASES.get(lower, lower)


def _neighborhood_pre_filter(
    listings: list[Listing],
    prefs: Preferences,
) -> list[Listing]:
    """Filter listings to only those in the user's preferred neighborhoods.

    This is a data quality fix, NOT preference enforcement. StreetEasy's
    RECOMMENDED sort ignores the areas filter, returning ~98% wrong-area results.
    This pre-filter compensates for that specific API bug.

    Normalizes BOTH preference names and listing names to canonical forms via
    NEIGHBORHOOD_ALIASES before matching. This handles bidirectional aliases:
    e.g. user says "Gramercy Park" and listing says "Gramercy" (or vice versa).

    If no neighborhoods are set in preferences, all listings pass through.
    """
    if not prefs.neighborhoods:
        return listings

    # Normalize preference neighborhoods to canonical names
    pref_canonical = set()
    for name in prefs.neighborhoods:
        pref_canonical.add(_normalize_hood(name))
        # Also add the raw lowered name for direct matches
        pref_canonical.add(name.lower())

    result = []
    for listing in listings:
        hood_lower = listing.neighborhood.lower()
        # Direct match against raw or canonical preference names
        if hood_lower in pref_canonical:
            result.append(listing)
            continue
        # Normalize listing neighborhood and check
        canonical = _normalize_hood(listing.neighborhood)
        if canonical in pref_canonical:
            result.append(listing)

    return result


def _filter_wrong_bedrooms(
    listings: list[Listing],
    prefs: Preferences,
) -> list[Listing]:
    """Drop listings with wrong bedroom count.

    Compensates for path URLs not filtering by bedrooms.
    If no bedrooms preference is set, all listings pass through.
    """
    if not prefs.bedrooms:
        return listings
    allowed = set(prefs.bedrooms)
    return [l for l in listings if l.bedrooms in allowed]


def _filter_wrong_bathrooms(
    listings: list[Listing],
    prefs: Preferences,
) -> list[Listing]:
    """Drop listings with fewer bathrooms than the user requires.

    This is a local hard filter that compensates for the rented Apify actor
    silently dropping the baths URL param.

    If min_bathrooms is not set, all listings pass through.
    """
    if not prefs.min_bathrooms:
        return listings
    return [l for l in listings if l.bathrooms >= prefs.min_bathrooms]


def _filter_broker_fee(
    listings: list[Listing],
    prefs: Preferences,
) -> list[Listing]:
    """Drop listings with broker fees when user requires no-fee only.

    Compensates for the rented Apify actor ignoring the no_fee URL param.
    If no_fee_only is False, all listings pass through.
    """
    if not prefs.no_fee_only:
        return listings
    return [l for l in listings if l.broker_fee is None]


def _filter_over_budget(
    listings: list[Listing],
    prefs: Preferences,
) -> list[Listing]:
    """Drop listings that clearly exceed the user's budget.

    Keeps a listing if EITHER gross price or net_effective_price is within
    budget — covers free-months concessions (gross over, net under) and the
    rare case where net > gross.
    If no budget_max is set, all listings pass through.
    """
    if not prefs.budget_max:
        return listings
    result = []
    for l in listings:
        if l.price <= prefs.budget_max:
            result.append(l)
        elif l.net_effective_price and l.net_effective_price <= prefs.budget_max:
            result.append(l)
    return result


def _filter_under_budget(
    listings: list[Listing],
    prefs: Preferences,
) -> list[Listing]:
    """Drop listings significantly below the user's minimum budget.

    Filters out suspiciously cheap listings (often stale or scam postings).
    If no budget_min is set, all listings pass through.
    """
    if not prefs.budget_min:
        return listings
    return [l for l in listings if l.price >= prefs.budget_min]


def _filter_stale_listings(listings: list[Listing]) -> list[Listing]:
    """Drop listings with available_date more than STALE_LISTING_MONTHS in the past.

    Listings with no available_date or unparseable dates pass through.
    """
    cutoff = date.today() - timedelta(days=STALE_LISTING_MONTHS * 30)
    result = []
    for listing in listings:
        if not listing.available_date:
            result.append(listing)
            continue
        try:
            avail = date.fromisoformat(listing.available_date)
            if avail >= cutoff:
                result.append(listing)
        except (ValueError, TypeError):
            result.append(listing)
    return result


def _apply_pre_filters(listings: list[Listing], prefs: Preferences) -> FilterResult:
    """Run all local pre-filters and return results with intermediate counts."""
    after_hood = _neighborhood_pre_filter(listings, prefs)
    after_beds = _filter_wrong_bedrooms(after_hood, prefs)
    after_baths = _filter_wrong_bathrooms(after_beds, prefs)
    after_stale = _filter_stale_listings(after_baths)
    after_budget = _filter_over_budget(after_stale, prefs)
    after_min_budget = _filter_under_budget(after_budget, prefs)
    after_fee = _filter_broker_fee(after_min_budget, prefs)
    return FilterResult(
        listings=after_fee,
        after_hood=len(after_hood),
        after_beds=len(after_beds),
        after_baths=len(after_baths),
        after_stale=len(after_stale),
        after_budget=len(after_budget),
        after_min_budget=len(after_min_budget),
        after_fee=len(after_fee),
    )


def _extract_building_address(address: str) -> str:
    """Extract building address from full address: '123 Main St #4A' -> '123 main st'."""
    return address.split("#")[0].strip().lower()


def _cap_per_building(listings: list[Listing]) -> list[Listing]:
    """Cap to MAX_PER_BUILDING listings per building. Assumes sorted by score desc."""
    building_counts: dict[str, int] = {}
    result = []
    for listing in listings:
        building = _extract_building_address(listing.address)
        count = building_counts.get(building, 0)
        if count < MAX_PER_BUILDING:
            result.append(listing)
            building_counts[building] = count + 1
    return result


def _canonical_neighborhood(hood: str) -> str:
    """Get canonical neighborhood name (lowered), resolving aliases."""
    return _normalize_hood(hood)


def _cap_per_neighborhood(listings: list[Listing]) -> list[Listing]:
    """Cap to MAX_PER_NEIGHBORHOOD listings per canonical neighborhood.

    Prevents one neighborhood from dominating results (e.g. Lincoln Square
    was 47% of results). Assumes input sorted by score descending.
    Uses NEIGHBORHOOD_ALIASES for canonical grouping.
    """
    hood_counts: dict[str, int] = {}
    result = []
    for listing in listings:
        canonical = _canonical_neighborhood(listing.neighborhood)
        count = hood_counts.get(canonical, 0)
        if count < MAX_PER_NEIGHBORHOOD:
            result.append(listing)
            hood_counts[canonical] = count + 1
    return result


def _interleave_by_neighborhood(listings: list[Listing]) -> list[Listing]:
    """Round-robin interleave listings across neighborhoods.

    Ensures the user sees variety in the first few listings rather than
    a cluster from one neighborhood. Neighborhoods are ordered by their
    best listing score (descending).
    """
    if not listings:
        return listings

    # Group by canonical neighborhood, preserving per-group order
    groups: dict[str, list[Listing]] = {}
    for listing in listings:
        canonical = _canonical_neighborhood(listing.neighborhood)
        if canonical not in groups:
            groups[canonical] = []
        groups[canonical].append(listing)

    # Order neighborhoods by best score (first element, since input is score-sorted)
    sorted_hoods = sorted(
        groups.keys(),
        key=lambda h: _score_key(groups[h][0]),
        reverse=True,
    )

    # Round-robin across neighborhoods
    result: list[Listing] = []
    indices = {h: 0 for h in sorted_hoods}
    while len(result) < len(listings):
        added_any = False
        for hood in sorted_hoods:
            idx = indices[hood]
            if idx < len(groups[hood]):
                result.append(groups[hood][idx])
                indices[hood] = idx + 1
                added_any = True
        if not added_any:
            break

    return result


def _format_listings_for_scoring(listings: list[Listing]) -> list[dict[str, Any]]:
    """Prepare listing data as compact dicts for the LLM scoring prompt."""
    listings_data = []
    for l in listings:
        confirmed_building = (
            l.confirmed_building_amenities if l.confirmed_building_amenities else l.building_amenities
        )
        confirmed_unit = (
            l.confirmed_unit_features if l.confirmed_unit_features else l.unit_features
        )
        amenity_text_dump = l.amenity_text_dump or _build_amenity_text_dump(
            confirmed_building,
            confirmed_unit,
            l.description,
        )
        amenity_signal_status = l.amenity_signal_status or _derive_amenity_signal_status(
            confirmed_building,
            confirmed_unit,
            l.description,
        )

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
        if confirmed_building:
            entry["confirmed_building_amenities"] = confirmed_building
        if confirmed_unit:
            entry["confirmed_unit_features"] = confirmed_unit
        if amenity_text_dump:
            entry["amenity_text_dump"] = amenity_text_dump
        entry["amenity_signal_status"] = amenity_signal_status
        if l.matched_amenities:
            entry["has_amenities"] = l.matched_amenities
        if l.missing_amenities:
            entry["missing_amenities"] = l.missing_amenities
        if l.description:
            entry["desc"] = l.description[:300]
        if l.net_effective_price and l.net_effective_price != l.price:
            entry["net_effective"] = l.net_effective_price
            entry["months_free"] = l.months_free
        # Phase 3: canonical neighborhood name for alias matches
        canonical = NEIGHBORHOOD_ALIASES.get(l.neighborhood.lower())
        if canonical:
            entry["hood_canonical"] = canonical.title()
        listings_data.append(entry)
    return listings_data


def _format_preferences_for_scoring(prefs: Preferences) -> str:
    """Format preferences as plain text for the LLM scoring prompt."""
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
    return "\n".join(pref_lines) if pref_lines else "No specific preferences."


def _parse_scoring_response(
    response_text: str,
    listings: list[Listing],
) -> ScoringResult:
    """Parse LLM scoring JSON, apply scores, filter by include + score floor."""
    text = response_text.strip()

    # Strip markdown code fences if present
    if text.startswith("```"):
        text = text.split("\n", 1)[1]
        text = text.rsplit("```", 1)[0].strip()

    # Extract JSON array — Claude may prepend/append text around the JSON.
    # Use raw_decode to parse exactly one JSON value, ignoring trailing text.
    start = text.find("[")
    if start == -1:
        raise ValueError("No JSON array found in scoring response")
    decoder = json.JSONDecoder()
    scores, _ = decoder.raw_decode(text, start)
    score_map = {item["id"]: item for item in scores}

    # Apply scores and filter
    for listing in listings:
        scored = score_map.get(listing.listing_id)
        if scored:
            listing.match_score = max(0, min(100, scored.get("score", 50)))
            listing.pros = scored.get("pros", [])[:MAX_PROS_DISPLAYED]
            listing.cons = scored.get("cons", [])[:MAX_CONS_DISPLAYED]
        else:
            # LLM omitted this listing — include with default score
            listing.match_score = 50

    # Filter: only include listings where LLM said include=true AND score >= SCORE_FLOOR
    included = []
    for listing in listings:
        scored_data = score_map.get(listing.listing_id)
        # If LLM omitted it, default to include=True
        llm_include = scored_data.get("include", True) if scored_data else True
        passes = llm_include and _score_key(listing) >= SCORE_FLOOR
        if passes:
            included.append(listing)

        # Per-listing decision logging
        canonical = NEIGHBORHOOD_ALIASES.get(listing.neighborhood.lower())
        hood_display = f"{listing.neighborhood}\u2192{canonical.title()}" if canonical else listing.neighborhood
        decision = "INCLUDED" if passes else "EXCLUDED"
        logger.info(
            "Listing %s [%s, $%s, %s]: include=%s score=%s -> %s",
            listing.listing_id,
            hood_display,
            f"{listing.price:,}",
            listing.address,
            llm_include,
            listing.match_score,
            decision,
        )

    logger.info(
        "LLM scoring: %d/%d included (floor=%d)",
        len(included),
        len(listings),
        SCORE_FLOOR,
    )

    # Fallback: if all excluded, return top 3 by score
    if not included and listings:
        listings.sort(key=_score_key, reverse=True)
        included = listings[:3]
        logger.info("All listings excluded by LLM filter, returning top 3 as fallback")
        return ScoringResult(listings=included, is_fallback=True)

    return ScoringResult(listings=included)


async def _llm_score_listings(
    listings: list[Listing],
    prefs: Preferences,
    current_apartment: CurrentApartment | None = None,
) -> ScoringResult:
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
        return ScoringResult(listings=listings)

    listings_data = _format_listings_for_scoring(listings)
    prefs_text = _format_preferences_for_scoring(prefs)

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
        apt_text = format_apartment_context(current_apartment)
        if apt_text:
            apt_text = "\nCurrent apartment:\n" + apt_text

    prompt = (
        f"Today's date is {date.today().isoformat()}.\n\n"
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
        "When 'hood_canonical' is provided, it means this listing's neighborhood is a "
        "sub-area of the user's preferred neighborhood. Treat it as matching.\n"
        "Amenity data sources (in order of reliability):\n"
        "1. 'confirmed_building_amenities' and 'confirmed_unit_features' — confirmed by StreetEasy detail pages. "
        "confirmed_building_amenities = building-level (Doorman, Gym, Elevator, Laundry in Building, etc.). "
        "confirmed_unit_features = unit-specific (Dishwasher, In-unit Laundry, Central AC, etc.). "
        "When these are present, use them to evaluate must-haves and nice-to-haves. "
        "A confirmed must-have should boost the score significantly. "
        "A must-have NOT in either list is likely absent — reduce score accordingly.\n"
        "2. 'has_amenities' / 'missing_amenities' — from search API annotation (when available). "
        "Same reliability as above but from a different data source.\n"
        "3. 'amenity_signal_status' summarizes confidence: present (structured), partial (text-only), missing (none). "
        "'amenity_text_dump' is a compact evidence summary. Use both to calibrate confidence.\n"
        "4. When no amenity fields are present — no amenity data available. "
        "Do not assume amenities are missing. Infer cautiously from building type, price, "
        "and neighborhood. A $15k/mo luxury high-rise almost certainly has doorman and gym. "
        "Score based on structural data (price, beds, baths, location) when amenity data "
        "is unavailable.\n"
        "When a listing has a net_effective price (from free months), use that for budget "
        "comparison, not the gross price. A listing at $4,500/mo gross with 2 months free "
        "has a net effective of ~$3,750/mo — that's the real cost.\n"
        "Must-haves matter more than nice-to-haves. "
        "2-3 pros, 1-2 cons per listing. Return ONLY the JSON array."
    )

    try:
        client = anthropic.AsyncAnthropic(
            api_key=os.environ.get("ANTHROPIC_API_KEY", ""),
        )
        response = await call_with_retry(
            lambda: client.messages.create(
                model=SCORING_MODEL,
                max_tokens=8192,
                messages=[{"role": "user", "content": prompt}],
            ),
            label="claude.score",
        )

        return _parse_scoring_response(response.content[0].text, listings)

    except Exception:
        logger.exception("LLM scoring failed, returning listings unscored")
        # Sort by price as fallback when scoring fails
        listings.sort(key=lambda l: l.price)
        return ScoringResult(listings=listings)


# ---------------------------------------------------------------------------
# Vision hero photo picker
# ---------------------------------------------------------------------------


def _sample_photo_keys(keys: list[str], max_count: int = 8) -> list[str]:
    """Sample up to max_count keys: first 3 + evenly spaced from remainder."""
    if len(keys) <= max_count:
        return list(keys)
    first = keys[:3]
    remaining = keys[3:]
    need = max_count - len(first)
    step = max(1, len(remaining) // need)
    sampled = [remaining[i * step] for i in range(need) if i * step < len(remaining)]
    return first + sampled[:need]


async def _download_photo(client: httpx.AsyncClient, url: str) -> tuple[str, bytes | None]:
    """Download a single photo, return (url, bytes | None)."""
    try:
        resp = await client.get(url, timeout=10.0, headers={"Referer": "https://streeteasy.com/"})
        if resp.status_code == 200 and resp.content:
            return (url, resp.content)
    except Exception:
        pass
    return (url, None)


async def _pick_hero_photos(listings: list[Listing]) -> dict[str, str]:
    """Pick the best hero photo for each listing using vision model.

    Returns dict mapping listing_id → best photo full-size URL.
    Returns {} on any failure (graceful fallback to photos[0]).
    """
    try:
        candidates = [l for l in listings if len(l.photo_keys) >= 2]
        if not candidates:
            return {}
        return await _vision_pick_heroes(candidates)
    except Exception:
        logger.exception("Hero photo picker failed, falling back to first photo")
        return {}


async def _vision_pick_heroes(listings: list[Listing]) -> dict[str, str]:
    """Core vision logic: download thumbnails, ask model to pick best hero photo."""
    # Build download tasks: {listing_id: [(key, thumb_url), ...]}
    download_plan: dict[str, list[tuple[str, str]]] = {}
    for listing in listings:
        sampled = _sample_photo_keys(listing.photo_keys)
        download_plan[listing.listing_id] = [
            (key, STREETEASY_PHOTO_URL_THUMB.format(key=key)) for key in sampled
        ]

    # Download all thumbnails in parallel with semaphore
    sem = asyncio.Semaphore(PHOTO_DOWNLOAD_SEMAPHORE)
    all_urls: list[tuple[str, str, str]] = []  # (listing_id, key, url)
    for lid, pairs in download_plan.items():
        for key, url in pairs:
            all_urls.append((lid, key, url))

    async with httpx.AsyncClient() as client:
        async def _bounded_download(url: str) -> tuple[str, bytes | None]:
            async with sem:
                return await _download_photo(client, url)

        results = await asyncio.gather(
            *[_bounded_download(url) for _, _, url in all_urls]
        )

    # Organize downloaded photos by listing
    # {listing_id: [(key, letter, b64_data), ...]}
    photos_by_listing: dict[str, list[tuple[str, str, str]]] = {}
    for (lid, key, url), (_, data) in zip(all_urls, results):
        if data is None:
            continue
        if lid not in photos_by_listing:
            photos_by_listing[lid] = []
        letter = chr(ord("A") + len(photos_by_listing[lid]))
        b64 = base64.standard_b64encode(data).decode("ascii")
        photos_by_listing[lid].append((key, letter, b64))

    # Need at least one listing with 2+ successfully downloaded photos
    viable = {lid: photos for lid, photos in photos_by_listing.items() if len(photos) >= 2}
    if not viable:
        return {}

    # Batch viable listings to stay under Claude API's 100-image limit
    # 12 listings × 8 photos = 96 images max per batch
    viable_ids = list(viable.keys())
    hero_map: dict[str, str] = {}

    for batch_start in range(0, len(viable_ids), MAX_LISTINGS_PER_BATCH):
        batch_ids = viable_ids[batch_start:batch_start + MAX_LISTINGS_PER_BATCH]
        batch_viable = {lid: viable[lid] for lid in batch_ids}
        batch_result = await _vision_pick_batch(batch_viable, listings)
        hero_map.update(batch_result)

    return hero_map


async def _vision_pick_batch(
    viable: dict[str, list[tuple[str, str, str]]],
    listings: list[Listing],
) -> dict[str, str]:
    """Pick hero photos for a single batch of listings via vision API.

    Args:
        viable: {listing_id: [(key, letter, b64_data), ...]} — listings with 2+ photos
        listings: full listing objects for address lookup
    """
    # Build vision API message
    content_blocks: list[dict[str, Any]] = []
    # Track key mapping: {listing_id: {letter: key}}
    letter_to_key: dict[str, dict[str, str]] = {}

    for lid, photos in viable.items():
        listing = next(l for l in listings if l.listing_id == lid)
        content_blocks.append({
            "type": "text",
            "text": f"Listing {lid} — {listing.address}:",
        })
        letter_to_key[lid] = {}
        for key, letter, b64 in photos:
            letter_to_key[lid][letter] = key
            content_blocks.append({
                "type": "image",
                "source": {"type": "base64", "media_type": "image/jpeg", "data": b64},
            })
            content_blocks.append({
                "type": "text",
                "text": f"  Photo {letter}",
            })

    listing_ids_str = ", ".join(viable.keys())
    content_blocks.append({
        "type": "text",
        "text": (
            f"\nFor each listing ({listing_ids_str}), pick the single best hero photo "
            "for a Telegram apartment card. Prefer wide-angle living room or main space shots. "
            "Avoid bathrooms, floor plans, building exteriors, and hallways.\n\n"
            'Return ONLY a JSON object mapping listing_id to the chosen photo letter, e.g.: '
            '{"4961650": "C", "4961651": "A"}'
        ),
    })

    client = anthropic.AsyncAnthropic(
        api_key=os.environ.get("ANTHROPIC_API_KEY", ""),
    )
    response = await call_with_retry(
        lambda: client.messages.create(
            model=VISION_MODEL,
            max_tokens=1024,
            system="You are a photo selector for apartment listings. Return ONLY valid JSON.",
            messages=[{"role": "user", "content": content_blocks}],
        ),
        label="claude.vision",
    )

    text = response.content[0].text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1]
        text = text.rsplit("```", 1)[0].strip()

    picks = json.loads(text)

    # Map letter picks back to full-size URLs
    hero_map: dict[str, str] = {}
    for lid, letter in picks.items():
        if lid not in letter_to_key:
            continue
        letter_upper = str(letter).upper()
        key = letter_to_key[lid].get(letter_upper)
        if key:
            hero_map[lid] = STREETEASY_PHOTO_URL.format(key=key)

    return hero_map
