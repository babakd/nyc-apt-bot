"""Daily apartment scanning: search StreetEasy, deduplicate, LLM-score, notify."""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import anthropic
import httpx
from apify_client import ApifyClientAsync

from src.formatter import format_listing_card, format_scan_header, listing_keyboard
from src.models import MAX_RECENT_LISTINGS, ChatState, CurrentApartment, Listing, Preferences
from src.apify_scraper import ApifyScraper, ApifyScraperError, STREETEASY_PHOTO_URL, STREETEASY_PHOTO_URL_THUMB
from src.storage import load_all_states, save_state

CACHE_MAX_AGE_HOURS = 48

logger = logging.getLogger(__name__)

SCORING_MODEL = "claude-opus-4-6"
SCORE_FLOOR = 25


@dataclass
class ScoringResult:
    """Result from LLM scoring, with fallback flag."""
    listings: list[Listing]
    is_fallback: bool = False

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

    # Search StreetEasy via Apify actor (with retry)
    try:
        raw_listings = await scraper.search_with_retry(prefs)
    except Exception:
        logger.exception("StreetEasy search failed")
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
                return
        await telegram_bot.send_text(
            state.chat_id,
            "\u26a0\ufe0f I had trouble searching StreetEasy today. I'll try again tomorrow.",
        )
        return

    if not raw_listings:
        await telegram_bot.send_text(state.chat_id, format_scan_header(0))
        return

    # Deduplicate (check only — marking as seen happens after successful send)
    new_listings = []
    for raw in raw_listings:
        lid = str(raw.get("listing_id", ""))
        if lid and lid not in state.seen_listing_ids:
            new_listings.append(raw)

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

    # Enrich with amenities/description from listing pages
    filtered = await _enrich_listings(filtered)

    # Layer 2: LLM filter + score
    scoring_result = await _llm_score_listings(filtered, prefs, state.current_apartment)
    scored = scoring_result.listings

    # Sort by score descending
    scored.sort(key=lambda l: l.match_score or 0, reverse=True)

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
        await telegram_bot.send_text(state.chat_id, format_scan_header(0))
    else:
        await telegram_bot.send_text(state.chat_id, format_scan_header(len(scored)))

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


def _extract_listing_detail(html: str) -> tuple[list[str], str | None]:
    """Extract amenities and description from a StreetEasy listing page HTML.

    Tries __NEXT_DATA__ JSON first, then falls back to JSON-LD structured data.
    Returns (amenities, description).
    """
    amenities: list[str] = []
    description: str | None = None

    # Try __NEXT_DATA__ (Next.js data)
    next_data_match = re.search(
        r'<script\s+id="__NEXT_DATA__"\s+type="application/json">(.*?)</script>',
        html,
        re.DOTALL,
    )
    if next_data_match:
        try:
            data = json.loads(next_data_match.group(1))
            # Navigate to listing props — structure: props.pageProps.listingData or similar
            props = data.get("props", {}).get("pageProps", {})
            # Try common paths for listing data
            listing_data = props.get("listingData") or props.get("listing") or {}
            if isinstance(listing_data, dict):
                # Amenities
                raw_amenities = listing_data.get("amenities") or listing_data.get("amenityList") or []
                if isinstance(raw_amenities, list):
                    for item in raw_amenities:
                        if isinstance(item, str):
                            amenities.append(item)
                        elif isinstance(item, dict):
                            name = item.get("name") or item.get("label") or ""
                            if name:
                                amenities.append(name)
                # Description
                desc = listing_data.get("description") or listing_data.get("listingDescription") or ""
                if desc and isinstance(desc, str):
                    description = desc.strip()
        except (json.JSONDecodeError, AttributeError, TypeError):
            pass

    # Fallback: JSON-LD structured data
    if not amenities and not description:
        ld_match = re.search(
            r'<script\s+type="application/ld\+json">(.*?)</script>',
            html,
            re.DOTALL,
        )
        if ld_match:
            try:
                ld_data = json.loads(ld_match.group(1))
                if isinstance(ld_data, list):
                    ld_data = ld_data[0] if ld_data else {}
                if isinstance(ld_data, dict):
                    # Schema.org Apartment/Residence
                    raw_amenities = ld_data.get("amenityFeature") or []
                    if isinstance(raw_amenities, list):
                        for item in raw_amenities:
                            if isinstance(item, str):
                                amenities.append(item)
                            elif isinstance(item, dict):
                                name = item.get("name") or item.get("value") or ""
                                if name:
                                    amenities.append(name)
                    desc = ld_data.get("description") or ""
                    if desc and isinstance(desc, str):
                        description = desc.strip()
            except (json.JSONDecodeError, AttributeError, TypeError):
                pass

    return amenities, description


async def _enrich_listings(listings: list[Listing]) -> list[Listing]:
    """Enrich listings with amenities and descriptions from individual listing pages.

    Uses Apify's cheerio-scraper actor to fetch listing pages through residential proxies
    (direct HTTP to streeteasy.com returns 403 from Imperva).

    Gracefully degrades: if enrichment fails for individual listings or entirely,
    the pipeline continues with unenriched data.
    """
    if not listings:
        return listings

    token = os.environ.get("APIFY_API_TOKEN", "")
    if not token:
        logger.warning("No APIFY_API_TOKEN, skipping enrichment")
        return listings

    urls = []
    url_to_listing: dict[str, Listing] = {}
    for listing in listings:
        if listing.url:
            urls.append(listing.url)
            url_to_listing[listing.url] = listing

    if not urls:
        return listings

    logger.info("Enriching %d listings via Apify cheerio-scraper", len(urls))

    try:
        client = ApifyClientAsync(token=token)
        run_input = {
            "startUrls": [{"url": u} for u in urls],
            "maxConcurrency": 3,
            "maxRequestRetries": 2,
            "pageFunction": """async function pageFunction(context) {
                return {
                    url: context.request.url,
                    html: context.body,
                };
            }""",
            "proxy": {
                "useApifyProxy": True,
                "apifyProxyGroups": ["RESIDENTIAL"],
                "countryCode": "US",
            },
        }

        run = await client.actor("apify/cheerio-scraper").call(
            run_input=run_input,
            timeout_secs=120,
            memory_mbytes=512,
        )

        if not run or run.get("status") != "SUCCEEDED":
            logger.warning("Enrichment actor run failed: %s", run)
            return listings

        dataset = client.dataset(run["defaultDatasetId"])
        result = await dataset.list_items()

        enriched_count = 0
        for item in result.items:
            page_url = item.get("url", "")
            html = item.get("html", "")
            if not html or page_url not in url_to_listing:
                continue

            listing = url_to_listing[page_url]
            try:
                page_amenities, page_description = _extract_listing_detail(html)
                if page_amenities and not listing.amenities:
                    listing.amenities = page_amenities
                if page_description and not listing.description:
                    listing.description = page_description
                if page_amenities or page_description:
                    enriched_count += 1
            except Exception:
                logger.debug("Failed to parse enrichment for %s", listing.listing_id)

        logger.info("Enriched %d/%d listings with amenities/description", enriched_count, len(listings))

    except Exception:
        logger.exception("Listing enrichment failed, continuing with unenriched data")

    return listings


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
        "When 'hood_canonical' is provided, it means this listing's neighborhood is a "
        "sub-area of the user's preferred neighborhood. Treat it as matching.\n"
        "Use the amenities list and description to evaluate must-haves and nice-to-haves. "
        "If amenities data is missing for a listing, do NOT penalize it — assume unknown.\n"
        "When a listing has a net_effective price (from free months), use that for budget "
        "comparison, not the gross price.\n"
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
            temperature=0,
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
            passes = llm_include and (listing.match_score or 0) >= SCORE_FLOOR
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
            listings.sort(key=lambda l: l.match_score or 0, reverse=True)
            included = listings[:3]
            logger.info("All listings excluded by LLM filter, returning top 3 as fallback")
            return ScoringResult(listings=included, is_fallback=True)

        return ScoringResult(listings=included)

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
    sem = asyncio.Semaphore(20)
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
    MAX_LISTINGS_PER_BATCH = 12
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
    response = await client.messages.create(
        model=SCORING_MODEL,
        max_tokens=1024,
        system="You are a photo selector for apartment listings. Return ONLY valid JSON.",
        messages=[{"role": "user", "content": content_blocks}],
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
