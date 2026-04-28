"""StreetEasy apartment search via Apify actor (replaces Playwright browser layer)."""

from __future__ import annotations

import asyncio
import logging
import os
import re
import time
from dataclasses import dataclass
from typing import Any

from apify_client import ApifyClientAsync

from src.config import (
    NEIGHBORHOODS,
    STREETEASY_RENTALS,
    POLL_INTERVAL_SECS,
    ABORT_AFTER_SECS_NO_ITEMS,
    ENRICHMENT_NO_ITEMS_ABORT_SECS,
    ENRICHMENT_MAX_WAIT_SECS,
    ENRICHMENT_RETRY_BATCH_SIZE,
    ENRICHMENT_TARGET_COVERAGE,
    ENRICHMENT_MAX_URLS,
)
from src.models import Preferences
from src.storage import (
    load_actor_build_pin,
    load_canary_urls,
    save_actor_build_pin,
    save_canary_urls,
)

logger = logging.getLogger(__name__)

_REQUEST_COUNTS_RE = re.compile(
    r"Total\s+\d+\s+requests:\s*(\d+)\s+succeeded,\s*(\d+)\s+failed",
    re.IGNORECASE,
)

# Backward-compatible user-agent used by local diagnostic scripts.
BROWSER_UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/131.0.0.0 Safari/537.36"
)


@dataclass
class ActorRunResult:
    """Raw actor run output with metadata useful for diagnostics and gating."""

    items: list[dict[str, Any]]
    run_id: str
    dataset_id: str
    build_id: str
    status: str
    status_message: str
    requests_succeeded: int | None = None
    requests_failed: int | None = None


@dataclass
class AmenityRunSummary:
    """Summary for one enrichment actor run/batch."""

    run_id: str
    dataset_id: str
    build_id: str
    status: str
    status_message: str
    rows_returned: int
    mapped_count: int
    requests_succeeded: int | None = None
    requests_failed: int | None = None


@dataclass
class AmenityEnrichmentResult:
    """Typed result for amenity enrichment diagnostics and scanner gating."""

    data_by_listing_id: dict[str, dict[str, Any]]
    coverage: float
    target_count: int
    run_summaries: list[AmenityRunSummary]
    failed: bool
    failure_reason: str | None = None


# Display names for amenity enum values from detail page data
AMENITY_DISPLAY_NAMES: dict[str, str] = {
    "DOORMAN": "Doorman",
    "GYM": "Gym",
    "LAUNDRY": "Laundry in Building",
    "DISHWASHER": "Dishwasher",
    "WASHER_DRYER": "In-unit Laundry",
    "ELEVATOR": "Elevator",
    "ROOF_DECK": "Roof Deck",
    "CENTRAL_AC": "Central AC",
    "OUTDOOR_SPACE": "Outdoor Space",
    "PETS": "Pets Allowed",
    "CATS": "Cats Allowed",
    "DOGS": "Dogs Allowed",
    "STORAGE": "Storage",
    "BIKE_ROOM": "Bike Room",
    "PARKING": "Parking",
    "POOL": "Pool",
    "CONCIERGE": "Concierge",
    "BALCONY": "Balcony",
    "TERRACE": "Terrace",
    "HARDWOOD_FLOORS": "Hardwood Floors",
    "LIVE_IN_SUPER": "Live-in Super",
    "PREWAR": "Pre-war",
    "NEW_DEVELOPMENT": "New Development",
    "NO_FEE": "No Fee",
}


def _normalize_amenity(value: str) -> str:
    """Convert an amenity enum value to a human-readable display name."""
    return AMENITY_DISPLAY_NAMES.get(value, value.replace("_", " ").title())


def _parse_request_counts(status_message: str) -> tuple[int | None, int | None]:
    """Parse request success/failure counts from actor status message when present."""
    if not status_message:
        return (None, None)
    match = _REQUEST_COUNTS_RE.search(status_message)
    if not match:
        return (None, None)
    try:
        return (int(match.group(1)), int(match.group(2)))
    except ValueError:
        return (None, None)


def _row_looks_like_listing(item: dict[str, Any]) -> bool:
    """Return True if an actor row looks like a real listing (carries an id).

    Real rows always carry a listing id somewhere — either a search-shape
    id (``node.id`` / ``node_id`` / top-level ``id``) or a detail-shape id
    (``basicInfo.id`` / ``federatedData.rentalByListingId.id``).

    Placeholder rows the actor writes on a total WAF block (e.g.
    ``{"message": "No results found"}``) carry none of these.
    """
    nested = item.get("node") if isinstance(item.get("node"), dict) else None
    if nested and nested.get("id"):
        return True
    if item.get("node_id") or item.get("id"):
        return True
    basic_info = item.get("basicInfo") if isinstance(item.get("basicInfo"), dict) else None
    if basic_info and basic_info.get("id"):
        return True
    fed = item.get("federatedData") if isinstance(item.get("federatedData"), dict) else None
    if fed:
        rental = fed.get("rentalByListingId") if isinstance(fed.get("rentalByListingId"), dict) else None
        if rental and rental.get("id"):
            return True
    return False


def _is_unhealthy_empty_run(run_result: ActorRunResult) -> bool:
    """Detect SUCCEEDED runs that returned no useful data because all requests failed.

    Three WAF-block shapes are covered:
      1. Empty dataset + all requests failed (the classic signature).
      2. Non-empty dataset where the status message reports
         requests_succeeded=0, requests_failed>0 — the actor wrote error
         placeholder rows but still failed every request.
      3. Non-empty dataset where the status message *omits* request counts
         (we have seen ``requests_succeeded=None requests_failed=None`` from
         the actor on partial WAF days), but every row is a placeholder with
         no listing id. Without this branch we silently surface "no listings
         today" while the dataset is full of ``{"message": ...}`` rows.
    """
    if run_result.status != "SUCCEEDED":
        return False

    rs, rf = run_result.requests_succeeded, run_result.requests_failed
    if rs is not None and rf is not None and rs == 0 and rf > 0:
        return True

    if run_result.items and not any(_row_looks_like_listing(it) for it in run_result.items):
        return True

    return False


class ApifyScraperError(Exception):
    """Error from Apify actor run."""


class ApifyScraper:
    """Searches StreetEasy via the memo23/apify-streeteasy-cheerio Apify actor."""

    def __init__(self) -> None:
        token = os.environ["APIFY_API_TOKEN"]
        self._client = ApifyClientAsync(token=token)
        self._actor_id = "memo23/apify-streeteasy-cheerio"

    async def _latest_build_id(self) -> str:
        """Resolve actor's current latest build id (Apify internal id, e.g. 'R8oKY...')."""
        actor_data = await self._client.actor(self._actor_id).get()
        tagged = (actor_data or {}).get("taggedBuilds") or {}
        latest = tagged.get("latest") or {}
        return str(latest.get("buildId") or "")

    async def _latest_build_number(self) -> str:
        """Resolve actor's current latest build version (e.g. '0.0.95').

        Apify's `build=` parameter accepts a tag (e.g. 'latest') or a
        version number — NOT a build id. We pin on the version so the pin
        remains usable.
        """
        actor_data = await self._client.actor(self._actor_id).get()
        tagged = (actor_data or {}).get("taggedBuilds") or {}
        latest = tagged.get("latest") or {}
        return str(latest.get("buildNumber") or "")

    async def _build_number_for_id(self, build_id: str) -> str:
        """Look up an Apify build's version number from its internal id."""
        if not build_id:
            return ""
        try:
            data = await self._client.build(build_id).get()
        except Exception:
            return ""
        return str((data or {}).get("buildNumber") or "")

    @staticmethod
    def _looks_like_build_id(value: str) -> bool:
        """Heuristic: Apify build ids are ~17 chars of [A-Za-z0-9], no dots."""
        return bool(value) and "." not in value and len(value) >= 12

    @staticmethod
    def _effective_pin() -> str:
        """Resolve active pin from env override first, then persisted pin.

        Returns a value safe to pass as Apify `build=` (tag or version like
        '0.0.95'). If the persisted pin still contains a legacy build id
        (older deployments saved ids), drop it — Apify will reject it.
        """
        env_pin = os.environ.get("APIFY_ACTOR_BUILD_PIN", "").strip()
        if env_pin:
            return env_pin
        persisted = load_actor_build_pin().strip()
        if persisted and ApifyScraper._looks_like_build_id(persisted):
            logger.warning(
                "Discarding legacy build-id pin %r (Apify expects tag or version)",
                persisted,
            )
            return ""
        return persisted

    @staticmethod
    def _can_persist_pin() -> bool:
        """Persisted pin updates are disabled when env pin override is set."""
        return not bool(os.environ.get("APIFY_ACTOR_BUILD_PIN", "").strip())

    async def _run_actor(
        self,
        start_urls: list[dict[str, str]],
        max_items: int = 1000,
        abort_after_secs_no_items: int | None = ABORT_AFTER_SECS_NO_ITEMS,
        max_wait_secs: int | None = None,
        poll_context: str = "",
        build: str | None = None,
    ) -> ActorRunResult:
        """Run the actor with given startUrls and poll until completion."""
        run_input = {
            "startUrls": start_urls,
            "maxItems": max_items,
            "maxRequestRetries": 15,
            "proxy": {
                "useApifyProxy": True,
                "apifyProxyGroups": ["RESIDENTIAL"],
                "countryCode": "US",
            },
        }

        run_info = await self._client.actor(self._actor_id).start(
            run_input=run_input,
            memory_mbytes=512,
            build=build,
        )

        run_id = run_info["id"]
        dataset_id = run_info["defaultDatasetId"]
        run_client = self._client.run(run_id)
        dataset_client = self._client.dataset(dataset_id)
        elapsed = 0
        run_data: dict[str, Any] | None = None

        while True:
            await asyncio.sleep(POLL_INTERVAL_SECS)
            elapsed += POLL_INTERVAL_SECS

            run_data = await run_client.get()
            status = (run_data or {}).get("status", "")

            if status in ("SUCCEEDED", "FAILED", "ABORTED", "TIMED-OUT"):
                break

            ds_info = await dataset_client.get()
            item_count = (ds_info or {}).get("itemCount", 0)
            logger.info(
                "Apify poll%s: %ds elapsed, status=%s, items=%d",
                f" [{poll_context}]" if poll_context else "",
                elapsed,
                status,
                item_count,
            )

            if max_wait_secs is not None and elapsed >= max_wait_secs:
                logger.warning(
                    "Aborting Apify run after %ds total runtime%s",
                    elapsed,
                    f" [{poll_context}]" if poll_context else "",
                )
                await run_client.abort()
                raise ApifyScraperError(
                    f"Actor run aborted after reaching max wait ({elapsed}s)"
                )

            if (
                abort_after_secs_no_items is not None
                and elapsed >= abort_after_secs_no_items
                and item_count == 0
            ):
                logger.warning(
                    "Aborting Apify run after %ds with 0 items (likely WAF block)%s",
                    elapsed,
                    f" [{poll_context}]" if poll_context else "",
                )
                await run_client.abort()
                raise ApifyScraperError(
                    f"Actor run aborted after {elapsed}s with 0 items"
                )

        status = str((run_data or {}).get("status", ""))
        if status != "SUCCEEDED":
            raise ApifyScraperError(f"Actor run failed with status: {status}")

        result = await dataset_client.list_items()
        items = list(result.items)
        status_message = str((run_data or {}).get("statusMessage", "") or "")
        requests_succeeded, requests_failed = _parse_request_counts(status_message)

        return ActorRunResult(
            items=items,
            run_id=run_id,
            dataset_id=dataset_id,
            build_id=str((run_data or {}).get("buildId") or ""),
            status=status,
            status_message=status_message,
            requests_succeeded=requests_succeeded,
            requests_failed=requests_failed,
        )

    @staticmethod
    def _log_run_summary(context: str, run_result: ActorRunResult) -> None:
        logger.info(
            "Apify %s run summary: build_id=%s run_id=%s dataset_id=%s status=%s "
            "requests_succeeded=%s requests_failed=%s rows=%d status_message=%s",
            context,
            run_result.build_id or "<unknown>",
            run_result.run_id,
            run_result.dataset_id,
            run_result.status,
            run_result.requests_succeeded,
            run_result.requests_failed,
            len(run_result.items),
            run_result.status_message,
        )

    async def _run_actor_with_build_policy(
        self,
        *,
        start_urls: list[dict[str, str]],
        max_items: int,
        abort_after_secs_no_items: int | None,
        max_wait_secs: int | None,
        poll_context: str,
        fail_on_unhealthy_empty: bool,
        force_build: str | None,
        allow_build_fallback: bool,
        auto_promote_on_fallback: bool,
    ) -> ActorRunResult:
        """Run actor with pinned build + latest fallback and optional auto-promotion."""
        pin_before = self._effective_pin()

        if force_build:
            build_candidates = [force_build]
        else:
            build_candidates: list[str | None] = []
            if pin_before:
                build_candidates.append(pin_before)
                if allow_build_fallback and pin_before != "latest":
                    build_candidates.append("latest")
            else:
                build_candidates.append("latest")

        # preserve candidate order while removing duplicates
        seen: set[str | None] = set()
        ordered_candidates: list[str | None] = []
        for candidate in build_candidates:
            if candidate in seen:
                continue
            seen.add(candidate)
            ordered_candidates.append(candidate)

        last_error: Exception | None = None
        for idx, build_candidate in enumerate(ordered_candidates):
            try:
                run_result = await self._run_actor(
                    start_urls=start_urls,
                    max_items=max_items,
                    abort_after_secs_no_items=abort_after_secs_no_items,
                    max_wait_secs=max_wait_secs,
                    poll_context=poll_context,
                    build=build_candidate,
                )
            except Exception as e:
                last_error = e
                logger.warning(
                    "Apify run failed for build=%s context=%s: %s",
                    build_candidate or "<default>",
                    poll_context,
                    e,
                )
                continue

            self._log_run_summary(poll_context or "run", run_result)

            if fail_on_unhealthy_empty and _is_unhealthy_empty_run(run_result):
                last_error = ApifyScraperError(
                    "Actor run succeeded with empty dataset while all requests failed"
                )
                logger.warning(
                    "Treating run as failure (unhealthy empty): build=%s run_id=%s "
                    "status_message=%s",
                    run_result.build_id or build_candidate or "<unknown>",
                    run_result.run_id,
                    run_result.status_message,
                )
                continue

            fallback_used = idx > 0
            if (
                fallback_used
                and auto_promote_on_fallback
                and self._can_persist_pin()
                and not force_build
            ):
                # Save the build VERSION number (e.g. '0.0.95'), not the
                # internal build id — Apify only accepts tag/version in build=.
                promoted_build = await self._build_number_for_id(run_result.build_id)
                if not promoted_build:
                    promoted_build = await self._latest_build_number()
                promotion_applied = bool(
                    promoted_build and promoted_build != pin_before
                )
                if promotion_applied:
                    save_actor_build_pin(promoted_build)
                    logger.warning(
                        "Actor build promotion: pin_before=%s tested_latest=%s promotion_applied=True",
                        pin_before or "<none>",
                        build_candidate,
                    )
                logger.info(
                    "Actor build fallback result: pin_before=%s tested_latest=%s promotion_applied=%s",
                    pin_before or "<none>",
                    build_candidate or "<default>",
                    promotion_applied,
                )

            return run_result

        raise last_error or ApifyScraperError("Actor run failed for all build candidates")

    async def search_streeteasy(self, prefs: Preferences) -> list[dict[str, Any]]:
        """Call actor with a pipe-style StreetEasy URL, return listing dicts."""
        url = _build_streeteasy_url(prefs)
        logger.info("Searching StreetEasy via Apify (pipe URL): %s", url)

        run_result = await self._run_actor_with_build_policy(
            start_urls=[{"url": url}],
            max_items=1000,
            abort_after_secs_no_items=ABORT_AFTER_SECS_NO_ITEMS,
            max_wait_secs=None,
            poll_context="search-pipe",
            fail_on_unhealthy_empty=True,
            force_build=None,
            allow_build_fallback=True,
            auto_promote_on_fallback=True,
        )

        listings = [_map_apify_item(item) for item in run_result.items]
        listings = [l for l in listings if l is not None]
        logger.info("Fetched %d listings from Apify (pipe URL)", len(listings))
        return listings

    async def search_by_neighborhoods(self, prefs: Preferences, max_items: int = 2000) -> list[dict[str, Any]]:
        """Search using path-style URLs for each neighborhood in a single actor run."""
        start_urls = _build_path_urls(prefs)
        if not start_urls:
            raise ApifyScraperError("No neighborhood URLs to search")

        logger.info(
            "Searching StreetEasy via Apify (path URLs): %d neighborhoods",
            len(start_urls),
        )

        run_result = await self._run_actor_with_build_policy(
            start_urls=start_urls,
            max_items=max_items,
            abort_after_secs_no_items=ABORT_AFTER_SECS_NO_ITEMS,
            max_wait_secs=None,
            poll_context="search-path",
            fail_on_unhealthy_empty=True,
            force_build=None,
            allow_build_fallback=True,
            auto_promote_on_fallback=True,
        )

        listings = [_map_apify_item(item) for item in run_result.items]
        listings = [l for l in listings if l is not None]
        logger.info("Fetched %d listings from Apify (path URLs)", len(listings))
        return listings

    async def enrich_with_amenities(
        self,
        listing_urls: list[str],
        url_to_id: dict[str, str],
        *,
        max_total_wait_secs: int | None = None,
        required_coverage: float = ENRICHMENT_TARGET_COVERAGE,
        no_items_abort_secs: int | None = ENRICHMENT_NO_ITEMS_ABORT_SECS,
        force_build: str | None = None,
        allow_build_fallback: bool = True,
        auto_promote_on_fallback: bool = True,
    ) -> AmenityEnrichmentResult:
        """Run actor on detail URLs and return typed enrichment diagnostics."""
        if not listing_urls or not url_to_id:
            return AmenityEnrichmentResult(
                data_by_listing_id={},
                coverage=1.0,
                target_count=0,
                run_summaries=[],
                failed=False,
                failure_reason=None,
            )

        unique_urls: list[str] = []
        seen_urls: set[str] = set()
        for url in listing_urls:
            if not url or url in seen_urls:
                continue
            seen_urls.add(url)
            unique_urls.append(url)

        if len(unique_urls) > ENRICHMENT_MAX_URLS:
            logger.info(
                "Capping amenity enrichment candidates from %d to %d",
                len(unique_urls),
                ENRICHMENT_MAX_URLS,
            )
            unique_urls = unique_urls[:ENRICHMENT_MAX_URLS]

        target_listing_ids = {url_to_id[url] for url in unique_urls if url in url_to_id}
        target_count = len(target_listing_ids)
        if not target_listing_ids:
            return AmenityEnrichmentResult(
                data_by_listing_id={},
                coverage=1.0,
                target_count=0,
                run_summaries=[],
                failed=False,
                failure_reason=None,
            )

        logger.info("Enriching %d listings with amenity data via detail pages", target_count)

        result: dict[str, dict[str, Any]] = {}
        run_summaries: list[AmenityRunSummary] = []
        errors: list[str] = []
        started_at = time.monotonic()

        def _remaining_budget_secs() -> int | None:
            if max_total_wait_secs is None:
                return ENRICHMENT_MAX_WAIT_SECS
            remaining = int(max_total_wait_secs - (time.monotonic() - started_at))
            return max(0, remaining)

        async def _run_and_map(batch_urls: list[str], label: str) -> None:
            remaining = _remaining_budget_secs()
            if remaining is not None and remaining <= 0:
                raise ApifyScraperError("Amenity enrichment budget exhausted")

            run_result = await self._run_actor_with_build_policy(
                start_urls=[{"url": url} for url in batch_urls],
                max_items=len(batch_urls),
                abort_after_secs_no_items=no_items_abort_secs,
                max_wait_secs=remaining,
                poll_context=label,
                fail_on_unhealthy_empty=True,
                force_build=force_build,
                allow_build_fallback=allow_build_fallback,
                auto_promote_on_fallback=auto_promote_on_fallback,
            )

            mapped = 0
            for item in run_result.items:
                source_url, source_listing_id, amenity_data = _map_detail_amenities(item)
                listing_id = url_to_id.get(source_url, "")
                if not listing_id and source_listing_id in target_listing_ids:
                    listing_id = source_listing_id
                if listing_id:
                    result[listing_id] = amenity_data
                    mapped += 1

            run_summaries.append(
                AmenityRunSummary(
                    run_id=run_result.run_id,
                    dataset_id=run_result.dataset_id,
                    build_id=run_result.build_id,
                    status=run_result.status,
                    status_message=run_result.status_message,
                    rows_returned=len(run_result.items),
                    mapped_count=mapped,
                    requests_succeeded=run_result.requests_succeeded,
                    requests_failed=run_result.requests_failed,
                )
            )

            logger.info(
                "Amenity enrichment batch %s: build_id=%s run_id=%s rows=%d mapped=%d "
                "requests_succeeded=%s requests_failed=%s coverage=%.1f%%",
                label,
                run_result.build_id or "<unknown>",
                run_result.run_id,
                len(run_result.items),
                mapped,
                run_result.requests_succeeded,
                run_result.requests_failed,
                (len(result) / target_count) * 100,
            )

        try:
            await _run_and_map(unique_urls, "bulk")
        except ApifyScraperError as e:
            errors.append(str(e))
            logger.warning("Amenity bulk enrichment failed: %s", e)

        if (len(result) / target_count) < required_coverage:
            remaining_urls = [
                url
                for url in unique_urls
                if (url_to_id.get(url) and url_to_id[url] not in result)
            ]
            batches = [
                remaining_urls[i:i + ENRICHMENT_RETRY_BATCH_SIZE]
                for i in range(0, len(remaining_urls), ENRICHMENT_RETRY_BATCH_SIZE)
            ]
            for i, batch in enumerate(batches, 1):
                if (len(result) / target_count) >= required_coverage:
                    break
                try:
                    await _run_and_map(batch, f"retry-{i}/{len(batches)}")
                except ApifyScraperError as e:
                    errors.append(str(e))
                    logger.warning("Amenity retry batch %d failed: %s", i, e)

        # Opportunistically store successful detail URLs for daily canary checks.
        successful_urls = [
            url
            for url in unique_urls
            if url_to_id.get(url) and url_to_id[url] in result
        ]
        if successful_urls:
            try:
                existing = load_canary_urls()
                save_canary_urls(successful_urls + existing)
            except Exception:
                # Canary URL persistence should never fail the enrichment path.
                logger.exception("Failed to persist canary URL sample")

        coverage = len(result) / target_count
        failure_reason = "; ".join(errors[:3]) if errors else None
        failed = bool(errors) and not result

        logger.info(
            "Amenity enrichment final: enriched=%d/%d coverage=%.1f%% failed=%s reason=%s",
            len(result),
            target_count,
            coverage * 100,
            failed,
            failure_reason or "",
        )

        return AmenityEnrichmentResult(
            data_by_listing_id=result,
            coverage=coverage,
            target_count=target_count,
            run_summaries=run_summaries,
            failed=failed,
            failure_reason=failure_reason,
        )

    async def search_with_retry(
        self,
        prefs: Preferences,
        max_retries: int = 2,
        retry_delays: tuple[int, ...] = (30, 60),
    ) -> list[dict[str, Any]]:
        """Search with path URLs (primary) falling back to pipe URLs, with retries."""
        last_error: Exception | None = None

        for attempt in range(1 + max_retries):
            try:
                logger.info("search_with_retry: attempt %d/%d", attempt + 1, 1 + max_retries)

                # Try path-based search first when neighborhoods are specified
                if prefs.neighborhoods:
                    try:
                        results = await self.search_by_neighborhoods(prefs)
                        return results
                    except Exception as e:
                        logger.warning(
                            "Path-based search failed, falling back to pipe URL: %s", e
                        )

                # Pipe URL fallback (or primary when no neighborhoods)
                results = await self.search_streeteasy(prefs)
                return results
            except Exception as e:
                logger.warning("search_with_retry: attempt %d failed: %s", attempt + 1, e)
                last_error = e

            if attempt < max_retries:
                delay = retry_delays[attempt] if attempt < len(retry_delays) else retry_delays[-1]
                logger.info("search_with_retry: waiting %ds before retry", delay)
                await asyncio.sleep(delay)

        raise last_error or ApifyScraperError("All retries exhausted")


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


def _build_path_urls(prefs: Preferences) -> list[dict[str, str]]:
    """Build path-style startUrls for per-neighborhood search.

    Path URLs (/for-rent/west-village) produce in-area results,
    unlike pipe URLs which the actor ignores for area filtering.
    """
    urls = []
    for name in prefs.neighborhoods:
        slug = NEIGHBORHOODS.get(name)
        if not slug:
            slug = name.lower().replace(" ", "-")
        urls.append({"url": f"{STREETEASY_RENTALS}/{slug}"})
    return urls


def _map_detail_amenities(item: dict[str, Any]) -> tuple[str, str, dict[str, Any]]:
    """Extract amenity data from a detail page actor result.

    Returns: (source_url, source_listing_id, {"building_amenities": [...], "unit_features": [...], "description": "..."})
    """
    source_url = str(item.get("url") or item.get("originalUrl") or "")

    fed = item.get("federatedData") or {}
    rental = fed.get("rentalByListingId") or {}
    basic_info = item.get("basicInfo") or {}
    source_listing_id = str(basic_info.get("id") or rental.get("id") or "")
    props = rental.get("propertyDetails") or {}

    raw_amenities = props.get("amenities") or {}
    raw_features = props.get("features") or {}

    amenity_list = raw_amenities.get("list") or []
    feature_list = raw_features.get("list") or []

    building_amenities = [_normalize_amenity(a) for a in amenity_list if isinstance(a, str)]
    unit_features = [_normalize_amenity(f) for f in feature_list if isinstance(f, str)]

    description = rental.get("description", "")

    return source_url, source_listing_id, {
        "building_amenities": building_amenities,
        "unit_features": unit_features,
        "description": description,
    }


STREETEASY_PHOTO_URL = "https://photos.zillowstatic.com/fp/{key}-se_large_800_400.jpg"
STREETEASY_PHOTO_URL_THUMB = "https://photos.zillowstatic.com/fp/{key}-se_medium_500_250.jpg"


def _map_apify_item(item: dict[str, Any]) -> dict[str, Any] | None:
    """Map an Apify actor output item to the raw listing dict format expected by scanner.py.

    The memo23 actor returns GraphQL-shaped items. Two schemas have been seen:

    Legacy (nested):
      { "node": { "id", "areaName", "price", "bedroomCount", "street", "urlPath", ... } }

    Current (flattened, as of build 0.0.95):
      { "node___typename": "...", "node_id", "node_areaName", "node_price",
        "node_bedroomCount", "node_street", "node_urlPath",
        "node_photos_json": "[{\"key\":\"...\"}, ...]", ... }

    We support both. _g(key) reads either nested-style or flattened-style.
    """
    nested = item.get("node") if isinstance(item.get("node"), dict) else None

    def _g(field: str, default: Any = None) -> Any:
        if nested is not None and field in nested:
            return nested.get(field, default)
        flat_key = "node_" + field
        if flat_key in item:
            return item.get(flat_key, default)
        if field in item:
            return item.get(field, default)
        return default

    listing_id = str(_g("id", "") or "")
    if not listing_id:
        return None

    url_path = _g("urlPath", "") or ""
    listing_url = f"https://streeteasy.com{url_path}" if url_path else ""

    street = _g("street", "") or ""
    unit = _g("unit", "") or ""
    address = f"{street} #{unit}".strip(" #") if street else f"Listing {listing_id}"

    # Photos: legacy returns a list under node.photos; flattened serializes to
    # node_photos_json (a JSON string).
    photo_dicts: list[dict[str, Any]] = []
    raw_photos = _g("photos")
    if isinstance(raw_photos, list):
        photo_dicts = [p for p in raw_photos if isinstance(p, dict)]
    else:
        photos_json = _g("photos_json")
        if isinstance(photos_json, str) and photos_json:
            try:
                import json as _json
                parsed = _json.loads(photos_json)
                if isinstance(parsed, list):
                    photo_dicts = [p for p in parsed if isinstance(p, dict)]
            except Exception:
                photo_dicts = []
    # Fall back to the lead media photo key when photos array is missing.
    if not photo_dicts:
        lead_key = _g("leadMedia_photo_key") or _g("leadMediaPhotoKey")
        if isinstance(lead_key, str) and lead_key:
            photo_dicts = [{"key": lead_key}]

    photos: list[str] = []
    photo_keys: list[str] = []
    for p in photo_dicts:
        key = p.get("key")
        if isinstance(key, str) and key:
            photos.append(STREETEASY_PHOTO_URL.format(key=key))
            photo_keys.append(key)

    full_bath = int(_g("fullBathroomCount", 0) or 0)
    half_bath = int(_g("halfBathroomCount", 0) or 0)
    bathrooms = full_bath + 0.5 * half_bath if (full_bath or half_bath) else 1.0

    living_area = _g("livingAreaSize", 0) or 0
    sqft = int(living_area) if living_area else None

    no_fee = bool(_g("noFee", False))
    net_effective = _g("netEffectivePrice")
    months_free = _g("monthsFree")

    return {
        "listing_id": listing_id,
        "url": listing_url,
        "address": address,
        "neighborhood": _g("areaName", "") or "",
        "price": int(_g("price", 0) or 0),
        "bedrooms": int(_g("bedroomCount", 0) or 0),
        "bathrooms": bathrooms,
        "sqft": sqft,
        "photos": photos,
        "photo_keys": photo_keys,
        "available_date": _g("availableAt"),
        "broker_fee": None if no_fee else "Broker fee",
        "net_effective_price": int(net_effective) if net_effective else None,
        "months_free": float(months_free) if months_free else None,
    }
