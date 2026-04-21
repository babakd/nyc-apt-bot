# StreetEasy Bot

NYC apartment hunting bot that monitors StreetEasy listings via Apify scraper, sends daily notifications via Telegram, and facilitates agent outreach. All user interaction is natural language — no commands, FSMs, or rigid parsing.

## Stack

- **Runtime**: Modal (serverless — webhooks, cron, on-demand functions)
- **StreetEasy scraping**: `memo23/apify-streeteasy-cheerio` rented actor via Apify — path-style URLs per neighborhood (primary), pipe-style URL (fallback). Two-pass: search + detail-page amenity enrichment with strict send gating.
- **Conversation / scoring**: Claude API (Opus 4.7) with `tool_use`. Hero-photo vision picker uses Haiku 4.5. (Bumped 2026-04-18 in commit `1235a1a`; Opus 4.7 deprecated the `temperature` parameter, so the scoring call no longer sends it.)
- **Messaging**: Telegram Bot API
- **Listing detail pages**: Telegraph (Instant View)
- **Models**: Pydantic v2
- **HTTP client**: httpx (async)
- **Language**: Python 3.11, fully async

## Production Update (2026-02-22)

Recent production fixes that agents must preserve:
- Webhook handling is split into two stages:
  - ingress (`TelegramWebhook.webhook`) claims `update_id` in a shared `modal.Dict` and enqueues worker processing;
  - worker (`process_telegram_update`) does conversation/scanning.
- Dedup is now multi-layer:
  - distributed update claims in `modal.Dict` (primary);
  - persistent marker files in `/data/telegram_updates/*.seen` (secondary);
  - in-memory guard in `TelegramBot` (tertiary).
- Manual scan execution is single-flight per chat via a distributed lease (`scan_locks` dict). Concurrent triggers for the same chat are suppressed.
- Search health hardening:
  - `SUCCEEDED + 0 rows + 0 succeeded + failed>0` is treated as failure (not true "no listings");
  - failure streak/cooldown is tracked in `ChatState` (`search_failure_streak`, `search_cooldown_until`).
- Actor build reliability:
  - runtime build policy uses pinned build with latest fallback and auto-promotion;
  - daily canary (`actor_build_canary`) tests latest build, records status, and promotes pin when healthy.

## Production Update (2026-04-18)

Commit `1235a1a` landed fixes for a cluster of live-traffic problems:
- **Actor output schema flattened** (build 0.0.96+): rows now have `node_id`, `node_areaName`, `node_photos_json` (JSON-encoded string) instead of the nested `{node: {id, areaName, photos: [...]}}` shape. `_map_apify_item` reads both shapes via a `_g(field)` helper and `node_photos_json` parser; keep both paths working when touching it.
- **Actor build pinning is now by version number** (e.g. `"0.0.95"`), not the internal build id. Apify's `build=` parameter only accepts tags or version numbers — legacy build-id pins are auto-discarded on load with a warning (`_looks_like_build_id`). `_build_number_for_id(build_id)` translates at promotion time.
- **Scoring switched to `claude-opus-4-7`**: the deprecated `temperature` parameter is no longer sent. Do NOT reintroduce it — the API rejects the call.
- **Hero-photo vision picker switched to `claude-haiku-4-5`** (cheaper for image classification).

## Production Update (2026-04-20)

Commit `dcade7b` + `f41cc45` on PR #3 (`claude/e2e-cleanup-20260420`) relaxed the strict amenity gate and fixed a latent crash:
- **Amenity gate softened** (was: hard-block all sends when coverage < 0.95). New behavior:
  - If `amenity_failure_reason and enriched_count == 0` → hard-block (zero signal).
  - If user has `must_haves` AND `coverage < AMENITY_MIN_COVERAGE_WITH_MUST_HAVES` (0.20) → hard-block.
  - Otherwise → proceed. A user-visible "Heads up: StreetEasy blocked some amenity lookups..." warning is sent when coverage < 0.95 but the gate passed, and the LLM scorer calibrates confidence via `amenity_signal_status`.
  - Motivation: a partial WAF block on detail pages used to silently suppress all listings; now the user gets what we *can* verify plus a clear warning.
- **Enrichment budget tightened** so it can't eat the whole daily scan:
  - `AMENITY_ENRICHMENT_MAX_WAIT_SECS`: 600 → 300
  - `AMENITY_ENRICHMENT_NO_ITEMS_ABORT_SECS`: 120 → 90
- **Daily scan Modal timeout**: 600s → 1500s (`modal_app.daily_scan` decorator).
- **Context-aware rate-limit fallback**: search-exhausted messages differ for `is_daily=True` ("I'll try again tomorrow") vs `is_daily=False` ("Please try again in a few minutes").
- **Latent NameError fix**: `_run_actor_with_build_policy` referenced `promotion_applied` in a log line after the 1235a1a refactor without defining it. Any successful build fallback on a system configured to auto-promote would crash there — the daily canary's whole purpose is to exercise that path. Now defined from the promotion condition.
- **Tests repaired**: `test_pinned_build_failure_falls_back_and_promotes`, `test_apify_api_error_falls_back_to_latest`, `test_temperature_zero` — first two now mock `_build_number_for_id`/`_latest_build_number`; third was renamed to `test_temperature_not_set_for_opus_4_7` and flipped to assert the parameter is absent.

## External Dependencies / Known Issues

- **StreetEasy WAF is PerimeterX** (`PXcZdhF737`, `px-cloud.net`) on the internal GraphQL endpoint `api-internal.streeteasy.com/graphql`. Not Imperva. Blocks show as HTTP 403 with a JSON body containing `appId`, `blockScript`, and `classification: px=true html=false` in recent actor logs.
- **Actor maintainer (memo23) is actively iterating** on WAF bypass. Bug filed: https://console.apify.com/actors/78UWNeqywwtKfp5z6/issues/oPEhSorpEYz4q4P8n (public view: https://apify.com/memo23/apify-streeteasy-cheerio/issues/oPEhSorpEYz4q4P8n). Builds 0.0.96→0.0.102 were bypass attempts; blocking is partial/intermittent as of 2026-04-20. Agents investigating a "no listings" day should check whether it's this (external, retry fallback path exercised) before blaming code.
- **Secondary actor bugs filed in the same issue**: SUCCEEDED status on runs where `requests_succeeded=0`; and the actor writes `{"message": "No results found"}` placeholder rows into the main dataset when all requests fail. `_is_unhealthy_empty_run` accounts for both.

## Architecture

```
Telegram webhook
  → modal_app.TelegramWebhook.webhook (auth + update_id claim + enqueue)
  → modal_app.process_telegram_update (async worker)
    → telegram_handler.py → conversation.py (Claude API + tools)
      ↓ tool calls
      update_preferences       → models.py (ChatState.preferences)
      search_apartments        → scanner.py → apify_scraper.py
      show_preferences         → formatter.py
      mark_ready               → models.py (preferences_ready=True)
      clear_search_history     → models.py (seen_listing_ids.clear())
      pause_daily_scans        → models.py (preferences_ready=False)
      get_liked_listings       → models.py (liked_listings)
      remove_liked_listing     → models.py
      reset_preferences        → models.py
      remove_neighborhoods     → models.py
      get_listing_details      → models.py (recent/liked listings)
      compare_listings         → models.py (recent/liked listings)
      draft_outreach           → outreach.py
      update_current_apartment → models.py (current_apartment)
      show_current_apartment   → models.py (current_apartment)
    ↓
    Claude text response → Telegram
```

The conversation engine (`src/conversation.py`) sends every user message to Claude with:
- System prompt containing bot personality + current preferences + constraint_context + current apartment context
- 16 tool definitions (see list above)
- Full conversation history (last 30 turns)

Claude extracts preferences from natural language and calls tools as needed. No regex, no FSM, no slash commands.

## File Map

| File | Purpose |
|---|---|
| `modal_app.py` | Modal app wiring: webhook ingress (claim + enqueue), async update worker, daily scan cron, actor canary cron, outreach sender |
| `src/models.py` | Pydantic models: `ChatState`, `Preferences`, `Listing`, `Draft`, `ConversationTurn`, `CurrentApartment` |
| `src/conversation.py` | LLM conversation engine — `ConversationEngine.handle_message()` calls Claude API with 16 tools |
| `src/claude_client.py` | Thin async wrapper around Anthropic SDK — handles tool_use response loop. Default model: `claude-opus-4-7` |
| `src/telegram_handler.py` | Telegram Bot API: message/callback handling, photo fallback chain, optional worker-mode dedup bypass, scan callback handoff with `update_id` |
| `src/formatter.py` | Telegram HTML formatting: listing cards (clean middot style), preferences summary, keyboards |
| `src/scanner.py` | Scan pipeline: search retrieval, local filters, strict amenity verification gate, LLM scoring, send path, and search failure cooldown state |
| `src/apify_scraper.py` | StreetEasy actor client: pinned/latest build policy, unhealthy-empty detection, path/pipe search, typed amenity enrichment diagnostics, mapping via `originalUrl` + listing-id fallback |
| `src/telegraph_pages.py` | Telegraph (Instant View) page creation for detailed listing views |
| `src/outreach.py` | Draft generation (Claude API), revision, copy+link send flow |
| `src/storage.py` | JSON persistence on Modal Volume with atomic writes (`/data/chats/{chat_id}.json`), update marker files, and actor canary/pin files |
| `src/config.py` | Constants, NYC neighborhood map (display name → slug), amenity maps (`AMENITIES` for URL params), scoring weights |
| `Dockerfile.modal` | Container image: Python 3.11 slim (no browser/Chromium) |
| `diagnose_amenities.py` | Diagnostic script: tests Apify search with/without amenity params to determine enrichment approach (not production) |

## Key Models (`src/models.py`)

- **`ChatState`**: Per-chat state — `preferences`, `preferences_ready`, `conversation_history`, `seen_listing_ids`, `liked_listing_ids`, `liked_listings` (dict of full Listing data), `recent_listings` (capped at 50), `amenity_cache` (detail enrichment cache by `listing_id`), `active_drafts`, `current_apartment`, and search reliability state (`search_failure_streak`, `last_search_failure_at`, `search_cooldown_until`)
- **`Preferences`**: Budget, bedrooms, neighborhoods, commute, must-haves, nice-to-haves, no-fee, move-in date, `constraint_context`
- **`ConversationTurn`**: `role` ("user"/"assistant") + `content` string
- **`Listing`**: Scraped apartment data with match scoring. Includes `building_amenities`/`unit_features`, plus Option-E contract fields `confirmed_building_amenities`, `confirmed_unit_features`, `amenity_text_dump`, and `amenity_signal_status`. Also has `matched_amenities`/`missing_amenities` for search API annotations. **Note**: `broker_fee` is `Optional[str]` (e.g. `"Broker fee"` or `None`), NOT a bool.
- **`Draft`**: Outreach message with status lifecycle (pending → sent/cancelled)
- **`CurrentApartment`**: User's current living situation — address, neighborhood, price, bedrooms, move_out_date, pros, cons, notes

There is no FSM state enum. The only state flag is `preferences_ready: bool`.

### Listing Storage

- **`recent_listings`**: Dict of recently shown listings (capped at `MAX_RECENT_LISTINGS = 50`). Populated by `scanner.py` after sending listing cards. Allows tools like `get_listing_details`, `compare_listings`, and `draft_outreach` to look up full listing data.
- **`liked_listings`**: Dict of liked listings with full `Listing` data. Populated when user taps the "Like" button — the full listing is copied from `recent_listings`.
- Both stores map `listing_id → Listing` and are persisted in `ChatState`.

## Conversation Tools (16 total)

| Tool | Purpose |
|---|---|
| `update_preferences` | Extract and save apartment preferences from natural language (includes `constraint_context`) |
| `show_preferences` | Display current saved preferences |
| `search_apartments` | Trigger StreetEasy search via Apify |
| `mark_ready` | Confirm preferences, enable daily scans |
| `clear_search_history` | Clear `seen_listing_ids` so next search shows all matches |
| `pause_daily_scans` | Pause daily scans (keeps preferences) |
| `get_liked_listings` | List all liked listings with details |
| `remove_liked_listing` | Remove a listing from liked list |
| `reset_preferences` | Reset all preferences to defaults, pause scans |
| `remove_neighborhoods` | Remove specific neighborhoods from preferences |
| `get_listing_details` | Look up full details for a listing by ID |
| `compare_listings` | Side-by-side comparison of 2-5 listings |
| `draft_outreach` | Trigger draft message creation for a listing's agent |
| `update_current_apartment` | Store/update user's current apartment info |
| `show_current_apartment` | Display saved current apartment info |

## Scan Pipeline (`src/scanner.py`)

The scan pipeline uses a two-pass architecture with multi-layer filtering:

```
Cooldown gate (search reliability)
    → If search_cooldown_until is active, skip actor calls and send retry-later message
    ↓
Pass 1: Retrieval — search_with_retry()
    ↓
Path A (primary): Path-style URLs per neighborhood
    → search_by_neighborhoods() → _build_path_urls()
    → One path URL per neighborhood (/for-rent/west-village, /for-rent/chelsea, ...)
    → ALL passed as startUrls in a single actor run
    → 100% in-area results (path URLs produce correct area filtering)
    → No beds/baths/price filtering — compensated by local filters
    → maxItems=2000
    ↓
Path B (fallback): Pipe-style URL
    → search_streeteasy() → _build_streeteasy_url()
    → Rented actor only respects `beds` and `price` URL params
    → Actor silently drops: area, baths, sort_by, amenities, no_fee
    → Returns up to 1000 results
    → Local filters in scanner.py compensate for dropped params
    ↓
Deduplicate by listing_id within run, then skip seen_listing_ids
    ↓
Parse (raw dicts → Listing models)
    ↓
Layer 1: Local pre-filters (compensate for path/pipe URL limitations)
    ↓
_neighborhood_pre_filter() — data quality fix
    → Drop listings from neighborhoods the user didn't search for
    → Uses NEIGHBORHOOD_ALIASES for naming variants (West Chelsea→Chelsea, etc.)
    → Case-insensitive matching
    → No neighborhoods in prefs → skip filter (all pass through)
    ↓
_filter_wrong_bedrooms() — local hard filter
    → Drop listings with wrong bedroom count
    → Compensates for path URLs not filtering by bedrooms
    ↓
_filter_wrong_bathrooms() — local hard filter
    → Drop listings with fewer bathrooms than prefs.min_bathrooms
    → Compensates for actor ignoring baths URL param
    ↓
_filter_stale_listings() — drop listings with available_date > 3 months in past
    → Listings with no date or unparseable dates pass through
    → STALE_LISTING_MONTHS = 3
    ↓
_filter_over_budget() — pre-LLM budget filter
    → Drop listings where both gross and net effective price > budget_max
    → Borderline cases (gross over, net under) kept for LLM judgment
    ↓
_filter_broker_fee() — local no-fee filter
    → When no_fee_only=True, drop listings with broker_fee != None
    → Compensates for actor ignoring no_fee URL param
    ↓
Pass 2: Detail page amenity enrichment (skipped if no must_haves/nice_to_haves)
    → Use ChatState.amenity_cache first (by listing_id), fetch only cache misses
    → enrich_with_amenities() runs actor on listing detail URLs
    → Maps detail rows by originalUrl and listing-id fallback (`basicInfo.id` / `rentalByListingId.id`)
    → Applies retries up to wait budget (`AMENITY_ENRICHMENT_MAX_WAIT_SECS = 300`)
    → Produces confirmed_building_amenities, confirmed_unit_features, amenity_text_dump, amenity_signal_status
    ↓
Strict amenity send gate (amenity-sensitive searches only)
    → Require coverage >= `AMENITY_REQUIRED_COVERAGE` (0.95), counting cache hits + fresh enrichment
    → If coverage below threshold or enrichment hard-fails: do NOT send listings
    → Send one failure notice instead:
      manual search: "Couldn't verify amenities reliably right now. Please retry shortly."
      daily scan: "Today's scan couldn't verify amenities reliably; I'll try again next run."
    ↓
Layer 2: _llm_score_listings() — single Opus call, ALL preference decisions
    → Input: survivors + structured preferences + constraint_context + amenity evidence contract
    → Listings include confirmed_building_amenities/confirmed_unit_features + amenity_text_dump + amenity_signal_status
    → Also has_amenities/missing_amenities from search API (when available)
    → Output: { include: bool, score: 0-100, pros, cons } per listing
    → Hard constraints (from constraint_context) → include: false
    → Soft constraints → affect score only
    → Score floor: include=true but score < 25 → excluded as safety net
    → LLM omits a listing → included with default score 50
    → LLM failure → all listings returned unscored, sorted by price
    ↓
Sort by score descending
    ↓
_cap_per_building() — max 2 listings per building address
    → MAX_PER_BUILDING = 2
    → Assumes sorted by score desc, keeps top N per building
    ↓
_cap_per_neighborhood() — max 5 listings per canonical neighborhood
    → MAX_PER_NEIGHBORHOOD = 5
    → Prevents one neighborhood from dominating results
    → Uses NEIGHBORHOOD_ALIASES for canonical grouping
    ↓
_interleave_by_neighborhood() — round-robin across neighborhoods
    → Neighborhoods ordered by best listing score
    → User sees variety in first few listings
    ↓
Fallback: if all excluded → return top 3 by score with caveat
    ↓
Send included listings

Funnel logged: Raw | Dedup | Hood | Beds | Baths | Stale | Budget | Fee | Enriched | LLM | Bldg | Sent
```

### `constraint_context`

A `Optional[str]` field on `Preferences` that the conversation engine's Claude populates with a natural language summary of what's firm vs flexible. Examples:

- *"Budget $3,500 is a firm max — user said 'not a dollar over.' 2BR needed for kids (non-negotiable). East Village preferred but open to nearby. Dishwasher non-negotiable. Gym nice but not critical."*
- *"Budget around $4k, could stretch for the right place. Ideally 2BR but would consider large 1BR. No-fee strongly preferred but not absolute."*

The LLM scorer uses this to decide what's a dealbreaker (→ `include: false`) vs a preference (→ score penalty). When `constraint_context` is `None` (e.g. first search), the LLM uses its own judgment.

The conversation engine's system prompt instructs Claude to maintain `constraint_context` based on user language: "must have" / "absolutely need" / "not a dollar over" → hard constraint; "ideally" / "would be nice" / "prefer" → soft preference.

### Path-Style URLs (`_build_path_urls`) — Primary Search

`_build_path_urls(prefs)` generates one path URL per neighborhood from preferences. Path URLs (`/for-rent/west-village`) produce 100% in-area results with the rented actor, unlike pipe-style URLs which the actor ignores for area filtering. All neighborhood URLs are passed as `startUrls` in a single actor run via `search_by_neighborhoods()`. Path URLs don't filter by beds/baths/price — local filters compensate.

### Pipe-Style URL (`_build_streeteasy_url`) — Fallback Search

`_build_streeteasy_url()` includes `area`, `price`, `beds`, and `no_fee` in a pipe-style URL. The rented actor only passes through `beds` and `price` — it silently drops `area`, `baths`, `sort_by`, amenities, and `no_fee`. Used as fallback when path-based search fails or when no neighborhoods are specified.

### Detail Page Enrichment (`enrich_with_amenities`)

The second pass runs the same rented actor on listing detail URLs to extract amenity data. Each detail page result includes `federatedData.rentalByListingId.propertyDetails` with:
- `amenities.list` — building-level amenities (Doorman, Gym, Elevator, etc.) → stored as `Listing.building_amenities`
- `features.list` — unit-specific features (Dishwasher, In-unit Laundry, Central AC, etc.) → stored as `Listing.unit_features`
- `description` — listing description text

Amenity enum values (e.g. `DOORMAN`, `WASHER_DRYER`) are normalized to human-readable display names via `AMENITY_DISPLAY_NAMES` in `apify_scraper.py`.

Important detail actor mapping notes:
- Detail rows commonly use `originalUrl` (not `url`) and include `basicInfo.id` / `federatedData.rentalByListingId.id`.
- Mapping must support both URL-based and listing-id-based joins to avoid zero-enrichment runs.

Reliability controls:
- Enrichment is **skipped** when user has no amenity preferences (`must_haves` and `nice_to_haves` both empty).
- Cache is checked first (`ChatState.amenity_cache`) to avoid re-fetching known listing ids.
- `enrich_with_amenities()` returns a typed `AmenityEnrichmentResult` with coverage, batch run summaries, request health counters, and failure reason.
- **Two-tier send gate** (as of 2026-04-20, was strict single-threshold before):
  - `AMENITY_REQUIRED_COVERAGE` (0.95) — the "verified" threshold. If combined cache+fresh coverage is at or above this, the scan proceeds silently.
  - `AMENITY_MIN_COVERAGE_WITH_MUST_HAVES` (0.20) — the hard floor. If the user declared `must_haves` AND coverage is below this, OR enrichment hard-failed with zero enriched listings, the send is blocked and a failure notice is sent.
  - Between the two: the scan proceeds and a user-visible "Heads up: StreetEasy blocked some amenity lookups..." warning is sent. The LLM scorer is told to calibrate confidence using `amenity_signal_status` (`present` / `partial` / `missing`).
- Rationale for softening: when StreetEasy's WAF partially blocks detail pages, the old strict gate silently suppressed the entire scan even though 30-70% of listings were verifiable. The LLM already knows how to handle missing amenity data.

### Local Pre-Filters

The scanner pipeline includes several local pre-filters that compensate for the rented actor's limitations:

- **`_neighborhood_pre_filter()`** — drops listings from wrong neighborhoods (data quality fix, mainly needed for pipe URL fallback)
- **`_filter_wrong_bedrooms()`** — drops listings with wrong bedroom count (path URLs don't filter by bedrooms)
- **`_filter_wrong_bathrooms()`** — drops listings with fewer bathrooms than required (actor ignores baths param)
- **`_filter_over_budget()`** — drops listings clearly over budget, respecting net effective price
- **`_filter_broker_fee()`** — drops fee listings when user requires no-fee (actor ignores no_fee param)
- **`_filter_stale_listings()`** — drops listings with available_date > 3 months in the past

### Neighborhood Pre-Filter

The pre-filter is a **data quality fix**, NOT preference enforcement. With path-style URLs (primary), most results are already in-area. With pipe-style URLs (fallback), ~95% of results may be from wrong neighborhoods.

`NEIGHBORHOOD_ALIASES` in `scanner.py` maps ~30 StreetEasy naming variants to canonical preference names (e.g. West Chelsea→Chelsea, Yorkville→Upper East Side, North/South Williamsburg→Williamsburg, NoMad→Flatiron, etc.).

### Key Constants

- `SCORING_MODEL = "claude-opus-4-7"` — model used for LLM scoring. **Don't send `temperature` to this model** — it's deprecated.
- `SCORE_FLOOR = 25` — minimum score for included listings
- `max_tokens = 8192` — for the scoring API call
- `STALE_LISTING_MONTHS = 3` — listings with `available_date` older than this are dropped
- `MAX_PER_BUILDING = 2` — max listings per building address after scoring
- `MAX_PER_NEIGHBORHOOD = 5` — max listings per canonical neighborhood after building dedup
- `AMENITY_REQUIRED_COVERAGE = 0.95` — the "verified" coverage threshold. At or above: scan proceeds silently. Below but above the must-have floor: scan proceeds WITH a user-visible "partial coverage" warning (`src/scanner.py`).
- `AMENITY_MIN_COVERAGE_WITH_MUST_HAVES = 0.20` — hard floor for users with declared `must_haves`. Below this with zero enrichment data, the send is blocked and a failure notice is sent.
- `AMENITY_ENRICHMENT_NO_ITEMS_ABORT_SECS = 90` and `AMENITY_ENRICHMENT_MAX_WAIT_SECS = 300` — verification budget used by scanner (`src/scanner.py`). Tightened 2026-04-20 so one stuck enrichment batch can't eat the entire daily-scan timeout.
- `ENRICHMENT_RETRY_BATCH_SIZE = 25` and `ENRICHMENT_MAX_URLS = 120` — internal pass-2 retry controls in scraper (`src/apify_scraper.py`)
- `SEARCH_FAILURE_STREAK_FOR_COOLDOWN = 2` — failures required before cooldown
- `SEARCH_FAILURE_COOLDOWN_BASE_SECS = 180` and `SEARCH_FAILURE_COOLDOWN_MAX_SECS = 1800` — exponential backoff bounds
- Daily-scan Modal task timeout: `1500s` (raised from 600s on 2026-04-20 to give retry chains headroom).

## Telegram Listing Cards

Listing cards use `sendPhoto` with a compact HTML caption (< 1024 chars) and inline keyboard buttons.

### Card Format

Clean, middot-separated layout:
```
#1  123 Main St #4A
East Village · $3,650/mo · NO FEE

2 BR · 1 BA · 750 sqft
Available Mar 1, 2026

████████░░  80% match

▸ In-unit laundry, dishwasher
▸ 5th floor walk-up

View on StreetEasy →
```

- Score bar uses Unicode block characters (`█░`), not emoji
- Pros/cons use `▸` triangle bullets, limited to 3 pros + 2 cons
- "NO FEE" shown inline with price when `broker_fee` is `None`

### Inline Keyboard

```
[👍 Like]     [👎 Pass]
[📋 Details]  [🔗 StreetEasy ↗]
```

- Like/Pass: callback buttons (bot records action)
- Details: callback button (creates Telegraph Instant View page)
- StreetEasy: **URL button** (opens listing page directly in browser, no callback)

### Photo Fallback Chain (`send_listing_photo`)

Photos are sent with a 4-step fallback to handle CDN issues:

1. **Direct URL** — `sendPhoto` with the imgix URL (fastest, zero bandwidth)
2. **Download + re-upload** — Our server downloads the image with `Referer: https://streeteasy.com/` header, then uploads bytes to Telegram
3. **Link preview** — `sendMessage` with `link_preview_options` using `prefer_large_media: True` + `show_above_text: True` (uses StreetEasy's OG image, gets 4096-char text limit)
4. **Text only** — Plain `sendMessage` with no photo

This ensures listings always appear even if imgix URLs are blocked or inaccessible.

## Telegraph Integration (`src/telegraph_pages.py`)

When a user taps the "Details" button on a listing card, a Telegraph page is created with:
- Up to 8 photos
- Full listing details (price, beds, baths, fee, sqft, availability, match score)
- Pros/cons lists
- Full description text
- Amenity list
- StreetEasy link

The page URL opens as an **Instant View** inside Telegram — a fast, in-app reader with no character limits. Falls back to a text-based detail message if Telegraph fails.

The Telegraph client is lazy-initialized on first use via `_get_telegraph()`.

## Outreach Flow

1. User says "draft a message for listing #1" → Claude calls `draft_outreach` tool
2. `create_draft()` in `outreach.py` generates a personalized message via Claude API, including context about the user's preferences and current apartment
3. Draft is shown to user with Send/Edit/Cancel buttons
4. On Send: `send_approved_draft()` provides the message in `<pre>` tags (easy to copy) + a direct link to the listing's StreetEasy page for the user to paste into the contact form
5. On Edit: user provides feedback, `revise_draft()` sends to Claude for revision

StreetEasy's contact form cannot be automated (Imperva bot protection). The bot drafts the message; the user copy-pastes it.

## Apify Integration

StreetEasy search uses a two-pass approach through the `memo23/apify-streeteasy-cheerio` rented actor:

**Pass 1 — Retrieval**: Path-style URLs per neighborhood (primary) or pipe-style URL (fallback). Runs under build policy (pinned build first, latest fallback, auto-promotion on healthy fallback success). Unhealthy runs — `SUCCEEDED + requests_succeeded=0 + requests_failed>0` regardless of item count — are treated as failures (actor writes placeholder rows on WAF block).

**Pass 2 — Enrichment**: Detail page URLs extract building amenities and unit features. Skipped when user has no amenity preferences. Uses bulk + retry batches and returns typed diagnostics (`AmenityEnrichmentResult`) including per-run `AmenityRunSummary` entries.

Total pipeline latency is variable (actor + WAF dependent). Enrichment is hard-bounded by `AMENITY_ENRICHMENT_MAX_WAIT_SECS` (300s) so it can't consume the daily-scan timeout (1500s).

The `ApifyScraper` class in `src/apify_scraper.py` provides:

- `search_by_neighborhoods(prefs, max_items=2000)` — path-style URLs per neighborhood in a single actor run. 100% in-area results. Primary search method when neighborhoods are specified.
- `search_streeteasy(prefs, max_items=1000)` — pipe-style URL fallback. Only `beds` and `price` URL params are functional. Used when path search fails or no neighborhoods specified.
- `enrich_with_amenities(listing_urls, url_to_id)` — runs actor on listing detail URLs, maps by `originalUrl` then listing-id fallback, and returns `AmenityEnrichmentResult` (`data_by_listing_id`, `coverage`, `run_summaries`, `failed`, `failure_reason`).
- `search_with_retry(prefs)` — tries path-style search first (when neighborhoods set); on failure falls back to pipe-style; retries with exponential backoff.
- `_run_actor_with_build_policy(...)` — central build pin/latest fallback policy with unhealthy-empty detection and optional auto-promotion.
- `_run_actor(start_urls, max_items, abort_after_secs_no_items, max_wait_secs, poll_context)` — low-level polling/abort wrapper returning run metadata.
- No lifecycle management needed — the Apify client is stateless HTTP

Actor build reliability files are stored under `/data/system/`:
- `apify_actor_build_pin.json` — active pinned build id
- `apify_actor_canary_status.json` — latest daily canary result
- `apify_actor_canary_urls.json` — sampled known-good detail URLs from production enrichment

### Apify Actor Output Format

The actor has emitted **two schemas** across its builds. `_map_apify_item()` handles both.

**Legacy nested (pre-0.0.96):**

```python
{
    "__typename": "OrganicRentalEdge",
    "node": {
        "id": "4961650",           # listing ID (string)
        "areaName": "East Village", # neighborhood name
        "price": 3650,             # monthly rent (int)
        "bedroomCount": 1,         # bedrooms (int, 0 = studio)
        "fullBathroomCount": 1,    # full baths (int)
        "halfBathroomCount": 0,    # half baths (int)
        "street": "123 Main St",   # street address
        "unit": "4A",              # apartment unit
        "urlPath": "/building/...",# path for listing URL
        "noFee": false,            # true if no broker fee
        "availableAt": "2026-03-01",
        "photos": [{"key": "abc123"}, ...],  # photo keys
        "livingAreaSize": 750,     # sqft (0 if unknown)
        "status": "ACTIVE",
    }
}
```

**Current flattened (0.0.96+):**

```python
{
    "__typename": "OrganicRentalEdge",
    "node___typename": "SearchRentalListing",
    "node_id": "4961650",
    "node_areaName": "East Village",
    "node_price": 3650,
    "node_bedroomCount": 1,
    "node_fullBathroomCount": 1,
    "node_halfBathroomCount": 0,
    "node_street": "123 Main St",
    "node_unit": "4A",
    "node_urlPath": "/building/...",
    "node_noFee": False,
    "node_availableAt": "2026-03-01",
    "node_photos_json": '[{"__typename":"Photo","key":"abc123"}, ...]',  # note: JSON STRING
    "node_leadMedia_photo_key": "abc123",  # fallback when node_photos_json missing
    "node_livingAreaSize": 750,
    "node_status": "ACTIVE",
    # ...various node_leadMedia_*, node_geoPoint_*, etc.
}
```

Photo URLs are constructed from photo keys as: `https://photos.zillowstatic.com/fp/{key}-se_large_800_400.jpg` (large card) and `...-se_medium_500_250.jpg` (thumb). The imgix host is deprecated.

`_map_apify_item()` reads both shapes via a `_g(field)` helper (tries `node.field`, then `node_field`, then top-level `field`) and decodes `*_json` fields as JSON. It also maps item-level amenity annotation fields (`matchedAmenities`, `missingAmenities`) when present. **Important**: All values must match `Listing` model types — e.g. `broker_fee` must be `Optional[str]`, not bool.

### Rented Actor

- Actor: `memo23/apify-streeteasy-cheerio` (rented, $19/mo)
- Actor uses StreetEasy's internal GraphQL API via `SearchDeeplink` operation (translates URLs to search params)
- Residential proxy group `RESIDENTIAL` is used for reliability
- **Path-style URLs** (`/for-rent/west-village`): Actor correctly filters by neighborhood. Used as primary search method.
- **Pipe-style URLs** (`/for-rent/area:west-village|price:...|beds:...`): Actor silently drops most URL params — only `beds` and `price` are functional. Used as fallback.
- **Detail page URLs** (`/building/.../unit`): Actor returns `federatedData` with full listing details including amenities. Used for enrichment pass.
- `maxItems` default 2000 for path-style, 1000 for pipe-style
- Local filters in scanner.py compensate for missing URL param filtering

## Modal Entry Points (`modal_app.py`)

1. **`TelegramWebhook.webhook`** (FastAPI ingress, `@modal.concurrent(max_inputs=10)`) — verifies webhook secret, claims `update_id` in shared `update_claims`, writes persistent marker, enqueues worker.
2. **`process_telegram_update`** (async worker) — runs `TelegramBot.process_update(..., skip_update_dedup=True)` and triggers scans via callback.
3. **`daily_scan`** (Cron, `30 16 * * *`) — scans StreetEasy for all users with `preferences_ready=True`.
4. **`actor_build_canary`** (Cron, `0 16 * * *`) — runs latest-build amenity canary, persists status, auto-promotes pin when healthy.
5. **`send_agent_message`** (on-demand) — prepares approved outreach drafts with listing link for user to send manually.
6. **`setup`/`_setup_webhook`** (manual helper) — registers Telegram webhook URL.

### Webhook Scan Flow

When a user triggers a search via conversation:
1. `TelegramWebhook.webhook` claims `update_id` and returns quickly after calling `process_telegram_update.spawn(data)`.
2. Worker `process_telegram_update` runs conversation/tool flow and executes scan callback.
3. Per-chat single-flight lock (`scan_locks`) suppresses concurrent manual scans.
4. `scan_send_receipts` blocks duplicate sends for the same `(chat_id, update_id)` trigger.

## Telegram Deduplication

Dedup is three-layer for webhook updates:
- Distributed claim dict: `update_claims` in Modal Dict (primary barrier at ingress)
- Persistent marker files: `storage.mark_update_seen(update_id)` → `/data/telegram_updates/{update_id}.seen`
- In-memory set: `TelegramBot._seen_update_ids` (fallback process-local guard)

Worker mode intentionally calls `process_update(..., skip_update_dedup=True)` because ingress already applied distributed + persistent dedup.

## Secrets

- `ANTHROPIC_API_KEY` — Claude API (Modal secret: `anthropic`)
- `SE_TELEGRAM_BOT_TOKEN` — Telegram bot token (Modal secret: `streeteasy-telegram`)
- `APIFY_API_TOKEN` — Apify API token (Modal secret: `apify`)

## Commands

```bash
# Run tests
python3 -m pytest tests/ -v

# Deploy to Modal
modal deploy modal_app.py

# Set up Telegram webhook (after deploy)
modal run modal_app.py::setup

# View live logs
modal app logs streeteasy-bot
```

## Agent Handoff (Amenity Reliability)

Use this checklist before/after modifying amenity enrichment:

1. Run targeted tests:
   - `python3 -m pytest -q tests/test_modal_app.py tests/test_telegram_handler.py tests/test_scanner.py tests/test_apify_scraper.py tests/test_storage.py tests/test_models.py`
2. Confirm send-gating in logs. **The log format changed 2026-04-20** — the old `send_allowed=` boolean is gone:
   - `Amenity enrichment: cache_hits=... fetched=... enriched=... coverage=...`
   - `Amenity send decision: coverage=... verified_threshold=0.950 min_with_must_haves=0.200 failure_reason=...`
   - On the happy-but-partial path: `Proceeding with partial amenity coverage: coverage=X.XXX` (sends the user-facing "Heads up: StreetEasy blocked some amenity lookups..." notice).
   - On hard-block: either "Couldn't verify amenities reliably right now" (manual) or "Today's scan couldn't verify amenities reliably" (daily).
3. Confirm actor-run health logs:
   - `Apify ... run summary: build_id=... requests_succeeded=... requests_failed=...`
   - Unhealthy runs should log `Treating run as failure (unhealthy empty)`. This now catches `SUCCEEDED` with `requests_succeeded=0, requests_failed>0` **regardless of item count** (the actor writes placeholder rows on total WAF block).
4. Confirm distributed scan protections:
   - `Scan lock acquired/released ...`
   - duplicate trigger suppression via `scan_send_receipts`.
5. If coverage drops, verify mapping fields (`originalUrl`, `basicInfo.id`, `federatedData.rentalByListingId.id`) before changing retry or timeout values.
6. If the symptom is "no listings today" on daily cron, check the Apify run log first for `[TRANSLATE_URL] entering: 403` and `classification: px=true` — that's StreetEasy's PerimeterX WAF, external, nothing to fix in our code. The code correctly falls through to the rate-limit fallback message in that case.

## Ops — Local e2e Routine (macOS)

A launchd agent runs a full end-to-end scan every 4 days (`StartInterval 345600`). It's wired up locally — not in the repo — and serves as an outside-of-cron regression canary:
- Agent plist: `~/Library/LaunchAgents/com.babakd.streeteasybot.e2e.plist`
- Script: `~/Library/streeteasybot/e2e_check.sh`
- Logs: `~/Library/Logs/streeteasybot/run-<timestamp>/`
- What it runs: `git pull origin main` (only when on main), `pytest`, then `modal run modal_app.py::daily_scan` against live services (real Apify, real Telegram delivery to the owner chat).
- Failure path: spawns a headless `claude -p --permission-mode bypassPermissions` session with a scope-narrowed prompt (no `modal deploy`, no push-to-main, no destructive commands, no weakening tests, 15-min timebox) to diagnose, fix, and open a PR.
- Summary: DMs the owner's Telegram with pass/fail + log dir + PR url.
- Load/unload: `launchctl load -w ~/Library/LaunchAgents/com.babakd.streeteasybot.e2e.plist` / `launchctl unload ~/Library/LaunchAgents/com.babakd.streeteasybot.e2e.plist`.

The routine is **intentionally not in the repo** — it holds a Telegram bot token file at `~/Library/streeteasybot/.tgbot_token` and runs on the owner's machine only. Other agents should not try to replicate it in-repo.

## Testing

Tests use `pytest` + `pytest-asyncio`. The conversation engine tests mock the `ClaudeClient` (no real Claude API calls). Scanner tests mock `anthropic.AsyncAnthropic` for scoring.

- `tests/test_conversation.py` — conversation engine, all 16 tools, system prompt, constraint_context
- `tests/test_modal_app.py` — webhook ingress claim/enqueue flow, worker scan lock + duplicate trigger suppression, canary behavior
- `tests/test_scanner.py` — neighborhood pre-filter, bathroom filter, budget filter, broker fee filter, stale filter, building dedup, neighborhood cap, interleaving, amenity annotations, LLM filter+score, end-to-end scan_for_chat, hero photo picker, scan cache fallback
- `tests/test_apify_scraper.py` — Apify polling/abort/retry, URL builder, resilient enrichment mapping (originalUrl + listing-id fallback), coverage retry behavior
- `tests/test_models.py` — Pydantic model creation, validation, JSON roundtrip including CurrentApartment, listing stores, constraint_context, amenity fields
- `tests/test_formatter.py` — HTML formatting, listing cards, keyboards, payload builders
- `tests/test_storage.py` — save/load/delete state, atomicity, persistent update-id dedup markers
- `tests/test_telegram_handler.py` — webhook routing, group chat, draft editing, in-memory + persistent update dedup flow
- `tests/test_telegraph_pages.py` — Telegraph page HTML escaping

## Latest Verification Snapshot

**April 20, 2026** (PR #3, commit `f41cc45`, deployed as Modal v53):
- Full test suite: `437 passed` (was 250 in Feb).
- Live end-to-end scan during a partial PerimeterX WAF event: 164 raw → 3 cards delivered to owner chat on Telegram. Partial-coverage amenity warning sent as expected (coverage=33.3% < 0.95 verified threshold but above 0.20 must-have floor).
- Build promotion path exercised via restored tests (mocks cover `_build_number_for_id` + `_latest_build_number`).
- Deploy commit is clean (no `*` dirty marker in `modal app history`).

**February 22, 2026** (baseline):
- Targeted reliability suite passed: `250 passed`
  - Command: `python3 -m pytest -q tests/test_modal_app.py tests/test_telegram_handler.py tests/test_scanner.py tests/test_apify_scraper.py tests/test_storage.py tests/test_models.py`
- Full `pytest -q` had one pre-existing unrelated collection issue:
  - `test_url_formats.py::test_url` missing fixture `proxy_url` (resolved since).

## Gotchas / Lessons Learned

- **Webhook must stay ingress-only**: Keep `TelegramWebhook.webhook` fast (claim + enqueue). Heavy work belongs in `process_telegram_update` worker to avoid Telegram retry storms and duplicate scans.
- **Apify field mapping handles TWO schemas**: Legacy nested (`node.bedroomCount`, `node.areaName`) and current flattened (`node_bedroomCount`, `node_areaName`, `node_photos_json` as a JSON string). `_map_apify_item()` uses a `_g(field)` helper that falls through nested → flattened → top-level, and a JSON-string parser for `*_json` list fields. Type mismatches (e.g. bool vs str for `broker_fee`) cause silent per-listing failures — test both shapes when touching this.
- **`seen_listing_ids` persistence**: Listing IDs are marked as seen only after successful scoring and sending. Parse failures and LLM-excluded listings are NOT marked seen, so they'll reappear in future scans.
- **Modal `allow_concurrent_inputs`**: Deprecated since 2025-04. Use `@modal.concurrent(max_inputs=N)` decorator instead.
- **Telegram retry behavior**: Telegram retries webhooks after ~60s. Robust dedup requires all three barriers (`update_claims` + persistent marker files + in-memory guard).
- **Scan single-flight is mandatory**: Without `scan_locks` and `scan_send_receipts`, duplicate user triggers can produce repeated "Searching now" and duplicated sends.
- **Telegram sendPhoto caption limit**: 1024 characters max. Listing cards are designed to stay under ~400 chars. Use Telegraph for detailed views.
- **Telegram sendPhoto URL failures**: StreetEasy's imgix CDN may block Telegram's servers. The 4-step fallback chain in `send_listing_photo` handles this by downloading with a Referer header, then falling back to link preview, then text only.
- **Telegraph lazy init**: The Telegraph client in `telegraph_pages.py` is initialized on first use. If Telegraph is unreachable, the details view falls back to a text-based message.
- **StreetEasy contact form**: Cannot be automated due to Imperva bot protection. The outreach flow generates draft messages for the user to copy-paste manually.
- **`listing_keyboard` URL button**: The StreetEasy button is a Telegram URL button (`"url": listing_url`), not a callback. It opens the browser directly without hitting the bot.
- **Path-style URLs are the primary search method**: `search_by_neighborhoods()` uses path-style URLs (`/for-rent/west-village`) that produce 100% in-area results. Pipe-style URLs (`/for-rent/area:west-village|...`) are the fallback — the actor silently ignores the `area` param in pipe URLs, returning ~95% wrong-neighborhood results. The neighborhood pre-filter compensates for the fallback path.
- **Path URLs don't filter by beds/baths/price**: Path-style URLs pass no query parameters to the actor. Local filters (`_filter_wrong_bedrooms`, `_filter_wrong_bathrooms`, `_filter_over_budget`, `_filter_broker_fee`) compensate for all of these.
- **Pipe-style URL (fallback) drops most params**: Only `beds` and `price` work in pipe-style URLs. `area`, `baths`, `sort_by`, amenities, and `no_fee` are silently dropped. Local filters compensate.
- **Two-pass enrichment latency is highly variable**: Pass 2 can be fast or slow depending on actor/WAF behavior. Do not hard-code short assumptions; use coverage-targeted retries and cache.
- **Amenity gating is two-tier, not strict single-send**: Above `AMENITY_REQUIRED_COVERAGE` (0.95) → silent send. Between the floor (`AMENITY_MIN_COVERAGE_WITH_MUST_HAVES`, 0.20) and 0.95 → send listings WITH a "heads up" warning and let the LLM calibrate via `amenity_signal_status`. Below the floor with must-haves → hard block, amenity failure notice. This was changed 2026-04-20 after the strict gate silently suppressed scans on partial-WAF-block days.
- **Detail row mapping is critical**: Pass-2 detail rows may have `originalUrl` (no `url`) and ids in `basicInfo.id` / `federatedData.rentalByListingId.id`. Mapping only `item["url"]` causes zero amenity coverage.
- **Unhealthy actor runs are not "no results"**: `SUCCEEDED` status can still mean total request failure. Treat `requests_succeeded=0 and requests_failed>0` as an error path **regardless of item count** — the actor writes `{"message": "No results found", ...}` placeholder rows into the dataset when all requests 403, which used to be silently treated as "no listings today" before `_is_unhealthy_empty_run` was widened (2026-04-20).
- **Build pinning policy matters**: Search/enrichment run pinned build first, then latest fallback. If fallback succeeds, auto-promote pin unless env override `APIFY_ACTOR_BUILD_PIN` is set.
- **Daily canary is latest-only**: `actor_build_canary` forces `build="latest"` and only promotes on coverage + request-health pass; otherwise pin remains unchanged.
- **Neighborhood aliases are one-directional**: `NEIGHBORHOOD_ALIASES` maps listing neighborhoods to preference names (e.g. "West Chelsea" → "Chelsea"). The map must be lowercase keys. Adding new aliases requires only adding to the dict — no code changes needed.
- **LLM scoring fallback**: If all listings get `include: false` from the LLM, the top 3 by score are returned as fallback to avoid showing nothing. If the LLM API fails entirely, listings are returned unscored and sorted by price.
- **`constraint_context` is optional**: Old persisted state without `constraint_context` loads correctly (defaults to `None`). The LLM scorer works fine without it — it uses its own judgment when `constraint_context` is `None`.
- **Amenity contract for scoring**: Detail enrichment populates `confirmed_building_amenities`, `confirmed_unit_features`, `amenity_text_dump`, and `amenity_signal_status` (plus legacy `building_amenities`/`unit_features`) for LLM scoring.
- **Amenities must NEVER be in the search URL filter**: The API hard-filters by amenities, silently dropping listings. Amenity data flows to the LLM scorer only; the LLM decides include/exclude. `_build_streeteasy_url()` intentionally excludes amenities.
- **Building dedup assumes score-sorted input**: `_cap_per_building()` must be called AFTER sorting by score descending, so it keeps the highest-scoring listings per building. Similarly, `_cap_per_neighborhood()` and `_interleave_by_neighborhood()` must run after building dedup.
- **Stale filter uses 30-day months**: `_filter_stale_listings()` uses `STALE_LISTING_MONTHS * 30` days, not calendar months. Listings with no `available_date` or unparseable dates pass through.
- **`diagnose_amenities.py` is diagnostic only**: Not production code. Reads APIFY_API_TOKEN from `.env`. Used to determine which amenity enrichment approach is viable.
- **StreetEasy has no public API**: The internal GraphQL endpoint (`api-internal.streeteasy.com/graphql`) is protected by **PerimeterX** (signature `PXcZdhF737`, `px-cloud.net` challenge pages). Direct scraping is blocked. We use Apify's rented `memo23/apify-streeteasy-cheerio` actor with residential proxies to bypass WAF; PerimeterX rotates its fingerprint blocks periodically, which shows up here as intermittent "no listings" days. (The StreetEasy public contact form is separately blocked by Imperva — we don't automate it; `outreach.py` generates copy-paste drafts instead.)

## Conventions

- All IO is async (`async def` / `await`)
- Imports use `from __future__ import annotations` everywhere
- Logging via `logging.getLogger(__name__)`
- HTML escaping for all Telegram output (`_escape_html()` in formatter.py and outreach.py)
- State is always loaded at the start of a handler and saved after mutations
- Atomic file writes via tempfile + `os.replace()`
- No emojis in code comments; emojis only in user-facing Telegram messages
- Default Claude model: `claude-opus-4-7` (set in `src/claude_client.py`)
- Before making substantial changes, add tests (unit and e2e when possible) that test those conditions, and make sure they pass after implementing the changes
