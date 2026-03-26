"""Modal App: 3 entry points — Telegram webhook, daily cron scanner, agent outreach sender."""

from __future__ import annotations

import json
import logging
import os
import time
import uuid
from typing import Any

import modal
from starlette.requests import Request
from src.storage import mark_update_seen

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# --- Modal App Setup ---

app = modal.App("streeteasy-bot")

# Container image from Dockerfile
image = modal.Image.from_dockerfile("Dockerfile.modal")

# Persistent volume for state
volume = modal.Volume.from_name("streeteasy-data", create_if_missing=True)
update_claims = modal.Dict.from_name("streeteasy-update-claims", create_if_missing=True)
scan_locks = modal.Dict.from_name("streeteasy-scan-locks", create_if_missing=True)
scan_send_receipts = modal.Dict.from_name("streeteasy-scan-send-receipts", create_if_missing=True)

# Secrets
secrets = [
    modal.Secret.from_name("anthropic"),            # ANTHROPIC_API_KEY
    modal.Secret.from_name("streeteasy-telegram"),   # SE_TELEGRAM_BOT_TOKEN
    modal.Secret.from_name("apify"),                 # APIFY_API_TOKEN
]

UPDATE_CLAIM_TTL_SECS = 6 * 60 * 60
SCAN_LOCK_TTL_SECS = 20 * 60


def _claim_update(update_id: int) -> tuple[bool, str]:
    """Best-effort distributed claim for a Telegram update id."""
    key = str(update_id)
    now = time.time()
    existing = update_claims.get(key)
    if isinstance(existing, dict):
        claimed_at = float(existing.get("claimed_at", 0) or 0)
        status = str(existing.get("status", ""))
        if status in {"queued", "processing", "done"} and (now - claimed_at) < UPDATE_CLAIM_TTL_SECS:
            return (False, status or "existing")

    token = uuid.uuid4().hex
    candidate = {
        "token": token,
        "claimed_at": now,
        "status": "queued",
    }
    update_claims.put(key, candidate)
    current = update_claims.get(key) or {}
    if current.get("token") != token:
        return (False, "lost_race")
    return (True, token)


def _set_update_status(update_id: int, token: str, status: str, **extra: Any) -> None:
    """Update claim status only if this worker still owns the token."""
    key = str(update_id)
    current = update_claims.get(key) or {}
    if current.get("token") != token:
        return
    payload = {
        "token": token,
        "claimed_at": float(current.get("claimed_at", time.time()) or time.time()),
        "status": status,
    }
    payload.update(extra)
    update_claims.put(key, payload)


def _acquire_scan_lock(chat_id: int, owner: str, ttl_secs: int = SCAN_LOCK_TTL_SECS) -> bool:
    """Acquire a per-chat distributed lease. Last-writer verification picks one winner."""
    key = str(chat_id)
    now = time.time()
    current = scan_locks.get(key)
    if isinstance(current, dict):
        expires_at = float(current.get("expires_at", 0) or 0)
        current_owner = str(current.get("owner", "") or "")
        if expires_at > now and current_owner and current_owner != owner:
            return False

    token = uuid.uuid4().hex
    candidate = {
        "owner": owner,
        "token": token,
        "claimed_at": now,
        "expires_at": now + ttl_secs,
    }
    scan_locks.put(key, candidate)
    accepted = scan_locks.get(key) or {}
    return accepted.get("token") == token and accepted.get("owner") == owner


def _release_scan_lock(chat_id: int, owner: str) -> None:
    """Release scan lock when still owned by this caller."""
    key = str(chat_id)
    current = scan_locks.get(key) or {}
    if current.get("owner") == owner:
        scan_locks.delete(key)


# --- 1. Telegram Webhook Handler ---

@app.cls(
    image=image,
    volumes={"/data": volume},
    secrets=secrets,
    scaledown_window=300,
    memory=2048,
    cpu=2.0,
    timeout=1200,
)
@modal.concurrent(max_inputs=10)
class TelegramWebhook:
    """Receives Telegram webhook POSTs and queues async processing."""

    @modal.enter()
    def start(self):
        """Initialize ingress container."""
        logger.info("Webhook ingress ready")

    @modal.fastapi_endpoint(method="POST")
    async def webhook(self, request: Request, data: dict):
        """Receive Telegram webhook POST and enqueue async processing."""
        # Verify webhook secret if configured
        expected_secret = os.environ.get("SE_WEBHOOK_SECRET", "")
        if expected_secret:
            provided_secret = request.headers.get("x-telegram-bot-api-secret-token", "")
            if provided_secret != expected_secret:
                logger.warning("Webhook auth failed: invalid secret token")
                return {"ok": False}

        logger.info("Received webhook update: %s", json.dumps(data)[:200])
        update_id = data.get("update_id")

        if update_id is not None:
            claimed, claim_status = _claim_update(int(update_id))
            if not claimed:
                logger.info(
                    "Dropping duplicate webhook update: update_id=%s reason=%s",
                    update_id,
                    claim_status,
                )
                return {"ok": True, "duplicate": True}

            # Persist marker immediately for a second dedup barrier.
            try:
                mark_update_seen(int(update_id))
            except Exception:
                logger.exception("Persistent marker write failed for update_id=%s", update_id)

        try:
            process_telegram_update.spawn(data)
        except Exception:
            logger.exception("Failed to enqueue webhook update")
            if update_id is not None:
                update_claims.delete(str(update_id))
            return {"ok": False}
        finally:
            volume.commit()

        return {"ok": True, "queued": True}

    @modal.fastapi_endpoint(method="GET")
    async def health(self):
        """Health check endpoint."""
        return {"status": "ok"}


# --- 1b. Async Update Worker ---

@app.function(
    image=image,
    volumes={"/data": volume},
    secrets=secrets,
    scaledown_window=300,
    memory=2048,
    cpu=2.0,
    timeout=1800,
    max_containers=8,
)
@modal.concurrent(max_inputs=1)
async def process_telegram_update(data: dict):
    """Process a queued Telegram update and trigger scans with per-chat single-flight."""
    from src.apify_scraper import ApifyScraper
    from src.scanner import scan_for_chat
    from src.telegram_handler import TelegramBot

    volume.reload()
    update_id = data.get("update_id")
    update_token = ""
    if update_id is not None:
        existing = update_claims.get(str(update_id)) or {}
        update_token = str(existing.get("token", "") or "")
        if update_token:
            _set_update_status(int(update_id), update_token, "processing", started_at=time.time())

    scraper = ApifyScraper()
    bot = TelegramBot(token=os.environ["SE_TELEGRAM_BOT_TOKEN"])

    async def do_scan(chat_id: int, state, trigger_update_id: int | None):
        trigger_key = f"{chat_id}:{trigger_update_id}" if trigger_update_id is not None else f"{chat_id}:na"
        if trigger_update_id is not None and scan_send_receipts.contains(trigger_key):
            logger.info("Skipping duplicate scan send for key=%s", trigger_key)
            return

        owner = f"{trigger_key}:{uuid.uuid4().hex}"
        if not _acquire_scan_lock(chat_id, owner):
            logger.warning(
                "Scan already in flight for chat_id=%s, suppressing duplicate trigger=%s",
                chat_id,
                trigger_key,
            )
            await bot.send_text(
                chat_id,
                "A search is already running. I'll send results once it finishes.",
            )
            return

        logger.info(
            "Scan lock acquired: chat_id=%s trigger=%s owner=%s",
            chat_id,
            trigger_key,
            owner,
        )
        try:
            await scan_for_chat(scraper, bot, state, is_daily=False)
            if trigger_update_id is not None:
                scan_send_receipts.put(
                    trigger_key,
                    {"sent_at": time.time(), "chat_id": chat_id},
                )
        finally:
            _release_scan_lock(chat_id, owner)
            logger.info(
                "Scan lock released: chat_id=%s trigger=%s owner=%s",
                chat_id,
                trigger_key,
                owner,
            )

    bot.set_scan_callback(do_scan)

    try:
        await bot.process_update(data, skip_update_dedup=True)
        if update_id is not None and update_token:
            _set_update_status(
                int(update_id),
                update_token,
                "done",
                finished_at=time.time(),
            )
    except Exception:
        logger.exception("Worker failed processing update")
        if update_id is not None and update_token:
            _set_update_status(
                int(update_id),
                update_token,
                "failed",
                finished_at=time.time(),
            )
        raise
    finally:
        await bot.close()
        volume.commit()


# --- 2. Daily Scanner (Cron) ---

@app.function(
    image=image,
    volumes={"/data": volume},
    secrets=secrets,
    memory=2048,
    cpu=2.0,
    timeout=600,
    schedule=modal.Cron("30 16 * * *"),
)
async def daily_scan():
    """Run daily at 8 AM — scan StreetEasy for all registered chats."""
    from src.apify_scraper import ApifyScraper
    from src.scanner import run_daily_scan
    from src.telegram_handler import TelegramBot

    scraper = ApifyScraper()
    bot = TelegramBot(token=os.environ["SE_TELEGRAM_BOT_TOKEN"])

    volume.reload()
    try:
        await run_daily_scan(scraper, bot)
    except Exception:
        logger.exception("Daily scan failed")
    finally:
        await bot.close()
        volume.commit()


# --- 2b. Actor Canary (Cron) ---

async def _run_actor_build_canary_once() -> dict[str, object]:
    """Run one canary check against latest build and persist status."""
    from src.apify_scraper import ApifyScraper
    from src.config import (
        AMENITY_ENRICHMENT_MAX_WAIT_SECS,
        AMENITY_ENRICHMENT_NO_ITEMS_ABORT_SECS,
        AMENITY_REQUIRED_COVERAGE,
    )
    from src.storage import (
        load_actor_build_pin,
        load_canary_urls,
        save_actor_build_pin,
        save_canary_status,
    )

    pin_before = load_actor_build_pin()
    urls = load_canary_urls()
    if not urls:
        status_payload: dict[str, object] = {
            "ok": False,
            "skipped": True,
            "failure_reason": "no_canary_urls",
            "pin_before": pin_before,
            "pin_after": pin_before,
            "promotion_applied": False,
            "coverage": 0.0,
            "threshold": AMENITY_REQUIRED_COVERAGE,
            "target_count": 0,
            "requests_succeeded": 0,
            "requests_failed": 0,
            "has_request_counts": False,
            "tested_latest": "latest",
            "run_summaries": [],
            "request_health_good": False,
        }
        save_canary_status(status_payload)
        logger.warning("Actor canary skipped: no canary URLs in storage")
        return status_payload

    sample_urls = urls[: min(20, len(urls))]
    url_to_id = {url: f"canary-{idx}" for idx, url in enumerate(sample_urls, start=1)}
    scraper = ApifyScraper()

    try:
        result = await scraper.enrich_with_amenities(
            sample_urls,
            url_to_id,
            max_total_wait_secs=AMENITY_ENRICHMENT_MAX_WAIT_SECS,
            required_coverage=AMENITY_REQUIRED_COVERAGE,
            no_items_abort_secs=AMENITY_ENRICHMENT_NO_ITEMS_ABORT_SECS,
            force_build="latest",
            allow_build_fallback=False,
            auto_promote_on_fallback=False,
        )

        requests_succeeded = sum(
            summary.requests_succeeded or 0
            for summary in result.run_summaries
        )
        requests_failed = sum(
            summary.requests_failed or 0
            for summary in result.run_summaries
        )
        has_request_counts = any(
            summary.requests_succeeded is not None or summary.requests_failed is not None
            for summary in result.run_summaries
        )
        request_health_good = requests_failed == 0 and (
            requests_succeeded > 0 or not has_request_counts
        )

        latest_build = ""
        for summary in result.run_summaries:
            if summary.build_id:
                latest_build = summary.build_id
                break
        if not latest_build:
            latest_build = await scraper._latest_build_id()

        canary_ok = (
            not result.failed
            and result.coverage >= AMENITY_REQUIRED_COVERAGE
            and request_health_good
        )

        env_pin_override = os.environ.get("APIFY_ACTOR_BUILD_PIN", "").strip()
        promotion_applied = False
        pin_after = pin_before
        if canary_ok and latest_build:
            if env_pin_override:
                logger.info(
                    "Actor canary pass but env pin override is active; skipping promotion "
                    "(pin_before=%s latest_build=%s)",
                    pin_before or "<none>",
                    latest_build,
                )
            else:
                pin_after = latest_build
                if latest_build != pin_before:
                    save_actor_build_pin(latest_build)
                    promotion_applied = True

        status_payload = {
            "ok": canary_ok,
            "skipped": False,
            "failure_reason": result.failure_reason,
            "pin_before": pin_before,
            "pin_after": pin_after,
            "promotion_applied": promotion_applied,
            "coverage": result.coverage,
            "threshold": AMENITY_REQUIRED_COVERAGE,
            "target_count": result.target_count,
            "requests_succeeded": requests_succeeded,
            "requests_failed": requests_failed,
            "has_request_counts": has_request_counts,
            "tested_latest": "latest",
            "latest_build_id": latest_build,
            "run_summaries": [
                {
                    "run_id": summary.run_id,
                    "dataset_id": summary.dataset_id,
                    "build_id": summary.build_id,
                    "status": summary.status,
                    "status_message": summary.status_message,
                    "rows_returned": summary.rows_returned,
                    "mapped_count": summary.mapped_count,
                    "requests_succeeded": summary.requests_succeeded,
                    "requests_failed": summary.requests_failed,
                }
                for summary in result.run_summaries
            ],
            "request_health_good": request_health_good,
        }
        save_canary_status(status_payload)

        if canary_ok:
            logger.info(
                "Actor canary passed: coverage=%.3f threshold=%.3f pin_before=%s "
                "tested_latest=%s promotion_applied=%s",
                result.coverage,
                AMENITY_REQUIRED_COVERAGE,
                pin_before or "<none>",
                latest_build or "<unknown>",
                promotion_applied,
            )
        else:
            logger.warning(
                "Actor canary failed: coverage=%.3f threshold=%.3f request_health_good=%s "
                "pin_before=%s tested_latest=%s promotion_applied=%s failure_reason=%s",
                result.coverage,
                AMENITY_REQUIRED_COVERAGE,
                request_health_good,
                pin_before or "<none>",
                latest_build or "<unknown>",
                promotion_applied,
                result.failure_reason or "",
            )
        return status_payload
    except Exception as e:
        logger.exception("Actor canary failed with exception")
        status_payload = {
            "ok": False,
            "skipped": False,
            "failure_reason": str(e),
            "pin_before": pin_before,
            "pin_after": pin_before,
            "promotion_applied": False,
            "coverage": 0.0,
            "threshold": AMENITY_REQUIRED_COVERAGE,
            "target_count": len(sample_urls),
            "requests_succeeded": 0,
            "requests_failed": 0,
            "has_request_counts": False,
            "tested_latest": "latest",
            "run_summaries": [],
            "request_health_good": False,
        }
        save_canary_status(status_payload)
        return status_payload


@app.function(
    image=image,
    volumes={"/data": volume},
    secrets=secrets,
    memory=2048,
    cpu=2.0,
    timeout=1200,
    schedule=modal.Cron("0 16 * * *"),
)
async def actor_build_canary():
    """Run daily canary on actor latest build and auto-promote pin when healthy."""
    volume.reload()
    try:
        await _run_actor_build_canary_once()
    finally:
        volume.commit()


# --- 3. Agent Outreach Sender (On-Demand) ---

@app.function(
    image=image,
    volumes={"/data": volume},
    secrets=secrets,
    timeout=120,
)
async def send_agent_message(
    chat_id: int,
    draft_id: str,
):
    """Prepare an approved outreach message for the user to send manually."""
    from src.outreach import send_approved_draft
    from src.telegram_handler import TelegramBot

    bot = TelegramBot(token=os.environ["SE_TELEGRAM_BOT_TOKEN"])

    try:
        await send_approved_draft(bot, chat_id, draft_id)
    except Exception:
        logger.exception("Outreach send failed")
        await bot.send_text(
            chat_id,
            "⚠️ Failed to prepare the message. Please try again.",
        )
    finally:
        await bot.close()
        volume.commit()


# --- Setup Helper ---

@app.function(image=image, secrets=secrets)
def _setup_webhook(webhook_url: str):
    """Set up the Telegram webhook (runs on Modal with access to secrets)."""
    import httpx

    token = os.environ["SE_TELEGRAM_BOT_TOKEN"]
    print(f"Setting Telegram webhook to: {webhook_url}")

    webhook_payload = {"url": webhook_url}
    webhook_secret = os.environ.get("SE_WEBHOOK_SECRET", "")
    if webhook_secret:
        webhook_payload["secret_token"] = webhook_secret
        print("Including secret_token in webhook registration")

    resp = httpx.post(
        f"https://api.telegram.org/bot{token}/setWebhook",
        json=webhook_payload,
    )
    print(f"Response: {resp.json()}")


@app.local_entrypoint()
def setup():
    """Set up the Telegram webhook. Run after 'modal deploy modal_app.py'."""
    webhook_url = TelegramWebhook().webhook.web_url
    # modal run creates -dev URLs; strip the -dev suffix to use production URL
    webhook_url = webhook_url.replace("-dev.modal.run", ".modal.run")
    print(f"Resolved webhook URL: {webhook_url}")
    _setup_webhook.remote(webhook_url)
