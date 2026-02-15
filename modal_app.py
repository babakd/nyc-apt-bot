"""Modal App: 3 entry points — Telegram webhook, daily cron scanner, agent outreach sender."""

from __future__ import annotations

import json
import logging
import os

import modal

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# --- Modal App Setup ---

app = modal.App("streeteasy-bot")

# Container image from Dockerfile
image = modal.Image.from_dockerfile("Dockerfile.modal")

# Persistent volume for state
volume = modal.Volume.from_name("streeteasy-data", create_if_missing=True)

# Secrets
secrets = [
    modal.Secret.from_name("anthropic"),            # ANTHROPIC_API_KEY
    modal.Secret.from_name("streeteasy-telegram"),   # SE_TELEGRAM_BOT_TOKEN
    modal.Secret.from_name("apify"),                 # APIFY_API_TOKEN
]


# --- 1. Telegram Webhook Handler ---

@app.cls(
    image=image,
    volumes={"/data": volume},
    secrets=secrets,
    scaledown_window=300,
    memory=2048,
    cpu=2.0,
)
@modal.concurrent(max_inputs=10)
class TelegramWebhook:
    """Receives Telegram webhook POSTs, processes messages via conversation engine."""

    @modal.enter()
    def start(self):
        """Initialize scraper and bot when container boots."""
        from src.apify_scraper import ApifyScraper
        from src.scanner import scan_for_chat
        from src.telegram_handler import TelegramBot

        self.scraper = ApifyScraper()
        self.bot = TelegramBot(token=os.environ["SE_TELEGRAM_BOT_TOKEN"])

        # Set up scan callback so conversation engine can trigger scans
        async def do_scan(chat_id, state):
            await scan_for_chat(self.scraper, self.bot, state)

        self.bot.set_scan_callback(do_scan)
        logger.info("Webhook handler ready (Apify scraper)")

    @modal.exit()
    def stop(self):
        """Close bot when container shuts down."""
        import asyncio
        try:
            asyncio.get_event_loop().run_until_complete(self.bot.close())
        except Exception:
            logger.exception("Error during shutdown")

    @modal.fastapi_endpoint(method="POST")
    async def webhook(self, data: dict):
        """Receive Telegram webhook POST. Returns 200 immediately, processes async."""
        logger.info("Received webhook update: %s", json.dumps(data)[:200])

        # Process the update — conversation engine handles everything
        # including triggering scans via tool calls
        try:
            await self.bot.process_update(data)
        except Exception:
            logger.exception("Error processing webhook update")

        return {"ok": True}

    @modal.fastapi_endpoint(method="GET")
    async def health(self):
        """Health check endpoint."""
        return {"status": "ok"}


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

    try:
        await run_daily_scan(scraper, bot)
    except Exception:
        logger.exception("Daily scan failed")
    finally:
        await bot.close()
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

@app.local_entrypoint()
def setup():
    """Local helper to set up the Telegram webhook."""
    import httpx

    token = os.environ.get("SE_TELEGRAM_BOT_TOKEN")
    if not token:
        print("Set SE_TELEGRAM_BOT_TOKEN environment variable first")
        return

    # Get the webhook URL from Modal
    webhook_url = TelegramWebhook.webhook.web_url
    print(f"Setting Telegram webhook to: {webhook_url}")

    resp = httpx.post(
        f"https://api.telegram.org/bot{token}/setWebhook",
        json={"url": webhook_url},
    )
    print(f"Response: {resp.json()}")
