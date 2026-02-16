"""Telegram webhook processing, message routing, and Bot API calls."""

from __future__ import annotations

import json
import logging
from typing import Any

import httpx

from src.conversation import ConversationEngine, Response
from src.models import ChatState
from src.storage import load_state, save_state

logger = logging.getLogger(__name__)

TELEGRAM_API = "https://api.telegram.org/bot{token}"


class TelegramBot:
    """Handles Telegram Bot API interactions."""

    def __init__(self, token: str):
        self.token = token
        self.api_base = TELEGRAM_API.format(token=token)
        self.client = httpx.AsyncClient(timeout=30)
        self._scan_callback = None
        self._seen_update_ids: set[int] = set()
        self._bot_username: str | None = None

    def set_scan_callback(self, callback):
        """Set a callback to trigger apartment scans: async fn(chat_id, state)."""
        self._scan_callback = callback

    async def close(self) -> None:
        await self.client.aclose()

    # --- Incoming Update Processing ---

    async def process_update(self, update: dict[str, Any]) -> None:
        """Route an incoming Telegram update to the appropriate handler."""
        update_id = update.get("update_id")
        if update_id is not None:
            if update_id in self._seen_update_ids:
                logger.debug("Skipping duplicate update_id %s", update_id)
                return
            self._seen_update_ids.add(update_id)

        if "message" in update:
            await self._handle_message(update["message"])
        elif "callback_query" in update:
            await self._handle_callback_query(update["callback_query"])
        else:
            logger.debug("Ignoring update type: %s", list(update.keys()))

    async def _ensure_bot_username(self) -> str:
        """Fetch and cache the bot's username via getMe."""
        if self._bot_username is None:
            result = await self._api_call("getMe", {})
            if result and result.get("ok"):
                self._bot_username = result["result"].get("username", "")
            else:
                logger.warning("getMe failed, will retry on next call")
                return ""
        return self._bot_username

    @staticmethod
    def _extract_sender_name(message: dict[str, Any]) -> str | None:
        """Extract a display name from a message sender. Returns None for private chats."""
        chat_type = message.get("chat", {}).get("type", "private")
        if chat_type == "private":
            return None
        sender = message.get("from", {})
        first = sender.get("first_name", "")
        last = sender.get("last_name", "")
        name = f"{first} {last}".strip()
        return name or sender.get("username") or "Someone"

    def _is_group_chat(self, message: dict[str, Any]) -> bool:
        """Check if a message is from a group or supergroup chat."""
        chat_type = message.get("chat", {}).get("type", "private")
        return chat_type in ("group", "supergroup")

    async def _handle_message(self, message: dict[str, Any]) -> None:
        """Handle an incoming text message via the LLM conversation engine."""
        chat_id = message["chat"]["id"]
        is_group = self._is_group_chat(message)

        # Handle bot added to group
        new_members = message.get("new_chat_members", [])
        if new_members:
            bot_username = await self._ensure_bot_username()
            for member in new_members:
                if member.get("username") == bot_username:
                    state = load_state(chat_id)
                    state.is_group = True
                    save_state(state)
                    await self.send_text(
                        chat_id,
                        "👋 Hey everyone! I'm your apartment hunting assistant. "
                        "Mention me with @" + bot_username + " to chat. "
                        "Preferences and likes are shared across the group — "
                        "let's find your next place together!",
                    )
                    return

        text = message.get("text", "")
        if not text:
            return

        # In group chats, only respond to @mentions or replies to the bot
        sender_name: str | None = None
        if is_group:
            bot_username = await self._ensure_bot_username()

            is_mention = bot_username and f"@{bot_username}" in text
            is_reply_to_bot = False
            reply_msg = message.get("reply_to_message")
            if reply_msg and reply_msg.get("from", {}).get("username") == bot_username:
                is_reply_to_bot = True

            if not is_mention and not is_reply_to_bot:
                return  # Ignore group messages not directed at the bot

            # Strip the @mention from the text
            if bot_username and is_mention:
                text = text.replace(f"@{bot_username}", "").strip()
                if not text:
                    return

            sender_name = self._extract_sender_name(message)

        logger.info("Message from chat %s: %s", chat_id, text[:100])

        state = load_state(chat_id)
        if is_group and not state.is_group:
            state.is_group = True

        # Check for pending draft edit — route to revise_draft directly
        if state.pending_draft_edit:
            draft_id = state.pending_draft_edit
            state.pending_draft_edit = None
            save_state(state)
            try:
                from src.outreach import revise_draft
                await revise_draft(self, chat_id, draft_id, text)
            except Exception:
                logger.exception("Draft revision failed")
                await self.send_text(chat_id, "⚠️ Failed to revise the draft. Please try again.")
            return

        engine = ConversationEngine(state)
        result = await engine.handle_message(text, sender_name=sender_name)

        save_state(state)

        # Send text responses
        await self._send_responses(chat_id, result.responses)

        # Run scan inline — update_id dedup prevents Telegram retries from
        # causing duplicate messages, and Modal's allow_concurrent_inputs
        # lets other webhooks proceed while this one waits for Apify.
        if result.trigger_search and self._scan_callback:
            try:
                await self._scan_callback(chat_id, state)
            except Exception:
                logger.exception("Scan triggered by conversation failed")
                await self.send_text(
                    chat_id, "⚠️ Scan failed. Please try again later."
                )

        # Draft outreach if triggered by conversation
        if result.trigger_draft_listing_id:
            listing_id = result.trigger_draft_listing_id
            listing = (
                state.recent_listings.get(listing_id)
                or state.liked_listings.get(listing_id)
            )
            if listing:
                try:
                    from src.outreach import create_draft
                    await create_draft(self, chat_id, listing)
                except Exception:
                    logger.exception("Draft creation failed")
                    await self.send_text(
                        chat_id, "⚠️ Failed to create draft. Please try again."
                    )
            else:
                await self.send_text(
                    chat_id,
                    f"I don't have details for listing {listing_id}. "
                    "Try liking a listing from search results first, then ask me to draft a message.",
                )

    async def _handle_callback_query(self, callback: dict[str, Any]) -> None:
        """Handle an inline keyboard button press."""
        chat_id = callback["message"]["chat"]["id"]
        message_id = callback["message"]["message_id"]
        data = callback.get("data", "")
        callback_id = callback["id"]

        logger.info("Callback from chat %s: %s", chat_id, data)

        # Always answer the callback to remove loading state
        await self._answer_callback(callback_id)

        state = load_state(chat_id)

        # Listing action callbacks
        if data.startswith("like:"):
            listing_id = data.split(":", 1)[1]
            state.liked_listing_ids.add(listing_id)
            # Store full listing data if available
            if listing_id in state.recent_listings:
                state.liked_listings[listing_id] = state.recent_listings[listing_id]
            save_state(state)
            await self.send_text(chat_id, f"👍 Liked! I'll keep this in mind for your search.")
            return

        if data.startswith("pass:"):
            listing_id = data.split(":", 1)[1]
            save_state(state)
            await self.send_text(chat_id, "👎 Passed. Moving on!")
            return

        if data.startswith("details:"):
            listing_id = data.split(":", 1)[1]
            await self._send_listing_details(chat_id, listing_id)
            return

        # Draft action callbacks
        if data.startswith("draft_send:"):
            draft_id = data.split(":", 1)[1]
            await self._send_draft(chat_id, draft_id)
            return

        if data.startswith("draft_edit:"):
            draft_id = data.split(":", 1)[1]
            # Validate draft exists and is pending
            draft = state.active_drafts.get(draft_id)
            if not draft or draft.status != "pending":
                await self.send_text(chat_id, "This draft is no longer available for editing.")
                return
            state.pending_draft_edit = draft_id
            save_state(state)
            await self.send_text(chat_id, "✏️ What would you like to change? Send me your feedback.")
            return

        if data.startswith("draft_cancel:"):
            draft_id = data.split(":", 1)[1]
            if draft_id in state.active_drafts:
                state.active_drafts[draft_id].status = "cancelled"
                save_state(state)
            await self.send_text(chat_id, "❌ Draft cancelled.")
            return

        logger.warning("Unknown callback data: %s", data)

    # --- Outgoing Messages ---

    async def send_text(
        self,
        chat_id: int,
        text: str,
        keyboard: list[list[dict[str, str]]] | None = None,
        parse_mode: str = "HTML",
    ) -> dict | None:
        """Send a text message."""
        payload: dict[str, Any] = {
            "chat_id": chat_id,
            "text": text,
            "parse_mode": parse_mode,
            "disable_web_page_preview": True,
        }
        if keyboard is not None:
            payload["reply_markup"] = {"inline_keyboard": keyboard}
        return await self._api_call("sendMessage", payload)

    async def send_photo(
        self,
        chat_id: int,
        photo_url: str,
        caption: str | None = None,
        keyboard: list[list[dict[str, str]]] | None = None,
    ) -> dict | None:
        """Send a photo with optional caption and keyboard."""
        payload: dict[str, Any] = {
            "chat_id": chat_id,
            "photo": photo_url,
        }
        if caption:
            payload["caption"] = caption
            payload["parse_mode"] = "HTML"
        if keyboard is not None:
            payload["reply_markup"] = {"inline_keyboard": keyboard}
        return await self._api_call("sendPhoto", payload)

    async def send_photo_bytes(
        self,
        chat_id: int,
        photo_bytes: bytes,
        caption: str | None = None,
    ) -> dict | None:
        """Send a photo from bytes (e.g., screenshot)."""
        url = f"{self.api_base}/sendPhoto"
        data = {"chat_id": str(chat_id)}
        if caption:
            data["caption"] = caption
            data["parse_mode"] = "HTML"
        files = {"photo": ("screenshot.png", photo_bytes, "image/png")}
        try:
            resp = await self.client.post(url, data=data, files=files)
            result = resp.json()
            if not result.get("ok"):
                logger.error("sendPhoto failed: %s", result)
            return result
        except Exception:
            logger.exception("Failed to send photo bytes")
            return None

    async def send_listing_photo(
        self,
        chat_id: int,
        listing_url: str,
        photo_url: str,
        caption: str,
        keyboard: list[list[dict[str, str]]] | None = None,
    ) -> dict | None:
        """Send a listing photo with 4-step fallback chain.

        1. sendPhoto with direct URL (fastest)
        2. Download image ourselves and re-upload as bytes
        3. sendMessage with link_preview_options (OG image from listing page)
        4. Plain text fallback
        """
        # Step 1: Try sending the photo URL directly
        result = await self.send_photo(chat_id, photo_url, caption, keyboard)
        if result and result.get("ok"):
            return result
        logger.warning("Direct photo URL failed for %s, trying download", photo_url)

        # Step 2: Download and re-upload with proper headers
        try:
            resp = await self.client.get(
                photo_url,
                timeout=15,
                headers={"Referer": "https://streeteasy.com/"},
            )
            if resp.status_code == 200 and len(resp.content) > 1000:
                url = f"{self.api_base}/sendPhoto"
                data: dict[str, Any] = {"chat_id": str(chat_id)}
                if caption:
                    data["caption"] = caption
                    data["parse_mode"] = "HTML"
                if keyboard is not None:
                    data["reply_markup"] = json.dumps({"inline_keyboard": keyboard})
                files = {"photo": ("listing.jpg", resp.content, "image/jpeg")}
                try:
                    upload_resp = await self.client.post(url, data=data, files=files)
                    upload_result = upload_resp.json()
                    if upload_result.get("ok"):
                        return upload_result
                except Exception:
                    logger.exception("Photo upload failed")
        except Exception:
            logger.warning("Photo download failed for %s", photo_url)

        # Step 3: sendMessage with link preview (OG image from listing page)
        if listing_url:
            payload: dict[str, Any] = {
                "chat_id": chat_id,
                "text": caption,
                "parse_mode": "HTML",
                "link_preview_options": {
                    "url": listing_url,
                    "prefer_large_media": True,
                    "show_above_text": True,
                },
            }
            if keyboard is not None:
                payload["reply_markup"] = {"inline_keyboard": keyboard}
            result = await self._api_call("sendMessage", payload)
            if result and result.get("ok"):
                return result
            logger.warning("Link preview fallback failed for %s", listing_url)

        # Step 4: Plain text fallback
        return await self.send_text(chat_id, caption, keyboard)

    async def edit_message_keyboard(
        self,
        chat_id: int,
        message_id: int,
        keyboard: list[list[dict[str, str]]] | None = None,
    ) -> None:
        """Edit an existing message's inline keyboard."""
        payload: dict[str, Any] = {
            "chat_id": chat_id,
            "message_id": message_id,
        }
        if keyboard is not None:
            payload["reply_markup"] = {"inline_keyboard": keyboard}
        else:
            payload["reply_markup"] = {"inline_keyboard": []}
        await self._api_call("editMessageReplyMarkup", payload)

    # --- Webhook Setup ---

    async def set_webhook(self, url: str) -> dict | None:
        """Set the Telegram webhook URL."""
        return await self._api_call("setWebhook", {"url": url})

    async def delete_webhook(self) -> dict | None:
        """Remove the webhook."""
        return await self._api_call("deleteWebhook", {})

    # --- Internal Helpers ---

    async def _api_call(self, method: str, payload: dict[str, Any]) -> dict | None:
        """Make a Telegram Bot API call."""
        url = f"{self.api_base}/{method}"
        try:
            resp = await self.client.post(url, json=payload)
            result = resp.json()
            if not result.get("ok"):
                logger.error("Telegram API %s failed: %s", method, result)
            return result
        except Exception:
            logger.exception("Telegram API call failed: %s", method)
            return None

    async def _answer_callback(self, callback_id: str, text: str | None = None) -> None:
        """Answer a callback query to remove loading indicator."""
        payload: dict[str, Any] = {"callback_query_id": callback_id}
        if text:
            payload["text"] = text
        await self._api_call("answerCallbackQuery", payload)

    async def _send_responses(self, chat_id: int, responses: list[Response]) -> None:
        """Send a list of Response objects to a chat."""
        for resp in responses:
            if resp.photo_url:
                await self.send_photo(chat_id, resp.photo_url, resp.text, resp.keyboard)
            else:
                await self.send_text(chat_id, resp.text, resp.keyboard)

    async def _send_listing_details(self, chat_id: int, listing_id: str) -> None:
        """Fetch and send detailed listing info via Telegraph Instant View."""
        state = load_state(chat_id)
        listing = (
            state.recent_listings.get(listing_id)
            or state.liked_listings.get(listing_id)
        )
        if not listing:
            await self.send_text(
                chat_id,
                f'Listing details not available.\n'
                f'<a href="https://streeteasy.com/rental/{listing_id}">View on StreetEasy</a>',
            )
            return

        # Try to create a Telegraph page for Instant View
        try:
            from src.telegraph_pages import create_listing_page
            page_url = await create_listing_page(listing)
        except Exception:
            logger.exception("Telegraph page creation failed")
            page_url = None

        if page_url:
            await self.send_text(
                chat_id,
                f'<b>{listing.address}</b>\n\n'
                f'<a href="{page_url}">View Full Details (Instant View)</a>\n'
                f'<a href="{listing.url}">View on StreetEasy</a>',
            )
        else:
            # Fallback: send detailed text
            from src.formatter import format_listing_detail
            detail_text = format_listing_detail(listing)
            await self.send_text(chat_id, detail_text)

    async def _send_draft(self, chat_id: int, draft_id: str) -> None:
        """Send an approved draft — provides copy-paste message + StreetEasy link."""
        try:
            from src.outreach import send_approved_draft
            await send_approved_draft(self, chat_id, draft_id)
        except Exception:
            logger.exception("Failed to send draft %s", draft_id)
            await self.send_text(chat_id, "⚠️ Failed to prepare the message. Please try again.")
