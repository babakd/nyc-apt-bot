"""Agent outreach: draft, review, and send messages via Claude API."""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone

from src.claude_client import ClaudeClient
from src.formatter import draft_keyboard, format_draft_message
from src.models import ChatState, Draft, Listing
from src.storage import load_state, save_state

logger = logging.getLogger(__name__)


def _escape_html(text: str) -> str:
    """Escape HTML special characters for Telegram."""
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


# Default message template used as a fallback
DEFAULT_TEMPLATE = (
    "Hi, I'm interested in the {bedrooms}-bedroom apartment at {address} "
    "listed for ${price:,}/month. I'd love to schedule a viewing at your "
    "earliest convenience. Could you let me know available times? Thank you!"
)

DRAFT_SYSTEM_PROMPT = """\
You are helping a user draft a personalized message to a real estate agent \
about an apartment listing on StreetEasy. Write a brief, professional, and \
friendly message that expresses genuine interest. Keep it concise (3-5 sentences). \
Do not use markdown. Just return the message text, nothing else."""

REVISE_SYSTEM_PROMPT = """\
You are helping a user revise their message to a real estate agent. \
Apply the user's feedback to improve the message. Keep it concise (3-5 sentences). \
Do not use markdown. Just return the revised message text, nothing else."""


async def create_draft(
    telegram_bot: "TelegramBot",
    chat_id: int,
    listing: Listing,
    claude: ClaudeClient | None = None,
) -> None:
    """Draft a personalized outreach message and send it for user review."""
    state = load_state(chat_id)
    claude = claude or ClaudeClient()

    # Generate a personalized draft via Claude API
    try:
        prompt = (
            f"Write a message to the listing agent for this apartment:\n"
            f"Address: {listing.address}\n"
            f"Neighborhood: {listing.neighborhood}\n"
            f"Price: ${listing.price:,}/month\n"
            f"Bedrooms: {listing.bedrooms}\n"
            f"Bathrooms: {listing.bathrooms}\n"
        )
        if listing.amenities:
            prompt += f"Amenities: {', '.join(listing.amenities)}\n"

        prefs = state.preferences
        prompt += f"\nUser's preferences:\n"
        if prefs.budget_max:
            prompt += f"Budget: up to ${prefs.budget_max:,}/month\n"
        if prefs.must_haves:
            prompt += f"Must-haves: {', '.join(prefs.must_haves)}\n"

        # Add current apartment context for personalization
        apt = state.current_apartment
        if apt:
            prompt += f"\nUser's current situation:\n"
            if apt.address:
                prompt += f"Currently at: {apt.address}\n"
            if apt.price:
                prompt += f"Current rent: ${apt.price:,}/month\n"
            if apt.cons:
                prompt += f"Looking to improve: {', '.join(apt.cons)}\n"

        message_text = await claude.chat(
            system=DRAFT_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
            tools=[],
            tool_handler=_noop_tool_handler,
        )
    except Exception:
        logger.exception("Failed to generate draft via Claude API, using template")
        message_text = ""

    if not message_text:
        message_text = DEFAULT_TEMPLATE.format(
            bedrooms=listing.bedrooms,
            address=listing.address,
            price=listing.price,
        )

    # Create draft record
    draft = Draft(
        draft_id=str(uuid.uuid4())[:8],
        listing_id=listing.listing_id,
        message_text=message_text,
        status="pending",
        created_at=datetime.now(timezone.utc),
    )
    state.active_drafts[draft.draft_id] = draft
    save_state(state)

    # Send draft for review
    formatted = format_draft_message(draft.message_text, listing)
    keyboard = draft_keyboard(draft.draft_id)
    await telegram_bot.send_text(chat_id, formatted, keyboard=keyboard)


async def revise_draft(
    telegram_bot: "TelegramBot",
    chat_id: int,
    draft_id: str,
    feedback: str,
    claude: ClaudeClient | None = None,
) -> None:
    """Revise a draft based on user feedback."""
    state = load_state(chat_id)
    draft = state.active_drafts.get(draft_id)
    if not draft:
        await telegram_bot.send_text(chat_id, "Draft not found. Please start over.")
        return

    claude = claude or ClaudeClient()

    # Revise via Claude API
    try:
        prompt = (
            f"Original message:\n{draft.message_text}\n\n"
            f"User feedback:\n{feedback}\n\n"
            f"Please revise the message based on the feedback."
        )
        revised = await claude.chat(
            system=REVISE_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
            tools=[],
            tool_handler=_noop_tool_handler,
        )
    except Exception:
        logger.exception("Failed to revise draft via Claude API")
        await telegram_bot.send_text(
            chat_id,
            "I had trouble revising. Please type out your preferred message directly.",
        )
        return

    # Update draft
    draft.message_text = revised
    save_state(state)

    # Re-send for review — look up actual listing if available
    listing = (
        state.recent_listings.get(draft.listing_id)
        or state.liked_listings.get(draft.listing_id)
    )
    if not listing:
        listing = Listing(
            listing_id=draft.listing_id,
            url=f"https://streeteasy.com/rental/{draft.listing_id}",
            address=f"Listing {draft.listing_id}",
            neighborhood="",
            price=0,
            bedrooms=0,
            bathrooms=0,
        )
    formatted = format_draft_message(draft.message_text, listing)
    keyboard = draft_keyboard(draft.draft_id)
    await telegram_bot.send_text(chat_id, formatted, keyboard=keyboard)


async def send_approved_draft(
    telegram_bot: "TelegramBot",
    chat_id: int,
    draft_id: str,
) -> None:
    """Mark a draft as sent and give the user their message + listing link to paste manually."""
    state = load_state(chat_id)
    draft = state.active_drafts.get(draft_id)
    if not draft:
        await telegram_bot.send_text(chat_id, "Draft not found.")
        return

    if draft.status != "pending":
        await telegram_bot.send_text(chat_id, f"This draft is already {draft.status}.")
        return

    # Look up the actual listing for a proper URL
    listing = (
        state.recent_listings.get(draft.listing_id)
        or state.liked_listings.get(draft.listing_id)
    )
    listing_url = listing.url if listing else f"https://streeteasy.com/rental/{draft.listing_id}"
    listing_address = listing.address if listing else f"Listing {draft.listing_id}"

    draft.status = "sent"
    draft.sent_at = datetime.now(timezone.utc)
    save_state(state)

    await telegram_bot.send_text(
        chat_id,
        f"✅ <b>Message ready for {_escape_html(listing_address)}</b>\n\n"
        f"<b>Your message:</b>\n"
        f"<pre>{_escape_html(draft.message_text)}</pre>\n\n"
        f"Open the listing and paste into the contact form:\n"
        f'<a href="{listing_url}">Contact Agent on StreetEasy →</a>\n\n'
        f"Tap the message above to copy it.",
    )


async def _noop_tool_handler(name: str, input_data: dict) -> str:
    """No-op tool handler for Claude calls that don't use tools."""
    return "Tool not available."
