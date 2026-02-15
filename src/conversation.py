"""LLM-powered conversation engine using Claude API with tool_use."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any

from src.claude_client import ChatResult, ClaudeClient
from src.formatter import format_preferences_summary
from src.models import ChatState, ConversationTurn, CurrentApartment, Preferences

logger = logging.getLogger(__name__)

MAX_HISTORY_TURNS = 30

SYSTEM_PROMPT = """\
You are a friendly, knowledgeable NYC apartment hunting assistant on Telegram. \
Your job is to help users find their perfect rental apartment on StreetEasy.

You need to learn the user's preferences through natural conversation. The key info you need:
- Monthly budget (min and/or max)
- Number of bedrooms (0 = studio)
- Minimum number of bathrooms (optional)
- Preferred neighborhoods in NYC
- Commute destination and max commute time (optional)
- Must-have amenities like dishwasher, laundry, doorman, elevator, etc. (optional)
- Nice-to-have amenities — things they'd prefer but aren't dealbreakers (optional)
- Whether they want no-fee only (optional)
- Move-in date (optional)

Guidelines:
- Be conversational and natural. Don't ask rigid questions — flow with the conversation.
- Extract preferences from natural language. "about 3 grand" = budget_max of 3000. \
"studio or one bedroom in the village" = bedrooms [0, 1] and neighborhoods including Village areas.
- When the user mentions neighborhoods, map them to standard NYC neighborhood names. \
For example: "the village" could mean West Village, East Village, or Greenwich Village — ask if unclear.
- Use the update_preferences tool whenever you extract new preference info from the user's message.
- Call show_preferences when the user wants to see their current preferences.
- Call search_apartments when the user wants to search or is ready to find apartments.
- Call mark_ready when the user has confirmed their preferences look good and they're ready for daily scans.
- You can update preferences at any time — users can change their mind.
- Keep responses concise (2-4 sentences usually). This is a chat, not an essay.
- Use plain text, not markdown. Telegram doesn't render markdown in bot messages.
- If the user seems done setting preferences, summarize what you have and ask if they'd like to search.

IMPORTANT — Tool usage rules:
- You MUST use the search_apartments tool to run a search. Never say "searching" or \
describe results without actually calling the tool.
- You MUST use the appropriate tool for any action. Never simulate tool output in text.
- If a user asks to search, search again, or find apartments, always call search_apartments.
- If a search fails or returns no results, say so honestly — do not fabricate listings.
- Every action (search, update preferences, clear history, etc.) must go through its tool call.

IMPORTANT — Constraint context:
When updating preferences, also set the constraint_context field to summarize what's firm \
(dealbreakers) vs flexible (nice-to-haves) based on how the user talks about their criteria. \
Pay attention to language like "must have", "absolutely need", "not a dollar over", "deal breaker" \
(these are hard constraints) vs "ideally", "would be nice", "prefer", "if possible" \
(these are soft preferences). Update constraint_context whenever the user expresses firmness \
or flexibility about any criterion. This context is used by the scoring system to decide \
what should disqualify a listing vs just lower its score.

Current user preferences:
{preferences_context}
Preferences confirmed: {preferences_ready}
{constraint_context}\
{current_apartment_context}\
{liked_count_context}\
"""

TOOLS = [
    {
        "name": "update_preferences",
        "description": (
            "Update the user's apartment search preferences. Only include fields the user "
            "has mentioned or changed. All parameters are optional — only pass what's new."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "budget_min": {
                    "type": "integer",
                    "description": "Minimum monthly rent in dollars",
                },
                "budget_max": {
                    "type": "integer",
                    "description": "Maximum monthly rent in dollars",
                },
                "bedrooms": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "description": "Acceptable bedroom counts. 0 = studio.",
                },
                "neighborhoods": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "NYC neighborhood names (e.g. 'East Village', 'Chelsea', 'Park Slope')",
                },
                "commute_address": {
                    "type": "string",
                    "description": "Where the user commutes to (address or area name)",
                },
                "commute_max_minutes": {
                    "type": "integer",
                    "description": "Maximum acceptable commute time in minutes",
                },
                "min_bathrooms": {
                    "type": "integer",
                    "description": "Minimum number of bathrooms required",
                },
                "must_haves": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Required amenities (e.g. 'Dishwasher', 'Laundry in Unit', 'Doorman')",
                },
                "nice_to_haves": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Preferred but not required amenities (e.g. 'Gym', 'Roof Deck', 'Pool')",
                },
                "no_fee_only": {
                    "type": "boolean",
                    "description": "Whether to only show no-fee listings",
                },
                "move_in_date": {
                    "type": "string",
                    "description": "Desired move-in date",
                },
                "constraint_context": {
                    "type": "string",
                    "description": (
                        "Natural language summary of what's firm vs flexible in the user's "
                        "preferences. Capture dealbreakers vs nice-to-haves based on how the "
                        "user talks about their criteria. Example: 'Budget $3,500 is a firm max. "
                        "2BR needed for kids (non-negotiable). East Village preferred but open to "
                        "nearby. Dishwasher non-negotiable. Gym nice but not critical.'"
                    ),
                },
            },
            "required": [],
        },
    },
    {
        "name": "show_preferences",
        "description": (
            "Show the user their current saved search preferences. Call this tool whenever "
            "the user asks to see, review, or confirm their preferences."
        ),
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    {
        "name": "search_apartments",
        "description": (
            "Trigger a StreetEasy apartment search based on the user's current preferences. "
            "This is the ONLY way to search for apartments — you must call this tool whenever "
            "the user asks to search, look for apartments, try again, or see what's available. "
            "Requires at least a budget or neighborhood preference. "
            "Results will be sent as listing cards with photos directly to the chat."
        ),
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    {
        "name": "mark_ready",
        "description": (
            "Mark the user's preferences as confirmed and ready for daily apartment scans. "
            "Call this when the user confirms their preferences look good."
        ),
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    {
        "name": "clear_search_history",
        "description": (
            "Clear the list of previously seen listing IDs so the next search returns all "
            "matching listings, including ones shown before. Use when the user asks to "
            "start fresh, reset their search, clear history, or see all listings again."
        ),
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    {
        "name": "pause_daily_scans",
        "description": (
            "Pause daily apartment scans. The user's preferences are kept but they won't "
            "receive daily notifications until they re-enable scans with mark_ready."
        ),
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    {
        "name": "get_liked_listings",
        "description": (
            "Get the user's liked/saved listings with full details. Call this tool whenever "
            "the user asks to see their liked, saved, or favorited listings."
        ),
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    {
        "name": "remove_liked_listing",
        "description": "Remove a listing from the user's liked/saved list.",
        "input_schema": {
            "type": "object",
            "properties": {
                "listing_id": {
                    "type": "string",
                    "description": "The listing ID to remove from liked list",
                },
            },
            "required": ["listing_id"],
        },
    },
    {
        "name": "reset_preferences",
        "description": (
            "Reset all search preferences to defaults. Clears budget, bedrooms, neighborhoods, "
            "amenities, and all other preferences. Also pauses daily scans. Use when the user "
            "wants to start over with a completely new search."
        ),
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    {
        "name": "remove_neighborhoods",
        "description": (
            "Remove specific neighborhoods from the user's preferred neighborhoods list "
            "without affecting the rest. Use when the user says to drop or remove certain areas."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "neighborhoods": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Neighborhood names to remove",
                },
            },
            "required": ["neighborhoods"],
        },
    },
    {
        "name": "get_listing_details",
        "description": (
            "Get full details for a specific listing by ID. Returns address, price, "
            "bedrooms, bathrooms, neighborhood, amenities, match score, and StreetEasy link. "
            "Works for recently shown listings and liked listings."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "listing_id": {
                    "type": "string",
                    "description": "The listing ID to look up",
                },
            },
            "required": ["listing_id"],
        },
    },
    {
        "name": "compare_listings",
        "description": (
            "Compare two or more listings side by side. Returns a comparison of price, "
            "bedrooms, neighborhood, score, and other details. Use when the user wants to "
            "decide between listings."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "listing_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Listing IDs to compare (2-5 listings)",
                },
            },
            "required": ["listing_ids"],
        },
    },
    {
        "name": "draft_outreach",
        "description": (
            "Draft an outreach message to the listing agent for a specific listing. "
            "The draft will be generated and shown to the user for review before sending. "
            "Only works for listings the user has seen or liked."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "listing_id": {
                    "type": "string",
                    "description": "The listing ID to draft a message for",
                },
            },
            "required": ["listing_id"],
        },
    },
    {
        "name": "update_current_apartment",
        "description": (
            "Store or update info about the user's current apartment. Helps compare "
            "new listings against their current situation. All fields are optional — "
            "only pass what the user mentions."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "address": {
                    "type": "string",
                    "description": "Current apartment address",
                },
                "neighborhood": {
                    "type": "string",
                    "description": "Current neighborhood",
                },
                "price": {
                    "type": "integer",
                    "description": "Current monthly rent in dollars",
                },
                "bedrooms": {
                    "type": "integer",
                    "description": "Number of bedrooms (0 = studio)",
                },
                "move_out_date": {
                    "type": "string",
                    "description": "When they plan to move out or lease ends",
                },
                "pros": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Things they like about their current place",
                },
                "cons": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Things they dislike about their current place",
                },
                "notes": {
                    "type": "string",
                    "description": "Any other notes about their current apartment",
                },
            },
            "required": [],
        },
    },
    {
        "name": "show_current_apartment",
        "description": (
            "Show the user's saved current apartment info. Call this tool whenever the user "
            "asks about their current apartment or living situation."
        ),
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
]


@dataclass
class Response:
    """A response to send back to Telegram."""
    text: str
    keyboard: list[list[dict[str, str]]] | None = None
    photo_url: str | None = None


@dataclass
class ConversationResult:
    """Result from handling a message, includes response and any triggered actions."""
    responses: list[Response] = field(default_factory=list)
    trigger_search: bool = False
    trigger_draft_listing_id: str | None = None


class ConversationEngine:
    """Manages conversation via Claude API with tool_use."""

    def __init__(self, state: ChatState, claude: ClaudeClient | None = None):
        self.state = state
        self.claude = claude or ClaudeClient()

    async def handle_message(self, text: str, sender_name: str | None = None) -> ConversationResult:
        """Handle a user message by sending it to Claude with tools.

        Returns a ConversationResult with text responses and any triggered actions.
        """
        # In group chats, prefix message with sender name so Claude sees who said what
        history_text = f"[{sender_name}]: {text}" if sender_name else text

        # Add user message to history
        self.state.conversation_history.append(
            ConversationTurn(role="user", content=history_text, sender_name=sender_name)
        )

        # Build Claude messages from conversation history
        messages = self._build_messages()

        # Build system prompt with current preferences
        system = self._build_system_prompt()

        result = ConversationResult()

        # Track tool calls during this interaction
        async def tool_handler(name: str, input_data: dict[str, Any]) -> str:
            return self._execute_tool(name, input_data, result)

        try:
            chat_result = await self.claude.chat(
                system=system,
                messages=messages,
                tools=TOOLS,
                tool_handler=tool_handler,
            )
            response_text = chat_result.text

            # Store intermediate tool messages (tool_use + tool_result turns)
            for msg in chat_result.tool_messages:
                self.state.conversation_history.append(
                    ConversationTurn(role=msg["role"], content=msg["content"])
                )
        except Exception:
            logger.exception("Claude API call failed")
            response_text = "I'm having a bit of trouble right now. Could you try again in a moment?"

        # Add final assistant text response to history
        self.state.conversation_history.append(
            ConversationTurn(role="assistant", content=response_text)
        )

        # Trim history with boundary awareness
        self._trim_history()

        result.responses.append(Response(text=response_text))
        return result

    def _build_messages(self) -> list[dict[str, Any]]:
        """Build Claude API messages from conversation history.

        Content can be str (text) or list[dict] (structured tool_use/tool_result blocks).
        The Claude API accepts both formats.
        """
        messages = []
        for turn in self.state.conversation_history:
            messages.append({"role": turn.role, "content": turn.content})
        return messages

    def _trim_history(self) -> None:
        """Trim conversation history with boundary awareness.

        After slicing to MAX_HISTORY_TURNS, ensures the first message is a user text
        message (not an orphaned tool_result or assistant tool_use mid-exchange).
        """
        history = self.state.conversation_history
        if len(history) <= MAX_HISTORY_TURNS:
            return
        trimmed = history[-MAX_HISTORY_TURNS:]
        # Drop leading messages until we find a user text message
        while trimmed and not (trimmed[0].role == "user" and isinstance(trimmed[0].content, str)):
            trimmed.pop(0)
        self.state.conversation_history = trimmed

    def _build_system_prompt(self) -> str:
        """Build system prompt with current preferences context."""
        prefs = self.state.preferences
        if any([prefs.budget_max, prefs.bedrooms, prefs.neighborhoods, prefs.min_bathrooms, prefs.must_haves]):
            prefs_context = format_preferences_summary(prefs)
        else:
            prefs_context = "No preferences set yet."

        # Constraint context
        if prefs.constraint_context:
            constraint_context = f"\nConstraint context: {prefs.constraint_context}\n"
        else:
            constraint_context = ""

        # Current apartment context
        apt = self.state.current_apartment
        if apt:
            apt_parts = ["\nCurrent apartment:"]
            if apt.address:
                apt_parts.append(f"  Address: {apt.address}")
            if apt.neighborhood:
                apt_parts.append(f"  Neighborhood: {apt.neighborhood}")
            if apt.price:
                apt_parts.append(f"  Rent: ${apt.price:,}/mo")
            if apt.bedrooms is not None:
                apt_parts.append(f"  Bedrooms: {apt.bedrooms}")
            if apt.move_out_date:
                apt_parts.append(f"  Moving out: {apt.move_out_date}")
            if apt.pros:
                apt_parts.append(f"  Likes: {', '.join(apt.pros)}")
            if apt.cons:
                apt_parts.append(f"  Dislikes: {', '.join(apt.cons)}")
            if apt.notes:
                apt_parts.append(f"  Notes: {apt.notes}")
            current_apt_context = "\n".join(apt_parts) + "\n"
        else:
            current_apt_context = ""

        # Liked listings count
        liked_count = len(self.state.liked_listing_ids)
        liked_context = f"\nLiked listings: {liked_count}\n" if liked_count else ""

        prompt = SYSTEM_PROMPT.format(
            preferences_context=prefs_context,
            preferences_ready=self.state.preferences_ready,
            constraint_context=constraint_context,
            current_apartment_context=current_apt_context,
            liked_count_context=liked_context,
        )

        if self.state.is_group:
            prompt += (
                "\n\nGROUP CHAT CONTEXT:\n"
                "This is a group chat (e.g. roommates searching together). "
                "Messages from group members are prefixed with [Name]. "
                "Preferences and liked listings are shared across the group.\n"
                "- Track which group member said what across the conversation history.\n"
                "- Help the group find consensus when preferences conflict.\n"
                "- Address people by name when relevant.\n"
            )

        return prompt

    def _execute_tool(self, name: str, input_data: dict[str, Any], result: ConversationResult) -> str:
        """Execute a tool call and return the result as a string."""
        if name == "update_preferences":
            return self._tool_update_preferences(input_data)
        elif name == "show_preferences":
            return self._tool_show_preferences()
        elif name == "search_apartments":
            return self._tool_search_apartments(result)
        elif name == "mark_ready":
            return self._tool_mark_ready()
        elif name == "clear_search_history":
            return self._tool_clear_search_history()
        elif name == "pause_daily_scans":
            return self._tool_pause_daily_scans()
        elif name == "get_liked_listings":
            return self._tool_get_liked_listings()
        elif name == "remove_liked_listing":
            return self._tool_remove_liked_listing(input_data)
        elif name == "reset_preferences":
            return self._tool_reset_preferences()
        elif name == "remove_neighborhoods":
            return self._tool_remove_neighborhoods(input_data)
        elif name == "get_listing_details":
            return self._tool_get_listing_details(input_data)
        elif name == "compare_listings":
            return self._tool_compare_listings(input_data)
        elif name == "draft_outreach":
            return self._tool_draft_outreach(input_data, result)
        elif name == "update_current_apartment":
            return self._tool_update_current_apartment(input_data)
        elif name == "show_current_apartment":
            return self._tool_show_current_apartment()
        else:
            return f"Unknown tool: {name}"

    def _tool_update_preferences(self, input_data: dict[str, Any]) -> str:
        """Update user preferences from tool call input."""
        prefs = self.state.preferences
        updated_fields = []

        if "budget_min" in input_data:
            prefs.budget_min = input_data["budget_min"]
            updated_fields.append(f"budget_min=${prefs.budget_min:,}")

        if "budget_max" in input_data:
            prefs.budget_max = input_data["budget_max"]
            updated_fields.append(f"budget_max=${prefs.budget_max:,}")

        if "bedrooms" in input_data:
            prefs.bedrooms = input_data["bedrooms"]
            beds_str = ", ".join("Studio" if b == 0 else f"{b}BR" for b in prefs.bedrooms)
            updated_fields.append(f"bedrooms=[{beds_str}]")

        if "neighborhoods" in input_data:
            prefs.neighborhoods = input_data["neighborhoods"]
            updated_fields.append(f"neighborhoods={prefs.neighborhoods}")

        if "commute_address" in input_data:
            prefs.commute_address = input_data["commute_address"]
            updated_fields.append(f"commute_address={prefs.commute_address}")

        if "commute_max_minutes" in input_data:
            prefs.commute_max_minutes = input_data["commute_max_minutes"]
            updated_fields.append(f"commute_max_minutes={prefs.commute_max_minutes}")

        if "min_bathrooms" in input_data:
            prefs.min_bathrooms = input_data["min_bathrooms"]
            updated_fields.append(f"min_bathrooms={prefs.min_bathrooms}")

        if "must_haves" in input_data:
            prefs.must_haves = input_data["must_haves"]
            updated_fields.append(f"must_haves={prefs.must_haves}")

        if "nice_to_haves" in input_data:
            prefs.nice_to_haves = input_data["nice_to_haves"]
            updated_fields.append(f"nice_to_haves={prefs.nice_to_haves}")

        if "no_fee_only" in input_data:
            prefs.no_fee_only = input_data["no_fee_only"]
            updated_fields.append(f"no_fee_only={prefs.no_fee_only}")

        if "move_in_date" in input_data:
            prefs.move_in_date = input_data["move_in_date"]
            updated_fields.append(f"move_in_date={prefs.move_in_date}")

        if "constraint_context" in input_data:
            prefs.constraint_context = input_data["constraint_context"]
            updated_fields.append("constraint_context updated")

        return f"Updated: {', '.join(updated_fields)}" if updated_fields else "No changes made."

    def _tool_show_preferences(self) -> str:
        """Return formatted preferences summary."""
        return format_preferences_summary(self.state.preferences)

    def _tool_search_apartments(self, result: ConversationResult) -> str:
        """Signal that a search should be triggered."""
        if not self.state.preferences.neighborhoods and not self.state.preferences.budget_max:
            return "Cannot search yet — need at least a budget or neighborhoods to search."
        result.trigger_search = True
        return "Search triggered. Results will be sent shortly."

    def _tool_mark_ready(self) -> str:
        """Mark preferences as confirmed for daily scans."""
        self.state.preferences_ready = True
        return "Preferences confirmed! Daily scans are now active."

    def _tool_clear_search_history(self) -> str:
        """Clear seen listing IDs so next search returns all matches."""
        count = len(self.state.seen_listing_ids)
        self.state.seen_listing_ids.clear()
        return f"Cleared {count} previously seen listings. Next search will show all matches."

    def _tool_pause_daily_scans(self) -> str:
        """Pause daily scans by unsetting preferences_ready."""
        if not self.state.preferences_ready:
            return "Daily scans are already paused."
        self.state.preferences_ready = False
        return "Daily scans paused. Your preferences are saved. Use mark_ready to resume."

    def _tool_get_liked_listings(self) -> str:
        """Return liked listings with full details if available."""
        if not self.state.liked_listing_ids:
            return "No liked listings yet."
        lines = []
        for lid in sorted(self.state.liked_listing_ids):
            listing = self.state.liked_listings.get(lid)
            if listing:
                score = f" (match: {listing.match_score}/100)" if listing.match_score is not None else ""
                fee = " | No fee" if not listing.broker_fee else ""
                lines.append(
                    f"- {listing.address} — ${listing.price:,}/mo, "
                    f"{listing.bedrooms}BR, {listing.neighborhood}{fee}{score}\n"
                    f"  {listing.url or f'https://streeteasy.com/rental/{lid}'}"
                )
            else:
                lines.append(f"- Listing {lid}: https://streeteasy.com/rental/{lid}")
        return f"{len(lines)} liked listing(s):\n" + "\n".join(lines)

    def _tool_remove_liked_listing(self, input_data: dict[str, Any]) -> str:
        """Remove a listing from liked lists."""
        listing_id = input_data.get("listing_id", "")
        if listing_id in self.state.liked_listing_ids:
            self.state.liked_listing_ids.discard(listing_id)
            self.state.liked_listings.pop(listing_id, None)
            return f"Removed listing {listing_id} from your liked list."
        return f"Listing {listing_id} is not in your liked list."

    def _tool_reset_preferences(self) -> str:
        """Reset all preferences to defaults."""
        self.state.preferences = Preferences()
        self.state.preferences_ready = False
        return "All preferences have been reset. Daily scans are paused. Let's start fresh!"

    def _tool_remove_neighborhoods(self, input_data: dict[str, Any]) -> str:
        """Remove specific neighborhoods from the list."""
        to_remove = input_data.get("neighborhoods", [])
        if not to_remove:
            return "No neighborhoods specified to remove."
        before = set(self.state.preferences.neighborhoods)
        # Case-insensitive matching
        remove_lower = {n.lower() for n in to_remove}
        remaining = [n for n in self.state.preferences.neighborhoods if n.lower() not in remove_lower]
        removed = before - set(remaining)
        self.state.preferences.neighborhoods = remaining
        if removed:
            return f"Removed: {', '.join(sorted(removed))}. Remaining: {remaining or 'none'}."
        return f"None of {to_remove} were in your neighborhoods list."

    def _tool_get_listing_details(self, input_data: dict[str, Any]) -> str:
        """Look up full details for a listing."""
        listing_id = input_data.get("listing_id", "")
        listing = (
            self.state.recent_listings.get(listing_id)
            or self.state.liked_listings.get(listing_id)
        )
        if not listing:
            return (
                f"Listing {listing_id} not found in recent or liked listings. "
                f"Link: https://streeteasy.com/rental/{listing_id}"
            )
        parts = [
            f"Address: {listing.address}",
            f"Neighborhood: {listing.neighborhood}",
            f"Price: ${listing.price:,}/mo",
            f"Bedrooms: {listing.bedrooms}",
            f"Bathrooms: {listing.bathrooms}",
        ]
        if listing.sqft:
            parts.append(f"Size: {listing.sqft:,} sqft")
        if listing.broker_fee:
            parts.append(f"Fee: {listing.broker_fee}")
        else:
            parts.append("Fee: No fee")
        if listing.available_date:
            parts.append(f"Available: {listing.available_date}")
        if listing.amenities:
            parts.append(f"Amenities: {', '.join(listing.amenities)}")
        if listing.match_score is not None:
            parts.append(f"Match score: {listing.match_score}/100")
        parts.append(f"URL: {listing.url or f'https://streeteasy.com/rental/{listing_id}'}")
        return "\n".join(parts)

    def _tool_compare_listings(self, input_data: dict[str, Any]) -> str:
        """Compare multiple listings side by side."""
        listing_ids = input_data.get("listing_ids", [])
        if len(listing_ids) < 2:
            return "Need at least 2 listing IDs to compare."

        found = []
        not_found = []
        for lid in listing_ids[:5]:
            listing = (
                self.state.recent_listings.get(lid)
                or self.state.liked_listings.get(lid)
            )
            if listing:
                found.append(listing)
            else:
                not_found.append(lid)

        if not found:
            return "None of those listings were found in recent or liked listings."

        if len(found) < 2:
            return f"Only found 1 listing. Missing: {', '.join(not_found)}."

        # Build comparison table
        lines = []
        for i, listing in enumerate(found, 1):
            fee_str = "No fee" if not listing.broker_fee else listing.broker_fee
            score_str = f"{listing.match_score}/100" if listing.match_score is not None else "N/A"
            lines.append(
                f"Listing {i}: {listing.address}\n"
                f"  Neighborhood: {listing.neighborhood}\n"
                f"  Price: ${listing.price:,}/mo\n"
                f"  Bedrooms: {listing.bedrooms} | Bathrooms: {listing.bathrooms}\n"
                f"  Fee: {fee_str}\n"
                f"  Match: {score_str}\n"
                f"  URL: {listing.url}"
            )
        result = "\n\n".join(lines)
        if not_found:
            result += f"\n\nNot found: {', '.join(not_found)}"
        return result

    def _tool_draft_outreach(self, input_data: dict[str, Any], result: ConversationResult) -> str:
        """Signal that an outreach draft should be created."""
        listing_id = input_data.get("listing_id", "")
        listing = (
            self.state.recent_listings.get(listing_id)
            or self.state.liked_listings.get(listing_id)
        )
        if not listing:
            return (
                f"Listing {listing_id} not found in recent or liked listings. "
                "I can only draft messages for listings you've seen or liked."
            )
        result.trigger_draft_listing_id = listing_id
        return "Draft will be prepared shortly."

    def _tool_update_current_apartment(self, input_data: dict[str, Any]) -> str:
        """Store or update current apartment info."""
        if self.state.current_apartment is None:
            self.state.current_apartment = CurrentApartment()
        apt = self.state.current_apartment
        updated = []

        if "address" in input_data:
            apt.address = input_data["address"]
            updated.append(f"address={apt.address}")
        if "neighborhood" in input_data:
            apt.neighborhood = input_data["neighborhood"]
            updated.append(f"neighborhood={apt.neighborhood}")
        if "price" in input_data:
            apt.price = input_data["price"]
            updated.append(f"rent=${apt.price:,}/mo")
        if "bedrooms" in input_data:
            apt.bedrooms = input_data["bedrooms"]
            updated.append(f"bedrooms={apt.bedrooms}")
        if "move_out_date" in input_data:
            apt.move_out_date = input_data["move_out_date"]
            updated.append(f"move_out={apt.move_out_date}")
        if "pros" in input_data:
            apt.pros = input_data["pros"]
            updated.append(f"pros={apt.pros}")
        if "cons" in input_data:
            apt.cons = input_data["cons"]
            updated.append(f"cons={apt.cons}")
        if "notes" in input_data:
            apt.notes = input_data["notes"]
            updated.append(f"notes={apt.notes}")

        return f"Updated current apartment: {', '.join(updated)}" if updated else "No changes made."

    def _tool_show_current_apartment(self) -> str:
        """Return current apartment info."""
        apt = self.state.current_apartment
        if not apt:
            return "No current apartment info saved yet."
        parts = ["Current apartment:"]
        if apt.address:
            parts.append(f"  Address: {apt.address}")
        if apt.neighborhood:
            parts.append(f"  Neighborhood: {apt.neighborhood}")
        if apt.price:
            parts.append(f"  Rent: ${apt.price:,}/mo")
        if apt.bedrooms is not None:
            parts.append(f"  Bedrooms: {apt.bedrooms}")
        if apt.move_out_date:
            parts.append(f"  Moving out: {apt.move_out_date}")
        if apt.pros:
            parts.append(f"  Likes: {', '.join(apt.pros)}")
        if apt.cons:
            parts.append(f"  Dislikes: {', '.join(apt.cons)}")
        if apt.notes:
            parts.append(f"  Notes: {apt.notes}")
        return "\n".join(parts)
