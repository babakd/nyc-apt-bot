"""Pydantic models for the StreetEasy apartment hunting bot."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal, Optional, Union

from pydantic import BaseModel, Field


class ConversationTurn(BaseModel):
    """A single turn in the conversation history.

    content is either:
    - str: simple text message (user input or assistant text-only reply)
    - list[dict]: structured content blocks (assistant tool_use or user tool_result)
    """
    role: Literal["user", "assistant"]
    content: Union[str, list[dict[str, Any]]]
    sender_name: Optional[str] = None


class Preferences(BaseModel):
    """User apartment search preferences."""
    budget_min: Optional[int] = None
    budget_max: Optional[int] = None
    bedrooms: list[int] = Field(default_factory=list, description="Acceptable bedroom counts (0=studio)")
    neighborhoods: list[str] = Field(default_factory=list, description="Preferred neighborhood names")
    commute_address: Optional[str] = None
    commute_max_minutes: Optional[int] = None
    min_bathrooms: Optional[int] = None
    must_haves: list[str] = Field(default_factory=list, description="Required amenities")
    nice_to_haves: list[str] = Field(default_factory=list, description="Preferred but not required amenities")
    no_fee_only: bool = False
    move_in_date: Optional[str] = None
    constraint_context: Optional[str] = None


class Listing(BaseModel):
    """A scraped apartment listing from StreetEasy."""
    listing_id: str
    url: str
    address: str
    neighborhood: str
    price: int = Field(description="Monthly rent in dollars")
    bedrooms: int
    bathrooms: float
    sqft: Optional[int] = None
    amenities: list[str] = Field(default_factory=list)
    photos: list[str] = Field(default_factory=list, description="Photo URLs")
    photo_keys: list[str] = Field(default_factory=list, description="Raw photo keys for CDN URL construction")
    broker_fee: Optional[str] = None
    available_date: Optional[str] = None
    description: Optional[str] = None
    match_score: Optional[int] = Field(None, ge=0, le=100)
    pros: list[str] = Field(default_factory=list)
    cons: list[str] = Field(default_factory=list)
    scraped_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class Draft(BaseModel):
    """An outreach draft message to a listing agent."""
    draft_id: str
    listing_id: str
    message_text: str
    status: Literal["pending", "approved", "sent", "cancelled"] = "pending"
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    sent_at: Optional[datetime] = None


class CurrentApartment(BaseModel):
    """Info about the user's current living situation for comparison context."""
    address: Optional[str] = None
    neighborhood: Optional[str] = None
    price: Optional[int] = None
    bedrooms: Optional[int] = None
    move_out_date: Optional[str] = None
    pros: list[str] = Field(default_factory=list)
    cons: list[str] = Field(default_factory=list)
    notes: Optional[str] = None


MAX_RECENT_LISTINGS = 50


class ChatState(BaseModel):
    """Per-chat persistent state."""
    chat_id: int
    preferences: Preferences = Field(default_factory=Preferences)
    preferences_ready: bool = False
    is_group: bool = False
    conversation_history: list[ConversationTurn] = Field(default_factory=list)
    seen_listing_ids: set[str] = Field(default_factory=set)
    liked_listing_ids: set[str] = Field(default_factory=set)
    liked_listings: dict[str, Listing] = Field(default_factory=dict)
    recent_listings: dict[str, Listing] = Field(default_factory=dict)
    active_drafts: dict[str, Draft] = Field(default_factory=dict)
    pending_draft_edit: Optional[str] = Field(None)
    current_apartment: Optional[CurrentApartment] = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
