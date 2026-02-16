"""Tests for Telegram message formatting."""

from datetime import datetime, timezone

from src.formatter import (
    _escape_html,
    _score_bar,
    build_send_message_payload,
    draft_keyboard,
    format_listing_card,
    format_preferences_summary,
    format_scan_header,
    listing_keyboard,
)
from src.models import Listing, Preferences


class TestEscapeHtml:
    def test_basic_escaping(self):
        assert _escape_html("foo & bar") == "foo &amp; bar"
        assert _escape_html("<script>") == "&lt;script&gt;"

    def test_no_escaping_needed(self):
        assert _escape_html("hello world") == "hello world"


class TestScoreBar:
    def test_full_score(self):
        assert _score_bar(100) == "\u2588" * 10

    def test_zero_score(self):
        assert _score_bar(0) == "\u2591" * 10

    def test_half_score(self):
        bar = _score_bar(50)
        assert bar.count("\u2588") == 5
        assert bar.count("\u2591") == 5


class TestListingCard:
    def test_basic_card(self):
        listing = Listing(
            listing_id="123",
            url="https://streeteasy.com/rental/123",
            address="100 Main St",
            neighborhood="Chelsea",
            price=3500,
            bedrooms=1,
            bathrooms=1.0,
        )
        card = format_listing_card(listing)
        assert "100 Main St" in card
        assert "Chelsea" in card
        assert "$3,500" in card
        assert "1 BR" in card
        # New format uses middot separators
        assert "\u00b7" in card or "·" in card
        # No broker_fee means NO FEE shown
        assert "NO FEE" in card

    def test_card_with_broker_fee(self):
        listing = Listing(
            listing_id="123",
            url="https://streeteasy.com/rental/123",
            address="100 Main St",
            neighborhood="Chelsea",
            price=3500,
            bedrooms=1,
            bathrooms=1.0,
            broker_fee="Broker fee",
        )
        card = format_listing_card(listing)
        assert "NO FEE" not in card

    def test_card_with_score(self):
        listing = Listing(
            listing_id="123",
            url="https://streeteasy.com/rental/123",
            address="100 Main St",
            neighborhood="Chelsea",
            price=3500,
            bedrooms=1,
            bathrooms=1.0,
            match_score=85,
            pros=["Great location"],
            cons=["No laundry"],
        )
        card = format_listing_card(listing)
        assert "85% match" in card
        assert "Great location" in card
        assert "No laundry" in card
        # Uses triangle bullets
        assert "\u25b8" in card

    def test_card_with_rank(self):
        listing = Listing(
            listing_id="123",
            url="https://streeteasy.com/rental/123",
            address="100 Main St",
            neighborhood="Chelsea",
            price=3500,
            bedrooms=1,
            bathrooms=1.0,
        )
        card = format_listing_card(listing, rank=1)
        assert "#1" in card

    def test_card_studio(self):
        listing = Listing(
            listing_id="123",
            url="https://streeteasy.com/rental/123",
            address="200 Broadway",
            neighborhood="SoHo",
            price=2500,
            bedrooms=0,
            bathrooms=1.0,
        )
        card = format_listing_card(listing)
        assert "Studio" in card

    def test_card_with_available_date(self):
        listing = Listing(
            listing_id="123",
            url="https://streeteasy.com/rental/123",
            address="100 Main St",
            neighborhood="Chelsea",
            price=3500,
            bedrooms=1,
            bathrooms=1.0,
            available_date="2026-03-01",
        )
        card = format_listing_card(listing)
        assert "Available 2026-03-01" in card

    def test_card_with_sqft(self):
        listing = Listing(
            listing_id="123",
            url="https://streeteasy.com/rental/123",
            address="100 Main St",
            neighborhood="Chelsea",
            price=3500,
            bedrooms=1,
            bathrooms=1.0,
            sqft=750,
        )
        card = format_listing_card(listing)
        assert "750 sqft" in card

    def test_card_caption_length(self):
        """Card should fit within Telegram's 1024-char sendPhoto caption limit."""
        listing = Listing(
            listing_id="123",
            url="https://streeteasy.com/rental/123",
            address="123 Very Long Street Name Apartment #24B",
            neighborhood="Upper East Side",
            price=4500,
            bedrooms=2,
            bathrooms=1.5,
            sqft=900,
            broker_fee="Broker fee",
            available_date="2026-04-01",
            match_score=75,
            pros=["Great location", "Doorman building", "In-unit laundry"],
            cons=["No dishwasher", "5th floor walk-up"],
        )
        card = format_listing_card(listing, rank=1)
        assert len(card) < 1024


class TestPreferencesSummary:
    def test_full_preferences(self):
        prefs = Preferences(
            budget_min=2500,
            budget_max=4000,
            bedrooms=[1, 2],
            neighborhoods=["Chelsea", "SoHo"],
            must_haves=["Dishwasher"],
            no_fee_only=True,
        )
        summary = format_preferences_summary(prefs)
        assert "$2,500" in summary
        assert "$4,000" in summary
        assert "Chelsea" in summary
        assert "Dishwasher" in summary
        assert "No-fee" in summary

    def test_empty_preferences(self):
        prefs = Preferences()
        summary = format_preferences_summary(prefs)
        assert "Preferences" in summary


class TestScanHeader:
    def test_no_results(self):
        header = format_scan_header(0)
        assert "No new listings" in header
        assert "ranked" not in header

    def test_one_result(self):
        header = format_scan_header(1)
        assert "1" in header
        assert "listing" in header
        assert "ranked" in header

    def test_multiple_results(self):
        header = format_scan_header(5)
        assert "5" in header
        assert "listings" in header
        assert "ranked" in header


class TestKeyboards:
    def test_listing_keyboard_without_url(self):
        kb = listing_keyboard("123")
        assert len(kb) == 2  # Two rows
        assert any("Like" in btn["text"] for row in kb for btn in row)
        assert any("Details" in btn["text"] for row in kb for btn in row)
        # No URL button without listing_url
        assert not any("url" in btn for row in kb for btn in row)

    def test_listing_keyboard_with_url(self):
        kb = listing_keyboard("123", "https://streeteasy.com/rental/123")
        assert len(kb) == 2  # Two rows
        assert any("Like" in btn["text"] for row in kb for btn in row)
        assert any("Details" in btn["text"] for row in kb for btn in row)
        # URL button should be present
        url_btns = [btn for row in kb for btn in row if "url" in btn]
        assert len(url_btns) == 1
        assert url_btns[0]["url"] == "https://streeteasy.com/rental/123"
        assert "StreetEasy" in url_btns[0]["text"]

    def test_listing_keyboard_no_contact_button(self):
        """Contact Agent button was removed in favor of URL button."""
        kb = listing_keyboard("123", "https://streeteasy.com/rental/123")
        assert not any("Contact" in btn.get("text", "") for row in kb for btn in row)

    def test_draft_keyboard(self):
        kb = draft_keyboard("abc")
        assert len(kb) == 1  # One row
        assert any("Send" in btn["text"] for row in kb for btn in row)


class TestPayloadBuilder:
    def test_basic_payload(self):
        payload = build_send_message_payload(123, "Hello")
        assert payload["chat_id"] == 123
        assert payload["text"] == "Hello"
        assert payload["parse_mode"] == "HTML"

    def test_payload_with_keyboard(self):
        kb = [[{"text": "OK", "callback_data": "ok"}]]
        payload = build_send_message_payload(123, "Hello", reply_markup=kb)
        assert "reply_markup" in payload
