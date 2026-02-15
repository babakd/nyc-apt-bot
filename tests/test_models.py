"""Tests for Pydantic models."""

from datetime import datetime, timezone

from src.models import ChatState, ConversationTurn, CurrentApartment, Draft, Listing, Preferences


class TestPreferences:
    def test_defaults(self):
        prefs = Preferences()
        assert prefs.budget_min is None
        assert prefs.budget_max is None
        assert prefs.bedrooms == []
        assert prefs.neighborhoods == []
        assert prefs.min_bathrooms is None
        assert prefs.no_fee_only is False

    def test_full_preferences(self):
        prefs = Preferences(
            budget_min=2500,
            budget_max=4000,
            bedrooms=[1, 2],
            neighborhoods=["Upper East Side", "Chelsea"],
            min_bathrooms=2,
            must_haves=["Dishwasher", "Elevator"],
            no_fee_only=True,
        )
        assert prefs.budget_max == 4000
        assert len(prefs.bedrooms) == 2
        assert prefs.min_bathrooms == 2
        assert prefs.no_fee_only is True

    def test_json_roundtrip(self):
        prefs = Preferences(budget_max=3500, bedrooms=[1], neighborhoods=["SoHo"])
        data = prefs.model_dump_json()
        restored = Preferences.model_validate_json(data)
        assert restored.budget_max == 3500
        assert restored.neighborhoods == ["SoHo"]


class TestConversationTurn:
    def test_user_turn(self):
        turn = ConversationTurn(role="user", content="Hello")
        assert turn.role == "user"
        assert turn.content == "Hello"

    def test_assistant_turn(self):
        turn = ConversationTurn(role="assistant", content="Hi there!")
        assert turn.role == "assistant"
        assert turn.content == "Hi there!"

    def test_structured_assistant_content(self):
        """Assistant turn can contain structured tool_use blocks."""
        content = [
            {"type": "text", "text": "Let me search."},
            {"type": "tool_use", "id": "toolu_abc", "name": "search_apartments", "input": {}},
        ]
        turn = ConversationTurn(role="assistant", content=content)
        assert isinstance(turn.content, list)
        assert turn.content[1]["name"] == "search_apartments"

    def test_structured_user_content(self):
        """User turn can contain structured tool_result blocks."""
        content = [
            {"type": "tool_result", "tool_use_id": "toolu_abc", "content": "Search triggered."},
        ]
        turn = ConversationTurn(role="user", content=content)
        assert isinstance(turn.content, list)
        assert turn.content[0]["type"] == "tool_result"


class TestListing:
    def test_minimal_listing(self):
        listing = Listing(
            listing_id="123456",
            url="https://streeteasy.com/rental/123456",
            address="100 Main St #2A",
            neighborhood="Chelsea",
            price=3500,
            bedrooms=1,
            bathrooms=1.0,
        )
        assert listing.listing_id == "123456"
        assert listing.match_score is None
        assert listing.photos == []

    def test_full_listing(self):
        listing = Listing(
            listing_id="789",
            url="https://streeteasy.com/rental/789",
            address="200 Broadway",
            neighborhood="SoHo",
            price=5000,
            bedrooms=2,
            bathrooms=1.5,
            sqft=900,
            amenities=["doorman", "elevator"],
            match_score=85,
            pros=["Great location"],
            cons=["Pricey"],
        )
        assert listing.match_score == 85
        assert len(listing.amenities) == 2

    def test_score_validation(self):
        listing = Listing(
            listing_id="1",
            url="",
            address="",
            neighborhood="",
            price=1000,
            bedrooms=1,
            bathrooms=1,
            match_score=100,
        )
        assert listing.match_score == 100


class TestDraft:
    def test_draft_creation(self):
        draft = Draft(
            draft_id="abc123",
            listing_id="456",
            message_text="Hello, I'm interested...",
        )
        assert draft.status == "pending"
        assert draft.sent_at is None

    def test_draft_status_transition(self):
        draft = Draft(
            draft_id="abc",
            listing_id="456",
            message_text="Hi",
            status="sent",
            sent_at=datetime.now(timezone.utc),
        )
        assert draft.status == "sent"
        assert draft.sent_at is not None


class TestCurrentApartment:
    def test_defaults(self):
        apt = CurrentApartment()
        assert apt.address is None
        assert apt.price is None
        assert apt.pros == []
        assert apt.cons == []

    def test_full_apartment(self):
        apt = CurrentApartment(
            address="456 Oak St",
            neighborhood="Williamsburg",
            price=2800,
            bedrooms=1,
            move_out_date="2026-04-01",
            pros=["great light", "quiet"],
            cons=["no dishwasher"],
            notes="Lease ends April 1",
        )
        assert apt.address == "456 Oak St"
        assert apt.price == 2800
        assert len(apt.pros) == 2
        assert apt.notes == "Lease ends April 1"

    def test_json_roundtrip(self):
        apt = CurrentApartment(
            address="123 Main St",
            price=3000,
            pros=["doorman"],
            cons=["small"],
        )
        data = apt.model_dump_json()
        restored = CurrentApartment.model_validate_json(data)
        assert restored.address == "123 Main St"
        assert restored.pros == ["doorman"]


class TestChatState:
    def test_default_state(self):
        state = ChatState(chat_id=12345)
        assert state.preferences_ready is False
        assert state.preferences.budget_max is None
        assert len(state.seen_listing_ids) == 0
        assert state.conversation_history == []
        assert state.liked_listings == {}
        assert state.recent_listings == {}
        assert state.current_apartment is None

    def test_state_with_data(self):
        state = ChatState(
            chat_id=12345,
            preferences_ready=True,
            seen_listing_ids={"a", "b", "c"},
            liked_listing_ids={"a"},
        )
        assert len(state.seen_listing_ids) == 3
        assert "a" in state.liked_listing_ids
        assert state.preferences_ready is True

    def test_conversation_history(self):
        state = ChatState(chat_id=12345)
        state.conversation_history.append(
            ConversationTurn(role="user", content="Hi")
        )
        state.conversation_history.append(
            ConversationTurn(role="assistant", content="Hello!")
        )
        assert len(state.conversation_history) == 2
        assert state.conversation_history[0].role == "user"

    def test_json_roundtrip(self):
        state = ChatState(chat_id=99999)
        state.preferences.budget_max = 4000
        state.preferences.neighborhoods = ["Chelsea"]
        state.seen_listing_ids.add("listing1")
        state.preferences_ready = True
        state.conversation_history.append(
            ConversationTurn(role="user", content="Hi")
        )

        data = state.model_dump_json()
        restored = ChatState.model_validate_json(data)
        assert restored.chat_id == 99999
        assert restored.preferences.budget_max == 4000
        assert "listing1" in restored.seen_listing_ids
        assert restored.preferences_ready is True
        assert len(restored.conversation_history) == 1

    def test_json_roundtrip_with_listings_and_apartment(self):
        state = ChatState(chat_id=99999)
        listing = Listing(
            listing_id="100",
            url="https://streeteasy.com/rental/100",
            address="100 Main St",
            neighborhood="Chelsea",
            price=3500,
            bedrooms=2,
            bathrooms=1.0,
            match_score=80,
        )
        state.liked_listing_ids.add("100")
        state.liked_listings["100"] = listing
        state.recent_listings["100"] = listing
        state.current_apartment = CurrentApartment(
            address="456 Oak St", price=2800, pros=["quiet"]
        )

        data = state.model_dump_json()
        restored = ChatState.model_validate_json(data)
        assert "100" in restored.liked_listings
        assert restored.liked_listings["100"].address == "100 Main St"
        assert "100" in restored.recent_listings
        assert restored.current_apartment is not None
        assert restored.current_apartment.address == "456 Oak St"
        assert restored.current_apartment.pros == ["quiet"]


class TestGroupChatFields:
    def test_conversation_turn_sender_name_default(self):
        """sender_name defaults to None."""
        turn = ConversationTurn(role="user", content="Hello")
        assert turn.sender_name is None

    def test_conversation_turn_with_sender_name(self):
        """sender_name can be set on a turn."""
        turn = ConversationTurn(role="user", content="[Alice]: Hello", sender_name="Alice")
        assert turn.sender_name == "Alice"
        assert turn.content == "[Alice]: Hello"

    def test_conversation_turn_sender_name_json_roundtrip(self):
        """sender_name survives JSON serialization."""
        turn = ConversationTurn(role="user", content="[Bob]: Hi", sender_name="Bob")
        data = turn.model_dump_json()
        restored = ConversationTurn.model_validate_json(data)
        assert restored.sender_name == "Bob"

    def test_conversation_turn_backward_compat(self):
        """Old JSON without sender_name loads correctly."""
        old_json = '{"role": "user", "content": "Hello"}'
        turn = ConversationTurn.model_validate_json(old_json)
        assert turn.sender_name is None
        assert turn.content == "Hello"

    def test_chatstate_is_group_default(self):
        """is_group defaults to False."""
        state = ChatState(chat_id=12345)
        assert state.is_group is False

    def test_chatstate_is_group_set(self):
        """is_group can be set to True."""
        state = ChatState(chat_id=-100123, is_group=True)
        assert state.is_group is True

    def test_chatstate_is_group_json_roundtrip(self):
        """is_group survives JSON serialization."""
        state = ChatState(chat_id=-100123, is_group=True)
        data = state.model_dump_json()
        restored = ChatState.model_validate_json(data)
        assert restored.is_group is True

    def test_chatstate_backward_compat(self):
        """Old JSON without is_group loads correctly."""
        old_json = '{"chat_id": 12345}'
        state = ChatState.model_validate_json(old_json)
        assert state.is_group is False


class TestConstraintContext:
    def test_defaults_to_none(self):
        """constraint_context defaults to None."""
        prefs = Preferences()
        assert prefs.constraint_context is None

    def test_set_constraint_context(self):
        """constraint_context can be set."""
        prefs = Preferences(
            constraint_context="Budget $3,500 is firm max. 2BR non-negotiable."
        )
        assert prefs.constraint_context == "Budget $3,500 is firm max. 2BR non-negotiable."

    def test_json_roundtrip(self):
        """constraint_context survives JSON serialization."""
        prefs = Preferences(
            budget_max=3500,
            neighborhoods=["Chelsea"],
            constraint_context="Budget is firm. Chelsea preferred but open to nearby.",
        )
        data = prefs.model_dump_json()
        restored = Preferences.model_validate_json(data)
        assert restored.constraint_context == "Budget is firm. Chelsea preferred but open to nearby."
        assert restored.budget_max == 3500

    def test_backward_compatibility(self):
        """Old serialized state without constraint_context loads correctly."""
        # Simulate old JSON without constraint_context field
        old_json = '{"budget_max": 3000, "bedrooms": [1], "neighborhoods": ["SoHo"]}'
        prefs = Preferences.model_validate_json(old_json)
        assert prefs.constraint_context is None
        assert prefs.budget_max == 3000

    def test_in_model_dump(self):
        """constraint_context is included in model_dump."""
        prefs = Preferences(constraint_context="All flexible")
        dump = prefs.model_dump()
        assert "constraint_context" in dump
        assert dump["constraint_context"] == "All flexible"

    def test_chatstate_roundtrip_with_constraint_context(self):
        """ChatState with constraint_context survives JSON roundtrip."""
        state = ChatState(chat_id=12345)
        state.preferences.constraint_context = "Budget firm, bedrooms flexible"
        state.preferences.budget_max = 4000

        data = state.model_dump_json()
        restored = ChatState.model_validate_json(data)
        assert restored.preferences.constraint_context == "Budget firm, bedrooms flexible"
        assert restored.preferences.budget_max == 4000
