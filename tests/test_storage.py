"""Tests for storage module."""

import asyncio

import pytest

from src.models import ChatState


@pytest.fixture(autouse=True)
def temp_data_dir(monkeypatch, tmp_path):
    """Use a temporary directory for all storage tests."""
    monkeypatch.setattr("src.storage.DATA_DIR", str(tmp_path))
    return tmp_path


class TestStorage:
    def test_load_fresh_state(self):
        from src.storage import load_state
        state = load_state(12345)
        assert state.chat_id == 12345
        assert state.preferences_ready is False

    def test_save_and_load(self):
        from src.storage import load_state, save_state
        state = ChatState(chat_id=12345)
        state.preferences.budget_max = 4000
        state.preferences.neighborhoods = ["Chelsea", "SoHo"]
        state.preferences_ready = True
        state.seen_listing_ids.add("abc123")
        save_state(state)

        loaded = load_state(12345)
        assert loaded.preferences.budget_max == 4000
        assert loaded.preferences.neighborhoods == ["Chelsea", "SoHo"]
        assert loaded.preferences_ready is True
        assert "abc123" in loaded.seen_listing_ids

    def test_delete_state(self):
        from src.storage import delete_state, load_state, save_state
        state = ChatState(chat_id=99999)
        save_state(state)
        delete_state(99999)
        loaded = load_state(99999)
        # Should return fresh state after deletion
        assert loaded.preferences_ready is False

    def test_list_chat_ids(self):
        from src.storage import list_chat_ids, save_state
        save_state(ChatState(chat_id=111))
        save_state(ChatState(chat_id=222))
        save_state(ChatState(chat_id=333))
        ids = list_chat_ids()
        assert set(ids) == {111, 222, 333}

    def test_load_all_states(self):
        from src.storage import load_all_states, save_state
        save_state(ChatState(chat_id=111))
        save_state(ChatState(chat_id=222))
        states = load_all_states()
        assert len(states) == 2
        chat_ids = {s.chat_id for s in states}
        assert chat_ids == {111, 222}

    def test_updated_at_changes(self):
        from src.storage import load_state, save_state
        import time
        state = ChatState(chat_id=12345)
        save_state(state)
        first = load_state(12345).updated_at

        time.sleep(0.01)
        state.preferences.budget_max = 5000
        save_state(state)
        second = load_state(12345).updated_at

        assert second > first


class TestChatLock:
    @pytest.mark.asyncio
    async def test_chat_lock_serializes_access(self):
        """Concurrent access to the same chat_id is serialized by chat_lock."""
        from src.storage import chat_lock

        order = []

        async def writer(label: str):
            async with chat_lock(12345):
                order.append(f"{label}_start")
                await asyncio.sleep(0.05)
                order.append(f"{label}_end")

        await asyncio.gather(writer("a"), writer("b"))

        # One must fully complete before the other starts
        assert order[:2] in [["a_start", "a_end"], ["b_start", "b_end"]]
        assert order[2:] in [["a_start", "a_end"], ["b_start", "b_end"]]

    @pytest.mark.asyncio
    async def test_chat_lock_different_ids_parallel(self):
        """Different chat_ids can proceed in parallel (no cross-locking)."""
        from src.storage import chat_lock

        order = []

        async def writer(chat_id: int, label: str):
            async with chat_lock(chat_id):
                order.append(f"{label}_start")
                await asyncio.sleep(0.05)
                order.append(f"{label}_end")

        await asyncio.gather(writer(111, "a"), writer(222, "b"))

        # Both should start before either finishes (parallel execution)
        starts = [i for i, x in enumerate(order) if x.endswith("_start")]
        ends = [i for i, x in enumerate(order) if x.endswith("_end")]
        assert len(starts) == 2
        assert len(ends) == 2
        # Both starts should happen before both ends
        assert max(starts) < max(ends)
