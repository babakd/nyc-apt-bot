"""JSON-based state persistence on Modal Volume with atomic writes."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path

from src.models import ChatState, Preferences

logger = logging.getLogger(__name__)

_chat_locks: dict[int, asyncio.Lock] = {}


@asynccontextmanager
async def chat_lock(chat_id: int):
    """Acquire a per-chat asyncio lock to serialize state mutations."""
    if chat_id not in _chat_locks:
        _chat_locks[chat_id] = asyncio.Lock()
    async with _chat_locks[chat_id]:
        yield

DATA_DIR = os.environ.get("DATA_DIR", "/data")


def _state_path(chat_id: int) -> Path:
    return Path(DATA_DIR) / "chats" / f"{chat_id}.json"


def _ensure_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _atomic_write(path: Path, data: str) -> None:
    """Write data to file atomically using tempfile + rename."""
    _ensure_dir(path)
    fd, tmp_path = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(data)
        os.replace(tmp_path, path)
    except Exception:
        # Clean up temp file on failure
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def load_state(chat_id: int) -> ChatState:
    """Load chat state from disk. Returns fresh state if not found."""
    path = _state_path(chat_id)
    if path.exists():
        try:
            raw = path.read_text()
            return ChatState.model_validate_json(raw)
        except Exception:
            logger.exception("Failed to load state for chat %s, returning fresh state", chat_id)
    return ChatState(chat_id=chat_id)


def save_state(state: ChatState) -> None:
    """Persist chat state to disk atomically."""
    from datetime import datetime, timezone
    state.updated_at = datetime.now(timezone.utc)
    path = _state_path(state.chat_id)
    data = state.model_dump_json(indent=2)
    _atomic_write(path, data)
    logger.debug("Saved state for chat %s", state.chat_id)


def delete_state(chat_id: int) -> None:
    """Delete chat state file."""
    path = _state_path(chat_id)
    if path.exists():
        path.unlink()
        logger.info("Deleted state for chat %s", chat_id)


def list_chat_ids() -> list[int]:
    """List all chat IDs that have stored state."""
    chats_dir = Path(DATA_DIR) / "chats"
    if not chats_dir.exists():
        return []
    return [
        int(f.stem)
        for f in chats_dir.glob("*.json")
        if f.stem.lstrip("-").isdigit()
    ]


def load_all_states() -> list[ChatState]:
    """Load all chat states (for daily scan across all users)."""
    return [load_state(cid) for cid in list_chat_ids()]


def clear_all_conversation_histories() -> int:
    """Clear conversation history for all chats. Returns count of chats cleared.

    One-time migration helper to flush poisoned history that taught Claude
    to skip tool calls.
    """
    count = 0
    for chat_id in list_chat_ids():
        state = load_state(chat_id)
        if state.conversation_history:
            state.conversation_history.clear()
            save_state(state)
            count += 1
            logger.info("Cleared conversation history for chat %s", chat_id)
    return count
