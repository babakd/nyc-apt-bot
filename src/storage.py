"""JSON-based state persistence on Modal Volume with atomic writes."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import tempfile
from datetime import datetime, timezone
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from src.config import (
    APIFY_ACTOR_BUILD_PIN_FILE,
    APIFY_ACTOR_CANARY_FILE,
    APIFY_ACTOR_CANARY_URLS_FILE,
    UPDATE_MARKER_RETENTION,
    UPDATE_MARKER_PRUNE_EVERY,
)
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
UPDATE_MARKER_DIRNAME = "telegram_updates"


def _state_path(chat_id: int) -> Path:
    return Path(DATA_DIR) / "chats" / f"{chat_id}.json"


def _system_path(filename: str) -> Path:
    return Path(DATA_DIR) / "system" / filename


def _configured_path(path_str: str) -> Path:
    return Path(path_str)


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


def _read_json_file(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text())
    except Exception:
        logger.exception("Failed to read JSON file: %s", path)
        return default


def _write_json_file(path: Path, payload: Any) -> None:
    _atomic_write(path, json.dumps(payload, indent=2))


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


def _update_marker_dir() -> Path:
    return Path(DATA_DIR) / UPDATE_MARKER_DIRNAME


def _prune_update_markers(latest_update_id: int) -> None:
    """Best-effort cleanup for old update-id markers."""
    min_keep_id = latest_update_id - UPDATE_MARKER_RETENTION
    if min_keep_id <= 0:
        return

    marker_dir = _update_marker_dir()
    if not marker_dir.exists():
        return

    for marker in marker_dir.glob("*.seen"):
        stem = marker.stem
        if not stem.isdigit():
            continue
        try:
            marker_id = int(stem)
        except ValueError:
            continue
        if marker_id < min_keep_id:
            try:
                marker.unlink()
            except OSError:
                logger.debug("Failed to prune update marker %s", marker)


def mark_update_seen(update_id: int) -> bool:
    """Atomically mark a Telegram update id as seen.

    Returns True if this is the first time seeing the id, False when duplicate.
    """
    marker_dir = _update_marker_dir()
    marker_dir.mkdir(parents=True, exist_ok=True)
    marker = marker_dir / f"{update_id}.seen"

    try:
        fd = os.open(marker, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    except FileExistsError:
        return False

    with os.fdopen(fd, "w") as f:
        f.write(datetime.now(timezone.utc).isoformat())

    if update_id % UPDATE_MARKER_PRUNE_EVERY == 0:
        _prune_update_markers(update_id)

    return True


def load_actor_build_pin() -> str:
    """Load pinned actor build id from disk. Empty string means no pin."""
    path = _configured_path(APIFY_ACTOR_BUILD_PIN_FILE)
    payload = _read_json_file(path, {})
    build_id = payload.get("build_id")
    return str(build_id) if build_id else ""


def save_actor_build_pin(build_id: str) -> None:
    """Persist pinned actor build id to disk."""
    path = _configured_path(APIFY_ACTOR_BUILD_PIN_FILE)
    _write_json_file(
        path,
        {
            "build_id": build_id,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        },
    )


def load_canary_status() -> dict[str, Any]:
    """Load latest actor canary status payload."""
    path = _configured_path(APIFY_ACTOR_CANARY_FILE)
    return _read_json_file(path, {})


def save_canary_status(payload: dict[str, Any]) -> None:
    """Persist latest actor canary status payload."""
    path = _configured_path(APIFY_ACTOR_CANARY_FILE)
    wrapped = dict(payload)
    wrapped["updated_at"] = datetime.now(timezone.utc).isoformat()
    _write_json_file(path, wrapped)


def load_canary_urls() -> list[str]:
    """Load known-good detail URLs for actor canary checks."""
    path = _configured_path(APIFY_ACTOR_CANARY_URLS_FILE)
    payload = _read_json_file(path, {})
    urls = payload.get("urls")
    if not isinstance(urls, list):
        return []
    return [str(u) for u in urls if isinstance(u, str) and u]


def save_canary_urls(urls: list[str], max_urls: int = 100) -> None:
    """Persist canary detail URLs with stable de-duplication and cap."""
    unique_urls: list[str] = []
    seen: set[str] = set()
    for url in urls:
        if not url or url in seen:
            continue
        seen.add(url)
        unique_urls.append(url)
    unique_urls = unique_urls[:max_urls]

    path = _configured_path(APIFY_ACTOR_CANARY_URLS_FILE)
    _write_json_file(
        path,
        {
            "urls": unique_urls,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        },
    )


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
