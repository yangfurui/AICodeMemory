"""Provider-neutral dialogue model shared by all session extractors."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone


@dataclass
class Message:
    role: str  # "user" | "assistant"
    text: str


@dataclass
class SessionDialogue:
    source: str  # "claude" | "codex" | "cursor"
    session_id: str
    project: str
    date: str  # 最后一条有效消息的本地日期 YYYY-MM-DD
    file_mtime_ns: int
    file_size: int
    messages: list[Message] = field(default_factory=list)


def to_local_date(iso_ts: str) -> str:
    """ISO timestamp -> local YYYY-MM-DD; malformed input becomes empty."""
    ts = iso_ts.replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(ts)
    except (TypeError, ValueError):
        return ""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone().strftime("%Y-%m-%d")
