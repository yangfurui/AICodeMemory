"""Parse local Codex rollout JSONL into clean user/final-answer dialogue.

Codex rollout files are operational event logs, not a stable public wire format.
The parser is deliberately tolerant: it consumes only the small message subset we
need and ignores unknown events. Raw archives retain everything for future
re-extraction when the format evolves.
"""

from __future__ import annotations

import gzip
import json
import re
from pathlib import Path

from .dialogue import Message, SessionDialogue, to_local_date

DEFAULT_SOURCE = Path.home() / ".codex" / "sessions"

_SESSION_ID_RE = re.compile(
    r"([0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-"
    r"[0-9a-fA-F]{4}-[0-9a-fA-F]{12})\.jsonl(?:\.gz)?$"
)
_INJECTED_USER_PREFIXES = (
    "<environment_context>",
    "<permissions instructions>",
    "<collaboration_mode>",
)


def _open_text(path: Path):
    if path.name.endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8", errors="replace")
    return path.open(encoding="utf-8", errors="replace")


def session_id_of(path: Path) -> str:
    """Prefer session_meta.id; fall back to the UUID suffix in rollout names."""
    try:
        with _open_text(path) as f:
            for _ in range(32):
                line = f.readline()
                if not line:
                    break
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if row.get("type") == "session_meta":
                    payload = row.get("payload") or {}
                    if sid := payload.get("id") or payload.get("session_id"):
                        return str(sid)
    except OSError:
        pass
    if match := _SESSION_ID_RE.search(path.name):
        return match.group(1)
    return path.name.removesuffix(".gz").removesuffix(".jsonl")


def _content_text(content, expected_type: str) -> str:
    if not isinstance(content, list):
        return ""
    return "\n".join(
        block.get("text", "")
        for block in content
        if isinstance(block, dict) and block.get("type") == expected_type
    )


def _is_injected_user_text(text: str) -> bool:
    stripped = text.lstrip()
    return any(stripped.startswith(prefix) for prefix in _INJECTED_USER_PREFIXES)


def parse_session(path: Path) -> SessionDialogue | None:
    stat = path.stat()
    sid = ""
    cwd = ""
    meta_ts = ""
    last_ts = ""
    messages: list[Message] = []

    with _open_text(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue

            row_type = row.get("type")
            payload = row.get("payload") or {}
            if row_type == "session_meta":
                sid = str(payload.get("id") or payload.get("session_id") or sid)
                cwd = str(payload.get("cwd") or cwd)
                meta_ts = str(payload.get("timestamp") or row.get("timestamp") or meta_ts)
                continue
            if row_type != "response_item" or payload.get("type") != "message":
                continue

            role = payload.get("role")
            if role == "user":
                text = _content_text(payload.get("content"), "input_text").strip()
                if not text or _is_injected_user_text(text):
                    continue
            elif role == "assistant":
                # commentary is useful progress UI but noisy long-term memory.
                # Older rollouts may omit phase, so accept an empty phase too.
                if payload.get("phase") not in (None, "", "final", "final_answer"):
                    continue
                text = _content_text(payload.get("content"), "output_text").strip()
                if not text:
                    continue
            else:
                continue  # developer/system messages never enter the text projection

            last_ts = str(row.get("timestamp") or last_ts)
            if messages and messages[-1].role == role:
                messages[-1].text += "\n" + text
            else:
                messages.append(Message(role=role, text=text))

    if not messages:
        return None
    return SessionDialogue(
        source="codex",
        session_id=sid or session_id_of(path),
        project=Path(cwd).name if cwd else "unknown",
        date=to_local_date(last_ts or meta_ts),
        file_mtime_ns=stat.st_mtime_ns,
        file_size=stat.st_size,
        messages=messages,
    )
