"""Session source registry for Claude Code, Codex and Cursor."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .dialogue import SessionDialogue


@dataclass(frozen=True)
class SourceAdapter:
    name: str
    root: Path
    parse_session: Callable[[Path], SessionDialogue | None]
    session_id_of: Callable[[Path], str]
    should_index: Callable[[Path], bool]

    def iter_files(self):
        if self.root.is_dir():
            yield from sorted(self.root.rglob("*.jsonl"))


def build_adapters(
    claude_root: Path,
    codex_root: Path,
    cursor_root: Path,
) -> dict[str, SourceAdapter]:
    from . import codex_extractor
    from . import cursor_extractor
    from . import extractor as claude_extractor

    return {
        "claude": SourceAdapter(
            name="claude",
            root=claude_root,
            parse_session=claude_extractor.parse_session,
            session_id_of=claude_extractor.session_id_of,
            should_index=lambda p: not p.name.startswith("agent-"),
        ),
        "codex": SourceAdapter(
            name="codex",
            root=codex_root,
            parse_session=codex_extractor.parse_session,
            session_id_of=codex_extractor.session_id_of,
            should_index=lambda _p: True,
        ),
        "cursor": SourceAdapter(
            name="cursor",
            root=cursor_root,
            parse_session=cursor_extractor.parse_session,
            session_id_of=cursor_extractor.session_id_of,
            should_index=lambda _p: True,
        ),
    }
