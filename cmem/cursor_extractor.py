"""Export Cursor's local SQLite chat history into append-only session JSONL.

Cursor documents that regular Agent history is stored in a local SQLite
database, but does not publish that database as a stable API.  Recent builds
store session metadata under ``composerData:<id>`` and message bubbles under
``bubbleId:<composer-id>:<bubble-id>`` in ``cursorDiskKV``.

The database also contains hundreds of megabytes of tool state, attached code
and execution context.  AICodeMemory deliberately retains only the verbatim
user/assistant message text plus stable provenance.  Each session is converted
to a provider-neutral, append-only JSONL event log before it enters the normal
gzip raw archive.  If Cursor later edits an existing bubble, a new revision is
appended; the previous text is never overwritten.

Background Agent chats are remote-only according to Cursor's documentation and
are therefore outside this local source adapter.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import re
import sqlite3
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from urllib.parse import unquote, urlparse

from .dialogue import Message, SessionDialogue, to_local_date

EXPORT_SCHEMA = 1
_SAFE_NAME = re.compile(r"[^A-Za-z0-9._-]+")


class CursorSourceError(RuntimeError):
    """Cursor's local history could not be read without risking silent loss."""


@dataclass(frozen=True)
class CursorExportResult:
    files: tuple[Path, ...]
    sessions_seen: int
    sessions_exported: int
    messages_exported: int
    issues: tuple[str, ...] = ()


@dataclass
class _WorkspaceMetadata:
    projects: dict[str, str]
    composer_workspaces: dict[str, str]
    titles: dict[str, str]
    excluded_composers: set[str]


def default_source(home: Path | None = None) -> Path:
    """Return Cursor's per-user data directory for the current platform."""
    home = Path(home) if home is not None else Path.home()
    if sys.platform == "darwin":
        return home / "Library" / "Application Support" / "Cursor" / "User"
    if os.name == "nt":
        appdata = os.environ.get("APPDATA")
        root = Path(appdata) if appdata else home / "AppData" / "Roaming"
        return root / "Cursor" / "User"
    config = Path(os.environ.get("XDG_CONFIG_HOME", home / ".config"))
    return config / "Cursor" / "User"


DEFAULT_SOURCE = default_source()


def locate_database(source: Path = DEFAULT_SOURCE) -> tuple[Path, Path | None]:
    """Resolve a Cursor User directory (preferred) or a direct state.vscdb."""
    source = Path(source).expanduser().absolute()
    if source.is_file():
        user_dir = source.parent.parent if source.parent.name == "globalStorage" else None
        return source, user_dir

    candidates = (
        source / "globalStorage" / "state.vscdb",
        source / "User" / "globalStorage" / "state.vscdb",
    )
    for candidate in candidates:
        if candidate.is_file():
            user_dir = candidate.parent.parent
            return candidate, user_dir
    raise CursorSourceError(f"Cursor 本地历史数据库不存在: {source}")


def _connect_readonly(path: Path) -> sqlite3.Connection:
    try:
        connection = sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)
        connection.execute("PRAGMA query_only = ON")
        return connection
    except sqlite3.Error as exc:
        raise CursorSourceError(f"Cursor 本地历史数据库无法只读打开: {path}") from exc


def _table_exists(connection: sqlite3.Connection, name: str) -> bool:
    return connection.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)
    ).fetchone() is not None


def _json_object(value: object) -> dict[str, object] | None:
    if isinstance(value, memoryview):
        value = value.tobytes()
    try:
        decoded = json.loads(value) if isinstance(value, (str, bytes, bytearray)) else value
    except (UnicodeError, json.JSONDecodeError):
        return None
    return decoded if isinstance(decoded, dict) else None


def _workspace_project(path: Path) -> str:
    metadata = path / "workspace.json"
    try:
        document = json.loads(metadata.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return "unknown"
    raw = document.get("folder") or document.get("workspace")
    if not isinstance(raw, str) or not raw:
        return "unknown"
    parsed = urlparse(raw)
    candidate = unquote(parsed.path) if parsed.scheme else raw
    name = PurePosixPath(candidate.replace("\\", "/")).name
    return name or "unknown"


def _workspace_metadata(user_dir: Path | None) -> _WorkspaceMetadata:
    metadata = _WorkspaceMetadata({}, {}, {}, set())
    if user_dir is None:
        return metadata
    root = user_dir / "workspaceStorage"
    if not root.is_dir():
        return metadata

    for workspace in sorted(path for path in root.iterdir() if path.is_dir()):
        workspace_id = workspace.name
        metadata.projects[workspace_id] = _workspace_project(workspace)
        database = workspace / "state.vscdb"
        if not database.is_file():
            continue
        connection: sqlite3.Connection | None = None
        try:
            connection = _connect_readonly(database)
            with connection:
                if not _table_exists(connection, "ItemTable"):
                    continue
                row = connection.execute(
                    "SELECT value FROM ItemTable WHERE key='composer.composerData'"
                ).fetchone()
                document = _json_object(row[0]) if row else None
                composers = document.get("allComposers", []) if document else []
                if not isinstance(composers, list):
                    continue
                for composer in composers:
                    if not isinstance(composer, dict):
                        continue
                    composer_id = composer.get("composerId")
                    if not isinstance(composer_id, str) or not composer_id:
                        continue
                    metadata.composer_workspaces.setdefault(composer_id, workspace_id)
                    title = composer.get("name")
                    if isinstance(title, str) and title.strip():
                        metadata.titles.setdefault(composer_id, title.strip())
                    if composer.get("isBestOfNSubcomposer"):
                        metadata.excluded_composers.add(composer_id)
        except (CursorSourceError, sqlite3.Error):
            # Workspace metadata only improves the project label.  The global
            # database remains the authority for actual conversation text.
            continue
        finally:
            if connection is not None:
                connection.close()
    return metadata


def _apply_global_headers(
    connection: sqlite3.Connection,
    metadata: _WorkspaceMetadata,
) -> dict[str, dict[str, object]]:
    headers: dict[str, dict[str, object]] = {}
    if not _table_exists(connection, "composerHeaders"):
        return headers
    columns = {
        str(row[1]) for row in connection.execute("PRAGMA table_info(composerHeaders)")
    }
    if "composerId" not in columns:
        return headers

    def optional(name: str) -> str:
        return name if name in columns else f"NULL AS {name}"

    for row in connection.execute(
        "SELECT composerId, "
        + ", ".join(
            optional(name)
            for name in (
                "workspaceId",
                "createdAt",
                "lastUpdatedAt",
                "isArchived",
                "isSubagent",
                "value",
            )
        )
        + " FROM composerHeaders"
    ):
        composer_id, workspace_id, created, updated, archived, subagent, value = row
        if not isinstance(composer_id, str) or not composer_id:
            continue
        parsed = _json_object(value) or {}
        headers[composer_id] = {
            "createdAt": created,
            "lastUpdatedAt": updated,
            "isArchived": bool(archived),
            "isSubagent": bool(subagent),
            **parsed,
        }
        if isinstance(workspace_id, str) and workspace_id:
            metadata.composer_workspaces[composer_id] = workspace_id
        if subagent or parsed.get("isBestOfNSubcomposer"):
            metadata.excluded_composers.add(composer_id)
    return headers


def _iso_timestamp(value: object) -> str:
    if isinstance(value, (int, float)):
        seconds = float(value) / 1000 if abs(float(value)) >= 10**11 else float(value)
        try:
            return (
                datetime.fromtimestamp(seconds, tz=timezone.utc)
                .isoformat()
                .replace("+00:00", "Z")
            )
        except (OSError, OverflowError, ValueError):
            return ""
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.isdigit():
            return _iso_timestamp(int(stripped))
        return stripped
    return ""


def _rich_text_plain(value: object) -> str:
    """Best-effort plain text fallback for Cursor's JSON rich-text editor state."""
    if not isinstance(value, str) or not value.strip():
        return ""
    try:
        document = json.loads(value)
    except json.JSONDecodeError:
        return value.strip()

    parts: list[str] = []

    def visit(node: object) -> None:
        if isinstance(node, dict):
            text = node.get("text")
            if isinstance(text, str):
                parts.append(text)
            for key, child in node.items():
                if key != "text":
                    visit(child)
            if node.get("type") in {"paragraph", "linebreak", "hardBreak"}:
                parts.append("\n")
        elif isinstance(node, list):
            for child in node:
                visit(child)

    visit(document)
    return "".join(parts).strip()


def _query_bubbles(
    connection: sqlite3.Connection,
    composer_id: str,
) -> dict[str, dict[str, object]]:
    prefix = f"bubbleId:{composer_id}:"
    query = """SELECT substr(key, ?),
                      json_extract(value, '$.type'),
                      json_extract(value, '$.text'),
                      json_extract(value, '$.richText'),
                      json_extract(value, '$.createdAt')
               FROM cursorDiskKV
               WHERE key GLOB ? AND json_valid(value)"""
    bubbles: dict[str, dict[str, object]] = {}
    for bubble_id, bubble_type, text, rich_text, created_at in connection.execute(
        query, (len(prefix) + 1, prefix + "*")
    ):
        bubbles[str(bubble_id)] = {
            "bubbleId": str(bubble_id),
            "type": bubble_type,
            "text": text,
            "richText": rich_text,
            "createdAt": created_at,
        }
    return bubbles


def _message_records(
    connection: sqlite3.Connection,
    composer_id: str,
    composer: dict[str, object],
) -> tuple[list[dict[str, object]], list[str]]:
    issues: list[str] = []
    headers = composer.get("fullConversationHeadersOnly")
    ordered: list[tuple[int, object, dict[str, object]]] = []

    if isinstance(headers, list) and headers:
        bubbles = _query_bubbles(connection, composer_id)
        fallback_map = composer.get("conversationMap")
        fallback_map = fallback_map if isinstance(fallback_map, dict) else {}
        for ordinal, header in enumerate(headers):
            if not isinstance(header, dict):
                continue
            bubble_id = header.get("bubbleId")
            if not isinstance(bubble_id, str) or not bubble_id:
                continue
            bubble = bubbles.get(bubble_id)
            if bubble is None:
                fallback = fallback_map.get(bubble_id)
                bubble = fallback if isinstance(fallback, dict) else None
            if bubble is None:
                issues.append(f"Cursor 会话 {composer_id}: 缺少消息 {bubble_id}")
                continue
            ordered.append((ordinal, header.get("type"), bubble))
    else:
        # Older builds embedded full bubbles directly in composerData.
        conversation = composer.get("conversation")
        if isinstance(conversation, list):
            for ordinal, bubble in enumerate(conversation):
                if isinstance(bubble, dict):
                    ordered.append((ordinal, bubble.get("type"), bubble))

    records: list[dict[str, object]] = []
    for ordinal, header_type, bubble in ordered:
        raw_type = bubble.get("type", header_type)
        try:
            bubble_type = int(raw_type)
        except (TypeError, ValueError):
            continue
        role = "user" if bubble_type == 1 else "assistant" if bubble_type == 2 else ""
        if not role:
            continue
        direct_text = bubble.get("text")
        text = direct_text if isinstance(direct_text, str) else ""
        if role == "user" and not text.strip():
            text = _rich_text_plain(bubble.get("richText"))
        if not text.strip():
            continue  # tool calls and execution state are type=2 but have no dialogue text.
        bubble_id = bubble.get("bubbleId")
        if not isinstance(bubble_id, str) or not bubble_id:
            digest = hashlib.sha256(
                f"{composer_id}:{ordinal}:{role}".encode()
            ).hexdigest()[:24]
            bubble_id = f"generated-{digest}"
        records.append(
            {
                "kind": "message",
                "schema": EXPORT_SCHEMA,
                "session_id": composer_id,
                "bubble_id": bubble_id,
                "ordinal": ordinal,
                "role": role,
                "created_at": _iso_timestamp(bubble.get("createdAt")),
                "text": text,
            }
        )
    return records, issues


def _safe_session_name(session_id: str) -> str:
    safe = _SAFE_NAME.sub("_", session_id).strip("._")
    if safe and len(safe) <= 160:
        return safe
    return hashlib.sha256(session_id.encode()).hexdigest()


def _read_archive(path: Path) -> bytes:
    if not path.exists():
        return b""
    try:
        with gzip.open(path, "rb") as handle:
            return handle.read()
    except (OSError, EOFError, gzip.BadGzipFile) as exc:
        raise CursorSourceError(f"Cursor 现有底片无法读取,已停止更新: {path}") from exc


def _event_state(data: bytes) -> dict[str, dict[str, object]]:
    state: dict[str, dict[str, object]] = {}
    for number, raw_line in enumerate(data.splitlines(), 1):
        if not raw_line.strip():
            continue
        try:
            row = json.loads(raw_line)
        except (UnicodeError, json.JSONDecodeError) as exc:
            raise CursorSourceError(f"Cursor 底片第 {number} 行不是有效 JSON") from exc
        if not isinstance(row, dict):
            continue
        kind = row.get("kind")
        if kind == "session":
            state["session"] = row
        elif kind == "message" and isinstance(row.get("bubble_id"), str):
            state[f"message:{row['bubble_id']}"] = row
    return state


def _without_revision(record: dict[str, object]) -> dict[str, object]:
    return {key: value for key, value in record.items() if key != "revision"}


def _append_revision(
    output: bytearray,
    state: dict[str, dict[str, object]],
    key: str,
    record: dict[str, object],
) -> bool:
    previous = state.get(key)
    if previous is not None and _without_revision(previous) == record:
        return False
    revision = 1
    if previous is not None:
        try:
            revision = int(previous.get("revision", 0)) + 1
        except (TypeError, ValueError):
            revision = 1
    written = {**record, "revision": revision}
    output.extend(
        json.dumps(
            written,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        + b"\n"
    )
    state[key] = written
    return True


def materialize_sessions(
    source: Path,
    raw_dir: Path,
    output_dir: Path,
) -> CursorExportResult:
    """Create append-only per-session candidates ready for ``archive_session``."""
    database, user_dir = locate_database(source)
    metadata = _workspace_metadata(user_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = Path(raw_dir)
    issues: list[str] = []
    files: list[Path] = []
    sessions_seen = messages_exported = 0

    connection = _connect_readonly(database)
    try:
        connection.execute("BEGIN")  # one consistent view while Cursor may keep writing.
        if not _table_exists(connection, "cursorDiskKV"):
            raise CursorSourceError("Cursor 数据库缺少 cursorDiskKV,格式暂不受支持")
        headers = _apply_global_headers(connection, metadata)
        rows = connection.execute(
            "SELECT key, value FROM cursorDiskKV WHERE key GLOB 'composerData:*'"
        )
        for key, value in rows:
            sessions_seen += 1
            composer = _json_object(value)
            composer_id = str(key).split(":", 1)[-1]
            if composer is None:
                if composer_id == "empty-state-draft":
                    continue  # Cursor's reserved new-chat placeholder is not a session.
                issues.append(f"Cursor 会话 {composer_id}: composerData 不是有效 JSON")
                continue
            value_id = composer.get("composerId")
            if isinstance(value_id, str) and value_id:
                composer_id = value_id
            if composer_id in metadata.excluded_composers:
                continue

            records, record_issues = _message_records(
                connection, composer_id, composer
            )
            issues.extend(record_issues)
            if not records or not any(row["role"] == "user" for row in records):
                continue

            header = headers.get(composer_id, {})
            workspace_id = metadata.composer_workspaces.get(composer_id, "")
            identifier = composer.get("workspaceIdentifier")
            if not workspace_id and isinstance(identifier, dict):
                possible_id = identifier.get("id")
                if isinstance(possible_id, str):
                    workspace_id = possible_id
            project = metadata.projects.get(workspace_id, "unknown")
            created_at = _iso_timestamp(
                composer.get("createdAt") or header.get("createdAt")
            )
            updated_at = _iso_timestamp(
                composer.get("lastUpdatedAt")
                or header.get("lastUpdatedAt")
                or records[-1].get("created_at")
            )
            session_record: dict[str, object] = {
                "kind": "session",
                "schema": EXPORT_SCHEMA,
                "source": "cursor",
                "session_id": composer_id,
                "project": project,
                "workspace_id": workspace_id,
                "title": metadata.titles.get(composer_id, ""),
                "created_at": created_at,
                "updated_at": updated_at,
            }

            filename = _safe_session_name(composer_id) + ".jsonl"
            output = output_dir / filename
            archived = raw_dir / "cursor" / (filename + ".gz")
            base = _read_archive(archived)
            if base and not base.endswith(b"\n"):
                base += b"\n"
            state = _event_state(base)
            candidate = bytearray(base)
            changed = _append_revision(candidate, state, "session", session_record)
            for record in records:
                changed = (
                    _append_revision(
                        candidate,
                        state,
                        f"message:{record['bubble_id']}",
                        record,
                    )
                    or changed
                )
            output.write_bytes(candidate)
            mtime_ns = database.stat().st_mtime_ns
            if not changed and archived.exists():
                mtime_ns = archived.stat().st_mtime_ns
            os.utime(output, ns=(mtime_ns, mtime_ns))
            files.append(output)
            messages_exported += len(records)
    except sqlite3.Error as exc:
        raise CursorSourceError("Cursor 本地历史数据库查询失败") from exc
    finally:
        connection.close()

    return CursorExportResult(
        files=tuple(files),
        sessions_seen=sessions_seen,
        sessions_exported=len(files),
        messages_exported=messages_exported,
        issues=tuple(issues),
    )


def _open_text(path: Path):
    if path.name.endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8", errors="replace")
    return path.open(encoding="utf-8", errors="replace")


def session_id_of(path: Path) -> str:
    try:
        with _open_text(path) as handle:
            for line in handle:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if (
                    isinstance(row, dict)
                    and row.get("kind") == "session"
                    and row.get("session_id")
                ):
                    return str(row["session_id"])
    except OSError:
        pass
    return path.name.removesuffix(".gz").removesuffix(".jsonl")


def parse_session(path: Path) -> SessionDialogue | None:
    """Parse the latest revision of each normalized Cursor message event."""
    stat = path.stat()
    session: dict[str, object] = {}
    messages: dict[str, dict[str, object]] = {}
    with _open_text(path) as handle:
        for line in handle:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(row, dict):
                continue
            if row.get("kind") == "session":
                session = row
            elif row.get("kind") == "message" and isinstance(
                row.get("bubble_id"), str
            ):
                messages[str(row["bubble_id"])] = row

    ordered = sorted(
        messages.values(),
        key=lambda row: (
            int(row.get("ordinal", 0))
            if str(row.get("ordinal", 0)).lstrip("-").isdigit()
            else 0,
            str(row.get("bubble_id", "")),
        ),
    )
    dialogue: list[Message] = []
    pending_assistant = ""
    last_timestamp = ""
    for row in ordered:
        role = row.get("role")
        text = row.get("text")
        if role not in {"user", "assistant"} or not isinstance(text, str):
            continue
        text = text.strip()
        if not text:
            continue
        created_at = row.get("created_at")
        if isinstance(created_at, str) and created_at:
            last_timestamp = created_at
        if role == "user":
            if pending_assistant:
                dialogue.append(Message(role="assistant", text=pending_assistant))
                pending_assistant = ""
            dialogue.append(Message(role="user", text=text))
        else:
            # Cursor emits progress/tool narration and a final response as
            # consecutive assistant bubbles.  Keep the last non-empty bubble
            # before the next user turn, matching the Codex final-answer policy.
            pending_assistant = text
    if pending_assistant:
        dialogue.append(Message(role="assistant", text=pending_assistant))
    if not dialogue:
        return None

    session_id = str(session.get("session_id") or session_id_of(path))
    project = str(session.get("project") or "unknown")
    fallback_timestamp = str(
        session.get("updated_at") or session.get("created_at") or ""
    )
    return SessionDialogue(
        source="cursor",
        session_id=session_id,
        project=project,
        date=to_local_date(last_timestamp or fallback_timestamp),
        file_mtime_ns=stat.st_mtime_ns,
        file_size=stat.st_size,
        messages=dialogue,
    )
