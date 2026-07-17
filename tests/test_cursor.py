"""Cursor local SQLite extraction and append-only archive contract."""

from __future__ import annotations

import gzip
import json
import os
import sqlite3
from types import SimpleNamespace

import numpy as np

from cmem.cli import cmd_index
from cmem.cursor_extractor import materialize_sessions, parse_session
from cmem.embedder import DIM
from cmem.raw import archive_session, archive_source_of
from cmem.store import Store


SESSION_ID = "11111111-2222-4333-8444-555555555555"
WORKSPACE_ID = "workspace-hash"


def _bubble(bubble_id, bubble_type, text, timestamp, **extra):
    return {
        "bubbleId": bubble_id,
        "type": bubble_type,
        "text": text,
        "createdAt": timestamp,
        **extra,
    }


def build_cursor_fixture(tmp_path):
    user_dir = tmp_path / "Cursor" / "User"
    global_db = user_dir / "globalStorage" / "state.vscdb"
    global_db.parent.mkdir(parents=True)
    workspace = user_dir / "workspaceStorage" / WORKSPACE_ID
    workspace.mkdir(parents=True)
    (workspace / "workspace.json").write_text(
        json.dumps({"folder": "file:///tmp/demo-project"}),
        encoding="utf-8",
    )

    workspace_db = sqlite3.connect(workspace / "state.vscdb")
    workspace_db.execute("CREATE TABLE ItemTable(key TEXT UNIQUE, value BLOB)")
    workspace_db.execute(
        "INSERT INTO ItemTable VALUES(?, ?)",
        (
            "composer.composerData",
            json.dumps(
                {
                    "allComposers": [
                        {
                            "composerId": SESSION_ID,
                            "name": "Cursor fixture",
                            "isBestOfNSubcomposer": False,
                        }
                    ]
                }
            ),
        ),
    )
    workspace_db.commit()
    workspace_db.close()

    bubbles = [
        _bubble("user-1", 1, "第一个问题", "2026-07-15T01:00:00Z"),
        _bubble("progress-1", 2, "正在检查项目", "2026-07-15T01:00:01Z"),
        _bubble(
            "tool-1",
            2,
            "",
            "2026-07-15T01:00:02Z",
            capabilityType=15,
            codebaseContextChunks=[{"contents": "private-tool-payload"}],
        ),
        _bubble("final-1", 2, "最终答案一", "2026-07-15T01:00:03Z"),
        _bubble(
            "user-2",
            1,
            "",
            "2026-07-15T02:00:00Z",
            richText=json.dumps(
                {
                    "root": {
                        "children": [
                            {"type": "paragraph", "children": [{"text": "追问二"}]}
                        ]
                    }
                }
            ),
        ),
        _bubble("final-2", 2, "最终答案二", "2026-07-15T02:00:01Z"),
    ]
    composer = {
        "composerId": SESSION_ID,
        "createdAt": 1752541200000,
        "lastUpdatedAt": 1752544801000,
        "fullConversationHeadersOnly": [
            {"bubbleId": bubble["bubbleId"], "type": bubble["type"]}
            for bubble in bubbles
        ],
    }

    connection = sqlite3.connect(global_db)
    connection.executescript(
        """
        CREATE TABLE cursorDiskKV(key TEXT UNIQUE ON CONFLICT REPLACE, value BLOB);
        CREATE TABLE composerHeaders(
            composerId TEXT PRIMARY KEY, workspaceId TEXT, createdAt INTEGER,
            lastUpdatedAt INTEGER, isArchived INTEGER, isSubagent INTEGER,
            recency INTEGER, checkpointAt INTEGER, value TEXT
        );
        """
    )
    connection.execute(
        "INSERT INTO cursorDiskKV VALUES(?, ?)",
        (f"composerData:{SESSION_ID}", json.dumps(composer)),
    )
    connection.execute(
        "INSERT INTO cursorDiskKV VALUES(?, ?)",
        ("composerData:empty-state-draft", "reserved-placeholder"),
    )
    connection.executemany(
        "INSERT INTO cursorDiskKV VALUES(?, ?)",
        [
            (
                f"bubbleId:{SESSION_ID}:{bubble['bubbleId']}",
                json.dumps(bubble),
            )
            for bubble in bubbles
        ],
    )
    connection.execute(
        "INSERT INTO composerHeaders VALUES(?, ?, ?, ?, 0, 0, 0, 0, ?)",
        (SESSION_ID, WORKSPACE_ID, 1752541200000, 1752544801000, "{}"),
    )
    connection.commit()
    connection.close()
    return user_dir, global_db


def test_cursor_export_keeps_verbatim_dialogue_without_tool_payload(tmp_path):
    user_dir, _ = build_cursor_fixture(tmp_path)
    raw_dir = tmp_path / "raw"
    output_dir = tmp_path / "export"

    result = materialize_sessions(user_dir, raw_dir, output_dir)

    assert result.sessions_seen == 2
    assert result.sessions_exported == 1
    assert result.messages_exported == 5  # empty tool bubble is intentionally omitted.
    assert result.issues == ()
    exported = result.files[0]
    exported_text = exported.read_text(encoding="utf-8")
    assert "第一个问题" in exported_text
    assert "最终答案二" in exported_text
    assert "private-tool-payload" not in exported_text

    session = parse_session(exported)
    assert session is not None
    assert session.source == "cursor"
    assert session.session_id == SESSION_ID
    assert session.project == "demo-project"
    assert session.date == "2026-07-15"
    assert [(message.role, message.text) for message in session.messages] == [
        ("user", "第一个问题"),
        ("assistant", "最终答案一"),
        ("user", "追问二"),
        ("assistant", "最终答案二"),
    ]


def test_cursor_export_is_idempotent_and_appends_message_revisions(tmp_path):
    user_dir, global_db = build_cursor_fixture(tmp_path)
    raw_dir = tmp_path / "raw"
    first_dir = tmp_path / "first"
    first = materialize_sessions(user_dir, raw_dir, first_dir).files[0]
    stat = first.stat()
    assert archive_session(
        first,
        stat.st_mtime_ns,
        raw_dir,
        source="cursor",
        source_root=first_dir,
    )
    archived = raw_dir / "cursor" / (first.name + ".gz")
    with gzip.open(archived, "rb") as handle:
        original = handle.read()
    assert archive_source_of(archived, raw_dir) == "cursor"

    unchanged_dir = tmp_path / "unchanged"
    unchanged = materialize_sessions(user_dir, raw_dir, unchanged_dir).files[0]
    assert unchanged.read_bytes() == original
    unchanged_stat = unchanged.stat()
    assert not archive_session(
        unchanged,
        unchanged_stat.st_mtime_ns,
        raw_dir,
        source="cursor",
        source_root=unchanged_dir,
    )

    connection = sqlite3.connect(global_db)
    key = f"bubbleId:{SESSION_ID}:final-2"
    payload = json.loads(
        connection.execute(
            "SELECT value FROM cursorDiskKV WHERE key=?", (key,)
        ).fetchone()[0]
    )
    payload["text"] = "最终答案二（修订）"
    connection.execute(
        "UPDATE cursorDiskKV SET value=? WHERE key=?",
        (json.dumps(payload), key),
    )
    connection.commit()
    connection.close()
    db_stat = global_db.stat()
    os.utime(global_db, ns=(db_stat.st_atime_ns, db_stat.st_mtime_ns + 1))

    revised_dir = tmp_path / "revised"
    revised = materialize_sessions(user_dir, raw_dir, revised_dir).files[0]
    assert revised.read_bytes().startswith(original)
    assert len(revised.read_bytes()) > len(original)
    revised_stat = revised.stat()
    assert archive_session(
        revised,
        revised_stat.st_mtime_ns,
        raw_dir,
        source="cursor",
        source_root=revised_dir,
    )

    session = parse_session(archived)
    assert session is not None
    assert session.messages[-1].text == "最终答案二（修订）"
    with gzip.open(archived, "rb") as handle:
        updated = handle.read()
    assert updated.startswith(original), "Cursor 修订覆盖了旧底片,违反 append-only 契约"


def test_cursor_raw_survives_source_database_deletion(tmp_path):
    user_dir, global_db = build_cursor_fixture(tmp_path)
    raw_dir = tmp_path / "raw"
    output_dir = tmp_path / "export"
    exported = materialize_sessions(user_dir, raw_dir, output_dir).files[0]
    stat = exported.stat()
    archive_session(
        exported,
        stat.st_mtime_ns,
        raw_dir,
        source="cursor",
        source_root=output_dir,
    )
    archived = raw_dir / "cursor" / (exported.name + ".gz")

    global_db.unlink()

    session = parse_session(archived)
    assert session is not None
    assert session.session_id == SESSION_ID
    assert "最终答案二" in " ".join(message.text for message in session.messages)


def test_cursor_runs_through_the_real_incremental_index_pipeline(
    tmp_path, monkeypatch, capsys
):
    user_dir, _ = build_cursor_fixture(tmp_path)
    raw_dir = tmp_path / "raw"
    database = tmp_path / "memory.sqlite3"

    class FakeEmbedder:
        def encode_texts(self, texts):
            return np.zeros((len(texts), DIM), dtype=np.float32)

    monkeypatch.setattr("cmem.embedder.Embedder", FakeEmbedder)
    args = SimpleNamespace(
        raw_dir=str(raw_dir),
        source=str(tmp_path / "no-claude"),
        codex_source=str(tmp_path / "no-codex"),
        cursor_source=str(user_dir),
        provider=["cursor"],
        db=str(database),
        no_heartbeat=True,
    )

    assert cmd_index(args) == 0
    first_output = capsys.readouterr().out
    assert "导出 1 场 / 5 条消息" in first_output
    assert "本次索引 1 个会话" in first_output
    store = Store(database)
    assert store.stats()["sources"]["cursor"]["sessions"] == 1
    assert store.conn.execute(
        "SELECT COUNT(*) FROM chunks WHERE source='cursor'"
    ).fetchone()[0] == 2

    assert cmd_index(args) == 0
    second_output = capsys.readouterr().out
    assert "本次索引 0 个会话" in second_output
    assert "新存档 0 份" in second_output
