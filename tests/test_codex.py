"""Codex rollout parser and cross-source archive contracts."""

import json

import numpy as np

from cmem.chunker import chunk_session
from cmem.codex_extractor import parse_session, session_id_of
from cmem.embedder import DIM
from cmem.raw import (
    archive_path_for,
    archive_session,
    content_size,
    find_archive,
    verify_archives,
)
from cmem.store import Store


SID = "019f5979-c07c-7da1-a93d-2756433fc64b"


def fake_encode(texts):
    return np.ones((len(texts), DIM), dtype=np.float32)


def write_codex_rollout(root):
    day = root / "2026" / "07" / "13"
    day.mkdir(parents=True)
    path = day / f"rollout-2026-07-13T11-16-19-{SID}.jsonl"
    ts = "2026-07-13T03:16:19.242Z"
    rows = [
        {"type": "session_meta", "timestamp": ts, "payload": {
            "id": SID, "timestamp": ts, "cwd": "/tmp/AICodeMemory",
            "cli_version": "0.144.1", "source": "cli",
        }},
        {"type": "response_item", "timestamp": ts, "payload": {
            "type": "message", "role": "developer",
            "content": [{"type": "input_text", "text": "开发者指令不要入库"}],
        }},
        {"type": "response_item", "timestamp": ts, "payload": {
            "type": "message", "role": "user",
            "content": [{"type": "input_text", "text": "<environment_context>噪音</environment_context>"}],
        }},
        {"type": "response_item", "timestamp": ts, "payload": {
            "type": "message", "role": "user",
            "content": [{"type": "input_text", "text": "Codex 会保存原始会话吗？"}],
        }},
        {"type": "response_item", "timestamp": ts, "payload": {
            "type": "message", "role": "assistant", "phase": "commentary",
            "content": [{"type": "output_text", "text": "我先检查一下。"}],
        }},
        {"type": "response_item", "timestamp": ts, "payload": {
            "type": "custom_tool_call", "name": "exec", "input": "{}",
        }},
        {"type": "response_item", "timestamp": ts, "payload": {
            "type": "reasoning", "encrypted_content": "opaque",
        }},
        {"type": "event_msg", "timestamp": ts, "payload": {
            "type": "agent_message", "message": "重复的展示消息",
        }},
        {"type": "response_item", "timestamp": "2026-07-13T03:17:00Z", "payload": {
            "type": "message", "role": "assistant", "phase": "final_answer",
            "content": [{"type": "output_text", "text": "会，保存在 sessions 目录。"}],
        }},
        {"type": "future_unknown_event", "payload": {"anything": True}},
    ]
    path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8")
    return path


def test_codex_parser_keeps_only_user_and_final_answer(tmp_path):
    path = write_codex_rollout(tmp_path / "sessions")
    session = parse_session(path)
    assert session is not None
    assert session.source == "codex"
    assert session.session_id == SID == session_id_of(path)
    assert session.project == "AICodeMemory"
    assert [(m.role, m.text) for m in session.messages] == [
        ("user", "Codex 会保存原始会话吗？"),
        ("assistant", "会，保存在 sessions 目录。"),
    ]


def test_codex_raw_survives_source_deletion_and_reextracts(tmp_path):
    root = tmp_path / "sessions"
    raw_dir = tmp_path / "raw"
    path = write_codex_rollout(root)
    stat = path.stat()
    assert archive_session(
        path, stat.st_mtime_ns, raw_dir, source="codex", source_root=root
    )
    archived = archive_path_for(path, raw_dir, "codex", root)
    assert archived == raw_dir / "codex" / "2026" / "07" / "13" / (path.name + ".gz")
    assert archived.exists()
    assert find_archive(SID[:8], raw_dir, "codex") == [archived]
    assert archived.stat().st_mtime_ns == stat.st_mtime_ns
    assert content_size(archived) == path.stat().st_size
    path.unlink()

    restored = parse_session(archived)
    assert restored is not None and restored.session_id == SID
    assert "sessions 目录" in restored.messages[-1].text
    assert verify_archives(raw_dir) == (1, [])


def test_same_session_id_from_two_sources_does_not_collide(tmp_path):
    codex_path = write_codex_rollout(tmp_path / "sessions")
    codex_session = parse_session(codex_path)
    codex_chunks = chunk_session(codex_session)

    # Reuse normalized Codex content but tag it as Claude to exercise identity isolation.
    claude_session = parse_session(codex_path)
    claude_session.source = "claude"
    claude_chunks = chunk_session(claude_session)

    store = Store(tmp_path / "memory.sqlite3")
    store.pending_migration()
    stat = codex_path.stat()
    for source, chunks in (("codex", codex_chunks), ("claude", claude_chunks)):
        store.index_session(
            source, SID, stat.st_mtime_ns, stat.st_size,
            chunks, fake_encode([c.text for c in chunks]),
        )

    assert store.stats()["sessions"] == 2
    assert store.stats()["sources"]["claude"]["sessions"] == 1
    assert store.stats()["sources"]["codex"]["sessions"] == 1
    ids = [r[0] for r in store.conn.execute("SELECT id FROM chunks")]
    assert len(ids) == len(set(ids))
