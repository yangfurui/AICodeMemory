"""档案/投影契约测试 — 任何未来改动不得违反。

契约一:raw 是唯一永久档案;源消失不删 raw,截断/改写不覆盖 raw。
契约二:text/chunks 是当前检索投影;允许按会话原子替换,失败必须回滚。
契约三:模型升级不改 text;提取升级能从 raw 重建等价投影。

测试使用假向量,不加载真模型——契约锁定的是 store/raw 层的数据行为。
"""

import gzip
import json
import os
import sqlite3
from types import SimpleNamespace

import numpy as np
import pytest

from cmem.chunker import Chunk, chunk_session
from cmem.cli import cmd_index
from cmem.embedder import DIM, MODEL_NAME
from cmem.extractor import parse_session, session_id_of
from cmem.raw import (
    ArchiveConflictError,
    archive_path_for,
    archive_session,
    content_size,
    iter_archived_files,
)
from cmem.store import Store


def fake_encode(texts):
    rng = np.random.default_rng(42)
    return rng.random((len(texts), DIM), dtype=np.float32)


def projection_chunks(*texts):
    return [
        Chunk(
            source="claude",
            session_id="projection-session",
            project="proj-a",
            date="2026-07-14",
            index=i,
            text=text,
        )
        for i, text in enumerate(texts)
    ]


def write_session_jsonl(dir_, sid, user_text, assistant_text, ts="2026-07-08T03:00:00Z"):
    """造一个最小可解析的 Claude Code 会话文件。"""
    f = dir_ / f"{sid}.jsonl"
    rows = [
        {"type": "user", "cwd": "/tmp/proj-a", "timestamp": ts,
         "message": {"content": user_text}},
        {"type": "assistant", "cwd": "/tmp/proj-a", "timestamp": ts,
         "message": {"content": [{"type": "text", "text": assistant_text}]}},
    ]
    f.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    return f


def index_one(store, path, raw_dir):
    """cli 索引循环对单文件的等价操作(假向量),与 cmd_index 行为保持镜像。"""
    sid = session_id_of(path)
    stat = path.stat()
    size = content_size(path)
    if not path.name.endswith(".gz"):
        archive_session(path, stat.st_mtime_ns, raw_dir)
    if sid.startswith("agent-"):
        return  # 侧链只进底片不进索引
    if not store.should_process("claude", sid, stat.st_mtime_ns, size):
        return
    sess = parse_session(path)
    chunks = chunk_session(sess)
    store.index_session(
        "claude", sid, stat.st_mtime_ns, size,
        chunks, fake_encode([c.text for c in chunks]),
    )


@pytest.fixture
def env(tmp_path):
    source = tmp_path / "projects" / "-tmp-proj-a"
    source.mkdir(parents=True)
    raw_dir = tmp_path / "raw"
    store = Store(tmp_path / "memory.sqlite3")
    store.pending_migration()  # fresh,落 meta
    return source, raw_dir, store


def test_contract_source_deletion_keeps_archive(env):
    """契约一:源消失 → 块仍在、raw 存档仍在、且可从存档完整重提取。"""
    source, raw_dir, store = env
    f1 = write_session_jsonl(source, "sess-1", "问题一", "答案一:根因是缺模块")
    f2 = write_session_jsonl(source, "sess-2", "问题二", "答案二")
    index_one(store, f1, raw_dir)
    index_one(store, f2, raw_dir)
    n_before = store.stats()["chunks"]
    assert n_before > 0

    # 模拟 Claude Code 的 30 天清理:sess-1 源文件消失
    gz1 = archive_path_for(f1, raw_dir)
    f1.unlink()

    # 再跑一轮索引(源只剩 sess-2 + raw 全量)——任何路径都不得删 sess-1 的块
    index_one(store, f2, raw_dir)
    for gz in iter_archived_files(raw_dir):
        index_one(store, gz, raw_dir)

    assert store.stats()["chunks"] == n_before, "源消失导致块丢失,违反档案契约"
    assert gz1.exists(), "raw 存档被删除,违反档案契约"
    # 存档可独立重提取出同一会话
    sess = parse_session(gz1)
    assert sess is not None and sess.session_id == "sess-1"
    assert "根因是缺模块" in " ".join(m.text for m in sess.messages)


def test_upgrade_rebuilds_equivalent_projection_without_touching_raw(env):
    """reembed 不改 text;reextract 可从同一 raw 重建等价投影。"""
    source, raw_dir, store = env
    for i in range(3):
        index_one(store, write_session_jsonl(source, f"sess-{i}", f"问题{i}", f"答案{i}"), raw_dir)
    texts_before = set(r[0] for r in store.conn.execute("SELECT text FROM chunks"))
    raw_before = len(list(iter_archived_files(raw_dir)))
    assert texts_before

    # 路径一:模型变更 → reembed(text 一字不动)
    store.conn.execute("UPDATE meta SET value='some-old-model' WHERE key='model'")
    store.conn.commit()
    assert store.pending_migration() == "reembed"
    store.reembed_all(fake_encode)
    texts_after_reembed = set(r[0] for r in store.conn.execute("SELECT text FROM chunks"))
    assert texts_after_reembed == texts_before, "reembed 改动了 text,违反档案契约"
    assert store.pending_migration() == "none"

    # 路径二:提取算法变更 → reextract(清账本,投影由 raw 重建)
    store.conn.execute("UPDATE meta SET value='0' WHERE key='extract_version'")
    store.conn.commit()
    assert store.pending_migration() == "reextract"
    store.reset_processed_ledger()
    assert store.stats()["chunks"] == len(texts_before), "reset 账本不得动 chunks"
    # 源已全部"消失"也无妨:从 raw 全量重提取
    for f in source.iterdir():
        f.unlink()
    for gz in iter_archived_files(raw_dir):
        index_one(store, gz, raw_dir)
    store.finalize_migration()

    texts_after = set(r[0] for r in store.conn.execute("SELECT text FROM chunks"))
    assert texts_after == texts_before, "reextract 后 text 集合变化,违反档案契约"
    assert len(list(iter_archived_files(raw_dir))) == raw_before
    assert store.pending_migration() == "none"


def test_session_update_replaces_current_projection_without_stale_chunks(env):
    """新投影可以比旧投影少;旧块不得残留在检索结果中。"""
    _, _, store = env
    old = projection_chunks("保留的内容", "已过期的旧块")
    store.index_session("claude", "projection-session", 1, 100, old, fake_encode(old))

    new = projection_chunks("更新后的当前内容")
    store.index_session("claude", "projection-session", 2, 80, new, fake_encode(new))

    rows = store.conn.execute(
        "SELECT chunk_index, text FROM chunks WHERE session_id = ? ORDER BY chunk_index",
        ("projection-session",),
    ).fetchall()
    assert rows == [(0, "更新后的当前内容")]
    assert store.conn.execute(
        "SELECT COUNT(*) FROM chunks_fts WHERE session_id = ?",
        ("projection-session",),
    ).fetchone()[0] == 1


def test_session_projection_replacement_rolls_back_on_failure(env):
    """删旧与插新同事务;插入中途失败时旧 chunks/FTS/账本都保留。"""
    _, _, store = env
    old = projection_chunks("旧投影一", "旧投影二")
    store.index_session("claude", "projection-session", 1, 100, old, fake_encode(old))
    store.conn.execute(
        """CREATE TRIGGER reject_projection BEFORE INSERT ON chunks
           WHEN NEW.text = '触发回滚'
           BEGIN SELECT RAISE(ABORT, '模拟插入失败'); END"""
    )
    store.conn.commit()

    broken = projection_chunks("触发回滚")
    with pytest.raises(sqlite3.IntegrityError, match="模拟插入失败"):
        store.index_session(
            "claude", "projection-session", 2, 50, broken, fake_encode(broken)
        )

    assert store.conn.execute(
        "SELECT text FROM chunks WHERE session_id = ? ORDER BY chunk_index",
        ("projection-session",),
    ).fetchall() == [("旧投影一",), ("旧投影二",)]
    assert store.conn.execute(
        "SELECT COUNT(*) FROM chunks_fts WHERE session_id = ?",
        ("projection-session",),
    ).fetchone()[0] == 2
    assert store.conn.execute(
        "SELECT mtime_ns, size FROM processed WHERE source = ? AND session_id = ?",
        ("claude", "projection-session"),
    ).fetchone() == (1, 100)


def test_raw_accepts_append_only_update(tmp_path):
    """新源完整包含旧底片时,可安全更新 canonical raw。"""
    source = tmp_path / "project" / "session.jsonl"
    source.parent.mkdir()
    raw_dir = tmp_path / "raw"
    first = '{"message":"第一条"}\n'
    source.write_text(first, encoding="utf-8")
    assert archive_session(source, 10**18, raw_dir)

    appended = first + '{"message":"追加内容"}\n'
    source.write_text(appended, encoding="utf-8")
    assert archive_session(source, 10**18 + 1, raw_dir)

    with gzip.open(archive_path_for(source, raw_dir), "rt", encoding="utf-8") as f:
        assert f.read() == appended


def test_raw_rejects_truncation_or_rewrite_and_keeps_old_archive(tmp_path):
    """源截断/改写时显式失败,已存档的唯一底片一字不动。"""
    source = tmp_path / "project" / "session.jsonl"
    source.parent.mkdir()
    raw_dir = tmp_path / "raw"
    original = '{"message":"不可丢的旧内容"}\n'
    source.write_text(original, encoding="utf-8")
    assert archive_session(source, 10**18, raw_dir)

    source.write_text('{"message":"被改写"}\n', encoding="utf-8")
    with pytest.raises(ArchiveConflictError, match="拒绝覆盖"):
        archive_session(source, 10**18 + 1, raw_dir)

    archived = archive_path_for(source, raw_dir)
    with gzip.open(archived, "rt", encoding="utf-8") as f:
        assert f.read() == original
    assert archived.stat().st_mtime_ns == 10**18


def test_index_reports_raw_conflict_as_failure(tmp_path, capsys):
    """非追加改写不能只跳过单文件;整次 index 必须非零退出且不刷心跳。"""
    source_root = tmp_path / "projects"
    project = source_root / "-tmp-project"
    project.mkdir(parents=True)
    source = project / "session.jsonl"
    original = '{"type":"unknown-a"}\n'
    source.write_text(original, encoding="utf-8")
    stat = source.stat()
    raw_dir = tmp_path / "raw"
    assert archive_session(source, stat.st_mtime_ns, raw_dir)

    source.write_text('{"type":"unknown-b"}\n', encoding="utf-8")
    os.utime(
        source,
        ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000),
    )
    args = SimpleNamespace(
        raw_dir=str(raw_dir),
        source=str(source_root),
        codex_source=str(tmp_path / "no-codex"),
        provider=["claude"],
        db=str(tmp_path / "memory.sqlite3"),
        no_heartbeat=True,
    )

    assert cmd_index(args) == 1
    captured = capsys.readouterr()
    assert "拒绝覆盖" in captured.err
    assert "不刷新心跳" in captured.err
    with gzip.open(archive_path_for(source, raw_dir), "rt", encoding="utf-8") as f:
        assert f.read() == original


def test_agent_sidechain_archived_but_not_indexed(env):
    """agent-* 侧链:进底片(原始数据完整性),不进检索库(不是"说过的话")。"""
    source, raw_dir, store = env
    f = write_session_jsonl(source, "agent-abc123", "子代理任务", "子代理过程输出")
    index_one(store, f, raw_dir)
    assert archive_path_for(f, raw_dir).exists(), "agent 侧链未拍底片,原始数据不完整"
    n = store.conn.execute(
        "SELECT COUNT(*) FROM chunks WHERE session_id LIKE 'agent-%'"
    ).fetchone()[0]
    assert n == 0, "agent 侧链混入了检索库"


def test_legacy_single_source_layout_migrates_without_losing_text(tmp_path):
    """v0.3 单来源表升级:原文原地保留并标记 claude,只重建可再生索引。"""
    db = tmp_path / "memory.sqlite3"
    conn = sqlite3.connect(db)
    conn.executescript(
        """
        CREATE TABLE chunks (
            id TEXT PRIMARY KEY, session_id TEXT NOT NULL, project TEXT NOT NULL,
            date TEXT NOT NULL, chunk_index INTEGER NOT NULL, text TEXT NOT NULL,
            embedding BLOB NOT NULL
        );
        CREATE TABLE processed (session_id TEXT PRIMARY KEY, mtime INTEGER NOT NULL);
        CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
        CREATE VIRTUAL TABLE chunks_fts USING fts5(
            tokens, id UNINDEXED, session_id UNINDEXED
        );
        """
    )
    vec = np.zeros(DIM, dtype=np.float32).tobytes()
    conn.execute(
        "INSERT INTO chunks VALUES(?, ?, ?, ?, ?, ?, ?)",
        ("old-id", "legacy-session", "proj", "2026-07-08", 0, "不可丢原文", vec),
    )
    conn.execute(
        "INSERT INTO chunks_fts VALUES(?, ?, ?)",
        ("不可 丢 原文", "old-id", "legacy-session"),
    )
    conn.execute("INSERT INTO processed VALUES(?, ?)", ("legacy-session", 123))
    conn.executemany(
        "INSERT INTO meta VALUES(?, ?)",
        [("schema_version", "2"), ("extract_version", "2"), ("model", MODEL_NAME)],
    )
    conn.commit()
    conn.close()

    store = Store(db)
    row = store.conn.execute(
        "SELECT source, session_id, text FROM chunks"
    ).fetchone()
    assert row == ("claude", "legacy-session", "不可丢原文")
    assert store.stats()["chunks"] == 1
    assert store.pending_migration() == "none"
    meta = dict(store.conn.execute("SELECT key, value FROM meta"))
    assert meta["schema_version"] == "3"
    processed_cols = {r[1] for r in store.conn.execute("PRAGMA table_info(processed)")}
    assert {"source", "session_id", "mtime_ns", "size"} <= processed_cols
    processed = store.conn.execute(
        "SELECT source, session_id, mtime_ns, size FROM processed"
    ).fetchone()
    assert processed == ("claude", "legacy-session", 123_000_000_000, -1)
    assert not store.should_process("claude", "legacy-session", 123_999_999_999, 999)
    assert store.should_process("claude", "legacy-session", 124_000_000_000, 999)
