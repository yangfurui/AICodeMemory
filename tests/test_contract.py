"""档案契约测试 — v0.3 数据安全版的两条不变量,任何未来改动不得违反。

契约一:源文件消失(Claude Code 30 天清理)后,库中的块与 raw 存档完整保留。
契约二:任何版本升级路径(模型变更/提取算法变更)都不减少 text 与 raw。

测试使用假向量,不加载真模型——契约锁定的是 store/raw 层的数据行为。
"""

import gzip
import json

import numpy as np
import pytest

from cmem.chunker import chunk_session
from cmem.embedder import DIM, MODEL_NAME
from cmem.extractor import parse_session, session_id_of
from cmem.raw import archive_path_for, archive_session, iter_archived_files
from cmem.store import Store


def fake_encode(texts):
    rng = np.random.default_rng(42)
    return rng.random((len(texts), DIM), dtype=np.float32)


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
    mtime = int(path.stat().st_mtime)
    archive_session(path, mtime, raw_dir)
    if sid.startswith("agent-"):
        return  # 侧链只进底片不进索引
    if not store.should_process(sid, mtime):
        return
    sess = parse_session(path)
    chunks = chunk_session(sess)
    store.index_session(sid, mtime, chunks, fake_encode([c.text for c in chunks]))


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


def test_contract_upgrade_never_loses_text(env):
    """契约二:reembed 与 reextract 两条升级路径,text 集合与 raw 均不减。"""
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

    # 路径二:提取算法变更 → reextract(清账本,块由 raw 重建覆盖,不减少)
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


def test_v02_legacy_meta_upgrades_smoothly(tmp_path):
    """v0.2 旧库(meta 无 extract_version)升级到三轴,不触发无谓重建。"""
    store = Store(tmp_path / "memory.sqlite3")
    store.conn.executemany(
        "INSERT INTO meta(key, value) VALUES(?, ?)",
        [("schema_version", "2"), ("model", MODEL_NAME)],
    )
    store.conn.commit()
    assert store.pending_migration() == "none"
    meta = dict(store.conn.execute("SELECT key, value FROM meta").fetchall())
    assert meta["extract_version"] == "2"
