"""SQLite storage — a single derived-cache file, rebuildable from source at any time.

设计要点:
- 块 ID = sha1(session_id:index),确定性 → INSERT OR REPLACE 天然幂等
- processed 表记录每个会话"处理时的源文件 mtime":mtime 变了(会话有新内容)
  就整会话重新索引(先删旧块再插,事务内完成)
- meta 表记 schema/模型版本:切块算法或 embedding 模型变更时,库整体作废重建
  (它只是缓存,重建无痛)
- 向量以 float32 BLOB 存储,检索时一次性载入 numpy 矩阵
"""

from __future__ import annotations

import hashlib
import sqlite3
from pathlib import Path

import numpy as np

from .chunker import Chunk
from .embedder import DIM, MODEL_NAME

DEFAULT_DB = Path.home() / ".claudememory" / "memory.sqlite3"
# v2:去噪清单扩充(task-notification/bash-*)+ FTS5 关键词索引
SCHEMA_VERSION = "2"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS chunks (
    id          TEXT PRIMARY KEY,
    session_id  TEXT NOT NULL,
    project     TEXT NOT NULL,
    date        TEXT NOT NULL,
    chunk_index INTEGER NOT NULL,
    text        TEXT NOT NULL,
    embedding   BLOB NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_chunks_session ON chunks(session_id);
CREATE TABLE IF NOT EXISTS processed (
    session_id  TEXT PRIMARY KEY,
    mtime       INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
-- 关键词召回通道:jieba 分词后空格拼接存入,MATCH 查询同用一套分词。
-- 存在意义:精确关键词(如错误码)命中但向量分平庸的块,必须有独立的
-- 参赛资格,不能被"向量 top-N 才能进候选池"焊死(v0.2 主修复)。
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
    tokens, id UNINDEXED, session_id UNINDEXED
);
"""


def chunk_id(session_id: str, index: int) -> str:
    return hashlib.sha1(f"{session_id}:{index}".encode()).hexdigest()


class Store:
    def __init__(self, db_path: Path = DEFAULT_DB) -> None:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(db_path)
        self.conn.executescript(_SCHEMA)
        self._check_versions()

    def _check_versions(self) -> None:
        """schema 或 embedding 模型变更 → 旧库作废,清空重建(缓存语义)。"""
        cur = self.conn.execute("SELECT key, value FROM meta")
        meta = dict(cur.fetchall())
        expected = {"schema_version": SCHEMA_VERSION, "model": MODEL_NAME}
        if meta and meta != expected:
            self.conn.executescript(
                "DELETE FROM chunks; DELETE FROM processed; DELETE FROM chunks_fts;"
            )
        self.conn.executemany(
            "INSERT OR REPLACE INTO meta(key, value) VALUES(?, ?)", expected.items()
        )
        self.conn.commit()

    # ---- 增量判定(借鉴 trace 的 sid+mtime 模式) ----

    def should_process(self, session_id: str, mtime: int) -> bool:
        row = self.conn.execute(
            "SELECT mtime FROM processed WHERE session_id = ?", (session_id,)
        ).fetchone()
        return row is None or mtime > row[0]

    # ---- 写入 ----

    def index_session(self, session_id: str, mtime: int, chunks: list[Chunk], vectors: np.ndarray) -> None:
        """一个会话的块整体落库(chunks + FTS 同步):先删旧再插新,单事务。"""
        from .tokenizer import tokenize

        assert len(chunks) == len(vectors)
        with self.conn:  # 事务:中途失败不留半成品
            self.conn.execute("DELETE FROM chunks WHERE session_id = ?", (session_id,))
            self.conn.execute("DELETE FROM chunks_fts WHERE session_id = ?", (session_id,))
            self.conn.executemany(
                "INSERT OR REPLACE INTO chunks VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    (
                        chunk_id(c.session_id, c.index),
                        c.session_id,
                        c.project,
                        c.date,
                        c.index,
                        c.text,
                        vectors[i].tobytes(),
                    )
                    for i, c in enumerate(chunks)
                ),
            )
            self.conn.executemany(
                "INSERT INTO chunks_fts(tokens, id, session_id) VALUES(?, ?, ?)",
                (
                    (" ".join(tokenize(c.text)), chunk_id(c.session_id, c.index), c.session_id)
                    for c in chunks
                ),
            )
            self.conn.execute(
                "INSERT OR REPLACE INTO processed(session_id, mtime) VALUES(?, ?)",
                (session_id, mtime),
            )

    def mark_processed(self, session_id: str, mtime: int) -> None:
        """无有效内容的会话也记账,避免每次增量都重扫。"""
        with self.conn:
            self.conn.execute(
                "INSERT OR REPLACE INTO processed(session_id, mtime) VALUES(?, ?)",
                (session_id, mtime),
            )

    # ---- 读取 ----

    def load_matrix(self):
        """全库载入:(行列表, (n,512) 矩阵)。行 = (id, session_id, project, date, chunk_index, text)。"""
        rows = self.conn.execute(
            "SELECT id, session_id, project, date, chunk_index, text, embedding FROM chunks"
        ).fetchall()
        if not rows:
            return [], np.empty((0, DIM), dtype=np.float32)
        matrix = np.frombuffer(b"".join(r[6] for r in rows), dtype=np.float32).reshape(len(rows), DIM)
        return [r[:6] for r in rows], matrix

    def fts_candidates(self, query_tokens: list[str], limit: int = 50) -> set[str]:
        """关键词召回:任一 token 命中即为候选(OR 语义),按 FTS5 内置 rank 取前 limit 个 id。
        排序精度不重要——候选随后会与向量候选合并统一重排。"""
        if not query_tokens:
            return set()
        match = " OR ".join('"' + t.replace('"', "") + '"' for t in query_tokens)
        try:
            rows = self.conn.execute(
                "SELECT id FROM chunks_fts WHERE chunks_fts MATCH ? ORDER BY rank LIMIT ?",
                (match, limit),
            ).fetchall()
        except sqlite3.OperationalError:  # 查询 token 触发 FTS5 语法边角(如纯符号)
            return set()
        return {r[0] for r in rows}

    def stats(self) -> dict:
        one = lambda q: self.conn.execute(q).fetchone()[0]
        return {
            "chunks": one("SELECT COUNT(*) FROM chunks"),
            "sessions": one("SELECT COUNT(DISTINCT session_id) FROM chunks"),
            "projects": one("SELECT COUNT(DISTINCT project) FROM chunks"),
            "date_min": one("SELECT MIN(date) FROM chunks WHERE date != ''"),
            "date_max": one("SELECT MAX(date) FROM chunks WHERE date != ''"),
        }
