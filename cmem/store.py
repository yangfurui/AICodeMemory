"""SQLite storage — this file is an ARCHIVE, not a cache.

源(~/.claude/projects/)被 Claude Code 按 30 天滚动清理,库中超过 30 天
的内容是唯一副本。由此立下本模块的最高契约(v0.3 起,不可违反):

    【text 列与 processed 之外的任何数据,永不被自动删除。】
    代码中不存在对 chunks 的全表清空;升级路径只做"重算"与"覆盖":
    - embedding 模型变更 → 从库内 text 重算向量(text 一字不动)
    - 提取算法变更     → 清 processed 账本,从 raw 存档 + 现存源
                          逐会话确定性 upsert 覆盖(旧块要么被同 ID
                          新块覆盖,要么原样保留,绝不凭空消失)

其余设计要点:
- 块 ID = sha1(session_id:index),确定性 → INSERT OR REPLACE 天然幂等
- processed 表是【可弃的增量账本】(非档案):记录每会话"处理时 mtime"
- meta 表存三根版本轴:schema_version / extract_version / model
- 向量以 float32 BLOB 存储,检索时一次性载入 numpy 矩阵
"""

from __future__ import annotations

import hashlib
import sqlite3
from pathlib import Path

import numpy as np

from .chunker import Chunk
from .embedder import DIM, MODEL_NAME
from .extractor import EXTRACT_VERSION

DEFAULT_DB = Path.home() / ".cmem" / "memory.sqlite3"
# 纯表结构的版本轴(列增删等);数据语义变化走 extract_version/model 两轴
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

    # ---- 版本轴与迁移(档案契约:识别动作,绝不在此删数据) ----

    def pending_migration(self) -> str:
        """比对三根版本轴,返回需要的迁移动作:
        'fresh'     — 全新库,直接索引
        'none'      — 一致,正常增量
        'reembed'   — 模型变更:从库内 text 重算向量(cli 执行)
        'reextract' — 提取算法/表结构变更:从 raw+源重提取覆盖(cli 执行)
        """
        meta = dict(self.conn.execute("SELECT key, value FROM meta").fetchall())
        if not meta:
            self._write_meta()
            return "fresh"
        # v0.2 旧库没有 extract_version 键:其数据即 "2" 代产出,补记而非重建
        old_extract = meta.get("extract_version", "2")
        old_schema = meta.get("schema_version", SCHEMA_VERSION)
        if old_schema != SCHEMA_VERSION or old_extract != EXTRACT_VERSION:
            return "reextract"
        if meta.get("model") != MODEL_NAME:
            return "reembed"
        self._write_meta()  # 补齐缺失键(如旧库首次升到三轴)
        return "none"

    def _write_meta(self) -> None:
        self.conn.executemany(
            "INSERT OR REPLACE INTO meta(key, value) VALUES(?, ?)",
            {
                "schema_version": SCHEMA_VERSION,
                "extract_version": EXTRACT_VERSION,
                "model": MODEL_NAME,
            }.items(),
        )
        self.conn.commit()

    def reembed_all(self, encode_texts, batch_size: int = 256) -> int:
        """模型升级:从库内 text 重算全部向量,text/FTS 一字不动。
        encode_texts: list[str] -> np.ndarray。完成后才更新 meta——中断则
        下次继续全量重算(幂等,只费时不丢数据)。"""
        ids_texts = self.conn.execute("SELECT id, text FROM chunks").fetchall()
        for i in range(0, len(ids_texts), batch_size):
            batch = ids_texts[i : i + batch_size]
            vectors = encode_texts([t for _, t in batch])
            with self.conn:
                self.conn.executemany(
                    "UPDATE chunks SET embedding = ? WHERE id = ?",
                    ((vectors[j].tobytes(), batch[j][0]) for j in range(len(batch))),
                )
        self._write_meta()
        return len(ids_texts)

    def reset_processed_ledger(self) -> None:
        """提取算法升级的第一步:清增量账本(chunks 不动!),让 raw+源
        全部重新过管线,逐会话确定性覆盖。完成由 cli 在索引后 finalize。"""
        with self.conn:
            self.conn.execute("DELETE FROM processed")

    def finalize_migration(self) -> None:
        """迁移(reextract)完成后由 cli 调用,落新版本号。"""
        self._write_meta()

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
        meta = dict(self.conn.execute("SELECT key, value FROM meta").fetchall())
        return {
            "chunks": one("SELECT COUNT(*) FROM chunks"),
            "sessions": one("SELECT COUNT(DISTINCT session_id) FROM chunks"),
            "projects": one("SELECT COUNT(DISTINCT project) FROM chunks"),
            "date_min": one("SELECT MIN(date) FROM chunks WHERE date != ''"),
            "date_max": one("SELECT MAX(date) FROM chunks WHERE date != ''"),
            "extract_version": meta.get("extract_version", "?"),
            "model": meta.get("model", "?"),
        }

    def integrity_check(self) -> str:
        """档案体检:quick_check 通过返回 'ok',否则返回错误摘要。"""
        row = self.conn.execute("PRAGMA quick_check").fetchone()
        return row[0] if row else "unknown"
