"""SQLite storage — the current searchable projection of the raw archive.

上游 Agent 的本地会话可能被清理或改变格式。不可再生的事实来源是
raw gzip 底片;本库的 chunks.text 是去噪、切块后的【当前检索投影】,
并非一份独立的不可变档案。

本模块的契约:
- 源文件消失时不删除任何投影;历史可继续从 raw 重建。
- 会话更新或提取算法变更时,允许在单个事务内用新投影整体替换旧投影;
  失败则保留该会话的全部旧投影。
- embedding 模型变更只重算向量,text 投影与 raw 都不动。
- 任何路径都不因版本变更全表清空 chunks;提取升级从 raw 逐会话重建。

其余设计要点:
- 块 ID = sha1(source:session_id:index),确定性 → INSERT OR REPLACE 天然幂等
- processed 表是【可弃的增量账本】(非档案):记录 mtime_ns + size
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
SCHEMA_VERSION = "3"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS chunks (
    id          TEXT PRIMARY KEY,
    source      TEXT NOT NULL DEFAULT 'claude',
    session_id  TEXT NOT NULL,
    project     TEXT NOT NULL,
    date        TEXT NOT NULL,
    chunk_index INTEGER NOT NULL,
    text        TEXT NOT NULL,
    embedding   BLOB NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_chunks_session ON chunks(session_id);
CREATE TABLE IF NOT EXISTS processed (
    source      TEXT NOT NULL,
    session_id  TEXT NOT NULL,
    mtime_ns    INTEGER NOT NULL,
    size        INTEGER NOT NULL,
    PRIMARY KEY(source, session_id)
);
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
-- 关键词召回通道:jieba 分词后空格拼接存入,MATCH 查询同用一套分词。
-- 存在意义:精确关键词(如错误码)命中但向量分平庸的块,必须有独立的
-- 参赛资格,不能被"向量 top-N 才能进候选池"焊死(v0.2 主修复)。
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
    tokens, id UNINDEXED, source UNINDEXED, session_id UNINDEXED
);
"""


def chunk_id(source: str, session_id: str, index: int) -> str:
    return hashlib.sha1(f"{source}:{session_id}:{index}".encode()).hexdigest()


class Store:
    def __init__(self, db_path: Path = DEFAULT_DB) -> None:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.conn.executescript(_SCHEMA)
        self._migrate_legacy_layout()

    def _migrate_legacy_layout(self) -> None:
        """把 v0.3 单来源表结构原地扩为多来源。

        chunks 只新增带默认值的 source 列;processed 与 FTS 都是可再生
        数据,允许原地重建。该结构迁移不改动已有 text 投影。
        """
        chunk_cols = {r[1] for r in self.conn.execute("PRAGMA table_info(chunks)")}
        if "source" not in chunk_cols:
            with self.conn:
                self.conn.execute(
                    "ALTER TABLE chunks ADD COLUMN source TEXT NOT NULL DEFAULT 'claude'"
                )

        processed_cols = {r[1] for r in self.conn.execute("PRAGMA table_info(processed)")}
        if not {"source", "session_id", "mtime_ns", "size"}.issubset(processed_cols):
            with self.conn:
                # processed 可再生,但保留旧账本能避免升级后把数百场未变化的
                # Claude 会话全部重嵌入。size=-1 标记秒级 legacy 指纹。
                self.conn.execute("ALTER TABLE processed RENAME TO processed_legacy")
                self.conn.execute(
                    """CREATE TABLE processed (
                           source TEXT NOT NULL,
                           session_id TEXT NOT NULL,
                           mtime_ns INTEGER NOT NULL,
                           size INTEGER NOT NULL,
                           PRIMARY KEY(source, session_id)
                       )"""
                )
                if {"session_id", "mtime"}.issubset(processed_cols):
                    self.conn.execute(
                        """INSERT INTO processed(source, session_id, mtime_ns, size)
                           SELECT 'claude', session_id, mtime * 1000000000, -1
                           FROM processed_legacy"""
                    )
                self.conn.execute("DROP TABLE processed_legacy")

        fts_cols = {r[1] for r in self.conn.execute("PRAGMA table_info(chunks_fts)")}
        if "source" not in fts_cols:
            from .tokenizer import tokenize

            rows = self.conn.execute(
                "SELECT id, source, session_id, text FROM chunks"
            ).fetchall()
            with self.conn:
                self.conn.execute("DROP TABLE chunks_fts")
                self.conn.execute(
                    """CREATE VIRTUAL TABLE chunks_fts USING fts5(
                           tokens, id UNINDEXED, source UNINDEXED, session_id UNINDEXED
                       )"""
                )
                self.conn.executemany(
                    "INSERT INTO chunks_fts(tokens, id, source, session_id) VALUES(?, ?, ?, ?)",
                    ((" ".join(tokenize(text)), cid, source, sid)
                     for cid, source, sid, text in rows),
                )
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_chunks_source_session "
            "ON chunks(source, session_id)"
        )
        self.conn.commit()

    # ---- 版本轴与迁移(识别重算/重提取动作) ----

    def pending_migration(self) -> str:
        """比对三根版本轴,返回需要的迁移动作:
        'fresh'     — 全新库,直接索引
        'none'      — 一致,正常增量
        'reembed'   — 模型变更:从库内 text 重算向量(cli 执行)
        'reextract' — 提取算法变更:从 raw+源重提取覆盖(cli 执行)
        """
        meta = dict(self.conn.execute("SELECT key, value FROM meta").fetchall())
        if not meta:
            self._write_meta()
            return "fresh"
        # v0.2 旧库没有 extract_version 键:其数据即 "2" 代产出,补记而非重建
        old_extract = meta.get("extract_version", "2")
        if old_extract != EXTRACT_VERSION:
            return "reextract"
        if meta.get("model") != MODEL_NAME:
            return "reembed"
        # 纯表结构升级已由 _migrate_legacy_layout 原地完成;这里只补记新版本,
        # 不让 source 列的加入触发无意义的全量重提取/重嵌入。
        self._write_meta()
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

    def should_process(
        self, source: str, session_id: str, mtime_ns: int, size: int
    ) -> bool:
        row = self.conn.execute(
            "SELECT mtime_ns, size FROM processed WHERE source = ? AND session_id = ?",
            (source, session_id),
        ).fetchone()
        if row is None:
            return True
        old_mtime_ns, old_size = row
        if old_size < 0:  # v0.3 只有秒级 mtime;同一秒视为未变化
            return mtime_ns // 1_000_000_000 > old_mtime_ns // 1_000_000_000
        return (mtime_ns, size) != row

    # ---- 写入 ----

    def index_session(
        self,
        source: str,
        session_id: str,
        mtime_ns: int,
        size: int,
        chunks: list[Chunk],
        vectors: np.ndarray,
    ) -> None:
        """原子替换一个会话的当前投影(chunks + FTS)。

        先删旧再插新可避免提取算法改变后残留过期块;二者在同一
        SQLite 事务中,任何中途失败都会回滚至完整旧投影。raw 档案不在此库内,
        由调用方保证先安全存档再调用本方法。
        """
        from .tokenizer import tokenize

        assert len(chunks) == len(vectors)
        assert all(c.source == source and c.session_id == session_id for c in chunks)
        with self.conn:  # 事务:中途失败不留半成品
            self.conn.execute(
                "DELETE FROM chunks WHERE source = ? AND session_id = ?",
                (source, session_id),
            )
            self.conn.execute(
                "DELETE FROM chunks_fts WHERE source = ? AND session_id = ?",
                (source, session_id),
            )
            self.conn.executemany(
                """INSERT OR REPLACE INTO chunks(
                       id, source, session_id, project, date, chunk_index, text, embedding
                   ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    (
                        chunk_id(c.source, c.session_id, c.index),
                        c.source,
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
                """INSERT INTO chunks_fts(tokens, id, source, session_id)
                   VALUES(?, ?, ?, ?)""",
                (
                    (" ".join(tokenize(c.text)),
                     chunk_id(c.source, c.session_id, c.index), c.source, c.session_id)
                    for c in chunks
                ),
            )
            self.conn.execute(
                """INSERT OR REPLACE INTO processed(
                       source, session_id, mtime_ns, size
                   ) VALUES(?, ?, ?, ?)""",
                (source, session_id, mtime_ns, size),
            )

    def mark_processed(
        self, source: str, session_id: str, mtime_ns: int, size: int
    ) -> None:
        """无有效内容的会话也记账,避免每次增量都重扫。"""
        with self.conn:
            self.conn.execute(
                """INSERT OR REPLACE INTO processed(
                       source, session_id, mtime_ns, size
                   ) VALUES(?, ?, ?, ?)""",
                (source, session_id, mtime_ns, size),
            )

    # ---- 读取 ----

    def load_matrix(self):
        """全库载入:(行列表, (n,512) 矩阵)。

        行 = (id, session_id, project, date, chunk_index, text, source)。
        """
        rows = self.conn.execute(
            """SELECT id, session_id, project, date, chunk_index, text, source, embedding
               FROM chunks"""
        ).fetchall()
        if not rows:
            return [], np.empty((0, DIM), dtype=np.float32)
        matrix = np.frombuffer(b"".join(r[7] for r in rows), dtype=np.float32).reshape(len(rows), DIM)
        return [r[:7] for r in rows], matrix

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
        sources = {
            source: {"chunks": chunks, "sessions": sessions}
            for source, chunks, sessions in self.conn.execute(
                """SELECT source, COUNT(*), COUNT(DISTINCT session_id)
                   FROM chunks GROUP BY source ORDER BY source"""
            )
        }
        return {
            "chunks": one("SELECT COUNT(*) FROM chunks"),
            "sessions": one(
                "SELECT COUNT(*) FROM (SELECT 1 FROM chunks GROUP BY source, session_id)"
            ),
            "projects": one("SELECT COUNT(DISTINCT project) FROM chunks"),
            "date_min": one("SELECT MIN(date) FROM chunks WHERE date != ''"),
            "date_max": one("SELECT MAX(date) FROM chunks WHERE date != ''"),
            "extract_version": meta.get("extract_version", "?"),
            "model": meta.get("model", "?"),
            "sources": sources,
        }

    def integrity_check(self) -> str:
        """SQLite 投影体检:quick_check 通过返回 'ok',否则返回错误摘要。"""
        row = self.conn.execute("PRAGMA quick_check").fetchone()
        return row[0] if row else "unknown"
