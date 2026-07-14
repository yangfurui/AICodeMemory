"""Reusable read service shared by the CLI and the MCP server.

The service owns the long-lived search cache.  SQLite remains the authority for
session/status reads; rows + the numpy matrix are reloaded whenever the database
fingerprint changes, while the embedding model stays warm for the process lifetime.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from .heartbeat import HEARTBEAT, STALE_AFTER_H, describe, staleness_warning
from .raw import RAW_DIR, stats as raw_stats
from .searcher import Hit, search
from .store import DEFAULT_DB, Store
from .tokenizer import tokenize

SESSION_CONTEXT_RADIUS = 2
MAX_SEARCH_K = 20


class QueryEmbedder(Protocol):
    def encode_query(self, query: str) -> np.ndarray: ...


class MemoryServiceError(ValueError):
    """Expected user-facing read error, safe to expose through CLI/MCP."""


@dataclass(frozen=True)
class SessionChunk:
    index: int
    text: str


@dataclass(frozen=True)
class SessionView:
    source: str
    session_id: str
    project: str
    date: str
    total_chunks: int
    chunks: list[SessionChunk]
    around: int | None = None

    @property
    def session_key(self) -> str:
        return f"{self.source}:{self.session_id}"

    def as_dict(self) -> dict[str, Any]:
        return {
            "session_key": self.session_key,
            "source": self.source,
            "session_id": self.session_id,
            "project": self.project,
            "date": self.date,
            "total_chunks": self.total_chunks,
            "around": self.around,
            "truncated": len(self.chunks) < self.total_chunks,
            "chunks": [
                {"index": chunk.index, "text": chunk.text} for chunk in self.chunks
            ],
        }


class MemoryService:
    def __init__(
        self,
        db_path: Path = DEFAULT_DB,
        raw_dir: Path = RAW_DIR,
        heartbeat_path: Path = HEARTBEAT,
        *,
        embedder_factory: Callable[[], QueryEmbedder] | None = None,
        store: Store | None = None,
    ) -> None:
        self.db_path = store.db_path if store is not None else Path(db_path).expanduser()
        self.raw_dir = Path(raw_dir).expanduser()
        self.heartbeat_path = Path(heartbeat_path).expanduser()
        self._embedder_factory = embedder_factory
        self._store = store
        self._embedder: QueryEmbedder | None = None
        self._rows: list | None = None
        self._matrix: np.ndarray | None = None
        self._matrix_fingerprint: tuple[int, int] | None = None

    def _get_store(self) -> Store:
        if self._store is None:
            self._store = Store(self.db_path)
        return self._store

    def _get_embedder(self) -> QueryEmbedder:
        if self._embedder is None:
            if self._embedder_factory is None:
                from .embedder import Embedder

                self._embedder = Embedder()
            else:
                self._embedder = self._embedder_factory()
        return self._embedder

    def _db_fingerprint(self) -> tuple[int, int] | None:
        try:
            stat = self.db_path.stat()
        except FileNotFoundError:
            return None
        return stat.st_mtime_ns, stat.st_size

    def _load_index(self) -> tuple[list, np.ndarray]:
        store = self._get_store()
        fingerprint = self._db_fingerprint()
        if (
            self._rows is None
            or self._matrix is None
            or fingerprint != self._matrix_fingerprint
        ):
            # If index writes race with this read, retry once.  Storing the final
            # fingerprint also guarantees the next call reloads after later writes.
            rows, matrix = store.load_matrix()
            after = self._db_fingerprint()
            if after != fingerprint:
                rows, matrix = store.load_matrix()
                after = self._db_fingerprint()
            self._rows = rows
            self._matrix = matrix
            self._matrix_fingerprint = after
        return self._rows, self._matrix

    @staticmethod
    def _validate_date(value: str) -> str:
        if not value:
            return ""
        try:
            return date.fromisoformat(value).isoformat()
        except ValueError as exc:
            raise MemoryServiceError(
                f"日期必须是 YYYY-MM-DD,实际为: {value}"
            ) from exc

    @staticmethod
    def _validate_source(source: str) -> str:
        if source and source not in {"claude", "codex"}:
            raise MemoryServiceError("source 只能是 claude 或 codex")
        return source

    def search_history(
        self,
        query: str,
        *,
        k: int = 5,
        before: str = "",
        exclude_projects: Iterable[str] = (),
        source: str = "",
    ) -> list[Hit]:
        query = query.strip()
        if not query:
            raise MemoryServiceError("查询不能为空")
        if not 1 <= k <= MAX_SEARCH_K:
            raise MemoryServiceError(f"k 必须在 1~{MAX_SEARCH_K} 之间")
        before = self._validate_date(before)
        source = self._validate_source(source)
        excluded = tuple(p for p in exclude_projects if p)

        rows, matrix = self._load_index()
        if not rows:
            raise MemoryServiceError("记忆库是空的,先运行 cmem index")

        if before or excluded or source:
            excluded_set = set(excluded)
            mask = np.asarray(
                [
                    (not before or (row[3] and row[3] < before))
                    and row[2] not in excluded_set
                    and (not source or row[6] == source)
                    for row in rows
                ],
                dtype=bool,
            )
            rows = [row for row, keep in zip(rows, mask) if keep]
            matrix = matrix[mask]
            if not rows:
                raise MemoryServiceError("过滤条件下没有可检索的记忆")

        fts_ids = self._get_store().fts_candidates(
            tokenize(query),
            before=before,
            exclude_projects=excluded,
            source=source,
        )
        return search(
            rows,
            matrix,
            self._get_embedder().encode_query(query),
            query,
            k=k,
            fts_ids=fts_ids,
        )

    @staticmethod
    def _split_session_key(session: str, source: str) -> tuple[str, str]:
        session = session.strip()
        if not session:
            raise MemoryServiceError("会话 ID 不能为空")
        if ":" in session:
            prefix, possible_id = session.split(":", 1)
            if prefix in {"claude", "codex"}:
                if source and source != prefix:
                    raise MemoryServiceError("session_key 与 source 冲突")
                source, session = prefix, possible_id
        if not session:
            raise MemoryServiceError("会话 ID 不能为空")
        return session, MemoryService._validate_source(source)

    def get_session(
        self,
        session: str,
        *,
        source: str = "",
        around: int | None = None,
    ) -> SessionView:
        session, source = self._split_session_key(session, source)
        if around is not None and around < 0:
            raise MemoryServiceError("around 必须是非负块序号")

        store = self._get_store()
        identity_query = (
            "SELECT DISTINCT source, session_id FROM chunks "
            "WHERE substr(session_id, 1, length(?)) = ?"
        )
        params: list[str] = [session, session]
        if source:
            identity_query += " AND source = ?"
            params.append(source)
        identity_query += " ORDER BY source, session_id"
        identities = store.conn.execute(identity_query, params).fetchall()
        if not identities:
            raise MemoryServiceError(f"库中没有匹配 '{session}' 的会话")
        if len(identities) > 1:
            choices = ", ".join(f"{src}:{sid}" for src, sid in identities[:8])
            raise MemoryServiceError(
                f"匹配多个会话,请使用更长前缀或 session_key: {choices}"
            )

        resolved_source, resolved_id = identities[0]
        rows = store.conn.execute(
            """SELECT project, date, chunk_index, text FROM chunks
               WHERE source = ? AND session_id = ? ORDER BY chunk_index""",
            (resolved_source, resolved_id),
        ).fetchall()
        total = len(rows)
        selected = rows
        if around is not None:
            positions = [i for i, row in enumerate(rows) if row[2] == around]
            if not positions:
                raise MemoryServiceError(
                    f"会话 {resolved_id} 中没有块 #{around}"
                )
            position = positions[0]
            start = max(0, position - SESSION_CONTEXT_RADIUS)
            end = min(total, position + SESSION_CONTEXT_RADIUS + 1)
            selected = rows[start:end]

        project, session_date = rows[0][:2]
        return SessionView(
            source=resolved_source,
            session_id=resolved_id,
            project=project,
            date=session_date,
            total_chunks=total,
            chunks=[SessionChunk(index=row[2], text=row[3]) for row in selected],
            around=around,
        )

    def index_state(self) -> dict[str, Any]:
        warning = staleness_warning(self.heartbeat_path)
        return {
            "last_success": describe(self.heartbeat_path),
            "stale": warning is not None,
            "stale_after_hours": STALE_AFTER_H,
            "warning": warning,
        }

    def memory_status(self) -> dict[str, Any]:
        store = self._get_store()
        db = store.stats()
        raw = raw_stats(self.raw_dir)
        return {
            "database": str(self.db_path),
            "raw_directory": str(self.raw_dir),
            "chunks": db["chunks"],
            "sessions": db["sessions"],
            "projects": db["projects"],
            "date_min": db["date_min"],
            "date_max": db["date_max"],
            "sources": db["sources"],
            "raw_files": raw["files"],
            "raw_bytes": raw["bytes"],
            "raw_sources": raw["sources"],
            "integrity": store.integrity_check(),
            "extract_version": db["extract_version"],
            "model": db["model"],
            "index_state": self.index_state(),
        }
