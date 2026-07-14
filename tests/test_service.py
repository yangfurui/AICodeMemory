"""Shared read service: filters, cache invalidation, sessions and status."""

import os

import numpy as np
import pytest

from cmem.chunker import Chunk
from cmem.embedder import DIM
from cmem.service import MemoryService, MemoryServiceError
from cmem.store import Store


def unit_vector(axis: int) -> np.ndarray:
    vector = np.zeros(DIM, dtype=np.float32)
    vector[axis] = 1
    return vector


def chunks(
    source: str,
    session_id: str,
    texts: list[str],
    *,
    project: str = "project",
    date: str = "2026-07-14",
) -> list[Chunk]:
    return [
        Chunk(source, session_id, project, date, index, text)
        for index, text in enumerate(texts)
    ]


def index_chunks(
    store: Store,
    source: str,
    session_id: str,
    texts: list[str],
    vectors: list[np.ndarray],
    *,
    project: str = "project",
    date: str = "2026-07-14",
    mtime_ns: int = 1,
) -> None:
    projection = chunks(
        source, session_id, texts, project=project, date=date
    )
    store.index_session(
        source,
        session_id,
        mtime_ns,
        sum(len(text) for text in texts),
        projection,
        np.stack(vectors),
    )


class FakeEmbedder:
    def __init__(self, created: list[object]) -> None:
        created.append(self)

    def encode_query(self, query: str) -> np.ndarray:
        return unit_vector(0)


def test_search_cache_stays_hot_and_reloads_after_database_change(tmp_path):
    db = tmp_path / "memory.sqlite3"
    writer = Store(db)
    writer.pending_migration()
    index_chunks(
        writer,
        "claude",
        "session-1",
        ["alpha 原始结论"],
        [unit_vector(0)],
    )

    reader = Store(db)
    created: list[object] = []
    service = MemoryService(
        db,
        tmp_path / "raw",
        tmp_path / "heartbeat",
        embedder_factory=lambda: FakeEmbedder(created),
        store=reader,
    )
    original_load = reader.load_matrix
    load_count = 0

    def counted_load():
        nonlocal load_count
        load_count += 1
        return original_load()

    reader.load_matrix = counted_load

    assert service.search_history("alpha")[0].session_id == "session-1"
    assert service.search_history("alpha")[0].session_id == "session-1"
    assert load_count == 1
    assert len(created) == 1

    old_mtime = db.stat().st_mtime_ns
    index_chunks(
        writer,
        "claude",
        "session-2",
        ["alpha 追加结论"],
        [unit_vector(0)],
        mtime_ns=2,
    )
    stat = db.stat()
    os.utime(db, ns=(stat.st_atime_ns, max(stat.st_mtime_ns, old_mtime) + 1))

    assert len(service.search_history("alpha", k=2)) == 2
    assert load_count == 2
    assert len(created) == 1


def test_fts_filters_before_candidate_limit(tmp_path):
    """被排除项目的大量命中不得挤掉允许项目的精确关键词候选。"""
    store = Store(tmp_path / "memory.sqlite3")
    store.pending_migration()
    for index in range(55):
        index_chunks(
            store,
            "claude",
            f"excluded-{index}",
            [f"needle 排除内容 {index}"],
            [unit_vector(1)],
            project="excluded",
            mtime_ns=index + 1,
        )
    index_chunks(
        store,
        "claude",
        "allowed",
        ["needle 允许内容"],
        [unit_vector(1)],
        project="allowed",
        mtime_ns=100,
    )

    ids = store.fts_candidates(
        ["needle"], limit=50, exclude_projects=("excluded",)
    )
    allowed_id = store.conn.execute(
        "SELECT id FROM chunks WHERE session_id = 'allowed'"
    ).fetchone()[0]
    assert ids == {allowed_id}


def test_get_session_supports_source_key_and_around_window(tmp_path):
    store = Store(tmp_path / "memory.sqlite3")
    store.pending_migration()
    index_chunks(
        store,
        "claude",
        "same-session",
        [f"claude-{index}" for index in range(7)],
        [unit_vector(0) for _ in range(7)],
    )
    index_chunks(
        store,
        "codex",
        "same-session",
        ["codex-0"],
        [unit_vector(0)],
    )
    service = MemoryService(store=store)

    with pytest.raises(MemoryServiceError, match="匹配多个会话"):
        service.get_session("same")

    view = service.get_session("claude:same", around=3)
    assert view.session_key == "claude:same-session"
    assert view.total_chunks == 7
    assert [chunk.index for chunk in view.chunks] == [1, 2, 3, 4, 5]
    assert view.as_dict()["truncated"] is True

    with pytest.raises(MemoryServiceError, match="没有块 #99"):
        service.get_session("claude:same", around=99)


def test_memory_status_includes_pure_staleness_state(tmp_path):
    store = Store(tmp_path / "memory.sqlite3")
    store.pending_migration()
    service = MemoryService(
        raw_dir=tmp_path / "raw",
        heartbeat_path=tmp_path / "missing-heartbeat",
        store=store,
    )

    status = service.memory_status()
    assert status["chunks"] == 0
    assert status["integrity"] == "ok"
    assert status["index_state"]["stale"] is True
    assert "cmem index" in status["index_state"]["warning"]


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"query": ""}, "查询不能为空"),
        ({"query": "x", "k": 0}, "k 必须在"),
        ({"query": "x", "before": "2026/07/14"}, "YYYY-MM-DD"),
        ({"query": "x", "source": "unknown"}, "source 只能"),
    ],
)
def test_search_validates_public_inputs(tmp_path, kwargs, message):
    service = MemoryService(
        tmp_path / "memory.sqlite3",
        tmp_path / "raw",
        tmp_path / "heartbeat",
    )
    with pytest.raises(MemoryServiceError, match=message):
        service.search_history(**kwargs)
