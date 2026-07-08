"""cmem — index / search / status."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

DEFAULT_SOURCE = Path.home() / ".claude" / "projects"


def cmd_index(args) -> int:
    from .chunker import chunk_session
    from .extractor import iter_session_files, parse_session
    from .store import DEFAULT_DB, Store

    source = Path(args.source).expanduser()
    if not source.is_dir():
        print(f"数据源不存在: {source}", file=sys.stderr)
        return 1

    store = Store(Path(args.db).expanduser())
    embedder = None  # 惰性:全部会话都无需更新时,不加载模型
    t0 = time.time()
    n_seen = n_indexed = n_chunks = 0

    for f in iter_session_files(source):
        n_seen += 1
        mtime = int(f.stat().st_mtime)  # 读取内容前取,追加发生时下次会重索引
        if not store.should_process(f.stem, mtime):
            continue
        sess = parse_session(f)
        if sess is None or not (chunks := chunk_session(sess)):
            store.mark_processed(f.stem, mtime)
            continue
        if embedder is None:
            print("加载 embedding 模型(首次使用会下载 ~100MB)...")
            from .embedder import Embedder
            embedder = Embedder()
        vectors = embedder.encode_texts([c.text for c in chunks])
        store.index_session(sess.session_id, mtime, chunks, vectors)
        n_indexed += 1
        n_chunks += len(chunks)
        if n_indexed % 25 == 0:
            print(f"  ...已索引 {n_indexed} 个会话 / {n_chunks} 块")

    s = store.stats()
    print(
        f"完成:扫描 {n_seen} 个会话,本次索引 {n_indexed} 个(新增/更新 {n_chunks} 块),"
        f"耗时 {time.time() - t0:.1f}s\n"
        f"库中现有 {s['chunks']} 块 / {s['sessions']} 会话,覆盖 {s['date_min']} ~ {s['date_max']}"
    )
    return 0


def cmd_search(args) -> int:
    from .embedder import Embedder
    from .searcher import search
    from .store import Store

    store = Store(Path(args.db).expanduser())
    rows, matrix = store.load_matrix()
    if not rows:
        print("库是空的,先跑 cmem index", file=sys.stderr)
        return 1

    hits = search(rows, matrix, Embedder().encode_query(args.query), args.query, k=args.k)
    for rank, h in enumerate(hits, 1):
        print(f"[{rank}] {h.score:.3f} (cos {h.cos:.3f} · bm25 {h.bm25:.2f}) "
              f"· {h.date or '????-??-??'} · {h.project} · {h.session_id[:8]}")
        print("    " + h.text.replace("\n", "\n    "))
        print()
    return 0


def cmd_status(args) -> int:
    from .store import Store

    s = Store(Path(args.db).expanduser()).stats()
    if not s["chunks"]:
        print("库是空的,先跑 cmem index")
        return 0
    print(f"块:      {s['chunks']}\n会话:    {s['sessions']}\n项目:    {s['projects']}\n"
          f"日期覆盖: {s['date_min']} ~ {s['date_max']}")
    return 0


def main() -> None:
    from .store import DEFAULT_DB

    p = argparse.ArgumentParser(prog="cmem", description="Local semantic memory for Claude Code")
    p.add_argument("--db", default=str(DEFAULT_DB), help=f"索引库路径(默认 {DEFAULT_DB})")
    sub = p.add_subparsers(dest="command", required=True)

    sp = sub.add_parser("index", help="增量索引会话历史(首次即全量)")
    sp.add_argument("--source", default=str(DEFAULT_SOURCE), help=f"会话数据源(默认 {DEFAULT_SOURCE})")
    sp.set_defaults(fn=cmd_index)

    sp = sub.add_parser("search", help="语义检索历史会话")
    sp.add_argument("query")
    sp.add_argument("-k", type=int, default=5, help="返回条数(默认 5)")
    sp.set_defaults(fn=cmd_search)

    sp = sub.add_parser("status", help="索引库概况")
    sp.set_defaults(fn=cmd_status)

    args = p.parse_args()
    sys.exit(args.fn(args))
