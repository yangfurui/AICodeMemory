"""cmem — index / search / status."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

DEFAULT_SOURCE = Path.home() / ".claude" / "projects"


def cmd_index(args) -> int:
    from .chunker import chunk_session
    from .extractor import iter_session_files, parse_session, session_id_of
    from .raw import archive_session, iter_archived_files
    from .store import Store

    source = Path(args.source).expanduser()
    store = Store(Path(args.db).expanduser())

    embedder = None  # 惰性:全部会话都无需更新时,不加载模型

    def ensure_embedder():
        nonlocal embedder
        if embedder is None:
            print("加载 embedding 模型(首次使用会下载 ~100MB)...")
            from .embedder import Embedder
            embedder = Embedder()
        return embedder

    # ---- 版本轴检查:升级只做重算/覆盖,档案(text/raw)永不删 ----
    action = store.pending_migration()
    if action == "reembed":
        print("检测到 embedding 模型变更:从库内原文重算全部向量(text/raw 档案不动)...")
        n = store.reembed_all(ensure_embedder().encode_texts)
        print(f"向量重算完成:{n} 块")
    elif action == "reextract":
        print("检测到提取算法/表结构变更:从 raw 存档 + 现存源全量重提取(档案不丢)...")
        store.reset_processed_ledger()

    t0 = time.time()
    n_seen = n_indexed = n_chunks = n_archived = 0

    def index_file(f: Path, archive: bool) -> None:
        nonlocal n_seen, n_indexed, n_chunks, n_archived
        n_seen += 1
        sid = session_id_of(f)
        mtime = int(f.stat().st_mtime)  # 读取内容前取,追加发生时下次会重索引
        if archive:
            # 存档先于账本判断:v0.2 时代已索引但未存档的会话,在这里补齐底片
            n_archived += archive_session(f, mtime)
        if not store.should_process(sid, mtime):
            return
        sess = parse_session(f)
        if sess is None or not (chunks := chunk_session(sess)):
            store.mark_processed(sid, mtime)
            return
        vectors = ensure_embedder().encode_texts([c.text for c in chunks])
        store.index_session(sess.session_id, mtime, chunks, vectors)
        n_indexed += 1
        n_chunks += len(chunks)
        if n_indexed % 25 == 0:
            print(f"  ...已索引 {n_indexed} 个会话 / {n_chunks} 块")

    # 数据源一:现存源(权威版本,顺手写 raw 底片)
    if source.is_dir():
        for f in iter_session_files(source):
            index_file(f, archive=True)
    else:
        print(f"警告:源目录不存在({source}),仅从 raw 存档索引", file=sys.stderr)

    # 数据源二:raw 存档(源已被 30 天清理的历史,靠账本天然跳过与源重复的部分)
    for f in iter_archived_files():
        index_file(f, archive=False)

    if action == "reextract":
        store.finalize_migration()

    s = store.stats()
    print(
        f"完成:扫描 {n_seen}(源+存档),本次索引 {n_indexed} 个会话(新增/更新 {n_chunks} 块),"
        f"新存档 {n_archived} 份,耗时 {time.time() - t0:.1f}s\n"
        f"库中现有 {s['chunks']} 块 / {s['sessions']} 会话,覆盖 {s['date_min']} ~ {s['date_max']}"
    )
    return 0


def cmd_search(args) -> int:
    from .embedder import Embedder
    from .searcher import search
    from .store import Store
    from .tokenizer import tokenize

    store = Store(Path(args.db).expanduser())
    rows, matrix = store.load_matrix()
    if not rows:
        print("库是空的,先跑 cmem index", file=sys.stderr)
        return 1

    # 来源过滤:行与矩阵同步裁剪;FTS 候选靠 id 失配自然跟随,无需单独过滤
    if args.before or args.exclude_project:
        import numpy as np

        excl = set(args.exclude_project or [])
        # date 为空的块在 --before 下保守排除(无法判定时间)
        mask = [(not args.before or (r[3] and r[3] < args.before)) and r[2] not in excl
                for r in rows]
        rows = [r for r, m in zip(rows, mask) if m]
        matrix = matrix[np.array(mask)]
        if not rows:
            print("过滤条件下没有可检索的块", file=sys.stderr)
            return 1

    fts_ids = store.fts_candidates(tokenize(args.query))
    hits = search(rows, matrix, Embedder().encode_query(args.query), args.query,
                  k=args.k, fts_ids=fts_ids)
    for rank, h in enumerate(hits, 1):
        print(f"[{rank}] {h.score:.3f} (cos {h.cos:.3f} · bm25 {h.bm25:.2f}) "
              f"· {h.date or '????-??-??'} · {h.project} · {h.session_id[:8]}")
        print("    " + h.text.replace("\n", "\n    "))
        print()
    return 0


def cmd_status(args) -> int:
    from .raw import RAW_DIR
    from .raw import stats as raw_stats
    from .store import Store

    store = Store(Path(args.db).expanduser())
    s = store.stats()
    if not s["chunks"]:
        print("库是空的,先跑 cmem index")
        return 0
    r = raw_stats()
    print(f"块:      {s['chunks']}\n会话:    {s['sessions']}\n项目:    {s['projects']}\n"
          f"日期覆盖: {s['date_min']} ~ {s['date_max']}\n"
          f"raw 存档: {r['files']} 份 / {r['bytes'] / 1048576:.1f} MB({RAW_DIR})\n"
          f"完整性:  {store.integrity_check()}\n"
          f"版本:    extract={s['extract_version']} · model={s['model']}")
    return 0


def cmd_verify(args) -> int:
    from .raw import verify_archives
    from .store import Store

    total, bad = verify_archives()
    db_ok = Store(Path(args.db).expanduser()).integrity_check()
    print(f"raw 底片: {total} 份,损坏 {len(bad)} 份")
    for p, err in bad:
        print(f"  ✗ {p}  ({err})")
    print(f"索引库:   {db_ok}")
    if bad or db_ok != "ok":
        print("\n发现损坏——请从你的备份恢复 ~/.cmem 对应文件", file=sys.stderr)
        return 1
    print("档案完整 ✓")
    return 0


def cmd_show(args) -> int:
    import gzip

    from .raw import find_archive
    from .store import Store

    store = Store(Path(args.db).expanduser())

    if args.raw:
        # 字面意义的"读原始数据":解压底片原样输出(可管道给 jq/grep)
        matches = find_archive(args.session)
        if not matches:
            print(f"raw 层没有匹配 '{args.session}' 的底片", file=sys.stderr)
            return 1
        if len(matches) > 1:
            print("匹配多份,请用更长前缀:", file=sys.stderr)
            for m in matches:
                print(f"  {m}", file=sys.stderr)
            return 1
        sys.stdout.write(gzip.open(matches[0], "rt", encoding="utf-8", errors="replace").read())
        return 0

    rows = store.conn.execute(
        "SELECT session_id, project, date, chunk_index, text FROM chunks "
        "WHERE session_id LIKE ? ORDER BY session_id, chunk_index",
        (args.session + "%",),
    ).fetchall()
    if not rows:
        print(f"库中没有匹配 '{args.session}' 的会话", file=sys.stderr)
        return 1
    sids = {r[0] for r in rows}
    if len(sids) > 1:
        print("匹配多个会话,请用更长前缀:", file=sys.stderr)
        for s in sorted(sids):
            print(f"  {s}", file=sys.stderr)
        return 1

    sid, project, date = rows[0][0], rows[0][1], rows[0][2]
    print(f"会话 {sid} · {project} · {date} · {len(rows)} 块\n{'=' * 60}")
    for _, _, _, idx, text in rows:
        print(f"\n--- 块 #{idx} ---\n{text}")
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
    sp.add_argument("--before", metavar="YYYY-MM-DD",
                    help="只检索该日期之前的块(排除后来复述历史的会话)")
    sp.add_argument("--exclude-project", action="append", metavar="NAME",
                    help="排除指定项目,可重复使用")
    sp.set_defaults(fn=cmd_search)

    sp = sub.add_parser("status", help="索引库概况")
    sp.set_defaults(fn=cmd_status)

    sp = sub.add_parser("verify", help="档案体检:raw 底片逐份验 CRC + 库完整性")
    sp.set_defaults(fn=cmd_verify)

    sp = sub.add_parser("show", help="展开一场会话的完整上下文")
    sp.add_argument("session", help="会话 ID(前缀即可,如 search 结果里的 8 位)")
    sp.add_argument("--raw", action="store_true",
                    help="输出未去噪的原始 jsonl(解压底片,可管道给 jq)")
    sp.set_defaults(fn=cmd_show)

    args = p.parse_args()
    sys.exit(args.fn(args))
