"""cmem — archive and search Claude Code + Codex sessions."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

DEFAULT_SOURCE = Path.home() / ".claude" / "projects"  # legacy public option
DEFAULT_CODEX_SOURCE = Path.home() / ".codex" / "sessions"


def cmd_index(args) -> int:
    from .chunker import chunk_session
    from .heartbeat import mark_success
    from .raw import (
        archive_session,
        archive_source_of,
        content_size,
        iter_archived_files,
    )
    from .sources import build_adapters
    from .store import Store

    raw_dir = Path(args.raw_dir).expanduser()
    adapters = build_adapters(
        Path(args.source).expanduser(), Path(args.codex_source).expanduser()
    )
    selected = set(args.provider or adapters)
    store = Store(Path(args.db).expanduser())

    embedder = None  # 惰性:全部会话都无需更新时,不加载模型

    def ensure_embedder():
        nonlocal embedder
        if embedder is None:
            print("加载 embedding 模型(首次使用会下载 ~100MB)...")
            from .embedder import Embedder

            embedder = Embedder()
        return embedder

    # 升级只做重算/覆盖,档案 text/raw 永不全清。
    action = store.pending_migration()
    if action == "reembed":
        print("检测到 embedding 模型变更:从库内原文重算全部向量(text/raw 档案不动)...")
        n = store.reembed_all(ensure_embedder().encode_texts)
        print(f"向量重算完成:{n} 块")
    elif action == "reextract":
        print("检测到提取算法变更:从 raw 存档 + 现存源全量重提取(档案不丢)...")
        store.reset_processed_ledger()

    t0 = time.time()
    n_seen = n_indexed = n_chunks = n_archived = 0

    def index_file(f: Path, adapter, archive: bool) -> None:
        nonlocal n_seen, n_indexed, n_chunks, n_archived
        n_seen += 1
        stat = f.stat()  # 读取前取;处理中继续追加会在下一轮重索引
        mtime_ns = stat.st_mtime_ns
        size = content_size(f)
        sid = adapter.session_id_of(f)
        if archive:
            # 先存底片再查账本,保证已索引但未存档的会话也会被收编。
            n_archived += archive_session(
                f,
                mtime_ns,
                raw_dir,
                source=adapter.name,
                source_root=adapter.root,
            )
        if not adapter.should_index(f):
            return  # 子代理侧链等只进底片
        if not store.should_process(adapter.name, sid, mtime_ns, size):
            return

        sess = adapter.parse_session(f)
        if sess is None or not (chunks := chunk_session(sess)):
            store.mark_processed(adapter.name, sid, mtime_ns, size)
            return
        vectors = ensure_embedder().encode_texts([c.text for c in chunks])
        store.index_session(
            adapter.name, sess.session_id, mtime_ns, size, chunks, vectors
        )
        n_indexed += 1
        n_chunks += len(chunks)
        if n_indexed % 25 == 0:
            print(f"  ...已索引 {n_indexed} 个会话 / {n_chunks} 块")

    # 数据源一:现存源是权威版本,并顺手写 raw 底片。
    for name in sorted(selected):
        adapter = adapters[name]
        if adapter.root.is_dir():
            for f in adapter.iter_files():
                index_file(f, adapter, archive=True)
        else:
            print(
                f"警告:{name} 源目录不存在({adapter.root}),仅从 raw 存档索引",
                file=sys.stderr,
            )

    # 数据源二:历史 raw;账本会跳过与现存源相同的版本。
    for f in iter_archived_files(raw_dir):
        source = archive_source_of(f, raw_dir)
        if source in selected:
            index_file(f, adapters[source], archive=False)

    s = store.stats()
    print(
        f"完成:扫描 {n_seen}(源+存档),本次索引 {n_indexed} 个会话"
        f"(新增/更新 {n_chunks} 块),新存档 {n_archived} 份,"
        f"耗时 {time.time() - t0:.1f}s\n"
        f"库中现有 {s['chunks']} 块 / {s['sessions']} 会话,"
        f"覆盖 {s['date_min']} ~ {s['date_max']}"
    )
    if n_seen == 0:
        print("异常:一份会话文件都没扫到,视同失败(不刷新心跳)", file=sys.stderr)
        return 2
    if action == "reextract":
        store.finalize_migration()
    if not args.no_heartbeat:
        mark_success()
    return 0


def cmd_search(args) -> int:
    from .embedder import Embedder
    from .heartbeat import warn_if_stale
    from .searcher import search
    from .store import Store
    from .tokenizer import tokenize

    warn_if_stale()
    store = Store(Path(args.db).expanduser())
    rows, matrix = store.load_matrix()
    if not rows:
        print("库是空的,先跑 cmem index", file=sys.stderr)
        return 1

    # 行与矩阵同步裁剪;FTS 候选靠 id 失配自然跟随。
    if args.before or args.exclude_project or args.source:
        import numpy as np

        excl = set(args.exclude_project or [])
        mask = [
            (not args.before or (r[3] and r[3] < args.before))
            and r[2] not in excl
            and (not args.source or r[6] == args.source)
            for r in rows
        ]
        rows = [r for r, keep in zip(rows, mask) if keep]
        matrix = matrix[np.array(mask)]
        if not rows:
            print("过滤条件下没有可检索的块", file=sys.stderr)
            return 1

    fts_ids = store.fts_candidates(tokenize(args.query))
    hits = search(
        rows,
        matrix,
        Embedder().encode_query(args.query),
        args.query,
        k=args.k,
        fts_ids=fts_ids,
    )
    for rank, hit in enumerate(hits, 1):
        print(
            f"[{rank}] {hit.score:.3f} (cos {hit.cos:.3f} · bm25 {hit.bm25:.2f}) "
            f"· {hit.date or '????-??-??'} · {hit.source} · {hit.project} "
            f"· {hit.session_id[:8]}"
        )
        print("    " + hit.text.replace("\n", "\n    "))
        print()
    return 0


def cmd_status(args) -> int:
    from .heartbeat import describe, warn_if_stale
    from .raw import stats as raw_stats
    from .store import Store

    warn_if_stale()
    store = Store(Path(args.db).expanduser())
    s = store.stats()
    if not s["chunks"]:
        print("库是空的,先跑 cmem index")
        return 0
    raw_dir = Path(args.raw_dir).expanduser()
    r = raw_stats(raw_dir)
    source_summary = ", ".join(
        f"{name} {data['sessions']} 会话/{data['chunks']} 块"
        for name, data in s["sources"].items()
    ) or "无"
    raw_summary = ", ".join(
        f"{name} {count}" for name, count in sorted(r["sources"].items())
    ) or "无"
    print(
        f"块:      {s['chunks']}\n"
        f"会话:    {s['sessions']}\n"
        f"项目:    {s['projects']}\n"
        f"来源:    {source_summary}\n"
        f"日期覆盖: {s['date_min']} ~ {s['date_max']}\n"
        f"raw 存档: {r['files']} 份 / {r['bytes'] / 1048576:.1f} MB"
        f"({raw_summary}; {raw_dir})\n"
        f"完整性:  {store.integrity_check()}\n"
        f"上次成功索引: {describe()}\n"
        f"版本:    extract={s['extract_version']} · model={s['model']}"
    )
    return 0


def cmd_list(args) -> int:
    """列出档案会话;porcelain 前五列兼容 v0.3,末尾追加 source。"""
    from .store import Store

    store = Store(Path(args.db).expanduser())
    rows = store.conn.execute(
        """SELECT c.source, c.session_id, COALESCE(p.mtime_ns, 0),
                  c.date, c.project, COUNT(*)
           FROM chunks c LEFT JOIN processed p
             ON p.source = c.source AND p.session_id = c.session_id
           GROUP BY c.source, c.session_id
           ORDER BY COALESCE(p.mtime_ns, 0) DESC"""
    ).fetchall()
    if args.source:
        rows = [r for r in rows if r[0] == args.source]
    if args.since:
        rows = [r for r in rows if r[3] and r[3] >= args.since]
    if not rows:
        print("(空)", file=sys.stderr)
        return 1
    if args.porcelain:
        for source, sid, mtime_ns, date, project, chunks in rows:
            mtime = mtime_ns // 1_000_000_000
            print(f"{sid}\t{mtime}\t{date}\t{project}\t{chunks}\t{source}")
    else:
        print(f"{'来源':<9}{'会话':<38}{'日期':<12}{'项目':<24}{'块':>5}")
        for source, sid, _, date, project, chunks in rows:
            print(f"{source:<9}{sid:<38}{date or '?':<12}{project:<24}{chunks:>5}")
        print(f"\n共 {len(rows)} 场会话")
    return 0


def cmd_verify(args) -> int:
    from .raw import verify_archives
    from .store import Store

    total, bad = verify_archives(Path(args.raw_dir).expanduser())
    db_ok = Store(Path(args.db).expanduser()).integrity_check()
    print(f"raw 底片: {total} 份,损坏 {len(bad)} 份")
    for path, error in bad:
        print(f"  ✗ {path}  ({error})")
    print(f"索引库:   {db_ok}")
    if bad or db_ok != "ok":
        print("\n发现损坏——请从你的备份恢复 ~/.cmem 对应文件", file=sys.stderr)
        return 1
    print("档案完整 ✓")
    return 0


def cmd_show(args) -> int:
    import gzip

    from .heartbeat import warn_if_stale
    from .raw import find_archive
    from .store import Store

    warn_if_stale()
    store = Store(Path(args.db).expanduser())

    if args.raw:
        matches = find_archive(
            args.session, Path(args.raw_dir).expanduser(), args.source
        )
        if not matches:
            print(f"raw 层没有匹配 '{args.session}' 的底片", file=sys.stderr)
            return 1
        if len(matches) > 1:
            print("匹配多份,请用更长前缀或 --source:", file=sys.stderr)
            for match in matches:
                print(f"  {match}", file=sys.stderr)
            return 1
        sys.stdout.write(
            gzip.open(matches[0], "rt", encoding="utf-8", errors="replace").read()
        )
        return 0

    query = (
        "SELECT source, session_id, project, date, chunk_index, text FROM chunks "
        "WHERE session_id LIKE ?"
    )
    params: list[str] = [args.session + "%"]
    if args.source:
        query += " AND source = ?"
        params.append(args.source)
    query += " ORDER BY source, session_id, chunk_index"
    rows = store.conn.execute(query, params).fetchall()
    if not rows:
        print(f"库中没有匹配 '{args.session}' 的会话", file=sys.stderr)
        return 1
    identities = {(r[0], r[1]) for r in rows}
    if len(identities) > 1:
        print("匹配多个会话,请用更长前缀或 --source:", file=sys.stderr)
        for source, sid in sorted(identities):
            print(f"  {source}:{sid}", file=sys.stderr)
        return 1

    source, sid, project, date = rows[0][:4]
    print(f"会话 {sid} · {source} · {project} · {date} · {len(rows)} 块\n{'=' * 60}")
    for _, _, _, _, index, text in rows:
        print(f"\n--- 块 #{index} ---\n{text}")
    return 0


def main() -> None:
    from .raw import RAW_DIR
    from .store import DEFAULT_DB

    parser = argparse.ArgumentParser(
        prog="cmem", description="Local semantic memory for AI coding sessions"
    )
    parser.add_argument("--db", default=str(DEFAULT_DB), help=f"索引库路径(默认 {DEFAULT_DB})")
    parser.add_argument(
        "--raw-dir", default=str(RAW_DIR), help=f"raw 档案目录(默认 {RAW_DIR})"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sp = sub.add_parser("index", help="增量索引 Claude Code + Codex 会话")
    sp.add_argument(
        "--source", default=str(DEFAULT_SOURCE),
        help=f"Claude 会话目录(兼容旧参数;默认 {DEFAULT_SOURCE})",
    )
    sp.add_argument(
        "--codex-source", default=str(DEFAULT_CODEX_SOURCE),
        help=f"Codex 会话目录(默认 {DEFAULT_CODEX_SOURCE})",
    )
    sp.add_argument(
        "--provider", action="append", choices=("claude", "codex"),
        help="只索引指定来源;可重复。默认两者都索引",
    )
    sp.add_argument("--no-heartbeat", action="store_true", help=argparse.SUPPRESS)
    sp.set_defaults(fn=cmd_index)

    sp = sub.add_parser("search", help="语义检索历史会话")
    sp.add_argument("query")
    sp.add_argument("-k", type=int, default=5, help="返回条数(默认 5)")
    sp.add_argument("--before", metavar="YYYY-MM-DD", help="只检索该日期之前的块")
    sp.add_argument(
        "--exclude-project", action="append", metavar="NAME",
        help="排除指定项目,可重复使用",
    )
    sp.add_argument("--source", choices=("claude", "codex"), help="只检索指定来源")
    sp.set_defaults(fn=cmd_search)

    sp = sub.add_parser("status", help="索引库概况")
    sp.set_defaults(fn=cmd_status)

    sp = sub.add_parser("list", help="列出库中的会话(档案目录)")
    sp.add_argument("--since", metavar="YYYY-MM-DD", help="只列该日期(含)之后的会话")
    sp.add_argument("--source", choices=("claude", "codex"), help="只列指定来源")
    sp.add_argument(
        "--porcelain", action="store_true",
        help="TAB 输出(sid/mtime/date/project/chunks/source),供脚本用",
    )
    sp.set_defaults(fn=cmd_list)

    sp = sub.add_parser("verify", help="档案体检:raw CRC + 库完整性")
    sp.set_defaults(fn=cmd_verify)

    sp = sub.add_parser("show", help="展开一场会话的完整上下文")
    sp.add_argument("session", help="会话 ID(前缀即可)")
    sp.add_argument("--raw", action="store_true", help="输出未去噪的原始 jsonl")
    sp.add_argument("--source", choices=("claude", "codex"), help="限定会话来源")
    sp.set_defaults(fn=cmd_show)

    args = parser.parse_args()
    sys.exit(args.fn(args))
