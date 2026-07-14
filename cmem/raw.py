"""Raw archive layer — the permanent source of truth.

上游 jsonl(Claude Code / Codex)可能被清理或改变格式。SQLite 里的
text/chunks 是去噪、切块后的当前检索投影,而本层把源文件
**原样 gzip 存档**(底片):
任何未来的去噪/切块算法改进,都能凭它回溯应用到全部历史——包括源早已
消失的部分。

契约(不可违反):
- raw 层的文件【永不自动删除】。代码里不存在删除路径;源文件从磁盘
  消失不影响其存档。
- 存档以【源文件 mtime + size】为增量指纹;未变则零成本跳过。
- 已有底片只能被它的字节级追加版本更新。源被截断或改写时拒绝覆盖,
  保住旧底片并让整次索引显式失败。
- 原子写入:先写 .tmp 再 rename,中断不留半个存档。

Claude legacy 目录保持不动;Codex 新档案按来源隔离:
- ~/.cmem/raw/<Claude 项目>/<会话>.jsonl.gz
- ~/.cmem/raw/codex/YYYY/MM/DD/<rollout>.jsonl.gz
"""

from __future__ import annotations

import gzip
import os
import shutil
import struct
from pathlib import Path

RAW_DIR = Path.home() / ".cmem" / "raw"


class ArchiveConflictError(RuntimeError):
    """源文件不再是已存档内容的追加版本。

    宁可停止该会话的更新,也不得覆盖可能是唯一副本的旧底片。
    """


def _extends_archive(src: Path, archived: Path) -> bool:
    """src 是否以 archived 的完整解压内容为字节前缀。"""
    with gzip.open(archived, "rb") as old, src.open("rb") as new:
        while chunk := old.read(1 << 20):
            if new.read(len(chunk)) != chunk:
                return False
    return True


def archive_path_for(
    src: Path,
    raw_dir: Path = RAW_DIR,
    source: str = "claude",
    source_root: Path | None = None,
) -> Path:
    if source == "claude":
        # 保持 v0.3 已有布局,避免复制/移动唯一底片。
        return raw_dir / src.parent.name / (src.name + ".gz")
    try:
        rel = src.relative_to(source_root) if source_root else Path(src.name)
    except ValueError:
        rel = Path(src.name)
    return raw_dir / source / rel.parent / (rel.name + ".gz")


def archive_session(
    src: Path,
    mtime_ns: int | None = None,
    raw_dir: Path = RAW_DIR,
    *,
    source: str = "claude",
    source_root: Path | None = None,
) -> bool:
    """把源会话文件 gzip 存档;已是最新则跳过。返回是否实际写入。

    mtime_ns 由调用方在【读取源内容之前】stat 得到(与索引共用同一语义):
    存档期间源又有追加,gz 的 mtime 会小于源的新 mtime,下次触发重存。
    """
    if mtime_ns is None:
        mtime_ns = src.stat().st_mtime_ns
    elif mtime_ns < 10**15:  # 兼容 v0.3 内部调用传入的秒级 mtime
        mtime_ns *= 1_000_000_000
    dst = archive_path_for(src, raw_dir, source, source_root)
    if dst.exists():
        # mtime + 未压缩大小是增量指纹;与 processed 账本语义一致。
        if (dst.stat().st_mtime_ns, content_size(dst)) == (
            mtime_ns,
            src.stat().st_size,
        ):
            return False
        try:
            safe_to_replace = _extends_archive(src, dst)
        except (OSError, EOFError, gzip.BadGzipFile) as exc:
            raise ArchiveConflictError(
                f"无法验证现有底片是否安全(已保留 {dst}): {exc}"
            ) from exc
        if not safe_to_replace:
            raise ArchiveConflictError(
                f"源文件发生截断或改写,拒绝覆盖现有底片: {src} -> {dst}"
            )

    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".tmp")
    try:
        with src.open("rb") as f_in, gzip.open(tmp, "wb", compresslevel=6) as f_out:
            shutil.copyfileobj(f_in, f_out)
        os.utime(tmp, ns=(mtime_ns, mtime_ns))  # gz mtime = 源 mtime,作为增量账本
        tmp.replace(dst)  # 原子落位
        return True
    finally:
        tmp.unlink(missing_ok=True)


def iter_archived_files(raw_dir: Path = RAW_DIR):
    """遍历全部存档(重提取时的数据源)。"""
    if raw_dir.is_dir():
        yield from sorted(raw_dir.rglob("*.jsonl.gz"))


def archive_source_of(path: Path, raw_dir: Path = RAW_DIR) -> str:
    """v0.3 legacy 顶层目录均视为 Claude;显式 source 目录用于新来源。"""
    try:
        first = path.relative_to(raw_dir).parts[0]
    except (ValueError, IndexError):
        return "claude"
    return first if first in {"claude", "codex"} else "claude"


def content_size(path: Path) -> int:
    """Return logical uncompressed size for source files and gzip archives.

    gzip's ISIZE trailer lets source and raw copies share the same incremental
    fingerprint without decompressing the archive.
    """
    if not path.name.endswith(".gz"):
        return path.stat().st_size
    try:
        with path.open("rb") as f:
            f.seek(-4, os.SEEK_END)
            return struct.unpack("<I", f.read(4))[0]
    except (OSError, struct.error):
        return path.stat().st_size


def stats(raw_dir: Path = RAW_DIR) -> dict:
    files = list(iter_archived_files(raw_dir))
    sources: dict[str, int] = {}
    for f in files:
        source = archive_source_of(f, raw_dir)
        sources[source] = sources.get(source, 0) + 1
    return {
        "files": len(files),
        "bytes": sum(f.stat().st_size for f in files),
        "sources": sources,
    }


def verify_archives(raw_dir: Path = RAW_DIR) -> tuple[int, list[tuple[Path, str]]]:
    """底片体检:逐份完整解压——gzip 读到 EOF 时自动核对 CRC32,
    位翻转/截断都会在这里暴露。返回 (总数, [(损坏文件, 错误)])。"""
    total, bad = 0, []
    for gz in iter_archived_files(raw_dir):
        total += 1
        try:
            with gzip.open(gz, "rb") as f:
                while f.read(1 << 20):
                    pass
        except (OSError, EOFError, gzip.BadGzipFile) as e:
            bad.append((gz, str(e)))
    return total, bad


def find_archive(
    session_prefix: str,
    raw_dir: Path = RAW_DIR,
    source: str | None = None,
) -> list[Path]:
    """按会话 ID 前缀定位底片(可能多个项目下有同前缀,全部返回)。"""
    matches = []
    for path in iter_archived_files(raw_dir):
        provider = archive_source_of(path, raw_dir)
        if source and provider != source:
            continue
        if provider == "codex":
            from .codex_extractor import session_id_of

            matched = session_id_of(path).startswith(session_prefix)
        else:
            matched = path.name.startswith(session_prefix)
        if matched:
            matches.append(path)
    return matches
