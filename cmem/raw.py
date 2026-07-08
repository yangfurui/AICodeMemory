"""Raw archive layer — the negatives, not the prints.

源 jsonl(~/.claude/projects/)被 Claude Code 按 30 天滚动清理。text 档案层
存的是去噪后的对话(剪报),而本层把源文件**原样 gzip 存档**(底片):
任何未来的去噪/切块算法改进,都能凭它回溯应用到全部历史——包括源早已
消失的部分。

契约(与 text 档案层同级,不可违反):
- raw 层的文件【永不自动删除】。代码里不存在删除路径;源文件从磁盘
  消失不影响其存档。
- 存档以【源文件 mtime】为增量判定并写到 gz 文件自身的 mtime 上:
  源没变则跳过(零成本),源追加了内容则覆盖更新。
- 原子写入:先写 .tmp 再 rename,中断不留半个存档。

目录结构镜像源:~/.cmem/raw/<源父目录名>/<会话>.jsonl.gz
保留 Claude Code 的项目目录编码,重提取时与直读源目录的行为完全一致。
"""

from __future__ import annotations

import gzip
import os
import shutil
from pathlib import Path

RAW_DIR = Path.home() / ".cmem" / "raw"


def archive_path_for(src: Path, raw_dir: Path = RAW_DIR) -> Path:
    return raw_dir / src.parent.name / (src.name + ".gz")


def archive_session(src: Path, mtime: int, raw_dir: Path = RAW_DIR) -> bool:
    """把源会话文件 gzip 存档;已是最新则跳过。返回是否实际写入。

    mtime 由调用方在【读取源内容之前】stat 得到(与索引共用同一语义):
    存档期间源又有追加,gz 的 mtime 会小于源的新 mtime,下次触发重存。
    """
    dst = archive_path_for(src, raw_dir)
    if dst.exists() and int(dst.stat().st_mtime) >= mtime:
        return False  # 存档已覆盖该版本

    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".tmp")
    try:
        with src.open("rb") as f_in, gzip.open(tmp, "wb", compresslevel=6) as f_out:
            shutil.copyfileobj(f_in, f_out)
        os.utime(tmp, (mtime, mtime))  # gz mtime = 源 mtime,作为增量与重提取的账本
        tmp.replace(dst)  # 原子落位
        return True
    finally:
        tmp.unlink(missing_ok=True)


def iter_archived_files(raw_dir: Path = RAW_DIR):
    """遍历全部存档(重提取时的数据源)。"""
    if raw_dir.is_dir():
        yield from sorted(raw_dir.rglob("*.jsonl.gz"))


def stats(raw_dir: Path = RAW_DIR) -> dict:
    files = list(iter_archived_files(raw_dir))
    return {
        "files": len(files),
        "bytes": sum(f.stat().st_size for f in files),
    }
