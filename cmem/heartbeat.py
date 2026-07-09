"""Heartbeat — 静默失败告警(launchd 归档任务停摆的探测层)。

原则:失败者不能自报失败——cmem 命令本身失效(重装后路径变化/venv 损坏)时,
launchd 任务根本起不来,写在任务里的任何告警代码都不会执行。告警因此挂在
两处"必然还活着"的东西上:

- 使用路径(search / show / status):读心跳文件,超龄即向 stderr 告警。
  本工具的日常界面是 AI 会话,stderr 随工具结果被 AI 看到并转告用户,
  送达率高于系统通知;
- /bin/bash:launchd plist 用 bash 包一层 `cmem index || osascript 通知`,
  防的正是"命令没了/坏了"这种最可能的静默失败(bash 自身不会失效)。

心跳只在 cmem index【成功收尾且扫描数 > 0】时刷新:扫到 0 份(源目录改名/
迁移导致两个数据源都不可见)时进程虽正常退出,归档实已停摆——视同失败。

阈值 72h:每日 + 每次登录双触发下,3 天没成功必有异常;此时距 30 天
清理窗口还剩约 27 天裕量,足够从容修复。
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

HEARTBEAT = Path.home() / ".cmem" / "last-index-ok"
STALE_AFTER_H = 72


def mark_success(path: Path = HEARTBEAT) -> None:
    """cmem index 成功收尾时调用;文件 mtime 即"上次成功时刻"。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch()


def _age_hours(path: Path, now: float | None) -> float | None:
    """距上次成功的小时数;从未成功过(文件不存在)返回 None。"""
    try:
        mtime = path.stat().st_mtime
    except FileNotFoundError:
        return None
    return ((now if now is not None else time.time()) - mtime) / 3600


def describe(path: Path = HEARTBEAT, now: float | None = None) -> str:
    """人读的心跳状态(status 常驻展示)。"""
    h = _age_hours(path, now)
    if h is None:
        return "无记录(尚未成功跑过 index)"
    t = time.strftime("%Y-%m-%d %H:%M", time.localtime(path.stat().st_mtime))
    ago = f"{h * 60:.0f} 分钟前" if h < 1 else f"{h:.0f} 小时前" if h < 48 else f"{h / 24:.0f} 天前"
    return f"{t}({ago})"


def warn_if_stale(path: Path = HEARTBEAT, now: float | None = None) -> bool:
    """使用路径上的超龄探测;超龄或无记录时向 stderr 打一行告警。"""
    h = _age_hours(path, now)
    if h is not None and h <= STALE_AFTER_H:
        return False
    what = "从未成功运行过" if h is None else f"已 {h / 24:.1f} 天未成功更新"
    print(f"⚠️ 索引{what}——launchd 任务可能失效,请跑 cmem index,"
          f"并查看 ~/.cmem/launchd-index.log", file=sys.stderr)
    return True
