"""Parse Claude Code session jsonl into clean dialogue.

Claude Code 的会话文件是逐行 JSON(~/.claude/projects/<project>/<uuid>.jsonl),
一行一个事件。我们只要人和 AI 的对话文本,其余全是噪音:

- 行级:只收 type in {user, assistant};system / file-history-snapshot /
  permission-mode / mode / ai-title / last-prompt / attachment 等直接跳过
- 元素级:content 数组里只收 type=="text";tool_use / tool_result /
  thinking / image 全部丢弃(思维链和工具输出不属于"说过的话")
- 侧链:isSidechain==true 的行是子代理转录,跳过;agent-*.jsonl 文件同理
- 文本级:剥离 <system-reminder> 块;含斜杠命令样板(<command-name> 等)的
  user 消息整条丢弃
"""

from __future__ import annotations

import gzip
import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

# 去噪/切块算法的版本轴:本文件的过滤规则或 chunker 的切块策略发生语义
# 变化时 bump。库检测到不一致会触发"从 raw 存档重提取"(见 store 的
# pending_migration)——只在真正影响产出的变更时动它。
# "2" 对应 v0.2 的去噪清单,与既有库无缝衔接。
EXTRACT_VERSION = "2"

# 系统注入的提醒块,出现在 user 消息文本内
_SYSTEM_REMINDER_RE = re.compile(r"<system-reminder>.*?</system-reminder>", re.DOTALL)
# 出现任一标记即视为系统注入/样板消息,整条丢弃。
# 注意只按已知系统标记精准过滤,不做"见尖括号就杀"——对话正文里
# 大量出现 Set<string>、<uid> 这类代码占位符(2026-07-08 全量审计证实)。
_COMMAND_MARKERS = (
    "<command-name>",         # 斜杠命令样板
    "<local-command-stdout>",
    "<local-command-caveat>",
    "<task-notification>",    # 后台任务完成通知(伪装成 user 的系统消息,含 result/usage 全家族)
    "<bash-input>",           # 用户 ! 前缀本地命令的回显(操作记录,非对话)
    "<bash-stdout>",
    "<bash-stderr>",
)


@dataclass
class Message:
    role: str  # "user" | "assistant"
    text: str


@dataclass
class SessionDialogue:
    session_id: str
    project: str  # 会话 cwd 的目录名,如 "kb"、"AICodeMemory"
    date: str  # 最后一条消息的本地日期 YYYY-MM-DD
    file_mtime: int  # 处理时的源文件 mtime,供增量防重
    messages: list[Message] = field(default_factory=list)


def iter_session_files(source: Path) -> Iterator[Path]:
    """遍历数据源下所有主会话文件(排除子代理转录 agent-*)。"""
    yield from sorted(
        p for p in source.rglob("*.jsonl") if not p.name.startswith("agent-")
    )


def session_id_of(path: Path) -> str:
    """源文件与 raw 存档共用的会话 ID:xxx.jsonl 与 xxx.jsonl.gz → xxx。"""
    return path.name.removesuffix(".gz").removesuffix(".jsonl")


def _clean_user_text(text: str) -> str:
    if any(m in text for m in _COMMAND_MARKERS):
        return ""
    return _SYSTEM_REMINDER_RE.sub("", text).strip()


def _content_text(content) -> str:
    """message.content 可能是纯字符串或元素数组,只取 text 元素。"""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            e.get("text", "") for e in content
            if isinstance(e, dict) and e.get("type") == "text"
        )
    return ""


def _to_local_date(iso_ts: str) -> str:
    ts = iso_ts.replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(ts)
    except ValueError:
        return ""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone().strftime("%Y-%m-%d")


def parse_session(path: Path) -> SessionDialogue | None:
    """解析一个会话文件(源 .jsonl 或 raw 存档 .jsonl.gz);无有效对话时返回 None。"""
    # 在读取内容前取 mtime:处理期间若有追加,下次增量会重新处理(借鉴 trace)
    mtime = int(path.stat().st_mtime)
    messages: list[Message] = []
    cwd = ""
    last_ts = ""

    opener = (lambda: gzip.open(path, "rt", encoding="utf-8", errors="replace")) \
        if path.name.endswith(".gz") else \
        (lambda: path.open(encoding="utf-8", errors="replace"))
    with opener() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("type") not in ("user", "assistant"):
                continue
            if row.get("isSidechain"):
                continue
            cwd = row.get("cwd") or cwd
            last_ts = row.get("timestamp") or last_ts

            text = _content_text((row.get("message") or {}).get("content"))
            if row["type"] == "user":
                text = _clean_user_text(text)
            text = text.strip()
            if not text:
                continue

            # 同角色连续消息合并(assistant 常被工具调用切成多段)
            if messages and messages[-1].role == row["type"]:
                messages[-1].text += "\n" + text
            else:
                messages.append(Message(role=row["type"], text=text))

    if not messages:
        return None
    return SessionDialogue(
        session_id=session_id_of(path),
        project=Path(cwd).name if cwd else "unknown",
        date=_to_local_date(last_ts) if last_ts else "",
        file_mtime=mtime,
        messages=messages,
    )
