"""Safe, idempotent client integration for AICodeMemory.

The default mode registers the same local stdio MCP server with every installed
supported client.  Claude Code and Codex keep separate configuration, so each
registration is inspected before mutation and an unrelated server named
``cmem`` is never overwritten.
"""

from __future__ import annotations

import json
import os
import shlex
import shutil
import stat
import subprocess
import sys
import tempfile
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

SERVER_NAME = "cmem"
INSTRUCTION_MARKER_BEGIN = "<!-- cmem:begin -->"
INSTRUCTION_MARKER_END = "<!-- cmem:end -->"
# 兼容早期内部名称;同一受管区块同时适用 CLAUDE.md / AGENTS.md。
CLAUDE_MARKER_BEGIN = INSTRUCTION_MARKER_BEGIN
CLAUDE_MARKER_END = INSTRUCTION_MARKER_END
COMMAND_TIMEOUT_SECONDS = 15

Emit = Callable[[str], None]
Runner = Callable[[Sequence[str], float], subprocess.CompletedProcess[str]]
Which = Callable[[str], str | None]


class SetupError(RuntimeError):
    """A safe setup operation could not be completed."""


@dataclass(frozen=True)
class ClientSpec:
    key: str
    label: str
    executable: str

    def inspect_command(self) -> list[str]:
        command = [self.executable, "mcp", "get", SERVER_NAME]
        if self.key == "codex":
            command.append("--json")
        return command

    def add_command(self, server_command: Sequence[str]) -> list[str]:
        if self.key == "claude":
            return [
                self.executable,
                "mcp",
                "add",
                "--transport",
                "stdio",
                "--scope",
                "user",
                SERVER_NAME,
                "--",
                *server_command,
            ]
        return [
            self.executable,
            "mcp",
            "add",
            SERVER_NAME,
            "--",
            *server_command,
        ]

    def remove_command(self) -> list[str]:
        command = [self.executable, "mcp", "remove"]
        if self.key == "claude":
            command.extend(["--scope", "user"])
        command.append(SERVER_NAME)
        return command


@dataclass(frozen=True)
class ConfiguredServer:
    command: str
    args: tuple[str, ...]
    user_scoped: bool = True


@dataclass
class IntegrationReport:
    detected: list[str] = field(default_factory=list)
    changed: list[str] = field(default_factory=list)
    unchanged: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors


@dataclass(frozen=True)
class InstructionUpdate:
    path: Path
    updated: str
    managed_block: str
    remove: bool


def run_command(
    command: Sequence[str], timeout: float
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(command),
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )


def discover_clients(which: Which = shutil.which) -> list[ClientSpec]:
    clients = []
    for key, label in (("claude", "Claude Code"), ("codex", "Codex")):
        if executable := which(key):
            clients.append(
                ClientSpec(key, label, str(Path(executable).expanduser().absolute()))
            )
    return clients


def resolve_module_command(
    script: str,
    module: str,
    *,
    python: str = sys.executable,
    which: Which = shutil.which,
) -> list[str]:
    """Prefer the console script next to this interpreter, with a module fallback."""
    python_path = Path(python).expanduser().absolute()
    sibling = python_path.with_name(script)
    if sibling.is_file() and os.access(sibling, os.X_OK):
        return [str(sibling)]
    if found := which(script):
        return [str(Path(found).expanduser().absolute())]
    return [str(python_path), "-m", module]


def resolve_mcp_command(**kwargs) -> list[str]:
    return resolve_module_command("cmem-mcp", "cmem.mcp_server", **kwargs)


def resolve_cmem_command(**kwargs) -> list[str]:
    return resolve_module_command("cmem", "cmem.cli", **kwargs)


def _combined_output(result: subprocess.CompletedProcess[str]) -> str:
    return "\n".join(part for part in (result.stdout, result.stderr) if part).strip()


def _is_missing(result: subprocess.CompletedProcess[str]) -> bool:
    output = _combined_output(result).lower()
    return any(
        marker in output
        for marker in (
            "no mcp server named",
            "mcp server not found",
            "does not exist",
        )
    )


def _parse_claude_server(output: str) -> ConfiguredServer:
    values: dict[str, str] = {}
    for line in output.splitlines():
        stripped = line.strip()
        for key in ("Scope", "Command", "Args"):
            prefix = f"{key}:"
            if stripped.startswith(prefix):
                values[key] = stripped[len(prefix) :].strip()
    if "Command" not in values:
        raise SetupError("Claude Code 返回了无法识别的 MCP 配置")
    try:
        args = tuple(shlex.split(values.get("Args", "")))
    except ValueError as exc:
        raise SetupError("Claude Code MCP 参数无法解析") from exc
    return ConfiguredServer(
        command=values["Command"],
        args=args,
        user_scoped=values.get("Scope", "").startswith("User config"),
    )


def _parse_codex_server(output: str) -> ConfiguredServer:
    try:
        payload = json.loads(output)
        transport = payload["transport"]
        if transport["type"] != "stdio":
            raise SetupError("Codex 中同名 cmem 配置不是 stdio server")
        return ConfiguredServer(
            command=transport["command"],
            args=tuple(transport.get("args") or ()),
        )
    except (KeyError, TypeError, json.JSONDecodeError) as exc:
        raise SetupError("Codex 返回了无法识别的 MCP 配置") from exc


def _inspect_client(
    client: ClientSpec, runner: Runner
) -> tuple[ConfiguredServer | None, str | None]:
    try:
        result = runner(client.inspect_command(), COMMAND_TIMEOUT_SECONDS)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return None, f"{client.label}: 读取现有 MCP 配置失败({exc})"
    if result.returncode != 0:
        if _is_missing(result):
            return None, None
        return None, f"{client.label}: 读取现有 MCP 配置失败"
    try:
        configured = (
            _parse_claude_server(result.stdout)
            if client.key == "claude"
            else _parse_codex_server(result.stdout)
        )
    except SetupError as exc:
        return None, f"{client.label}: {exc}"
    return configured, None


def _matches(configured: ConfiguredServer, expected: Sequence[str]) -> bool:
    return (
        configured.command == expected[0]
        and configured.args == tuple(expected[1:])
        and configured.user_scoped
    )


def _belongs_to_cmem(configured: ConfiguredServer, expected: Sequence[str]) -> bool:
    if not configured.user_scoped:
        return False
    if _matches(configured, expected):
        return True
    executable = Path(configured.command).name.removesuffix(".exe")
    return executable == "cmem-mcp" or configured.args == ("-m", "cmem.mcp_server")


def _short_failure(result: subprocess.CompletedProcess[str]) -> str:
    output = _combined_output(result)
    if not output:
        return f"退出码 {result.returncode}"
    # add/remove 命令不包含凭据;仍限制长度,避免把客户端诊断整段注入终端。
    return output[-500:]


def configure_mcp(
    *,
    remove: bool = False,
    clients: Sequence[ClientSpec] | None = None,
    server_command: Sequence[str] | None = None,
    runner: Runner = run_command,
    emit: Emit = print,
) -> IntegrationReport:
    clients = list(clients if clients is not None else discover_clients())
    expected = list(server_command or resolve_mcp_command())
    report = IntegrationReport(detected=[client.key for client in clients])

    if not expected:
        report.errors.append("cmem-mcp 启动命令不能为空")
        return report

    if not clients:
        if remove:
            emit("未检测到 Claude Code 或 Codex CLI,MCP 配置无需处理")
        else:
            report.errors.append(
                "未检测到 Claude Code 或 Codex CLI;可安装客户端后重试,"
                "或使用 cmem setup --claude-md"
            )
        return report

    inspected: list[tuple[ClientSpec, ConfiguredServer | None]] = []
    for client in clients:
        configured, error = _inspect_client(client, runner)
        if error:
            report.errors.append(error)
        else:
            inspected.append((client, configured))

    # 先完成所有只读检查;发现同名冲突时不对任何客户端动手。
    for client, configured in inspected:
        if configured is None:
            continue
        if remove:
            if not _belongs_to_cmem(configured, expected):
                report.errors.append(
                    f"{client.label}: 同名 cmem 配置不属于 AICodeMemory,"
                    "已保留不动"
                )
        elif not _matches(configured, expected):
            report.errors.append(
                f"{client.label}: 已有不同的 cmem 配置,已保留不覆盖;"
                "请先确认并移除旧配置"
            )
    if report.errors:
        return report

    for client, configured in inspected:
        if remove and configured is None:
            report.unchanged.append(client.key)
            emit(f"{client.label}: 未配置 cmem,无需移除")
            continue
        if not remove and configured is not None:
            report.unchanged.append(client.key)
            emit(f"{client.label}: cmem 已正确配置")
            continue

        command = client.remove_command() if remove else client.add_command(expected)
        verb = "移除" if remove else "注册"
        emit(f"{client.label}: 将{verb} cmem\n  $ {shlex.join(command)}")
        try:
            result = runner(command, COMMAND_TIMEOUT_SECONDS)
        except (OSError, subprocess.TimeoutExpired) as exc:
            report.errors.append(f"{client.label}: {verb}失败({exc})")
            continue
        if result.returncode != 0:
            report.errors.append(
                f"{client.label}: {verb}失败: {_short_failure(result)}"
            )
            continue
        report.changed.append(client.key)
        emit(f"{client.label}: {verb}完成")

    return report


def instruction_block(cmem_command: Sequence[str]) -> str:
    search_command = shlex.join([*cmem_command, "search", "<问题>"])
    return (
        f"{INSTRUCTION_MARKER_BEGIN}\n"
        "## 历史会话检索\n"
        "被问到过去的讨论、决策、结论或原话时,先运行 "
        f"`{search_command}` 检索历史原文。\n"
        "基于检索结果回答并注明出处;查不到就明说,不要凭印象编造。\n"
        f"{INSTRUCTION_MARKER_END}"
    )


def codex_instruction_paths(codex_home: Path | None = None) -> tuple[Path, Path]:
    if codex_home is None:
        configured_home = os.environ.get("CODEX_HOME")
        codex_home = (
            Path(configured_home) if configured_home else Path.home() / ".codex"
        )
    home = Path(codex_home).expanduser().absolute()
    return home / "AGENTS.md", home / "AGENTS.override.md"


def active_codex_instruction_path(codex_home: Path | None = None) -> Path:
    """Return the global file Codex will actually read for the current profile."""
    agents, override = codex_instruction_paths(codex_home)
    if override.exists() and override.read_text(encoding="utf-8").strip():
        return override
    return agents


def _atomic_write(path: Path, content: str) -> None:
    target = path.resolve() if path.is_symlink() else path
    target.parent.mkdir(parents=True, exist_ok=True)
    mode = stat.S_IMODE(target.stat().st_mode) if target.exists() else 0o644
    fd, temporary = tempfile.mkstemp(
        prefix=f".{target.name}.cmem-", dir=target.parent
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, mode)
        os.replace(temporary, target)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def _prepare_instruction_update(
    path: Path,
    *,
    remove: bool = False,
    cmem_command: Sequence[str] | None = None,
) -> tuple[InstructionUpdate | None, str]:
    path = Path(path).expanduser()
    existing = path.read_text(encoding="utf-8") if path.exists() else ""
    begin_count = existing.count(INSTRUCTION_MARKER_BEGIN)
    end_count = existing.count(INSTRUCTION_MARKER_END)
    if begin_count != end_count or begin_count > 1:
        raise SetupError(
            f"{path} 中 cmem marker 不完整或重复,为避免破坏文件已停止"
        )

    if begin_count:
        start = existing.index(INSTRUCTION_MARKER_BEGIN)
        end_marker = existing.find(INSTRUCTION_MARKER_END, start)
        if end_marker < 0:
            raise SetupError(
                f"{path} 中 cmem marker 顺序错误,为避免破坏文件已停止"
            )
        end = end_marker + len(INSTRUCTION_MARKER_END)
    else:
        start = end = -1

    if remove:
        if not begin_count:
            return None, f"{path}: 没有 cmem 受管区块,无需移除"
        before = existing[:start].rstrip("\n")
        after = existing[end:].lstrip("\n")
        if before and after:
            updated = f"{before}\n\n{after}"
        elif before:
            updated = f"{before}\n"
        else:
            updated = after
        managed_block = existing[start:end]
    else:
        block = instruction_block(cmem_command or resolve_cmem_command())
        if begin_count:
            updated = f"{existing[:start]}{block}{existing[end:]}"
        else:
            separator = (
                "" if not existing else ("\n" if existing.endswith("\n") else "\n\n")
            )
            updated = f"{existing}{separator}{block}\n"
        if updated == existing:
            return None, f"{path}: cmem 受管区块已是最新,无需修改"
        managed_block = block

    return (
        InstructionUpdate(path, updated, managed_block, remove),
        "",
    )


def update_instruction_files(
    paths: Sequence[Path],
    *,
    remove: bool = False,
    cmem_command: Sequence[str] | None = None,
    emit: Emit = print,
) -> list[Path]:
    """Validate every file first, then update each managed block atomically."""
    unique_paths: list[Path] = []
    seen: set[Path] = set()
    for path in paths:
        normalized = Path(path).expanduser().absolute()
        if normalized not in seen:
            seen.add(normalized)
            unique_paths.append(normalized)

    updates: list[InstructionUpdate] = []
    noops: list[str] = []
    resolved_command = None if remove else list(cmem_command or resolve_cmem_command())
    for path in unique_paths:
        update, noop = _prepare_instruction_update(
            path,
            remove=remove,
            cmem_command=resolved_command,
        )
        if update is None:
            noops.append(noop)
        else:
            updates.append(update)

    # 所有 marker 都已验证完成;从这里开始才回显并写入。
    for message in noops:
        emit(message)
    for update in updates:
        action = "从" if update.remove else "写入"
        suffix = "移除以下" if update.remove else "的"
        emit(
            f"将{action} {update.path} {suffix}受管区块:\n"
            f"{update.managed_block}"
        )
    for update in updates:
        _atomic_write(update.path, update.updated)
        emit(f"{update.path}: 更新完成")

    return [update.path for update in updates]


def update_instruction_file(
    path: Path,
    *,
    remove: bool = False,
    cmem_command: Sequence[str] | None = None,
    emit: Emit = print,
) -> bool:
    return bool(
        update_instruction_files(
            [path],
            remove=remove,
            cmem_command=cmem_command,
            emit=emit,
        )
    )


def update_claude_md(
    path: Path,
    *,
    remove: bool = False,
    cmem_command: Sequence[str] | None = None,
    emit: Emit = print,
) -> bool:
    """Compatibility wrapper for the original Claude-only setup mode."""
    return update_instruction_file(
        path,
        remove=remove,
        cmem_command=cmem_command,
        emit=emit,
    )
