"""Client setup is safe, idempotent and reversible."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from cmem.setup import (
    CLAUDE_MARKER_BEGIN,
    CLAUDE_MARKER_END,
    ClientSpec,
    ConfiguredServer,
    SetupError,
    active_codex_instruction_path,
    codex_instruction_paths,
    configure_mcp,
    cursor_config_path,
    discover_cursor_config,
    resolve_module_command,
    update_claude_md,
    update_instruction_files,
)


CLIENTS = [
    ClientSpec("claude", "Claude Code", "/fake/claude"),
    ClientSpec("codex", "Codex", "/fake/codex"),
]


class FakeRunner:
    def __init__(self) -> None:
        self.configs: dict[str, ConfiguredServer | None] = {
            "claude": None,
            "codex": None,
        }
        self.calls: list[list[str]] = []

    @staticmethod
    def _client(command: list[str]) -> str:
        return Path(command[0]).name

    def __call__(
        self, command, timeout: float
    ) -> subprocess.CompletedProcess[str]:
        command = list(command)
        self.calls.append(command)
        client = self._client(command)
        operation = command[2]
        configured = self.configs[client]

        if operation == "get":
            if configured is None:
                return subprocess.CompletedProcess(
                    command, 1, "", "No MCP server named 'cmem' found."
                )
            if client == "claude":
                scope = (
                    "User config (available in all your projects)"
                    if configured.user_scoped
                    else "Local config"
                )
                output = (
                    "cmem:\n"
                    f"  Scope: {scope}\n"
                    "  Type: stdio\n"
                    f"  Command: {configured.command}\n"
                    f"  Args: {' '.join(configured.args)}\n"
                )
            else:
                output = json.dumps(
                    {
                        "name": "cmem",
                        "transport": {
                            "type": "stdio",
                            "command": configured.command,
                            "args": list(configured.args),
                        },
                    }
                )
            return subprocess.CompletedProcess(command, 0, output, "")

        if operation == "add":
            separator = command.index("--")
            server_command = command[separator + 1 :]
            self.configs[client] = ConfiguredServer(
                server_command[0], tuple(server_command[1:])
            )
            return subprocess.CompletedProcess(command, 0, "added", "")

        if operation == "remove":
            self.configs[client] = None
            return subprocess.CompletedProcess(command, 0, "removed", "")

        raise AssertionError(command)


def mutation_calls(runner: FakeRunner) -> list[list[str]]:
    return [call for call in runner.calls if call[2] in {"add", "remove"}]


def test_mcp_setup_registers_both_clients_and_is_idempotent():
    runner = FakeRunner()
    server = ["/venv/bin/cmem-mcp"]
    output: list[str] = []

    first = configure_mcp(
        clients=CLIENTS,
        server_command=server,
        runner=runner,
        emit=output.append,
    )
    assert first.ok
    assert first.changed == ["claude", "codex"]
    assert runner.configs == {
        "claude": ConfiguredServer(server[0], ()),
        "codex": ConfiguredServer(server[0], ()),
    }
    assert mutation_calls(runner) == [
        [
            "/fake/claude",
            "mcp",
            "add",
            "--transport",
            "stdio",
            "--scope",
            "user",
            "cmem",
            "--",
            server[0],
        ],
        ["/fake/codex", "mcp", "add", "cmem", "--", server[0]],
    ]

    second = configure_mcp(
        clients=CLIENTS,
        server_command=server,
        runner=runner,
        emit=output.append,
    )
    assert second.ok
    assert second.changed == []
    assert second.unchanged == ["claude", "codex"]
    assert len(mutation_calls(runner)) == 2
    assert any("将注册 cmem" in line for line in output)


def test_mcp_setup_refuses_conflict_before_mutating_other_client():
    runner = FakeRunner()
    runner.configs["codex"] = ConfiguredServer("/bin/unrelated", ())

    report = configure_mcp(
        clients=CLIENTS,
        server_command=["/venv/bin/cmem-mcp"],
        runner=runner,
        emit=lambda _: None,
    )

    assert not report.ok
    assert "保留不覆盖" in report.errors[0]
    assert runner.configs["claude"] is None
    assert mutation_calls(runner) == []


def test_mcp_setup_without_clients_fails_but_remove_is_a_noop():
    install = configure_mcp(
        clients=[],
        server_command=["cmem-mcp"],
        emit=lambda _: None,
    )
    remove = configure_mcp(
        remove=True,
        clients=[],
        server_command=["cmem-mcp"],
        emit=lambda _: None,
    )

    assert not install.ok and "--instructions" in install.errors[0]
    assert remove.ok


def test_cursor_config_registers_preserves_other_servers_and_is_idempotent(
    tmp_path,
):
    path = tmp_path / ".cursor" / "mcp.json"
    path.parent.mkdir()
    path.write_text(
        json.dumps(
            {
                "version": 1,
                "mcpServers": {
                    "other": {
                        "command": "/bin/other",
                        "env": {"TOKEN": "do-not-print"},
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    path.chmod(0o600)
    output: list[str] = []
    server = ["/venv/bin/cmem-mcp"]

    first = configure_mcp(
        clients=[],
        cursor_config=path,
        server_command=server,
        emit=output.append,
    )
    assert first.ok
    assert first.detected == ["cursor"]
    assert first.changed == ["cursor"]
    installed = json.loads(path.read_text(encoding="utf-8"))
    assert installed["version"] == 1
    assert installed["mcpServers"]["other"]["env"]["TOKEN"] == "do-not-print"
    assert installed["mcpServers"]["cmem"] == {
        "command": server[0],
        "args": [],
    }
    assert path.stat().st_mode & 0o777 == 0o600
    assert "do-not-print" not in "\n".join(output)
    installed_text = path.read_text(encoding="utf-8")

    second = configure_mcp(
        clients=[],
        cursor_config=path,
        server_command=server,
        emit=output.append,
    )
    assert second.ok
    assert second.changed == []
    assert second.unchanged == ["cursor"]
    assert path.read_text(encoding="utf-8") == installed_text


def test_new_cursor_config_uses_private_file_permissions(tmp_path):
    path = tmp_path / ".cursor" / "mcp.json"

    report = configure_mcp(
        clients=[],
        cursor_config=path,
        server_command=["/venv/bin/cmem-mcp"],
        emit=lambda _: None,
    )

    assert report.ok
    assert path.stat().st_mode & 0o777 == 0o600


def test_cursor_conflict_is_preflighted_before_cli_mutation(tmp_path):
    path = tmp_path / "mcp.json"
    original = json.dumps(
        {"mcpServers": {"cmem": {"command": "/bin/unrelated", "args": []}}}
    )
    path.write_text(original, encoding="utf-8")
    runner = FakeRunner()

    report = configure_mcp(
        clients=CLIENTS,
        cursor_config=path,
        server_command=["/venv/bin/cmem-mcp"],
        runner=runner,
        emit=lambda _: None,
    )

    assert not report.ok
    assert any("Cursor" in error and "保留不覆盖" in error for error in report.errors)
    assert mutation_calls(runner) == []
    assert path.read_text(encoding="utf-8") == original


def test_malformed_cursor_config_stops_before_cli_mutation(tmp_path):
    path = tmp_path / "mcp.json"
    path.write_text("{ broken", encoding="utf-8")
    runner = FakeRunner()

    report = configure_mcp(
        clients=CLIENTS,
        cursor_config=path,
        server_command=["/venv/bin/cmem-mcp"],
        runner=runner,
        emit=lambda _: None,
    )

    assert not report.ok
    assert any("Cursor" in error and "有效 JSON" in error for error in report.errors)
    assert mutation_calls(runner) == []
    assert path.read_text(encoding="utf-8") == "{ broken"


def test_cursor_remove_is_safe_reversible_and_idempotent(tmp_path):
    path = tmp_path / "mcp.json"
    path.write_text(
        json.dumps(
            {
                "mcpServers": {
                    "other": {"command": "/bin/other"},
                    "cmem": {"command": "/old/bin/cmem-mcp", "args": []},
                }
            }
        ),
        encoding="utf-8",
    )

    first = configure_mcp(
        remove=True,
        clients=[],
        cursor_config=path,
        server_command=["/venv/bin/cmem-mcp"],
        emit=lambda _: None,
    )
    assert first.ok and first.changed == ["cursor"]
    assert json.loads(path.read_text(encoding="utf-8"))["mcpServers"] == {
        "other": {"command": "/bin/other"}
    }

    second = configure_mcp(
        remove=True,
        clients=[],
        cursor_config=path,
        server_command=["/venv/bin/cmem-mcp"],
        emit=lambda _: None,
    )
    assert second.ok
    assert second.changed == []
    assert second.unchanged == ["cursor"]


def test_cursor_remove_preserves_unmanaged_same_name_server(tmp_path):
    path = tmp_path / "mcp.json"
    original = json.dumps(
        {"mcpServers": {"cmem": {"command": "/bin/unrelated", "args": []}}}
    )
    path.write_text(original, encoding="utf-8")

    report = configure_mcp(
        remove=True,
        clients=[],
        cursor_config=path,
        server_command=["/venv/bin/cmem-mcp"],
        emit=lambda _: None,
    )

    assert not report.ok
    assert "不属于 AICodeMemory" in report.errors[0]
    assert path.read_text(encoding="utf-8") == original


def test_cursor_detection_uses_global_config_directory_or_application(tmp_path):
    def nothing(_: str) -> None:
        return None

    assert discover_cursor_config(
        home=tmp_path,
        which=nothing,
        application_paths=[],
    ) is None

    cursor_dir = tmp_path / ".cursor"
    cursor_dir.mkdir()
    assert discover_cursor_config(
        home=tmp_path,
        which=nothing,
        application_paths=[],
    ) == cursor_config_path(tmp_path).absolute()

    other_home = tmp_path / "other"
    app = tmp_path / "Cursor.app"
    app.mkdir()
    assert discover_cursor_config(
        home=other_home,
        which=nothing,
        application_paths=[app],
    ) == cursor_config_path(other_home).absolute()


def test_mcp_remove_is_safe_reversible_and_idempotent():
    runner = FakeRunner()
    server = ["/venv/bin/cmem-mcp"]
    runner.configs = {
        "claude": ConfiguredServer(server[0], ()),
        "codex": ConfiguredServer(server[0], ()),
    }

    first = configure_mcp(
        remove=True,
        clients=CLIENTS,
        server_command=server,
        runner=runner,
        emit=lambda _: None,
    )
    second = configure_mcp(
        remove=True,
        clients=CLIENTS,
        server_command=server,
        runner=runner,
        emit=lambda _: None,
    )

    assert first.ok and first.changed == ["claude", "codex"]
    assert second.ok and second.unchanged == ["claude", "codex"]
    assert runner.configs == {"claude": None, "codex": None}
    assert mutation_calls(runner) == [
        ["/fake/claude", "mcp", "remove", "--scope", "user", "cmem"],
        ["/fake/codex", "mcp", "remove", "cmem"],
    ]


@pytest.mark.parametrize(
    "configured",
    [
        ConfiguredServer("/bin/unrelated", ()),
        ConfiguredServer("/venv/bin/cmem-mcp", (), user_scoped=False),
    ],
)
def test_mcp_remove_preserves_unmanaged_same_name_server(configured):
    runner = FakeRunner()
    runner.configs["claude"] = configured

    report = configure_mcp(
        remove=True,
        clients=CLIENTS,
        server_command=["/venv/bin/cmem-mcp"],
        runner=runner,
        emit=lambda _: None,
    )

    assert not report.ok
    assert "不属于 AICodeMemory" in report.errors[0]
    assert runner.configs["claude"] == configured
    assert mutation_calls(runner) == []


def test_claude_md_round_trip_is_idempotent_and_preserves_user_text(tmp_path):
    path = tmp_path / ".claude" / "CLAUDE.md"
    path.parent.mkdir()
    original = "# 我的指令\n"
    path.write_text(original, encoding="utf-8")
    output: list[str] = []

    assert update_claude_md(
        path,
        cmem_command=["/venv/bin/cmem"],
        emit=output.append,
    )
    installed = path.read_text(encoding="utf-8")
    assert installed.count(CLAUDE_MARKER_BEGIN) == 1
    assert installed.count(CLAUDE_MARKER_END) == 1
    assert "'/venv/bin/cmem'" not in installed
    assert "/venv/bin/cmem search '<问题>'" in installed
    assert output[0].startswith("将写入")

    assert not update_claude_md(
        path,
        cmem_command=["/venv/bin/cmem"],
        emit=output.append,
    )
    assert path.read_text(encoding="utf-8") == installed

    assert update_claude_md(path, remove=True, emit=output.append)
    assert path.read_text(encoding="utf-8") == original
    assert not update_claude_md(path, remove=True, emit=output.append)


def test_claude_md_stops_on_broken_markers(tmp_path):
    path = tmp_path / "CLAUDE.md"
    broken = f"user content\n{CLAUDE_MARKER_END}\n{CLAUDE_MARKER_BEGIN}\n"
    path.write_text(broken, encoding="utf-8")

    with pytest.raises(SetupError, match="marker 顺序错误"):
        update_claude_md(path, cmem_command=["cmem"], emit=lambda _: None)
    assert path.read_text(encoding="utf-8") == broken


def test_instruction_files_update_claude_and_codex_as_one_validated_batch(tmp_path):
    claude = tmp_path / ".claude" / "CLAUDE.md"
    codex = tmp_path / ".codex" / "AGENTS.md"
    claude.parent.mkdir()
    codex.parent.mkdir()
    claude_original = "# Claude preferences\n"
    codex_original = "# Codex preferences\n"
    claude.write_text(claude_original, encoding="utf-8")
    codex.write_text(
        f"{codex_original}{CLAUDE_MARKER_BEGIN}\n",
        encoding="utf-8",
    )

    # Codex 文件 marker 损坏时,Claude 文件也不得被提前改写。
    with pytest.raises(SetupError, match="marker 不完整"):
        update_instruction_files(
            [claude, codex],
            cmem_command=["cmem"],
            emit=lambda _: None,
        )
    assert claude.read_text(encoding="utf-8") == claude_original

    codex.write_text(codex_original, encoding="utf-8")
    changed = update_instruction_files(
        [claude, codex],
        cmem_command=["cmem"],
        emit=lambda _: None,
    )
    assert changed == [claude.absolute(), codex.absolute()]
    assert CLAUDE_MARKER_BEGIN in claude.read_text(encoding="utf-8")
    assert CLAUDE_MARKER_BEGIN in codex.read_text(encoding="utf-8")
    assert update_instruction_files(
        [claude, codex],
        cmem_command=["cmem"],
        emit=lambda _: None,
    ) == []

    update_instruction_files([claude, codex], remove=True, emit=lambda _: None)
    assert claude.read_text(encoding="utf-8") == claude_original
    assert codex.read_text(encoding="utf-8") == codex_original


def test_codex_instruction_path_honors_profile_and_nonempty_override(
    tmp_path, monkeypatch
):
    codex_home = tmp_path / "profile"
    agents, override = codex_instruction_paths(codex_home)
    assert agents == codex_home / "AGENTS.md"
    assert override == codex_home / "AGENTS.override.md"
    assert active_codex_instruction_path(codex_home) == agents

    codex_home.mkdir()
    override.write_text("", encoding="utf-8")
    assert active_codex_instruction_path(codex_home) == agents
    override.write_text("# temporary global override\n", encoding="utf-8")
    assert active_codex_instruction_path(codex_home) == override

    alternate = tmp_path / "alternate"
    monkeypatch.setenv("CODEX_HOME", str(alternate))
    assert codex_instruction_paths()[0] == alternate / "AGENTS.md"


def test_resolve_module_command_keeps_virtualenv_interpreter_path(tmp_path):
    bin_dir = tmp_path / "venv" / "bin"
    bin_dir.mkdir(parents=True)
    python = bin_dir / "python"
    python.touch()

    assert resolve_module_command(
        "cmem-mcp",
        "cmem.mcp_server",
        python=str(python),
        which=lambda _: None,
    ) == [str(python), "-m", "cmem.mcp_server"]
