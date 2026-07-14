"""MCP tool schemas and a real stdio protocol handshake."""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

import numpy as np
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from cmem.chunker import Chunk
from cmem.embedder import DIM
from cmem.mcp_server import create_server
from cmem.service import MemoryService
from cmem.store import Store


def test_server_exposes_only_the_three_read_tools(tmp_path):
    store = Store(tmp_path / "memory.sqlite3")
    store.pending_migration()
    vector = np.zeros(DIM, dtype=np.float32)
    vector[0] = 1
    store.index_session(
        "codex",
        "session-1",
        1,
        20,
        [Chunk("codex", "session-1", "demo", "2026-07-14", 0, "needle 结论")],
        np.stack([vector]),
    )

    class FakeEmbedder:
        def encode_query(self, query: str) -> np.ndarray:
            return vector

    server = create_server(
        MemoryService(
            raw_dir=tmp_path / "raw",
            heartbeat_path=tmp_path / "heartbeat",
            embedder_factory=FakeEmbedder,
            store=store,
        )
    )

    async def exercise_server():
        tools = await server.list_tools()
        status = await server.call_tool("memory_status", {})
        search = await server.call_tool("search_history", {"query": "needle"})
        session = await server.call_tool(
            "get_session", {"session_id": "codex:session-1", "around": 0}
        )
        return tools, status, search, session

    tools, (_, status), (_, search), (_, session) = asyncio.run(exercise_server())

    assert [tool.name for tool in tools] == [
        "search_history",
        "get_session",
        "memory_status",
    ]
    schemas = {tool.name: tool.inputSchema for tool in tools}
    assert all(tool.annotations.readOnlyHint is True for tool in tools)
    assert all(tool.annotations.destructiveHint is False for tool in tools)
    assert schemas["search_history"]["required"] == ["query"]
    assert set(schemas["search_history"]["properties"]) == {
        "query",
        "k",
        "before",
        "exclude_project",
    }
    assert schemas["get_session"]["required"] == ["session_id"]
    assert status["integrity"] == "ok"
    assert status["chunks"] == 1
    assert search["hits"][0]["session_key"] == "codex:session-1"
    assert search["hits"][0]["chunk_index"] == 0
    assert session["chunks"] == [{"index": 0, "text": "needle 结论"}]


def test_stdio_server_initializes_and_answers_memory_status(tmp_path):
    project_root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env["HOME"] = str(tmp_path)
    params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "cmem.mcp_server"],
        cwd=project_root,
        env=env,
    )
    with (tmp_path / "server.stderr").open("w+") as stderr:

        async def exercise_stdio():
            async with stdio_client(params, errlog=stderr) as (reader, writer):
                async with ClientSession(reader, writer) as session:
                    await session.initialize()
                    tools = await session.list_tools()
                    result = await session.call_tool("memory_status", {})
                    return tools, result

        tools, result = asyncio.run(exercise_stdio())
        stderr.seek(0)
        server_stderr = stderr.read()

    assert [tool.name for tool in tools.tools] == [
        "search_history",
        "get_session",
        "memory_status",
    ]
    assert result.isError is False
    assert result.structuredContent is not None
    assert result.structuredContent["integrity"] == "ok"
    assert result.structuredContent["database"] == str(
        tmp_path / ".cmem" / "memory.sqlite3"
    )
    assert "Traceback" not in server_stderr
    assert "ERROR" not in server_stderr
