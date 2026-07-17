"""AICodeMemory MCP server (stdio transport)."""

from __future__ import annotations

from typing import Any

from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.exceptions import ToolError
from mcp.types import ToolAnnotations

from .searcher import Hit
from .service import MemoryService, MemoryServiceError


READ_ONLY = ToolAnnotations(
    readOnlyHint=True,
    destructiveHint=False,
    idempotentHint=True,
    openWorldHint=False,
)


def _hit_dict(hit: Hit) -> dict[str, Any]:
    return {
        "session_key": f"{hit.source}:{hit.session_id}",
        "source": hit.source,
        "session_id": hit.session_id,
        "project": hit.project,
        "date": hit.date,
        "chunk_index": hit.chunk_index,
        "score": hit.score,
        "cosine": hit.cos,
        "bm25": hit.bm25,
        "text": hit.text,
    }


def create_server(service: MemoryService | None = None) -> FastMCP:
    memory = service or MemoryService()
    server = FastMCP(
        "AICodeMemory",
        instructions=(
            "Vendor-independent, local, read-only archive of Claude Code and "
            "Codex and Cursor sessions. Search it when the user needs cross-client "
            "history, verbatim evidence, older decisions, or how a coding problem "
            "was solved."
        ),
    )

    @server.tool(annotations=READ_ONLY)
    def search_history(
        query: str,
        k: int = 5,
        before: str = "",
        exclude_project: str = "",
        source: str = "",
    ) -> dict[str, Any]:
        """搜索过去的 AI 编码会话原话。

        当用户询问跨客户端历史、过去的讨论/决策、精确原话、
        稳定出处或曾经解决的问题时使用。
        before 为可选 YYYY-MM-DD(不含当天);exclude_project 可排除自指污染项目;
        source 可限定 claude/codex/cursor。
        结果中的 session_key + chunk_index 可传给 get_session 展开上下文。
        """
        try:
            hits = memory.search_history(
                query,
                k=k,
                before=before,
                exclude_projects=(exclude_project,) if exclude_project else (),
                source=source,
            )
        except MemoryServiceError as exc:
            raise ToolError(str(exc)) from exc
        return {
            "query": query,
            "count": len(hits),
            "hits": [_hit_dict(hit) for hit in hits],
            "index_state": memory.index_state(),
        }

    @server.tool(annotations=READ_ONLY)
    def get_session(
        session_id: str,
        around: int | None = None,
    ) -> dict[str, Any]:
        """展开一场历史会话的 text 当前投影。

        session_id 建议传 search_history 返回的 session_key(source:id),
        也可传唯一会话 ID 前缀。around 为命中的 chunk_index;传入时
        返回该块前各 2 块,不传则返回整场会话。
        """
        try:
            return memory.get_session(session_id, around=around).as_dict()
        except MemoryServiceError as exc:
            raise ToolError(str(exc)) from exc

    @server.tool(annotations=READ_ONLY)
    def memory_status() -> dict[str, Any]:
        """查看本地记忆覆盖范围、来源、完整性与索引新鲜度。"""
        return memory.memory_status()

    return server


mcp = create_server()


def main() -> None:
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
