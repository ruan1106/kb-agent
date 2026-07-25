"""
MCP server(把知识库工具标准化暴露)
==================================
对应概念文档 §9 MCP:工具方实现一次,任何 MCP 客户端(Claude Desktop 等)都能用。
暴露:search_concepts / search_pitfalls / search_notes / ingest_note。

跑起来(在 kb/ 目录):
    python -m kb.mcp_server        # stdio 模式,供 MCP 客户端连接
"""
from mcp.server.fastmcp import FastMCP

from kb import tools, vectorstore
from kb.ingest import ingest_text

vectorstore.ensure_collection()
mcp = FastMCP("personal-kb")


@mcp.tool()
def search_concepts(query: str, top_k: int = 5) -> str:
    """检索计算机原理/概念笔记。"""
    return tools.search_concepts.invoke({"query": query, "top_k": top_k})


@mcp.tool()
def search_pitfalls(query: str, top_k: int = 5) -> str:
    """检索踩坑避坑笔记(复现条件/根因/解法)。"""
    return tools.search_pitfalls.invoke({"query": query, "top_k": top_k})


@mcp.tool()
def search_notes(query: str, top_k: int = 5) -> str:
    """跨类型检索所有笔记(概念+踩坑)。"""
    return tools.search_notes.invoke({"query": query, "top_k": top_k})


@mcp.tool()
def ingest_note(title: str, content: str, note_type: str = "concept") -> str:
    """沉淀新笔记进知识库。note_type: concept(原理/概念) 或 pitfall(踩坑)。"""
    nid = ingest_text(title, content, note_type)
    return f"已沉淀 note_id={nid}({note_type}): {title}"


if __name__ == "__main__":
    mcp.run()
