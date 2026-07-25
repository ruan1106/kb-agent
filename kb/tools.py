"""
Agent 工具(Function Calling)
=============================
Agent 通过这些工具「行动」。每个 @tool 自动从签名+docstring 生成 JSON Schema。
    - search_concepts / search_pitfalls / search_notes:RAG 检索(向量+payload过滤+重排)
    - ingest_note:把新笔记沉淀进库(可被 graph 标记为需 HITL 确认)

检索结果带 note_id,供生成节点引用、审查节点做引用存在性校验。
对应概念文档:第 3 节 Function Calling、第 7.4 节 RAG 作为工具。
"""
from langchain_core.tools import tool

from . import embedding, rerank, vectorstore


def _format(hits: list[dict]) -> str:
    if not hits:
        return "(没检索到相关笔记)"
    parts = []
    for i, h in enumerate(hits, 1):
        p = h["payload"]
        head = f"[片段{i}] note_id={p.get('note_id')} type={p.get('type')} cat={p.get('category','')}"
        parts.append(f"{head}\n{p.get('text', '')}")
    return "\n\n".join(parts)


def _search(query: str, filters: list[dict] | None, top_k: int) -> str:
    """向量召回 + payload 过滤 + 重排。"""
    qv = embedding.embed_query(query)
    hits = vectorstore.search(qv, filters=filters, limit=top_k * 3)
    ranked = rerank.rerank(query, hits, top_n=top_k)
    return _format(ranked)


@tool
def search_concepts(query: str, top_k: int = 5) -> str:
    """检索计算机原理/概念笔记。当用户问概念性、原理性问题时使用。"""
    return _search(query, [{"key": "type", "match": {"value": "concept"}}], top_k)


@tool
def search_pitfalls(query: str, top_k: int = 5) -> str:
    """检索踩坑避坑笔记(含复现条件/根因/解法)。当用户问报错、踩坑、避坑时使用。"""
    return _search(query, [{"key": "type", "match": {"value": "pitfall"}}], top_k)


@tool
def search_notes(query: str, top_k: int = 5) -> str:
    """跨类型检索所有笔记(概念+踩坑)。当不确定该查概念还是踩坑时使用。"""
    return _search(query, None, top_k)


@tool
def ingest_note(title: str, content: str, note_type: str = "concept") -> str:
    """把一段新笔记沉淀进知识库。note_type 取值:concept(原理/概念) 或 pitfall(踩坑)。
    用于用户说"记一下/沉淀"时。"""
    from .ingest import ingest_text
    note_id = ingest_text(title, content, note_type)
    return f"已沉淀笔记 note_id={note_id}({note_type}): {title}"


# 工具注册表(供 graph 的 tools 节点分发)
TOOLS = [search_concepts, search_pitfalls, search_notes, ingest_note]
TOOL_MAP = {t.name: t for t in TOOLS}

# 标记需要人工确认的工具(写入知识库 = 有副作用,走 HITL)
DANGEROUS = {"ingest_note"}
