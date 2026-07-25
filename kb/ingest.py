"""
导入(离线建索引)
=================
把笔记变成可检索的向量库:
    解析 -> 切块 -> embedding -> Qdrant upsert(payload) -> SQLite 元数据
    + 自动打标(structured output)+ 抽踩坑结构化字段(structured output)
    + 哈希去重(确定性逻辑,不交给 LLM)

对应概念文档:第 3 节 Structured Output、第 7 节 RAG(离线建索引)。
"""
import hashlib
import os
from datetime import datetime

from . import embedding, storage, vectorstore
from .config import settings
from .llm import get_chat_llm, structured_invoke
from .models import ChunkPayload, NoteTags, PitfallFields


def chunk_text(text: str, max_len: int = 500) -> list[str]:
    """切块:按空行切段落,合并到 max_len;块太大=不精准,太小=丢上下文。"""
    paras = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks, buf = [], ""
    for p in paras:
        if len(buf) + len(p) < max_len:
            buf = (buf + "\n\n" + p).strip()
        else:
            if buf:
                chunks.append(buf)
            buf = p if len(p) <= max_len else p[:max_len]
    if buf:
        chunks.append(buf)
    return chunks or [text[:max_len]]


def _auto_tag(title: str, content: str) -> NoteTags:
    """LLM 自动打标:领域分类 + 关键词 + 摘要(structured output)。失败返回空。"""
    try:
        llm = get_chat_llm(temperature=0.0)
        prompt = f"给下面笔记打标(领域分类/关键词标签/一句话摘要)。\n标题:{title}\n内容:\n{content[:1500]}"
        return structured_invoke(llm, NoteTags, prompt)
    except Exception as e:
        print(f"[Ingest] 自动打标失败({e}),用空标签")
        return NoteTags()


def _extract_pitfall(content: str) -> dict:
    """从踩坑笔记抽结构化字段:复现条件/根因/解法/环境/状态。失败返回空 dict。"""
    try:
        llm = get_chat_llm(temperature=0.0)
        prompt = f"从踩坑笔记抽取结构化字段。\n{content[:1500]}"
        return structured_invoke(llm, PitfallFields, prompt).model_dump()
    except Exception as e:
        print(f"[Ingest] 抽踩坑字段失败({e})")
        return {}


def ingest_text(title: str, content: str, note_type: str = "concept",
                source: str = "manual") -> str:
    """把一段文本沉淀进知识库。返回 note_id。

    去重:按内容哈希。已存在则复用 SQLite 元数据(省掉自动打标 LLM 调用,含 tags),
    但向量仍重新 embed+upsert--降级模式向量在内存/落盘,进程重启可能没载入,
    必须补上,否则检索永远空。upsert 按 id 幂等覆盖,不产生重复点。
    """
    note_id = hashlib.md5(content.encode()).hexdigest()[:16]
    existing = storage.get_note(note_id)

    if existing:
        category = existing["category"]
        tag_list = existing.get("tags", [])
        summary = existing["summary"]
        pitfall = existing.get("pitfall", {}) or {}
        now = existing["created_at"]
        reused = True
    else:
        tags = _auto_tag(title, content)
        category, tag_list, summary = tags.category, tags.tags, tags.summary
        pitfall = _extract_pitfall(content) if note_type == "pitfall" else {}
        now = datetime.now().isoformat(timespec="seconds")
        reused = False

    chunks = chunk_text(content)
    vectors = embedding.embed_texts(chunks)
    points = []
    for i, (ch, vec) in enumerate(zip(chunks, vectors)):
        payload = ChunkPayload(
            note_id=note_id, type=note_type, category=category, title=title,
            tags=tag_list, source=source, chunk_index=i, created_at=now, text=ch,
        ).model_dump()
        points.append((f"{note_id}_{i}", vec, payload))
    vectorstore.upsert_chunks(points)

    if not existing:
        storage.save_note({
            "note_id": note_id, "type": note_type, "category": category,
            "title": title, "source": source, "summary": summary,
            "created_at": now, "hash": note_id, "pitfall": pitfall, "tags": tag_list,
        })
    msg = "复用元数据,补向量" if reused else f"沉淀 {note_type} 笔记"
    print(f"[Ingest] {msg} {note_id}({title}),{len(chunks)} 块")
    return note_id


def ingest_file(path: str, note_type: str = "concept") -> str:
    """导入一个 .md 文件。"""
    title = os.path.splitext(os.path.basename(path))[0]
    content = open(path, encoding="utf-8").read()
    return ingest_text(title, content, note_type, source=path)


def ingest_dir(dir_path: str, note_type: str = "concept") -> int:
    """导入目录下所有 .md。"""
    n = 0
    for fn in sorted(os.listdir(dir_path)):
        if fn.endswith(".md"):
            try:
                ingest_file(os.path.join(dir_path, fn), note_type)
                n += 1
            except Exception as e:
                print(f"[Ingest] 跳过 {fn}({e})")
    return n


# ============================================================
# 种子语料:用仓库现有概念笔记初始化知识库
# ============================================================
def seed() -> int:
    """把仓库里现成的 .md 概念笔记灌进库,开箱即有内容可问。"""
    repo = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    targets = [
        os.path.join(repo, "agent-development-concepts.md"),
        os.path.join(repo, "langgraph-state-guide.md"),
    ]
    # agent/ 下的知识点汇总也是好语料
    agent_dir = os.path.join(repo, "agent")
    if os.path.isdir(agent_dir):
        for fn in sorted(os.listdir(agent_dir)):
            if fn.endswith("-知识点汇总.md"):
                targets.append(os.path.join(agent_dir, fn))
    # 确保 collection 存在
    vectorstore.ensure_collection()
    n = 0
    for p in targets:
        if os.path.isfile(p):
            try:
                ingest_file(p, "concept")
                n += 1
            except Exception as e:
                print(f"[Seed] 跳过 {p}({e})")
    print(f"[Seed] 共灌入 {n} 篇概念笔记")
    return n
