"""
三层记忆
========
对应概念文档 §5.1 三层记忆:
    - 短期:State messages(在图里,随 checkpointer 落盘)
    - 长期:事实(SQLite facts 表)+ 近期问答(SQLite qa_history)← 本文件
    - 外部:RAG(Qdrant,见 tools/vectorstore)

load_long_term 把长期记忆拼成文本,喂给生成节点;save_turn 把一轮问答落库。
"""
from . import storage


def load_long_term(thread_id: str) -> str:
    """加载长期记忆:已知事实 + 近期问答。"""
    parts = []
    facts = storage.list_facts(thread_id)
    if facts:
        parts.append("已知事实:\n" + "\n".join(facts))
    qa = storage.list_qa(thread_id, limit=5)
    if qa:
        parts.append("近期问答:\n" + "\n".join(
            f"Q:{x['question']} A:{x['answer'][:80]}" for x in qa))
    return "\n\n".join(parts)


def save_turn(thread_id: str, question: str, answer: str, refs: list):
    """一轮问答结束,落进 qa_history(成为下次可召回的长期记忆)。"""
    storage.save_qa(thread_id, question, answer, refs)


def save_fact(thread_id: str, key: str, value: str):
    """抽取一条事实存进 facts(如「用户主要用 Python」)。"""
    storage.save_fact(thread_id, key, value)
