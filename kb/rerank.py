"""
重排
====
召回 top-k 后精排。两套:
    1. 手写多维加权公式(向量分 + 关键词重合 + 类型匹配 + 时间新鲜度)
    2. Cross-Encoder 精排(gte-rerank API,有 DASHSCOPE 才用)

可插拔:没配 DashScope -> 只用手写加权;配了 -> 先加权筛 20 条,再 cross-encoder 精排 top_n。
对应概念文档:第 7.3 节 RAG 的「重排序」步骤。
"""
from .config import settings


# ============================================================
# 1. 工具:关键词重合度
# ============================================================
def _tokens(text: str) -> set[str]:
    """分词。有 jieba 用 jieba,没有就按空白/字符切。"""
    try:
        import jieba
        jieba.setLogLevel(jieba.logging.WARNING)
        return {w for w in jieba.cut(text) if len(w.strip()) > 1}
    except Exception:
        return {w for w in text.lower().split() if len(w) > 1}


def _keyword_overlap(query: str, text: str) -> float:
    qt, tt = _tokens(query), _tokens(text)
    if not qt:
        return 0.0
    return len(qt & tt) / len(qt)


def _have_dashscope() -> bool:
    if not settings.dashscope_api_key:
        return False
    try:
        import dashscope  # noqa: F401
        return True
    except Exception:
        return False


# ============================================================
# 2. 手写多维加权公式
# ============================================================
def _weighted_score(query: str, hit: dict) -> float:
    """多维加权:向量余弦 + 关键词重合 + 类型匹配 + 时间新鲜度。权重可调。"""
    base = float(hit.get("score", 0.0))                       # 向量余弦(0~1)
    kw = _keyword_overlap(query, hit["payload"].get("text", ""))  # 0~1

    # 类型匹配:concept/pitfall 与查询意图匹配时加分(简化:问"坑"命中 pitfall 加分)
    ptype = hit["payload"].get("type", "")
    type_bonus = 0.15 if (ptype == "pitfall" and ("坑" in query or "报错" in query or "失败" in query)) else 0.0

    # 时间新鲜度:近 90 天内的笔记加分(简化处理)
    created = hit["payload"].get("created_at", "")
    recency = 0.05 if created and created >= "2026-04" else 0.0

    return 0.6 * base + 0.2 * kw + type_bonus + recency


def weighted_rerank(query: str, hits: list[dict], top_n: int = 20) -> list[dict]:
    """按加权公式重排。"""
    for h in hits:
        h["rerank_score"] = _weighted_score(query, h)
    hits.sort(key=lambda h: h["rerank_score"], reverse=True)
    return hits[:top_n]


# ============================================================
# 3. Cross-Encoder 精排(gte-rerank API)
# ============================================================
def _cross_rerank(query: str, hits: list[dict], top_n: int) -> list[dict]:
    import dashscope
    from dashscope import TextReRank

    dashscope.api_key = settings.dashscope_api_key
    docs = [h["payload"].get("text", "") for h in hits]
    resp = TextReRank.call(
        model=settings.rerank_model,
        query=query,
        documents=docs,
        top_n=min(top_n, len(docs)),
        return_documents=False,
    )
    if resp.status_code != 200:
        print(f"[Rerank] gte-rerank 失败({resp.code}),回退手写加权")
        return hits[:top_n]
    results = resp.output["results"]  # [{index, relevance_score}]
    out = []
    for r in results:
        h = dict(hits[r["index"]])
        h["rerank_score"] = float(r["relevance_score"])
        out.append(h)
    return out


# ============================================================
# 4. 统一入口
# ============================================================
def rerank(query: str, hits: list[dict], top_n: int = 5) -> list[dict]:
    """先手写加权筛,有 DashScope 再 cross-encoder 精排。"""
    if not hits:
        return []
    pre = weighted_rerank(query, hits, top_n=min(len(hits), 20))
    if _have_dashscope() and len(pre) > 1:
        try:
            return _cross_rerank(query, pre, top_n)
        except Exception as e:
            print(f"[Rerank] cross-encoder 异常({e}),回退手写加权")
    return pre[:top_n]
