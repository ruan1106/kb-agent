"""
Embedding(阿里通义 text-embedding-v3)
======================================
文本 -> 向量。批量上限 25 条/次,自动分批。维度与 Qdrant collection 对齐(1024)。
可插拔:没配 DASHSCOPE_API_KEY -> 降级到确定性哈希向量(演示流程,不懂语义)。
对应概念文档:第 7.2 节 RAG 的 embedding 步骤。
"""
import hashlib
import math

from .config import settings

_BATCH = 25  # DashScope text-embedding-v3 单次上限


def _have_dashscope() -> bool:
    if not settings.dashscope_api_key:
        return False
    try:
        import dashscope  # noqa: F401
        return True
    except Exception:
        return False


def _embed_real(texts: list[str]) -> list[list[float]]:
    import dashscope
    from dashscope import TextEmbedding

    dashscope.api_key = settings.dashscope_api_key
    out: list[list[float]] = []
    for i in range(0, len(texts), _BATCH):
        batch = texts[i:i + _BATCH]
        resp = TextEmbedding.call(
            model=settings.embed_model,
            input=batch,
            dimension=settings.embed_dim,
        )
        if resp.status_code != 200:
            raise RuntimeError(f"embedding 失败: {resp.code} {resp.message}")
        # 按 text_index 排序,保证顺序与输入一致
        embs = sorted(resp.output["embeddings"], key=lambda e: e["text_index"])
        out.extend([e["embedding"] for e in embs])
    return out


def _embed_fake(texts: list[str]) -> list[list[float]]:
    """降级:确定性哈希向量。相同文本 -> 相同向量,演示用,不懂语义。"""
    dim = settings.embed_dim
    vecs = []
    for t in texts:
        v = [0.0] * dim
        for i in range(dim):
            h = hashlib.md5(f"{t}|{i}".encode()).hexdigest()
            v[i] = int(h[:8], 16) / 0xFFFFFFFF - 0.5
        n = math.sqrt(sum(x * x for x in v)) or 1.0
        vecs.append([x / n for x in v])
    return vecs


_real: bool | None = None


def _use_real() -> bool:
    """是否用通义真 embedding。缓存结果;没配 key 或没装 dashscope -> 假向量。"""
    global _real
    if _real is None:
        _real = _have_dashscope()
        print("[Embedding] 用通义 text-embedding-v3" if _real
              else "[Embedding] 未配 DASHSCOPE_API_KEY,降级到哈希向量(演示用)")
    return _real


def embed_texts(texts: list[str]) -> list[list[float]]:
    """批量文本 -> 向量列表(顺序与输入一致)。"""
    if not texts:
        return []
    return _embed_real(texts) if _use_real() else _embed_fake(texts)


def embed_query(text: str) -> list[float]:
    """单条查询 -> 向量。"""
    return embed_texts([text])[0]
