"""
向量库 Qdrant
=============
study 的 rag_agent.py 用 numpy 存向量,撑不住规模、没法按字段过滤。
这里用 Qdrant 建「知识库 collection」,演示:
    1. 建 collection(锁维度)+ payload 索引(type/category/created_at/note_id)
    2. upsert 写入(向量 + payload)
    3. 混合检索(向量召回 + payload 过滤一次完成)★

可插拔(沿用 agent/qdrant_agent.py 的降级风格):
    - 装了 qdrant-client 且本地起了 Qdrant -> 走真 Qdrant
    - 否则降级到 numpy 迷你版(MiniQdrant),同样接口

对应概念文档:第 7 节 RAG(检索步)。
起服务:docker compose up -d qdrant
"""
import math
import os

from .config import settings


# ============================================================
# 0. 真 Qdrant 连接
# ============================================================
def make_client():
    """连真 Qdrant;连不上返回 None。"""
    try:
        from qdrant_client import QdrantClient
    except Exception:
        return None
    try:
        client = QdrantClient(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key or None,
        )
        client.get_collections()  # 探活
        print("[Qdrant] 连上真服务")
        return client
    except Exception as e:
        print(f"[Qdrant] 连不上真服务({e}),降级到 numpy 迷你版")
        return None


# ============================================================
# 1. numpy 迷你版(接口对齐真 Qdrant,教学降级)
# ============================================================
class MiniQdrant:
    """教学降级:用 numpy 实现 collection + payload 过滤 + 向量搜索。"""

    def __init__(self):
        import numpy as np
        self.np = np
        self._collections: dict[str, dict] = {}
        print("[Qdrant] 使用 numpy 迷尼版(教学降级)")

    def get_collections(self):
        return list(self._collections.keys())

    def recreate_collection(self, name, vectors_config, **_):
        self._collections[name] = {"size": vectors_config["size"], "points": []}

    def create_payload_index(self, name, field, field_schema, **_):
        pass  # 迷你版顺序扫描,不需要索引

    def upsert(self, name, points, **_):
        for p in points:
            self._collections[name]["points"].append({
                "id": p.id, "vector": list(p.vector), "payload": p.payload or {},
            })

    def search(self, name, query_vector, query_filter=None, limit=3, **_):
        np = self.np
        pts = self._collections[name]["points"]
        if query_filter and query_filter.get("must"):
            pts = [p for p in pts if _match_all(p["payload"], query_filter["must"])]
        qv = np.array(query_vector)
        scored = []
        for p in pts:
            v = np.array(p["vector"])
            sim = float(qv @ v)  # 已归一化,点积 = 余弦
            scored.append((sim, p))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [_Hit(s, p["payload"], p["id"]) for s, p in scored[:limit]]


class _Hit:
    """统一 hit 结构(score/payload/id),真 Qdrant 的 ScoredPoint 也有这些字段。"""
    def __init__(self, score, payload, id):
        self.score = score
        self.payload = payload
        self.id = id


class _Point:
    """迷你版用的简易 point(对齐 qdrant PointStruct)。"""
    def __init__(self, id, vector, payload):
        self.id, self.vector, self.payload = id, vector, payload


def _match_all(payload: dict, must: list) -> bool:
    """简化版 Filter.must 匹配:支持 match(MatchValue) 和 range(Range)。"""
    for cond in must:
        key, spec = cond["key"], cond
        val = payload.get(key)
        if "match" in spec:
            if val != spec["match"].get("value"):
                return False
        elif "range" in spec:
            r = spec["range"]
            if "gte" in r and (val is None or val < r["gte"]):
                return False
            if "lte" in r and (val is None or val > r["lte"]):
                return False
    return True


def _build_qdrant_filter(filters: list[dict] | None):
    """把 [{'key':'type','match':{'value':'concept'}}, ...] 转成真 Qdrant 的 Filter。"""
    if not filters:
        return None
    from qdrant_client.models import Filter, FieldCondition, MatchValue, Range
    must = []
    for cond in filters:
        key = cond["key"]
        if "match" in cond:
            must.append(FieldCondition(key=key, match=MatchValue(value=cond["match"]["value"])))
        elif "range" in cond:
            must.append(FieldCondition(key=key, range=Range(**cond["range"])))
    return Filter(must=must) if must else None


# ============================================================
# 2. 业务封装:建库 / 写入 / 三种检索
# ============================================================
_client = None


def get_client():
    """单例 client(真 Qdrant 或迷你版)。"""
    global _client
    if _client is None:
        _client = make_client() or MiniQdrant()
    return _client


def ensure_collection(name: str | None = None):
    """建 collection + payload 索引(幂等:已存在不重建,不抹数据)。"""
    client = get_client()
    name = name or settings.kb_collection
    existing = client.get_collections()
    # 真 Qdrant 返回 GetCollectionsResponse(.collections);迷你版返回 list[str]
    names = [c.name for c in existing.collections] if hasattr(existing, "collections") else list(existing)
    if name in names:
        return client, name
    if hasattr(client, "create_collection"):
        client.create_collection(name, vectors_config={"size": settings.embed_dim, "distance": "Cosine"})
        for field, schema in [("type", "keyword"), ("category", "keyword"),
                              ("note_id", "keyword"), ("created_at", "datetime")]:
            client.create_payload_index(name, field, schema)
    else:
        client.recreate_collection(name, vectors_config={"size": settings.embed_dim, "distance": "Cosine"})
    print(f"[Qdrant] 建 collection {name}(dim={settings.embed_dim})")
    return client, name


def upsert_chunks(points: list[tuple[str, list[float], dict]], name: str | None = None):
    """points: [(id, vector, payload), ...]。真 Qdrant 用 PointStruct,迷你版用 _Point。"""
    name = name or settings.kb_collection
    ensure_collection(name)  # 幂等:保证 collection 存在,任何调用顺序都安全
    client = get_client()
    try:
        from qdrant_client.models import PointStruct
        pts = [PointStruct(id=pid, vector=vec, payload=p) for pid, vec, p in points]
    except Exception:
        pts = [_Point(pid, vec, p) for pid, vec, p in points]
    client.upsert(name, points=pts)


def search(query_vector: list[float], filters: list[dict] | None = None,
           limit: int = 5, name: str | None = None) -> list[dict]:
    """混合检索:向量召回 + payload 过滤一次完成。返回 [{score, payload, id}]。"""
    name = name or settings.kb_collection
    ensure_collection(name)  # 空库首问时 collection 可能还没建
    client = get_client()
    if isinstance(client, MiniQdrant):
        qf = {"must": filters} if filters else None
    else:
        qf = _build_qdrant_filter(filters)
    hits = client.search(name, query_vector=query_vector, query_filter=qf, limit=limit)
    return [{"score": h.score, "payload": h.payload, "id": h.id} for h in hits]
