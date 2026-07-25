"""测向量库检索(MiniQdrant 降级版,不依赖真 Qdrant/embedding)。"""
from kb.vectorstore import MiniQdrant, _Point, _match_all


def test_mini_search_orders_by_similarity():
    c = MiniQdrant()
    c.recreate_collection("t", {"size": 4, "distance": "Cosine"})
    c.upsert("t", [
        _Point(1, [1.0, 0.0, 0.0, 0.0], {"type": "concept", "text": "a"}),
        _Point(2, [0.0, 1.0, 0.0, 0.0], {"type": "concept", "text": "b"}),
    ])
    hits = c.search("t", query_vector=[1.0, 0.0, 0.0, 0.0], limit=2)
    assert hits[0].id == 1          # 与查询同向,最相似
    assert hits[0].score > hits[1].score


def test_match_all_keyword_and_range():
    payload = {"type": "concept", "created_at": "2026-07-01"}
    assert _match_all(payload, [{"key": "type", "match": {"value": "concept"}}])
    assert not _match_all(payload, [{"key": "type", "match": {"value": "pitfall"}}])
    assert _match_all(payload, [{"key": "created_at", "range": {"gte": "2026-01-01"}}])
    assert not _match_all(payload, [{"key": "created_at", "range": {"gte": "2027-01-01"}}])


def test_mini_search_with_filter():
    c = MiniQdrant()
    c.recreate_collection("t", {"size": 4, "distance": "Cosine"})
    c.upsert("t", [
        _Point(1, [1.0, 0, 0, 0], {"type": "concept", "text": "a"}),
        _Point(2, [1.0, 0, 0, 0], {"type": "pitfall", "text": "b"}),
    ])
    flt = {"must": [{"key": "type", "match": {"value": "pitfall"}}]}
    hits = c.search("t", query_vector=[1.0, 0, 0, 0], query_filter=flt, limit=5)
    assert len(hits) == 1
    assert hits[0].payload["type"] == "pitfall"
