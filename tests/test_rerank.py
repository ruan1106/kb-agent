"""测重排公式(确定性,不依赖外部服务)。"""
from kb.rerank import _weighted_score, weighted_rerank


def test_keyword_overlap_boosts_ranking():
    """关键词重合的片段应排到前面(即使向量分略低)。"""
    no_kw = {"score": 0.4, "payload": {"text": "无关内容", "type": "concept", "created_at": ""}}
    with_kw = {"score": 0.3, "payload": {"text": "epoll 中断 处理", "type": "concept", "created_at": ""}}
    out = weighted_rerank("epoll 中断", [no_kw, with_kw], top_n=2)
    assert out[0]["payload"]["text"] == "epoll 中断 处理"


def test_pitfall_type_bonus():
    """问"坑"时命中 pitfall 类型加分。"""
    concept = {"score": 0.5, "payload": {"text": "x", "type": "concept", "created_at": ""}}
    pitfall = {"score": 0.5, "payload": {"text": "x", "type": "pitfall", "created_at": ""}}
    assert _weighted_score("这个报错怎么踩坑", pitfall) > _weighted_score("这个报错怎么踩坑", concept)


def test_high_vector_score_still_wins_without_keywords():
    """没有关键词重合时,向量分主导。"""
    high = {"score": 0.9, "payload": {"text": "abc", "type": "concept", "created_at": ""}}
    low = {"score": 0.1, "payload": {"text": "def", "type": "concept", "created_at": ""}}
    out = weighted_rerank("zzz", [low, high], top_n=2)
    assert out[0]["payload"]["text"] == "abc"
