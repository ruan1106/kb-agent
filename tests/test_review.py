"""测审查 Agent 的引用存在性校验(确定性规则,不依赖 LLM)。"""
from kb.models import ReviewVerdict
from kb.review import _cited_ids, review as do_review


class _StubResp:
    """模拟 bind_tools().invoke() 的返回:带 tool_calls。"""
    def __init__(self, verdict):
        self.tool_calls = [{"args": verdict.model_dump()}]


class _StubLLM:
    """模拟 chat llm:bind_tools -> invoke -> 带 tool_calls 的响应。"""
    def __init__(self, verdict):
        self.verdict = verdict

    def bind_tools(self, _tools, **_kw):
        return self

    def invoke(self, _prompt):
        return _StubResp(self.verdict)


def test_cited_ids_parse():
    assert _cited_ids("见 note_id=deadbeef 和 note_id=cafef00d") == {"deadbeef", "cafef00d"}
    assert _cited_ids("没有引用") == set()


def test_fake_citation_fails(monkeypatch):
    """引用了不存在的 note_id -> citation_real=False -> 不通过。"""
    # LLM 说一切 OK,但确定性规则应否决假引用
    stub_verdict = ReviewVerdict(citation_real=True, no_hallucination=True,
                                 on_topic=True, need_more_retrieval=False, reason="llm 说 ok")
    monkeypatch.setattr("kb.review.get_chat_llm", lambda *a, **k: _StubLLM(stub_verdict))
    monkeypatch.setattr("kb.review.storage.note_exists", lambda cid: False)

    retrieved = [{"note_id": "real1", "text": "真实片段"}]
    v, passed = do_review("q", "答案见 note_id=deadbeef", retrieved)
    assert v.citation_real is False
    assert passed is False


def test_real_citation_passes(monkeypatch):
    """引用的 note_id 在检索集里 -> citation_real=True。"""
    stub_verdict = ReviewVerdict(citation_real=True, no_hallucination=True,
                                 on_topic=True, need_more_retrieval=False, reason="ok")
    monkeypatch.setattr("kb.review.get_chat_llm", lambda *a, **k: _StubLLM(stub_verdict))
    monkeypatch.setattr("kb.review.storage.note_exists", lambda cid: False)

    retrieved = [{"note_id": "real1", "text": "真实片段"}]
    v, passed = do_review("q", "答案见 note_id=real1", retrieved)
    assert v.citation_real is True
    assert passed is True
