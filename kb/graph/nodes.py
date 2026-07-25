"""
节点 = 各角色 Agent(supervisor 多 Agent 协作)
=============================================
对应概念文档 §8 多 Agent:supervisor 编排,各专业 Agent 各司其职。
    - supervisor:确定性路由(谁先谁后 = 编排)
    - researcher:查询分解(§2.2 Plan-Execute)+ 混合检索 + 重排
    - writer:基于片段生成答案(judge 模型纯推理,不调工具;失败回退 chat)
    - reviewer:四道检查(§2.3 Reflection + §10 LLM-judge)
    - ask_human:低置信时 interrupt 暂停问人(§11.3 HITL)
"""
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END
from langgraph.types import interrupt

from .. import context, embedding, memory, rerank, vectorstore
from ..llm import get_chat_llm, get_judge_llm
from ..review import review as do_review
from .state import State


def _user_query(state: State) -> str:
    for m in reversed(state.get("messages") or []):
        if isinstance(m, HumanMessage):
            return m.content
    return ""


def _is_pitfall(q: str) -> bool:
    return any(k in q for k in ("坑", "报错", "失败", "异常", "error", "cannot"))


def _decompose(query: str, llm) -> list[str]:
    """查询分解(§2.2 Plan-Execute):拆成利于检索的子问题。失败回退原问题。"""
    try:
        msg = llm.invoke([
            SystemMessage("把问题拆成最多 2 个更利于检索的子问题,每行一个,不要编号不要解释。"
                          "不好拆就原样输出一行。"),
            HumanMessage(query),
        ])
        lines = [l.strip(" -0123456789.\t") for l in msg.content.splitlines() if l.strip()]
        return lines[:2] or [query]
    except Exception:
        return [query]


# ============================================================
# supervisor:确定性路由
# ============================================================
def supervisor(state: State) -> dict:
    q = state.get("query") or _user_query(state)
    retrieved = state.get("retrieved") or []
    answer = state.get("answer") or ""
    review_passed = state.get("review_passed")
    verdict = state.get("verdict") or {}

    if review_passed is True:
        nxt = "end"
    elif verdict.get("need_more_retrieval"):
        nxt = "researcher"
    elif review_passed is False:
        nxt = "ask_human"
    elif answer:
        nxt = "reviewer"
    elif retrieved:
        nxt = "writer"
    else:
        nxt = "researcher"
    return {"query": q, "next": nxt}


def route(state: State) -> str:
    n = state.get("next", "end")
    return END if n == "end" else n


# ============================================================
# researcher:检索 Agent
# ============================================================
def researcher(state: State) -> dict:
    q = state.get("query") or _user_query(state)
    llm = get_chat_llm(temperature=0.0)
    subs = _decompose(q, llm)

    raw = []
    for s in subs[:3]:
        qv = embedding.embed_query(s)
        flt = [{"key": "type", "match": {"value": "pitfall"}}] if _is_pitfall(s) else None
        raw += vectorstore.search(qv, filters=flt, limit=8)

    seen, dedup = set(), []
    for h in raw:
        if h["id"] not in seen:
            seen.add(h["id"])
            dedup.append(h)
    ranked = rerank.rerank(q, dedup, top_n=5)
    retrieved = [{
        "note_id": h["payload"].get("note_id"),
        "type": h["payload"].get("type"),
        "text": h["payload"].get("text", ""),
        "score": h.get("rerank_score", h.get("score")),
    } for h in ranked]
    print(f"  [researcher] 检索 {len(retrieved)} 条")
    # 重置下游状态,强制重新生成+审查
    return {"retrieved": retrieved, "answer": "", "review_passed": None, "verdict": {}}


# ============================================================
# writer:生成 Agent(judge 模型纯推理)
# ============================================================
def writer(state: State, config=None) -> dict:
    q = state.get("query") or _user_query(state)
    retrieved = state.get("retrieved") or []
    thread_id = (config or {}).get("configurable", {}).get("thread_id", "default")
    long_term = memory.load_long_term(thread_id)

    ctx = "\n\n".join(f"[note_id={h['note_id']}] {h['text']}" for h in retrieved) or "(无检索结果)"
    sys = SystemMessage(
        "你是知识库答题助手。只基于【检索片段】回答,不要编造概念。"
        "引用来源时写 note_id=xxx。片段不足就直说需要更多信息。\n"
        f"【长期记忆】{long_term or '无'}"
    )
    msgs = context.trim([sys, HumanMessage(f"问题:{q}\n\n【检索片段】\n{ctx}")])

    try:
        ans = get_judge_llm().invoke(msgs)   # judge(pro):纯推理,不调工具
        model_used = "judge"
    except Exception as e:
        print(f"  [writer] judge 失败({e}),回退 chat")
        ans = get_chat_llm().invoke(msgs)
        model_used = "chat"
    print(f"  [writer] 生成答案({model_used})")
    return {"answer": ans.content, "review_passed": None, "verdict": {}}


# ============================================================
# reviewer:审查 Agent(四道检查)
# ============================================================
def reviewer(state: State) -> dict:
    v, passed = do_review(state.get("query", ""), state.get("answer", ""),
                          state.get("retrieved") or [])
    print(f"  [reviewer] passed={passed} reason={v.reason}")
    return {"verdict": v.model_dump(), "review_passed": passed}


# ============================================================
# ask_human:人机协作(§11.3 interrupt)
# ============================================================
def ask_human(state: State) -> dict:
    approval = interrupt({
        "question": state.get("query", ""),
        "answer": state.get("answer", ""),
        "issue": (state.get("verdict") or {}).get("reason", "审查未通过,低置信"),
        "prompt": "同意此答案? 输入 yes 同意 / no 重新检索 / 或直接输入补充线索",
    })
    if isinstance(approval, str) and approval.strip().lower() in ("yes", "y"):
        return {"review_passed": True}
    if isinstance(approval, str) and approval.strip().lower() in ("no", "n"):
        return {"verdict": {**(state.get("verdict") or {}), "need_more_retrieval": True},
                "answer": "", "retrieved": []}
    # 补充线索:当成新查询重新检索
    return {"query": str(approval), "answer": "", "retrieved": [],
            "verdict": {}, "review_passed": None}
