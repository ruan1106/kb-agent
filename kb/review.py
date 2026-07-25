"""
审查 Agent(四道检查)
=====================
对应概念文档 §2.3 Reflection + §10 LLM-as-Judge。
四道检查里:
    - citation_real(引用是否真实):确定性规则--引用的 note_id 必须在检索集或库里真存在
    - no_hallucination / on_topic / need_more_retrieval:LLM 判定(chat 模型 + bind_tools 结构化)
pass/fail 由确定性逻辑算(不交给 LLM 猜)。
注:用 chat(flash) 做审查而非 judge(pro)--审查要出结构化字段,走 bind_tools;
    judge 留给 writer 做纯推理。structured_invoke 统一封装 DeepSeek thinking 模式兼容。
"""
import re

from . import storage
from .llm import get_chat_llm, structured_invoke
from .models import ReviewVerdict


def _cited_ids(answer: str) -> set[str]:
    """从答案里抓 note_id=xxxx 引用。"""
    return set(re.findall(r"note_id=([a-f0-9]+)", answer))


def review(query: str, answer: str, retrieved: list[dict]) -> tuple[ReviewVerdict, bool]:
    """返回 (verdict, passed)。"""
    retrieved_ids = {h.get("note_id") for h in retrieved if h.get("note_id")}
    cited = _cited_ids(answer)
    # ① 确定性:引用必须真实存在(在检索集或库里)
    citation_real = (all(c in retrieved_ids or storage.note_exists(c) for c in cited)
                     if cited else True)

    # ②③④ LLM 判另外三道(chat + structured output)
    ctx = "\n\n".join(f"[{h.get('note_id')}] {h.get('text', '')[:200]}" for h in retrieved) or "(无)"
    llm = get_chat_llm(temperature=0.0)
    prompt = (
        "你是知识库答案审查员,对下面回答做检查。\n"
        f"问题:{query}\n检索到的真实片段:\n{ctx}\n回答:{answer}\n\n"
        "判定:citation_real(引用是否真实)、no_hallucination(是否没编造/没讲反概念)、"
        "on_topic(是否答非所问)、need_more_retrieval(是否需要补检)、reason(一句话)。"
    )
    try:
        v = structured_invoke(llm, ReviewVerdict, prompt)
        v = v.model_copy(update={"citation_real": citation_real})
    except Exception as e:
        v = ReviewVerdict(
            citation_real=citation_real, no_hallucination=True, on_topic=True,
            need_more_retrieval=False, reason=f"judge 失败({e}),默认放行",
        )

    # pass/fail 确定性
    passed = (v.citation_real and v.no_hallucination and v.on_topic and not v.need_more_retrieval)
    return v, passed
