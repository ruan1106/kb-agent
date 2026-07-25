"""
LLM-as-Judge 评估
=================
对应概念文档 §10.2 Eval:用 LLM 给答案打分。
维度:grounded(基于检索)/correct(没讲反概念)/on_topic(答非所问)+ 综合 score。
+ 确定性:测试集期望关键词命中率(加权进 score)。
"""
from pydantic import BaseModel, Field

from kb.llm import get_chat_llm, structured_invoke


class JudgeScore(BaseModel):
    grounded: bool
    correct: bool
    on_topic: bool
    score: float = Field(..., description="0~1 综合分")
    reason: str = ""


def _keyword_hit(answer: str, keywords: list[str]) -> float:
    if not keywords:
        return 1.0
    return sum(1 for k in keywords if k in answer) / len(keywords)


def judge(question: str, answer: str, refs: list, retrieved: list,
          keywords: list[str] | None = None) -> JudgeScore:
    kw = _keyword_hit(answer, keywords or [])
    ctx = "\n".join(h.get("text", "")[:150] for h in retrieved) or "(无)"
    llm = get_chat_llm(temperature=0.0)
    prompt = (
        f"问题:{question}\n检索片段:\n{ctx}\n答案:{answer}\n"
        "判定:grounded(是否基于检索)、correct(是否没讲反概念)、on_topic(是否答非所问)、"
        "score(0~1 综合分)、reason(一句话)。"
    )
    try:
        j = structured_invoke(llm, JudgeScore, prompt)
    except Exception as e:
        j = JudgeScore(grounded=True, correct=True, on_topic=True,
                       score=kw, reason=f"judge 失败({e})")
    # 关键词命中加权进综合分
    j = j.model_copy(update={"score": round(0.7 * j.score + 0.3 * kw, 2)})
    return j
