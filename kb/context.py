"""
上下文窗口管理
==============
messages 越堆越长 -> 超窗口/注意力稀释/成本涨(§6.1)。三策略:
    - trim:滑动窗口,保留 system + 最近 N 条(§6.2 策略2)
    - maybe_summarize:旧消息压成摘要(§6.2 策略3)

对应概念文档:第 6 节 Context Window。
"""
from langchain_core.messages import SystemMessage


def trim(messages: list, keep_recent: int = 6) -> list:
    """滑动窗口:保留 system 消息 + 最近 keep_recent 条非 system 消息。"""
    sys = [m for m in messages if isinstance(m, SystemMessage)]
    rest = [m for m in messages if not isinstance(m, SystemMessage)]
    return sys + rest[-keep_recent:]


def maybe_summarize(messages: list, llm, keep_recent: int = 6) -> list:
    """旧消息摘要压缩:超过 keep_recent 的旧消息压成一条摘要放前面。失败回退 trim。"""
    rest = [m for m in messages if not isinstance(m, SystemMessage)]
    if len(rest) <= keep_recent:
        return messages
    sys = [m for m in messages if isinstance(m, SystemMessage)]
    old, recent = rest[:-keep_recent], rest[-keep_recent:]
    try:
        text = "\n".join(getattr(m, "content", "") for m in old)
        summ = llm.invoke([
            SystemMessage("把以下对话压缩成简短摘要,只保留关键事实。"),
            SystemMessage(text),
        ])
        return sys + [SystemMessage(f"【历史摘要】{summ.content}")] + recent
    except Exception:
        return trim(messages, keep_recent)
