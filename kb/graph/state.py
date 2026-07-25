"""
State(状态)
===========
对应概念文档 §5.2:把状态从「藏在变量里」变成显式数据结构,每个节点读它、改它。
"""
from typing import Annotated

from typing_extensions import TypedDict
from langgraph.graph.message import add_messages


class State(TypedDict):
    messages: Annotated[list, add_messages]   # 对话历史(短期记忆,reducer=add_messages)
    query: str                                 # 当前用户问题
    retrieved: list[dict]                      # 检索到的片段 [{note_id,type,text,score}]
    answer: str                                # 生成的答案
    verdict: dict                              # 审查四道检查结果
    review_passed: bool | None                 # 审查是否通过(True/False/None=未审)
    next: str                                  # supervisor 决定的路由
