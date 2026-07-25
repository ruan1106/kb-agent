"""
流式输出
========
对应概念文档 §11.2:让用户实时看到中间过程,而不是转圈等结果。
run_streaming 用 graph.stream(stream_mode="updates") 逐节点产出进展 + 答案。
(HITL 中断流式不处理,走 /chat 的 invoke 路径。)
"""
from langchain_core.messages import HumanMessage


def run_streaming(graph, question: str, config: dict):
    """yield 文本片段:检索进展 / 答案 / 审查结果。"""
    inputs = {"messages": [HumanMessage(question)], "query": question}
    for chunk in graph.stream(inputs, config, stream_mode="updates"):
        if not isinstance(chunk, dict):
            continue
        for node, upd in chunk.items():
            if not isinstance(upd, dict):
                continue
            if node == "researcher" and upd.get("retrieved"):
                yield f"[检索] 找到 {len(upd['retrieved'])} 条相关笔记\n"
            elif node == "writer" and upd.get("answer"):
                yield upd["answer"]
            elif node == "reviewer":
                tag = "通过" if upd.get("review_passed") else "未通过"
                yield f"\n[审查 {tag}]\n"
            elif node == "supervisor" and upd.get("next") == "end":
                yield ""
