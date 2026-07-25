"""
问答主图(supervisor 多 Agent)
==============================
对应概念文档 §1(骨架)、§4.3(有环图)、§8(多 Agent)。
拓扑:
    START -> supervisor --route--> researcher | writer | reviewer | ask_human | END
    researcher/writer/reviewer/ask_human -> supervisor(回环 = 循环)
停止:recursion_limit=MAX_ITERATIONS(§4.2)。
"""
from langgraph.graph import END, START, StateGraph

from ..reliability import MAX_ITERATIONS
from ..storage import get_persistent_checkpointer
from .nodes import ask_human, researcher, reviewer, route, supervisor, writer
from .state import State


def build_graph(checkpointer=None) -> object:
    b = StateGraph(State)
    b.add_node("supervisor", supervisor)
    b.add_node("researcher", researcher)
    b.add_node("writer", writer)
    b.add_node("reviewer", reviewer)
    b.add_node("ask_human", ask_human)

    b.add_edge(START, "supervisor")
    b.add_conditional_edges("supervisor", route,
                            ["researcher", "writer", "reviewer", "ask_human", END])
    b.add_edge("researcher", "supervisor")
    b.add_edge("writer", "supervisor")
    b.add_edge("reviewer", "supervisor")
    b.add_edge("ask_human", "supervisor")

    return b.compile(checkpointer=checkpointer)


def get_graph():
    """带 SQLite checkpointer 的图(状态落盘,可中断续跑)。进程级单例连接。"""
    return build_graph(get_persistent_checkpointer())


def default_config(thread_id: str = "default") -> dict:
    return {"configurable": {"thread_id": thread_id},
            "recursion_limit": MAX_ITERATIONS}
