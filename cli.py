"""
CLI 交互入口(流式打印 + HITL)
=============================
对应概念文档 §11:多轮对话、HITL(审查低置信时暂停问人)、长期记忆落盘。

跑起来(在 kb/ 目录):
    python cli.py                       # 交互 REPL
    python cli.py --seed                # 先灌种子语料(仓库现有 .md 概念笔记)
    python cli.py --ingest note.md pitfall   # 导入一个文件
"""
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from langchain_core.messages import HumanMessage
from langgraph.types import Command

from kb import ingest, vectorstore
from kb.graph.qa_graph import default_config, get_graph
from kb.memory import save_turn
from kb.observe import observe


def _interrupt_payload(state):
    for task in state.tasks:
        if task.interrupts:
            return task.interrupts[0].value
    return None


@observe(name="kb_chat")
def chat(graph, question: str, thread_id: str):
    """跑一轮问答,处理 HITL 中断,落长期记忆。返回 (answer, refs)。"""
    config = default_config(thread_id)
    graph.invoke({"messages": [HumanMessage(question)], "query": question}, config)

    while True:
        state = graph.get_state(config)
        if not state.next:
            break
        payload = _interrupt_payload(state)
        if payload is None:
            break
        print(f"\n[需人工确认] {payload['prompt']}")
        print(f"草拟答案:{payload['answer']}")
        print(f"问题点:{payload['issue']}")
        decision = input("> ").strip() or "no"
        graph.invoke(Command(resume=decision), config)

    vals = graph.get_state(config).values
    answer = vals.get("answer", "(无答案)")
    refs = [h.get("note_id") for h in vals.get("retrieved", [])]
    save_turn(thread_id, question, answer, refs)
    return answer, refs


def main():
    import argparse

    ap = argparse.ArgumentParser(description="个人知识库 CLI")
    ap.add_argument("--seed", action="store_true", help="灌种子语料")
    ap.add_argument("--ingest", nargs="+", metavar=("FILE", "TYPE"), help="导入文件 [concept|pitfall]")
    ap.add_argument("--thread", default="default", help="会话 id(隔离记忆)")
    args = ap.parse_args()

    vectorstore.ensure_collection()

    if args.seed:
        ingest.seed()
        return
    if args.ingest:
        path, ntype = args.ingest[0], args.ingest[1] if len(args.ingest) > 1 else "concept"
        ingest.ingest_file(path, ntype)
        return

    graph = get_graph()
    print("知识库 CLI。输入问题开始;/seed 灌种子;/ingest <file> [concept|pitfall];/quit 退出")
    while True:
        try:
            q = input("\n你> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not q:
            continue
        if q == "/quit":
            break
        if q == "/seed":
            ingest.seed()
            continue
        if q.startswith("/ingest"):
            parts = q.split()
            if len(parts) >= 2:
                ingest.ingest_file(parts[1], parts[2] if len(parts) > 2 else "concept")
            else:
                print("用法:/ingest <文件> [concept|pitfall]")
            continue

        print("Agent> ", end="", flush=True)
        answer, refs = chat(graph, q, args.thread)
        print(answer)
        if refs:
            print("引用:", ", ".join(refs))


if __name__ == "__main__":
    main()
