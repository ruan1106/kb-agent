"""
跑评估集
========
对应概念文档 §10.2:固定测试集 + Judge + 通过率,让质量可量化。
跑起来(在 kb/ 目录):
    python -m eval.run_eval
"""
import json
import os
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from langchain_core.messages import HumanMessage
from langgraph.types import Command

from kb import ingest, storage, vectorstore
from kb.eval.judge import judge as do_judge
from kb.graph.qa_graph import default_config, get_graph


def _interrupt(state):
    for t in state.tasks:
        if t.interrupts:
            return t.interrupts[0].value
    return None


def run_one(graph, q: str, thread_id: str):
    config = default_config(thread_id)
    graph.invoke({"messages": [HumanMessage(q)], "query": q}, config)
    # eval 自动接受 HITL(不阻塞)
    while True:
        state = graph.get_state(config)
        if not state.next or _interrupt(state) is None:
            break
        graph.invoke(Command(resume="yes"), config)
    vals = graph.get_state(config).values
    return (vals.get("answer", ""),
            [h.get("note_id") for h in vals.get("retrieved", [])],
            vals.get("retrieved", []))


def main():
    vectorstore.ensure_collection()
    if not storage.list_notes():
        print("[Eval] 知识库为空,先灌种子语料")
        ingest.seed()

    graph = get_graph()
    path = os.path.join(os.path.dirname(__file__), "testset.jsonl")
    cases = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]

    results = []
    for i, c in enumerate(cases):
        q, kw = c["q"], c.get("keywords", [])
        ans, refs, retrieved = run_one(graph, q, f"eval-{i}")
        j = do_judge(q, ans, refs, retrieved, keywords=kw)
        results.append((q, j))
        print(f"[{i+1}/{len(cases)}] score={j.score:.2f} grounded={j.grounded} "
              f"correct={j.correct} on_topic={j.on_topic} | {j.reason}")

    passed = sum(1 for _, j in results if j.score >= 0.6)
    print(f"\n通过率 {passed}/{len(cases)} = {passed/len(cases):.0%}")


if __name__ == "__main__":
    main()
