"""
FastAPI 服务
============
对应概念文档 §11 工程化:把 Agent 服务化。
端点:
    POST /chat         问答(遇审查低置信返回 need_approval,需 /resume)
    POST /resume       HITL 恢复(yes/no/补充线索)
    POST /chat/stream  流式(SSE,快乐路径)
    POST /ingest       沉淀新笔记
    GET  /notes        列笔记

跑起来(在 kb/ 目录):
    uvicorn api.server:app --reload --port 8000
"""
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from langchain_core.messages import HumanMessage
from langgraph.types import Command
from pydantic import BaseModel

from kb import ingest, storage
from kb.graph.qa_graph import default_config, get_graph
from kb.memory import save_turn
from kb.streaming import run_streaming

app = FastAPI(title="个人知识库 API")


class ChatIn(BaseModel):
    question: str
    thread_id: str = "default"


class ResumeIn(BaseModel):
    thread_id: str
    decision: str  # yes / no / 补充线索


class IngestIn(BaseModel):
    title: str
    content: str
    note_type: str = "concept"


def _check_interrupt(graph, config):
    """有中断返回 interrupt payload,否则返回 None。"""
    state = graph.get_state(config)
    for task in state.tasks:
        if task.interrupts:
            return task.interrupts[0].value
    return None


@app.post("/chat")
def chat(body: ChatIn):
    graph = get_graph()
    config = default_config(body.thread_id)
    graph.invoke({"messages": [HumanMessage(body.question)], "query": body.question}, config)

    it = _check_interrupt(graph, config)
    if it:
        return {"status": "need_approval", "interrupt": it}

    vals = graph.get_state(config).values
    answer = vals.get("answer", "")
    refs = [h.get("note_id") for h in vals.get("retrieved", [])]
    save_turn(body.thread_id, body.question, answer, refs)
    return {"status": "done", "answer": answer, "refs": refs}


@app.post("/resume")
def resume(body: ResumeIn):
    graph = get_graph()
    config = default_config(body.thread_id)
    graph.invoke(Command(resume=body.decision), config)

    it = _check_interrupt(graph, config)
    if it:
        return {"status": "need_approval", "interrupt": it}

    vals = graph.get_state(config).values
    return {"status": "done", "answer": vals.get("answer", "")}


@app.post("/chat/stream")
def chat_stream(body: ChatIn):
    graph = get_graph()
    config = default_config(body.thread_id)

    def gen():
        for chunk in run_streaming(graph, body.question, config):
            yield f"data: {chunk}\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream")


@app.post("/ingest")
def do_ingest(body: IngestIn):
    nid = ingest.ingest_text(body.title, body.content, body.note_type)
    return {"note_id": nid}


@app.get("/notes")
def notes(note_type: str | None = None):
    return storage.list_notes(note_type)
