"""
存储:SQLite(标准库 sqlite3)+ LangGraph Checkpointer
=====================================================
职责:
    - 业务表:notes(笔记元数据)/ qa_history(问答历史)/ facts(长期事实记忆)
    - Checkpointer:SqliteSaver,把图状态落盘,可中断续跑(对应 persistence_agent.py)

一个 .sqlite 文件,sqlite3 管业务表,SqliteSaver 管 checkpoint 表,互不干扰。
对应概念文档:第 5 节 State & Memory(长期记忆落盘)。
"""
import json
import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime

from langgraph.checkpoint.sqlite import SqliteSaver

from .config import settings

_init_done = False


def _db_file() -> str:
    p = settings.db_path
    d = os.path.dirname(p)
    if d:
        os.makedirs(d, exist_ok=True)
    return p


def _connect():
    return sqlite3.connect(_db_file(), check_same_thread=False)


def init_db():
    """建业务表(幂等)。"""
    global _init_done
    if _init_done:
        return
    conn = _connect()
    conn.executescript("""
    CREATE TABLE IF NOT EXISTS notes(
        note_id TEXT PRIMARY KEY, type TEXT, category TEXT, title TEXT,
        source TEXT, summary TEXT, created_at TEXT, hash TEXT, pitfall TEXT);
    CREATE TABLE IF NOT EXISTS qa_history(
        id INTEGER PRIMARY KEY AUTOINCREMENT, thread_id TEXT,
        question TEXT, answer TEXT, refs TEXT, created_at TEXT);
    CREATE TABLE IF NOT EXISTS facts(
        id INTEGER PRIMARY KEY AUTOINCREMENT, thread_id TEXT,
        key TEXT, value TEXT, created_at TEXT);
    CREATE INDEX IF NOT EXISTS idx_qa_thread ON qa_history(thread_id);
    CREATE INDEX IF NOT EXISTS idx_facts_thread ON facts(thread_id);
    """)
    conn.commit()
    conn.close()
    _init_done = True


@contextmanager
def _conn():
    init_db()
    c = _connect()
    try:
        yield c
        c.commit()
    finally:
        c.close()


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


# ---- Checkpointer ----
@contextmanager
def get_checkpointer():
    """SqliteSaver(沿用 persistence_agent.py 的写法)。with 结束连接关,db 文件留存。"""
    with SqliteSaver.from_conn_string(_db_file()) as saver:
        saver.setup()  # 建 checkpoint 表(幂等)
        yield saver


_persistent_saver = None


def get_persistent_checkpointer():
    """进程级长连接 checkpointer(FastAPI/CLI 复用,check_same_thread=False 跨线程安全)。"""
    global _persistent_saver
    if _persistent_saver is None:
        init_db()
        conn = sqlite3.connect(_db_file(), check_same_thread=False)
        _persistent_saver = SqliteSaver(conn)
        _persistent_saver.setup()
    return _persistent_saver


# ---- 业务 CRUD ----
def save_note(note: dict):
    with _conn() as c:
        c.execute(
            "INSERT OR REPLACE INTO notes(note_id,type,category,title,source,summary,created_at,hash,pitfall) "
            "VALUES(?,?,?,?,?,?,?,?,?)",
            (note["note_id"], note.get("type", ""), note.get("category", ""), note.get("title", ""),
             note.get("source", ""), note.get("summary", ""), note.get("created_at", ""),
             note.get("hash", ""), json.dumps(note.get("pitfall", {}), ensure_ascii=False)),
        )


def list_notes(note_type: str | None = None) -> list[dict]:
    with _conn() as c:
        if note_type:
            rows = c.execute(
                "SELECT note_id,type,category,title,source,summary,created_at,pitfall "
                "FROM notes WHERE type=?", (note_type,)).fetchall()
        else:
            rows = c.execute(
                "SELECT note_id,type,category,title,source,summary,created_at,pitfall "
                "FROM notes").fetchall()
    out = []
    for r in rows:
        try:
            pitfall = json.loads(r[7]) if r[7] else {}
        except Exception:
            pitfall = {}
        out.append({"note_id": r[0], "type": r[1], "category": r[2], "title": r[3],
                    "source": r[4], "summary": r[5], "created_at": r[6], "pitfall": pitfall})
    return out


def note_exists(note_id: str) -> bool:
    """引用存在性校验:回答里引用的 note_id 必须真存在(确定性逻辑,不交给 LLM)。"""
    with _conn() as c:
        return c.execute("SELECT 1 FROM notes WHERE note_id=?", (note_id,)).fetchone() is not None


def save_qa(thread_id: str, question: str, answer: str, refs: list):
    with _conn() as c:
        c.execute(
            "INSERT INTO qa_history(thread_id,question,answer,refs,created_at) VALUES(?,?,?,?,?)",
            (thread_id, question, answer, json.dumps(refs, ensure_ascii=False), _now()),
        )


def list_qa(thread_id: str, limit: int = 10) -> list[dict]:
    with _conn() as c:
        rows = c.execute(
            "SELECT question,answer,refs FROM qa_history WHERE thread_id=? ORDER BY id DESC LIMIT ?",
            (thread_id, limit)).fetchall()
    return [{"question": r[0], "answer": r[1], "refs": json.loads(r[2] or "[]")}
            for r in reversed(rows)]


def save_fact(thread_id: str, key: str, value: str):
    with _conn() as c:
        c.execute("INSERT INTO facts(thread_id,key,value,created_at) VALUES(?,?,?,?)",
                  (thread_id, key, value, _now()))


def list_facts(thread_id: str) -> list[str]:
    with _conn() as c:
        rows = c.execute("SELECT key,value FROM facts WHERE thread_id=?", (thread_id,)).fetchall()
    return [f"{r[0]}: {r[1]}" for r in rows]
