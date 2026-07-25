# 个人知识库 · kb

把 `agent-development-concepts.md` 的 13 节 Agent 概念，用一个「计算机原理/概念 + 踩坑避坑」个人知识库**全部落地成可跑代码**。这份 README 就是概念↔代码的对照表。

> 原理优先于框架 API。Loop、Tool、State、Context 这四个概念在这里都有对应实现。

---

## 🗺️ 13 节概念 ↔ 代码映射

| 概念文档章节 | 落地文件 | 怎么体现 |
|---|---|---|
| §0-1 定义/骨架 | `kb/graph/qa_graph.py` | 整张图 = LLM + 工具 + 循环 |
| §2.1 ReAct | `kb/graph/nodes.py` `researcher` | 检索 Agent:分解->检索->重排 |
| §2.2 Plan-Execute | `kb/graph/nodes.py` `_decompose` | 查询拆子问题再检索 |
| §2.3 Reflection | `kb/graph/review.py` + `nodes.py` `reviewer` | 生成->审查->改进循环 |
| §3 Function Calling/Structured Output | `kb/tools.py`、`kb/ingest.py`、`kb/review.py` | `@tool`、`with_structured_output` 打标/抽踩坑/审查 |
| §4 Reasoning Loop+停止条件 | `kb/graph/qa_graph.py`、`kb/reliability.py` | 条件边回环 + `recursion_limit`/熔断 |
| §5.1 三层记忆 | `kb/memory.py`、`kb/storage.py`、`kb/tools.py` | 短期(State)/长期(SQLite事实+问答)/外部(RAG) |
| §5.2 State | `kb/graph/state.py` | `TypedDict + add_messages` 显式状态 |
| §5.3 长期记忆落盘 | `kb/storage.py` `get_persistent_checkpointer` | SqliteSaver,重启不丢 |
| §6 Context Window | `kb/context.py` | 滑动窗口 `trim` + 摘要 `maybe_summarize` |
| §7 RAG | `kb/embedding.py`+`vectorstore.py`+`rerank.py`+`ingest.py` | 切块->embed->Qdrant混合检索->重排->生成 |
| §7.4 RAG 作为工具 | `kb/tools.py` `search_*` | 检索是 Agent 调的工具 |
| §8 多 Agent | `kb/graph/nodes.py`+`qa_graph.py` | supervisor 编排 researcher/writer/reviewer |
| §9 MCP | `kb/mcp_server.py` | 检索/导入工具暴露为 MCP server |
| §10.1 Tracing | `kb/observe.py` | Langfuse `@observe`,可降级本地打印 |
| §10.2 Eval | `eval/` | 测试集 + LLM-as-Judge + 通过率 |
| §11.1 工程化/可靠 | `kb/reliability.py` | tenacity 重试/超时/熔断 |
| §11.2 流式 | `kb/streaming.py`、`cli.py`、`api/server.py` | `graph.stream` 逐节点产出 |
| §11.3 HITL | `kb/graph/nodes.py` `ask_human` | `interrupt`+`Command(resume=)` |

---

## 📁 目录结构

```
kb/
├── kb/                    # 包代码
│   ├── config.py          # 配置(§4 dotenv)
│   ├── models.py          # Pydantic 模型(§3/§5)
│   ├── llm.py             # DeepSeek 封装(chat 工具 / judge 推理 + structured_invoke)
│   ├── embedding.py       # 通义 text-embedding-v3(可降级)
│   ├── vectorstore.py     # Qdrant 混合检索(可降级 MiniQdrant)
│   ├── rerank.py          # 手写加权 + gte-rerank
│   ├── storage.py         # SQLite + SqliteSaver(§5)
│   ├── ingest.py          # 导入+自动打标+抽踩坑(§3/§7)
│   ├── tools.py           # @tool 检索/导入(§3/§7.4)
│   ├── memory.py          # 三层记忆(§5)
│   ├── context.py         # 上下文管理(§6)
│   ├── reliability.py     # 重试/熔断(§4/§11)
│   ├── observe.py         # Langfuse trace(§10)
│   ├── streaming.py       # 流式(§11.2)
│   ├── review.py          # 审查四道检查(§2.3/§10)
│   ├── mcp_server.py      # MCP server(§9)
│   └── graph/             # LangGraph 图
│       ├── state.py  nodes.py  qa_graph.py
├── api/server.py          # FastAPI(§11)
├── cli.py                 # CLI 交互(流式+HITL)
├── eval/                  # 评估(§10)
├── tests/                 # pytest(§10)
├── docker-compose.yml     # Qdrant(+可选 Langfuse)
└── requirements.txt
```

---

## 🚀 跑起来

### 1. 装依赖 + 配 key

```bash
cd D:\items\langgraph\kb
pip install -r requirements.txt
cp .env.example .env          # 填 DEEPSEEK_API_KEY、DASHSCOPE_API_KEY
```

> 没配 `DASHSCOPE_API_KEY` -> embedding 降级哈希向量(演示用)。
> 没起 Qdrant -> 降级 numpy 迷你版。没配 Langfuse -> 降级本地 trace 打印。
> 三者皆可降级,先跑通原理,再接真服务。

### 2. (可选)起真服务

```bash
docker compose up -d qdrant    # 向量库(必需)
# Langfuse 建议先用云:在 .env 填 LANGFUSE_HOST/PUBLIC_KEY/SECRET_KEY
```

### 3. 灌种子语料 + 问答

```bash
python cli.py --seed          # 把仓库现有 .md 概念笔记灌进库
python cli.py                 # 进入 REPL 问问题
# > 什么是 ReAct?
# > 三层记忆是哪三层?
# /quit
```

### 4. API 服务

```bash
uvicorn api.server:app --port 8000
# POST /chat {"question":"什么是 RAG?"}
# POST /ingest {"title":"...","content":"...","note_type":"pitfall"}
# GET  /notes
```

### 5. 评估 + 测试

```bash
python -m eval.run_eval       # 跑测试集出通过率
pytest                        # 测重排公式/审查规则/检索
```

### 6. MCP(供 Claude Desktop 等调用)

```bash
python -m kb.mcp_server       # stdio 模式
```

---

## 🔑 模型分工(规避 DeepSeek thinking 模式与强制 tool_choice 冲突)

- `deepseek-v4-flash`(chat):工具调用 / 结构化输出节点(supervisor 路由、打标、审查四道检查)
- `deepseek-v4-pro`(judge):writer 纯推理生成答案,不调工具;失败回退 chat

> **踩坑**:DeepSeek v4 系列默认开 thinking,与强制 `tool_choice`(named/required)冲突,
> 与 `response_format` 的 json_schema 也冲突。`with_structured_output` 默认走这两条会 400。
> 解法:结构化输出统一走 `kb/llm.py::structured_invoke` -> `bind_tools(tool_choice="auto")`,
> 模型自行调工具返回字段(实测稳定命中);`deepseek-reasoner` 在该 API 不可用,judge 用 `pro`。

---

## ⚙️ 选型 & 已处理风险

| 项 | 选型 | 风险处理 |
|---|---|---|
| LLM | DeepSeek | thinking 模式下结构化输出只能 `tool_choice=auto`;judge 用 pro 纯推理 |
| 编排 | LangGraph | Checkpointer 落 SQLite(非 Qdrant) |
| 向量 | Qdrant | `ensure_collection` 幂等,不抹数据;连不上降级 |
| 重排 | gte-rerank API | 不引 torch,全 API |
| 存储 | SQLite | SqliteSaver + 业务表同文件 |
| 可观测 | Langfuse | 可降级本地 trace;云/自托管皆可 |
| 审查 | LLM-judge + 规则 | 引用存在性=确定性规则,pass/fail 不交给 LLM 猜 |

---

## 📝 内容模型

两类笔记(`type` payload 字段):
- `concept`:计算机原理/概念(按 OS/网络/语言/体系结构分类)
- `pitfall`:踩坑避坑(结构化抽取:复现条件/根因/解法/环境/状态)

导入时 LLM 自动打标 + 抽取踩坑字段;哈希去重(确定性)。

---

## 🔗 和 study/、agent/ 的关系

- `study/`:概念原理的手写教学版(20 主题)
- `agent/`:单个基础设施的落地 demo(Qdrant/Redis/MariaDB/...)
- `kb/`(本项目):把概念文档 13 节**串成完整系统**,真服务 + 完整链路
