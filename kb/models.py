"""
Pydantic 数据模型
=================
State 在 graph/state.py;这里放业务数据模型 + Qdrant payload + 结构化输出模型。
对应概念文档:第 3 节 Structured Output、第 5 节 State(建模)。
"""
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field


class NoteType(str, Enum):
    concept = "concept"   # 计算机原理/概念
    pitfall = "pitfall"   # 踩坑避坑


# ---- Qdrant payload:每个 chunk 的元数据(对应清单 payload schema + 索引)----
# use_enum_values:把 type 存成纯字符串,避免 NoteType enum 混进 LangGraph State ->
# checkpoint 反序列化告警(将来会变阻断);Qdrant payload 也只认字符串。
class ChunkPayload(BaseModel):
    model_config = ConfigDict(use_enum_values=True)

    note_id: str
    type: NoteType
    category: str = ""                       # OS/网络/语言/体系结构...(keyword 索引)
    title: str = ""
    tags: list[str] = Field(default_factory=list)
    source: str = ""                         # 来源文件
    chunk_index: int = 0
    created_at: str = ""                     # ISO 日期(datetime 索引)
    text: str = ""                           # 原文,便于直接展示


# ---- 踩坑结构化字段(ingest 时 structured output 抽取)----
class PitfallFields(BaseModel):
    """踩坑笔记的结构化抽取。把"经验"变成可检索的结构化字段。"""
    reproduce: str = Field("", description="复现条件/步骤")
    root_cause: str = Field("", description="根因")
    fix: str = Field("", description="解法")
    environment: str = Field("", description="环境/版本")
    status: str = Field("open", description="open/resolved/wontfix")


# ---- 自动打标输出(ingest 时 structured output)----
class NoteTags(BaseModel):
    category: str = Field("", description="领域分类,如 OS/网络/语言/体系结构")
    tags: list[str] = Field(default_factory=list, description="关键词标签")
    summary: str = Field("", description="一句话摘要")


# ---- 审查 Agent 四道检查(§2.3 Reflection + §10 LLM-judge)----
# 四个判定交给 LLM;pass/fail 由 review.py 用确定性逻辑算(不交给 LLM 猜)
class ReviewVerdict(BaseModel):
    citation_real: bool = Field(..., description="引用片段是否真实存在")
    no_hallucination: bool = Field(..., description="是否没讲反概念/没编造")
    on_topic: bool = Field(..., description="是否答非所问")
    need_more_retrieval: bool = Field(..., description="是否需要补检")
    reason: str = ""
