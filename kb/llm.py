"""
LLM 封装(DeepSeek,OpenAI 兼容)
================================
模型分工:
    - chat 模型(deepseek-v4-flash):工具调用 / 结构化输出节点(researcher 拆解、
      审查判定、自动打标/抽踩坑)。flash 默认开 thinking,结构化输出走 bind_tools。
    - judge 模型(deepseek-v4-pro):writer 的纯推理生成(出答案,不调工具)。
      pro 推理更强,适合基于片段作答。

踩坑:DeepSeek v4 系列默认开 thinking 模式,与强制 tool_choice(named/required)
冲突(报 "Thinking mode does not support this tool_choice");与 response_format
的 json_schema 也冲突。只有 tool_choice="auto" 可用。故结构化输出统一走
structured_invoke -> bind_tools(tool_choice=auto) + 模型自行调工具返回字段。

对应概念文档:第 1 节、第 3 节 Function Calling。
"""
import json
import re

from langchain_openai import ChatOpenAI

from .config import settings


def get_chat_llm(temperature: float = 0.3) -> ChatOpenAI:
    """工具/结构化节点用的对话模型。"""
    return ChatOpenAI(
        model=settings.kb_model,
        api_key=settings.deepseek_api_key,
        base_url="https://api.deepseek.com/v1",
        temperature=temperature,
    )


def get_judge_llm() -> ChatOpenAI:
    """writer 纯推理用的模型:出答案,不调工具。"""
    return ChatOpenAI(
        model=settings.kb_judge_model,
        api_key=settings.deepseek_api_key,
        base_url="https://api.deepseek.com/v1",
        temperature=0.0,
    )


def structured_invoke(llm, schema, prompt):
    """统一结构化输出入口(规避 DeepSeek thinking 模式与强制 tool_choice 的冲突)。

    用 bind_tools([schema], tool_choice="auto"):模型自行决定调工具,返回符合
    schema 的字段。命中 tool_calls 则 model_validate 还原成 Pydantic 对象;
    没命中则从文本里抠 JSON 兜底;都失败抛异常,由调用方 try/except 降级。
    """
    resp = llm.bind_tools([schema], tool_choice="auto").invoke(prompt)
    if resp.tool_calls:
        return schema.model_validate(resp.tool_calls[0]["args"])
    m = re.search(r"\{.*\}", resp.content or "", re.S)
    if m:
        return schema.model_validate(json.loads(m.group()))
    raise RuntimeError("模型未返回结构化结果")
