"""
配置加载
========
集中读 .env,全项目从这里取配置。用 os.getenv(和 study/agent 的 demo 一致)。
对应概念文档:第 4 节 Python 基础(dotenv)。
"""
import os

from dotenv import load_dotenv

# 先试项目内 .env,再试仓库根 .env(那里有 DEEPSEEK_API_KEY)
_HERE = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(_HERE, "..", ".env"))
load_dotenv(os.path.join(_HERE, "..", "..", ".env"))


class Settings:
    # LLM(DeepSeek,OpenAI 兼容)
    deepseek_api_key: str = os.getenv("DEEPSEEK_API_KEY", "")
    kb_model: str = os.getenv("KB_MODEL", "deepseek-v4-flash")        # 工具/结构化节点
    kb_judge_model: str = os.getenv("KB_JUDGE_MODEL", "deepseek-v4-pro")  # writer 纯推理(pro 更强)

    # Embedding / Rerank(阿里通义)
    dashscope_api_key: str = os.getenv("DASHSCOPE_API_KEY", "")
    embed_model: str = os.getenv("EMBED_MODEL", "text-embedding-v3")
    embed_dim: int = int(os.getenv("EMBED_DIM", "1024"))
    rerank_model: str = os.getenv("RERANK_MODEL", "gte-rerank-v2")

    # Qdrant
    qdrant_url: str = os.getenv("QDRANT_URL", "http://localhost:6333")
    qdrant_api_key: str = os.getenv("QDRANT_API_KEY", "")
    kb_collection: str = os.getenv("KB_COLLECTION", "kb_notes")

    # Langfuse(留空 -> 降级本地 trace)
    langfuse_host: str = os.getenv("LANGFUSE_HOST", "")
    langfuse_public_key: str = os.getenv("LANGFUSE_PUBLIC_KEY", "")
    langfuse_secret_key: str = os.getenv("LANGFUSE_SECRET_KEY", "")

    # 路径
    db_path: str = os.getenv("KB_DB_PATH", "data/kb.sqlite")


settings = Settings()
