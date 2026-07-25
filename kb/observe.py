"""
可观测:Langfuse trace
======================
可插拔(沿用 agent/langfuse_agent.py 风格):
    - 配了 LANGFUSE_HOST/KEY -> 接真 Langfuse,trace 上传可视化
    - 否则降级到本地 trace 打印(缩进嵌套树)

LangChain/LangGraph 配了 Langfuse 会自动 trace LLM 调用;@observe 给自定义函数加 span。
对应概念文档:第 10.1 节 Tracing。
"""
import contextvars
import os
import time

from .config import settings

_depth = contextvars.ContextVar("depth", default=0)
_indent = "  "


def _have_langfuse() -> bool:
    if not (settings.langfuse_host and settings.langfuse_public_key and settings.langfuse_secret_key):
        return False
    try:
        import langfuse  # noqa: F401
        return True
    except Exception:
        return False


def _local_observe(name=None):
    """降级版 @observe:打印 span 进出,缩进表示嵌套。"""
    def deco(fn):
        def wrapper(*args, **kwargs):
            d = _depth.get()
            print(f"{_indent * d}▶ {name or fn.__name__}  (span)")
            _depth.set(d + 1)
            t0 = time.time()
            try:
                return fn(*args, **kwargs)
            finally:
                _depth.set(d)
                print(f"{_indent * d}◀ {name or fn.__name__}  done {time.time()-t0:.2f}s")
        return wrapper
    return deco


if _have_langfuse():
    os.environ["LANGFUSE_HOST"] = settings.langfuse_host
    os.environ["LANGFUSE_PUBLIC_KEY"] = settings.langfuse_public_key
    os.environ["LANGFUSE_SECRET_KEY"] = settings.langfuse_secret_key
    from langfuse import observe as _lf_observe

    def observe(name=None):
        return _lf_observe(name=name) if name else _lf_observe()

    print("[Langfuse] 接真服务,trace 上传到", settings.langfuse_host)
else:
    observe = _local_observe
    print("[Langfuse] 未配置,降级本地 trace 打印")
