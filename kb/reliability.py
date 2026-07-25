"""
可靠性:重试 / 超时 / 熔断 / 最大步数
=====================================
Agent 循环必须有停止条件,否则是定时炸弹(§4.2)。
    - llm_retry:tenacity 指数退避重试(LLM/工具抖动)
    - MAX_ITERATIONS:图的 recursion_limit 强制熔断
    - CircuitBreaker:连续失败 N 次打开熔断

对应概念文档:第 4.2 节 停止条件、第 11.1 节 工程化。
"""
from tenacity import retry, stop_after_attempt, wait_exponential

# 图的 recursion_limit:超过强制停(§4.2 最大步数熔断)
MAX_ITERATIONS = 16

# LLM 调用重试:指数退避,最多 3 次,失败抛出
llm_retry = retry(
    reraise=True,
    stop=stop_after_attempt(3),
    wait=wait_exponential(min=1, max=4),
)


class CircuitBreaker:
    """连续失败 N 次打开熔断,阻止后续调用(§11.1 错误熔断)。"""

    def __init__(self, threshold: int = 3):
        self.threshold = threshold
        self.failures = 0
        self.open = False

    def record_success(self):
        self.failures = 0
        self.open = False

    def record_failure(self):
        self.failures += 1
        if self.failures >= self.threshold:
            self.open = True

    def allow(self) -> bool:
        return not self.open


breaker = CircuitBreaker()
