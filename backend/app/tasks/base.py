"""Task 抽象基类 — 定义统一的任务执行接口。"""

from abc import ABC, abstractmethod
from typing import Generator


class BaseTask(ABC):
    """所有 Task 的基类。"""

    name: str = ""
    required_tools: list[str] = []

    @abstractmethod
    def run(self, question: str, **kwargs) -> dict:
        """同步执行任务，返回结果字典。"""
        ...

    @abstractmethod
    def run_stream(self, question: str, **kwargs) -> Generator[dict, None, None]:
        """流式执行任务，yield SSE 事件。"""
        ...
