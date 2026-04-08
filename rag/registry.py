"""rag/registry.py — 通用插件注册表及全局实例。"""

from collections.abc import Callable
from typing import Any


class Registry:
    """通用插件注册表，支持按名称注册和链式执行。"""

    def __init__(self) -> None:
        self._handlers: dict[str, Callable[..., Any]] = {}

    def register(self, name: str, fn: Callable[..., Any]) -> None:
        """注册一个处理函数，已存在时覆盖。"""
        self._handlers[name] = fn

    def get(self, name: str) -> Callable[..., Any]:
        """获取已注册函数，不存在时抛 KeyError 并列出可用名称。"""
        if name not in self._handlers:
            raise KeyError(
                f"Handler '{name}' not registered. Available: {list(self._handlers.keys())}"
            )
        return self._handlers[name]

    def has(self, name: str) -> bool:
        """检查名称是否已注册。"""
        return name in self._handlers

    def names(self) -> list[str]:
        """返回所有已注册名称列表。"""
        return list(self._handlers.keys())

    def run_pipeline(self, names: list[str], initial: Any, **kwargs: Any) -> Any:
        """链式执行：前一个的输出作为后一个的输入。

        Args:
            names: 按执行顺序排列的处理函数名称列表。
            initial: 流水线初始输入值。
            **kwargs: 传递给每个处理函数的关键字参数。

        Returns:
            最后一个处理函数的输出。
        """
        result: Any = initial
        for name in names:
            result = self.get(name)(result, **kwargs)
        return result


# ---------------------------------------------------------------------------
# 全局注册表实例（各自独立）
# ---------------------------------------------------------------------------

transform_registry: Registry = Registry()
retriever_registry: Registry = Registry()
processor_registry: Registry = Registry()
grader_registry: Registry = Registry()
generator_registry: Registry = Registry()
