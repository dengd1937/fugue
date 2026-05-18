"""src/ragline/registry/__init__.py — 7 个全局 Registry 单例与插件发现。"""

import logging
from collections.abc import Callable
from importlib.metadata import entry_points
from typing import Any

from ragline.api.types import RaglineRegistryError

logger = logging.getLogger(__name__)


class Registry[H: Callable[..., Any]]:
    def __init__(self, name: str) -> None:
        self._name = name
        self._items: dict[str, H] = {}

    def register(self, name: str, fn: H) -> None:
        """注册一个 handler。名字已存在时覆盖并 warning（不抛错）。"""
        if name in self._items:
            logger.warning(
                "Handler '%s' already registered in %s registry; overwriting",
                name,
                self._name,
            )
        self._items[name] = fn

    def unregister(self, name: str) -> None:
        self._items.pop(name, None)

    def get(self, name: str) -> H:
        if name not in self._items:
            available = sorted(self._items.keys())
            raise RaglineRegistryError(f"Handler '{name}' not registered in {self._name}. Available: {available}")
        return self._items[name]

    def has(self, name: str) -> bool:
        return name in self._items

    def names(self) -> list[str]:
        return sorted(self._items.keys())

    def __call__(self, name: str) -> Callable[[H], H]:
        """装饰器糖。@registry("foo") 装饰后返回原函数。"""

        def decorator(fn: H) -> H:
            self.register(name, fn)
            return fn

        return decorator


# 7 个全局单例
transform_registry: Registry[Callable[..., Any]] = Registry("transform")
retriever_registry: Registry[Callable[..., Any]] = Registry("retriever")
processor_registry: Registry[Callable[..., Any]] = Registry("processor")
grader_registry: Registry[Callable[..., Any]] = Registry("grader")
generator_registry: Registry[Callable[..., Any]] = Registry("generator")
parser_registry: Registry[Callable[..., Any]] = Registry("parser")
chunker_registry: Registry[Callable[..., Any]] = Registry("chunker")


def discover_plugins() -> None:
    """扫描 entry_points group='ragline.handlers' 触发第三方插件注册。
    单个插件 raise 时不应阻塞其他插件加载（warning 日志）。
    """
    for ep in entry_points(group="ragline.handlers"):
        try:
            register_fn = ep.load()
            register_fn()
        except Exception as exc:
            logger.warning(
                "Failed to load entry point %s: %s",
                ep.name,
                exc,
            )
