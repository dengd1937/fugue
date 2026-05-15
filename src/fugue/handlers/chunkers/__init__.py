"""src/fugue/handlers/chunkers/__init__.py — 注册 + re-export。"""

from fugue.handlers.chunkers.recursive import recursive_chunker
from fugue.registry import chunker_registry


def register_chunkers() -> None:
    """注册 recursive 到 chunker_registry。无依赖。"""
    chunker_registry.register("recursive", recursive_chunker)


# 模块 import 即注册（无依赖）
register_chunkers()


__all__ = ["recursive_chunker", "register_chunkers"]
