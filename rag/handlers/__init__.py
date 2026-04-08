"""rag/handlers — 统一触发所有 handler 模块的注册。"""

from rag.handlers import generators, graders, intent_router, processors, retrievers, transforms

__all__ = [
    "transforms",
    "intent_router",
    "retrievers",
    "processors",
    "graders",
    "generators",
]
