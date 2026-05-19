"""src/ragline/handlers/processors/__init__.py — 注册函数 + re-export。"""

from ragline.handlers.processors.rerank import ProcessorFn, make_rerank
from ragline.handlers.processors.rrf import rrf_fn
from ragline.providers.reranker.base import Reranker
from ragline.registry import processor_registry


def register_processors(reranker: Reranker) -> None:
    """注册 rrf + rerank 到 processor_registry。

    rrf 是无状态函数；rerank 需 reranker 注入（闭包绑定）。
    """
    processor_registry.register("rrf", rrf_fn)
    processor_registry.register("rerank", make_rerank(reranker))


__all__ = [
    "ProcessorFn",
    "make_rerank",
    "register_processors",
    "rrf_fn",
]
