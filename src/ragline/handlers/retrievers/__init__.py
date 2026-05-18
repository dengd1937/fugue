"""src/ragline/handlers/retrievers/__init__.py — 注册函数 + re-export。"""

from ragline.handlers.retrievers.atoms import (
    RetrieverFn,
    make_bm25_search,
    make_vector_search,
)
from ragline.providers.bm25 import BM25Provider
from ragline.providers.embedding import EmbeddingClient
from ragline.providers.vector_store.base import VectorStore
from ragline.registry import retriever_registry


def register_retrievers(
    vector_store: VectorStore,
    embedding_client: EmbeddingClient,
    bm25_provider: BM25Provider,
) -> None:
    """注册 vector + bm25 retriever 到 retriever_registry。"""
    retriever_registry.register(
        "vector",
        make_vector_search(vector_store, embedding_client),
    )
    retriever_registry.register(
        "bm25",
        make_bm25_search(bm25_provider),
    )


__all__ = [
    "RetrieverFn",
    "make_bm25_search",
    "make_vector_search",
    "register_retrievers",
]
