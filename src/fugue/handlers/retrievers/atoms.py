"""src/fugue/handlers/retrievers/atoms.py — Retriever 工厂函数。"""

from collections.abc import Callable
from typing import Any

from fugue.api.types import Document
from fugue.providers.bm25 import BM25Provider
from fugue.providers.embedding import EmbeddingClient
from fugue.providers.vector_store.base import VectorStore

RetrieverFn = Callable[[str, dict[str, Any] | None], list[Document]]


def make_vector_search(
    vector_store: VectorStore,
    embedding_client: EmbeddingClient,
) -> RetrieverFn:
    """返回 retriever 闭包：query + metadata_filter → list[Document]。

    Document.source 设置为 "vector"。
    """

    def vector_search(
        query: str,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[Document]:
        emb = embedding_client.embed([query])[0]
        docs = vector_store.similarity_search(
            emb,
            k=20,
            metadata_filter=metadata_filter,
        )
        return [{**d, "source": "vector"} for d in docs]

    return vector_search


def make_bm25_search(bm25_provider: BM25Provider) -> RetrieverFn:
    """返回 BM25 retriever 闭包。

    BM25 MVP 不支持 metadata_filter，参数被忽略。Document.source 设置为 "bm25"。
    """

    def bm25_search(
        query: str,
        metadata_filter: dict[str, Any] | None = None,  # noqa: ARG001 - MVP 不支持
    ) -> list[Document]:
        docs = bm25_provider.search(query, k=20)
        return [{**d, "source": "bm25"} for d in docs]

    return bm25_search
