"""src/ragline/providers/vector_store/base.py — VectorStore Protocol。"""

from collections.abc import Iterator
from typing import Any, Protocol, runtime_checkable

from ragline.api.types import Chunk, Document


@runtime_checkable
class VectorStore(Protocol):
    """向量存储抽象。所有 vector store 实现遵循此协议。"""

    def add(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        """upsert chunks + embeddings 到底层存储。"""
        ...

    def similarity_search(
        self,
        query_embedding: list[float],
        k: int = 20,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[Document]:
        """向量检索 top-k，score ∈ [0, 1]。"""
        ...

    def delete_collection(self) -> None:
        """删除整个 collection（用于测试/重置）。"""
        ...

    def stats(self) -> dict[str, Any]:
        """返回统计信息字典，至少含 num_chunks。"""
        ...

    def iter_chunks(self, batch_size: int = 1000) -> Iterator[list[Chunk]]:
        """分批迭代所有 chunks，避免大语料 OOM。"""
        ...

    def close(self) -> None:
        """释放底层资源。"""
        ...
