"""src/ragline/providers/bm25.py — 内存中维护的 BM25 索引。"""

import threading

from rank_bm25 import BM25Okapi

from ragline.api.types import Chunk, Document


def _tokenize(text: str) -> list[str]:
    """MVP 用空白分词（中英文 MVP 接受；中文 P1 加 jieba）。"""
    return text.lower().split()


class BM25Provider:
    """内存中维护的 BM25 索引。

    启动时由 RAG 从 vector store 全量重建（rank_bm25 不支持持久化）。
    rank_bm25 不支持增量，update 触发全量重建。
    """

    def __init__(self) -> None:
        self._chunks: list[Chunk] = []
        self._bm25: BM25Okapi | None = None
        self._lock = threading.RLock()

    def rebuild(self, chunks: list[Chunk]) -> None:
        """全量重建索引。chunks 为空则清空。"""
        with self._lock:
            self._chunks = list(chunks)
            if not self._chunks:
                self._bm25 = None
                return
            tokenized = [_tokenize(c.content) for c in self._chunks]
            self._bm25 = BM25Okapi(tokenized)

    def update(self, new_chunks: list[Chunk]) -> None:
        """增量追加 + 全量重建（rank_bm25 限制）。"""
        with self._lock:
            self.rebuild(self._chunks + list(new_chunks))

    def search(self, query: str, k: int = 20) -> list[Document]:
        """返回 top-k Document，score 是 BM25 原始分数，source='bm25'。"""
        with self._lock:
            if self._bm25 is None or not self._chunks:
                return []
            query_tokens = _tokenize(query)
            scores = self._bm25.get_scores(query_tokens)
            # 索引按分数降序，截取 top-k
            indexed = sorted(
                enumerate(scores),
                key=lambda x: x[1],
                reverse=True,
            )[:k]
            documents: list[Document] = []
            for idx, score in indexed:
                chunk = self._chunks[idx]
                documents.append(
                    Document(
                        doc_id=chunk.chunk_id,
                        content=chunk.content,
                        score=float(score),
                        source="bm25",
                        metadata=dict(chunk.metadata),
                    )
                )
            return documents

    def close(self) -> None:
        """清理索引引用。"""
        with self._lock:
            self._chunks = []
            self._bm25 = None
