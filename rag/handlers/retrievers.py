"""rag/handlers/retrievers.py — 检索器 handler 实现。"""

from __future__ import annotations

from typing import Any, Protocol

from rag.registry import retriever_registry
from rag.types import Document

# ---------------------------------------------------------------------------
# RetrieverClientProtocol
# ---------------------------------------------------------------------------


class RetrieverClientProtocol(Protocol):
    """检索客户端协议（鸭子类型，供类型检查用）。"""

    def search(self, query: str, filter: dict[str, Any] | None, k: int) -> list[dict[str, Any]]: ...


# ---------------------------------------------------------------------------
# Lazy default client（Phase 2 为 stub，Phase 6 接真实客户端）
# ---------------------------------------------------------------------------


class _StubRetrieverClient:
    """开发阶段 stub 客户端，返回空列表。"""

    def search(self, query: str, filter: dict[str, Any] | None, k: int) -> list[dict[str, Any]]:
        return []


class _LazyDefaultRetrieverClient:
    """延迟构造的默认 retriever client。"""

    def __init__(self) -> None:
        self._client: _StubRetrieverClient | None = None

    def _get_client(self) -> _StubRetrieverClient:
        if self._client is None:
            self._client = _StubRetrieverClient()
        return self._client

    def search(self, query: str, filter: dict[str, Any] | None, k: int) -> list[dict[str, Any]]:
        return self._get_client().search(query, filter, k)


_default_client: _LazyDefaultRetrieverClient = _LazyDefaultRetrieverClient()

# ---------------------------------------------------------------------------
# 内部辅助
# ---------------------------------------------------------------------------

_DEFAULT_K = 10


def _raw_to_document(raw: dict[str, Any], source: str) -> Document:
    """将 client.search() 返回的原始 dict 转为 Document，强制写入 source 字段。"""
    return Document(
        doc_id=str(raw.get("doc_id", "")),
        content=str(raw.get("content", "")),
        score=float(raw.get("score", 0.0)),
        source=source,  # 防御性强制写入
        metadata=dict(raw.get("metadata", {})),
    )


def _retrieve(
    source_name: str,
    query: str,
    metadata_filter: dict[str, Any] | None,
    client: Any,
    k: int = _DEFAULT_K,
) -> list[Document]:
    """通用检索逻辑。"""
    raw_docs = client.search(query, metadata_filter, k)
    return [_raw_to_document(raw, source_name) for raw in raw_docs]


# ---------------------------------------------------------------------------
# 各 Retriever 函数
# ---------------------------------------------------------------------------


def vector_search_fn(
    query: str,
    metadata_filter: dict[str, Any] | None = None,
    client: Any = _default_client,
) -> list[Document]:
    """向量语义检索。"""
    return _retrieve("vector", query, metadata_filter, client)


def es_search_fn(
    query: str,
    metadata_filter: dict[str, Any] | None = None,
    client: Any = _default_client,
) -> list[Document]:
    """Elasticsearch 全文检索。"""
    return _retrieve("es", query, metadata_filter, client)


def kg_search_fn(
    query: str,
    metadata_filter: dict[str, Any] | None = None,
    client: Any = _default_client,
) -> list[Document]:
    """知识图谱检索。"""
    return _retrieve("kg", query, metadata_filter, client)


def web_search_fn(
    query: str,
    metadata_filter: dict[str, Any] | None = None,
    client: Any = _default_client,
) -> list[Document]:
    """Web 搜索检索。"""
    return _retrieve("web", query, metadata_filter, client)


def sql_search_fn(
    query: str,
    metadata_filter: dict[str, Any] | None = None,
    client: Any = _default_client,
) -> list[Document]:
    """SQL 数据库检索。"""
    return _retrieve("sql", query, metadata_filter, client)


def vector_parent_child_fn(
    query: str,
    metadata_filter: dict[str, Any] | None = None,
    client: Any = _default_client,
) -> list[Document]:
    """向量父子节点检索。"""
    return _retrieve("vector_parent_child", query, metadata_filter, client)


def vector_sentence_window_fn(
    query: str,
    metadata_filter: dict[str, Any] | None = None,
    client: Any = _default_client,
) -> list[Document]:
    """向量句子窗口检索。"""
    return _retrieve("vector_sentence_window", query, metadata_filter, client)


# ---------------------------------------------------------------------------
# 注册
# ---------------------------------------------------------------------------

retriever_registry.register("vector", vector_search_fn)
retriever_registry.register("es", es_search_fn)
retriever_registry.register("kg", kg_search_fn)
retriever_registry.register("web", web_search_fn)
retriever_registry.register("sql", sql_search_fn)
retriever_registry.register("vector_parent_child", vector_parent_child_fn)
retriever_registry.register("vector_sentence_window", vector_sentence_window_fn)
