"""tests/unit/test_handlers/test_retrievers.py — Retriever 工厂函数单元测试。"""

from unittest.mock import MagicMock

import pytest

from ragline.api.types import Document

# ===== Fixtures =====


@pytest.fixture
def mock_vector_store():
    """返回一个模拟 VectorStore 的 MagicMock。"""
    return MagicMock()


@pytest.fixture
def mock_embedding():
    """返回一个模拟 EmbeddingClient 的 MagicMock。"""
    return MagicMock()


@pytest.fixture
def mock_bm25():
    """返回一个模拟 BM25Provider 的 MagicMock。"""
    return MagicMock()


@pytest.fixture
def clean_retriever_registry():
    """清空再 yield，结束时再清空（防止污染其他测试）。"""
    from ragline.registry import retriever_registry

    saved = {n: retriever_registry.get(n) for n in retriever_registry.names()}
    for n in list(retriever_registry.names()):
        retriever_registry.unregister(n)
    yield retriever_registry
    for n in list(retriever_registry.names()):
        retriever_registry.unregister(n)
    for n, fn in saved.items():
        retriever_registry.register(n, fn)


# ===== 测试 1: vector_search 基础 =====


def test_vector_search_basic(mock_vector_store, mock_embedding):
    from ragline.handlers.retrievers.atoms import make_vector_search

    mock_embedding.embed.return_value = [[0.1, 0.2, 0.3]]
    mock_vector_store.similarity_search.return_value = [
        Document(doc_id="d1", content="c1", score=0.9, source="vector", metadata={}),
        Document(doc_id="d2", content="c2", score=0.8, source="vector", metadata={}),
    ]

    retriever = make_vector_search(mock_vector_store, mock_embedding)
    result = retriever("query")

    mock_embedding.embed.assert_called_once_with(["query"])
    mock_vector_store.similarity_search.assert_called_once_with(
        [0.1, 0.2, 0.3],
        k=20,
        metadata_filter=None,
    )
    assert len(result) == 2
    for doc in result:
        assert doc["source"] == "vector"


# ===== 测试 2: bm25_search 基础 =====


def test_bm25_search_basic(mock_bm25):
    from ragline.handlers.retrievers.atoms import make_bm25_search

    mock_bm25.search.return_value = [
        Document(doc_id="b1", content="b", score=2.5, source="bm25", metadata={}),
    ]

    retriever = make_bm25_search(mock_bm25)
    result = retriever("query")

    mock_bm25.search.assert_called_once_with("query", k=20)
    assert result[0]["source"] == "bm25"


# ===== 测试 3: vector_search metadata_filter 透传 =====


def test_vector_search_metadata_filter_passed_through(mock_vector_store, mock_embedding):
    from ragline.handlers.retrievers.atoms import make_vector_search

    mock_embedding.embed.return_value = [[0.5, 0.6, 0.7]]
    mock_vector_store.similarity_search.return_value = []

    retriever = make_vector_search(mock_vector_store, mock_embedding)
    retriever("q", metadata_filter={"year": 2024})

    mock_vector_store.similarity_search.assert_called_once_with(
        [0.5, 0.6, 0.7],
        k=20,
        metadata_filter={"year": 2024},
    )


# ===== 测试 4: bm25_search 忽略 metadata_filter（不报错）=====


def test_bm25_search_ignores_metadata_filter(mock_bm25):
    from ragline.handlers.retrievers.atoms import make_bm25_search

    mock_bm25.search.return_value = []

    retriever = make_bm25_search(mock_bm25)
    # 不抛错
    retriever("q", metadata_filter={"year": 2024})

    # BM25 search 被调用时不带 metadata_filter
    mock_bm25.search.assert_called_once_with("q", k=20)


# ===== 测试 5: register_retrievers 注册 =====


def test_register_retrievers_registers_all(mock_vector_store, mock_embedding, mock_bm25, clean_retriever_registry):
    from ragline.handlers.retrievers import register_retrievers

    register_retrievers(mock_vector_store, mock_embedding, mock_bm25)

    assert clean_retriever_registry.has("vector")
    assert clean_retriever_registry.has("bm25")


# ===== 测试 6: source 字段重写（不原地修改）=====


def test_vector_search_does_not_modify_original_documents(mock_vector_store, mock_embedding):
    from ragline.handlers.retrievers.atoms import make_vector_search

    original_doc = Document(doc_id="d1", content="c1", score=0.9, source="original_source", metadata={})
    mock_embedding.embed.return_value = [[0.1, 0.2, 0.3]]
    mock_vector_store.similarity_search.return_value = [original_doc]

    retriever = make_vector_search(mock_vector_store, mock_embedding)
    result = retriever("query")

    # 原始文档未被修改
    assert original_doc["source"] == "original_source"
    # 结果中 source 已被覆盖为 "vector"
    assert result[0]["source"] == "vector"
    # 结果是新字典，不是同一对象
    assert result[0] is not original_doc


# ===== 测试 7: source 覆盖（vector store 返回非 "vector" source 时）=====


def test_vector_search_overrides_source_field(mock_vector_store, mock_embedding):
    from ragline.handlers.retrievers.atoms import make_vector_search

    mock_embedding.embed.return_value = [[0.1, 0.2, 0.3]]
    mock_vector_store.similarity_search.return_value = [
        Document(doc_id="r1", content="raw content", score=0.85, source="raw", metadata={}),
    ]

    retriever = make_vector_search(mock_vector_store, mock_embedding)
    result = retriever("query")

    assert result[0]["source"] == "vector"


# ===== 测试 8: register_retrievers 注册的函数可调用 =====


def test_registered_vector_retriever_is_callable(
    mock_vector_store, mock_embedding, mock_bm25, clean_retriever_registry
):
    from ragline.handlers.retrievers import register_retrievers

    mock_embedding.embed.return_value = [[0.1, 0.2, 0.3]]
    mock_vector_store.similarity_search.return_value = [
        Document(doc_id="x1", content="cx", score=0.7, source="vector", metadata={}),
    ]

    register_retrievers(mock_vector_store, mock_embedding, mock_bm25)

    vector_retriever = clean_retriever_registry.get("vector")
    result = vector_retriever("test query")

    assert isinstance(result, list)
    assert len(result) == 1
    assert result[0]["source"] == "vector"
    mock_embedding.embed.assert_called_once_with(["test query"])
