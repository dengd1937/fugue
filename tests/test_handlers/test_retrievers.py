"""tests/test_handlers/test_retrievers.py — retrievers handler 测试。"""

from typing import Any

import pytest

# ---------------------------------------------------------------------------
# FakeRetrieverClient stub
# ---------------------------------------------------------------------------


class FakeRetrieverClient:
    """测试用 retriever client stub。"""

    def __init__(self, docs: list[dict[str, Any]] | None = None) -> None:
        self._docs = docs or []
        self.last_query: str | None = None
        self.last_filter: dict[str, Any] | None = None
        self.last_k: int | None = None

    def search(self, query: str, filter: dict[str, Any] | None, k: int) -> list[dict[str, Any]]:
        self.last_query = query
        self.last_filter = filter
        self.last_k = k
        return self._docs


def _make_raw_doc(
    doc_id: str = "d1",
    content: str = "content",
    score: float = 0.9,
    source: str = "",
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "doc_id": doc_id,
        "content": content,
        "score": score,
        "source": source,
        "metadata": metadata or {},
    }


# ---------------------------------------------------------------------------
# 共用辅助：验证 source 字段被写入
# ---------------------------------------------------------------------------

RETRIEVER_CASES = [
    ("vector_search_fn", "vector"),
    ("es_search_fn", "es"),
    ("kg_search_fn", "kg"),
    ("web_search_fn", "web"),
    ("sql_search_fn", "sql"),
    ("vector_parent_child_fn", "vector_parent_child"),
    ("vector_sentence_window_fn", "vector_sentence_window"),
]


class TestRetrieverSourceField:
    @pytest.mark.parametrize("fn_name,expected_source", RETRIEVER_CASES)
    def test_source_field_written(self, fn_name: str, expected_source: str) -> None:
        import rag.handlers.retrievers as mod

        fn = getattr(mod, fn_name)
        raw_doc = _make_raw_doc(source="wrong_source")
        client = FakeRetrieverClient([raw_doc])
        result = fn("test query", client=client)
        assert len(result) == 1
        assert result[0]["source"] == expected_source

    @pytest.mark.parametrize("fn_name,expected_source", RETRIEVER_CASES)
    def test_empty_client_returns_empty_list(self, fn_name: str, expected_source: str) -> None:
        import rag.handlers.retrievers as mod

        fn = getattr(mod, fn_name)
        client = FakeRetrieverClient([])
        result = fn("test query", client=client)
        assert result == []

    @pytest.mark.parametrize("fn_name,expected_source", RETRIEVER_CASES)
    def test_returns_list_of_document(self, fn_name: str, expected_source: str) -> None:
        import rag.handlers.retrievers as mod

        fn = getattr(mod, fn_name)
        raw_doc = _make_raw_doc()
        client = FakeRetrieverClient([raw_doc])
        result = fn("q", client=client)
        assert isinstance(result, list)
        assert len(result) == 1
        # 验证是 Document TypedDict 结构
        doc = result[0]
        assert "doc_id" in doc
        assert "content" in doc
        assert "score" in doc
        assert "source" in doc
        assert "metadata" in doc


class TestMetadataFilterPassing:
    @pytest.mark.parametrize("fn_name,_", RETRIEVER_CASES)
    def test_metadata_filter_passed_to_client(self, fn_name: str, _: str) -> None:
        import rag.handlers.retrievers as mod

        fn = getattr(mod, fn_name)
        client = FakeRetrieverClient([])
        metadata_filter = {"year": 2024, "category": "tech"}
        fn("q", metadata_filter=metadata_filter, client=client)
        assert client.last_filter == metadata_filter

    @pytest.mark.parametrize("fn_name,_", RETRIEVER_CASES)
    def test_no_filter_passes_none_to_client(self, fn_name: str, _: str) -> None:
        import rag.handlers.retrievers as mod

        fn = getattr(mod, fn_name)
        client = FakeRetrieverClient([])
        fn("q", client=client)
        assert client.last_filter is None


class TestRetrieverQueryPassing:
    def test_query_passed_to_client(self) -> None:
        from rag.handlers.retrievers import vector_search_fn

        client = FakeRetrieverClient([])
        vector_search_fn("specific query text", client=client)
        assert client.last_query == "specific query text"

    def test_multiple_docs_preserved(self) -> None:
        from rag.handlers.retrievers import vector_search_fn

        docs = [_make_raw_doc(doc_id=str(i)) for i in range(5)]
        client = FakeRetrieverClient(docs)
        result = vector_search_fn("q", client=client)
        assert len(result) == 5


class TestRetrieverRegistration:
    def test_all_retrievers_registered(self) -> None:
        import rag.handlers.retrievers  # noqa: F401
        from rag.registry import retriever_registry

        for name in [
            "vector",
            "es",
            "kg",
            "web",
            "sql",
            "vector_parent_child",
            "vector_sentence_window",
        ]:
            assert retriever_registry.has(name), f"'{name}' not registered"
