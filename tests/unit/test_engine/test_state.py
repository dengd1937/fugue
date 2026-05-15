"""tests/unit/test_engine/test_state.py — RAGState 与 merge_docs 测试。"""

from fugue.api.types import Document
from fugue.engine.state import RAGState, RetrieveInput, merge_docs


def _doc(source: str, doc_id: str, content: str = "x") -> Document:
    return Document(
        doc_id=doc_id, content=content, score=0.9, source=source, metadata={}
    )


def test_merge_docs_basic_dedup() -> None:
    """同 (source, doc_id) 重复时只保留 existing 中的版本。"""
    existing = [_doc("vector", "1", "old")]
    new = [_doc("vector", "1", "new")]
    merged = merge_docs(existing, new)
    assert len(merged) == 1
    assert merged[0]["content"] == "old"


def test_merge_docs_cross_source_same_doc_id() -> None:
    """跨 source 同 doc_id 两条都保留。"""
    existing = [_doc("vector", "1")]
    new = [_doc("bm25", "1")]
    merged = merge_docs(existing, new)
    assert len(merged) == 2


def test_merge_docs_order_preserved() -> None:
    """existing 在前，new 中未重复的按原顺序追加。"""
    existing = [_doc("vector", "1"), _doc("vector", "2")]
    new = [_doc("vector", "1"), _doc("bm25", "3"), _doc("vector", "4")]
    merged = merge_docs(existing, new)
    ids = [(d["source"], d["doc_id"]) for d in merged]
    assert ids == [("vector", "1"), ("vector", "2"), ("bm25", "3"), ("vector", "4")]


def test_merge_docs_empty_existing() -> None:
    merged = merge_docs([], [_doc("vector", "1")])
    assert len(merged) == 1


def test_merge_docs_empty_new() -> None:
    existing = [_doc("vector", "1")]
    merged = merge_docs(existing, [])
    assert merged == existing


def test_rag_state_fields_present() -> None:
    """RAGState 是 TypedDict，可以构造实例。"""
    state: RAGState = RAGState(
        original_query="q",
        rewritten_queries=[],
        documents=[],
        grade_score=0.0,
        grade_decision="insufficient",
        source="kb",
        retry_count=0,
        retrieval_history=[],
        ranked_documents=[],
        answer="",
    )
    assert state["original_query"] == "q"


def test_retrieve_input_construct() -> None:
    item: RetrieveInput = RetrieveInput(
        query="q",
        retriever_name="vector",
        source="kb",
        metadata_filter=None,
    )
    assert item["retriever_name"] == "vector"
