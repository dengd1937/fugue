"""tests/test_types.py — rag.types 单元测试（TDD RED 阶段先写）"""

from dataclasses import FrozenInstanceError

import pytest

from rag.types import Document, RetrieveInput, TransformResult

# ---------------------------------------------------------------------------
# Document
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_document_construction_and_field_access() -> None:
    doc: Document = {
        "doc_id": "d1",
        "content": "hello world",
        "score": 0.95,
        "source": "vector",
        "metadata": {"page": 1},
    }
    assert doc["doc_id"] == "d1"
    assert doc["content"] == "hello world"
    assert doc["score"] == 0.95
    assert doc["source"] == "vector"
    assert doc["metadata"] == {"page": 1}


@pytest.mark.unit
def test_document_with_empty_metadata() -> None:
    doc: Document = {
        "doc_id": "d2",
        "content": "text",
        "score": 0.0,
        "source": "es",
        "metadata": {},
    }
    assert doc["metadata"] == {}


@pytest.mark.unit
def test_document_score_zero_and_one() -> None:
    for score in (0.0, 1.0):
        doc: Document = {
            "doc_id": "dx",
            "content": "x",
            "score": score,
            "source": "web",
            "metadata": {},
        }
        assert doc["score"] == score


# ---------------------------------------------------------------------------
# TransformResult
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_transform_result_construction() -> None:
    tr = TransformResult(query="who is ada lovelace?")
    assert tr.query == "who is ada lovelace?"
    assert tr.metadata_filter is None


@pytest.mark.unit
def test_transform_result_with_metadata_filter() -> None:
    flt = {"year": 2024, "lang": "zh"}
    tr = TransformResult(query="recent papers", metadata_filter=flt)
    assert tr.metadata_filter == flt


@pytest.mark.unit
def test_transform_result_is_frozen_dataclass() -> None:
    tr = TransformResult(query="test")
    with pytest.raises(FrozenInstanceError):
        tr.query = "mutated"  # type: ignore[misc]


@pytest.mark.unit
def test_transform_result_metadata_filter_default_is_none() -> None:
    tr = TransformResult(query="q")
    assert tr.metadata_filter is None


@pytest.mark.unit
def test_transform_result_frozen_prevents_new_attr() -> None:
    tr = TransformResult(query="q")
    with pytest.raises(FrozenInstanceError):
        tr.metadata_filter = {"k": "v"}  # type: ignore[misc]


# ---------------------------------------------------------------------------
# RetrieveInput
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_retrieve_input_construction() -> None:
    ri: RetrieveInput = {
        "query": "what is rag?",
        "retriever_name": "vector_store",
        "source": "vector",
        "metadata_filter": None,
    }
    assert ri["query"] == "what is rag?"
    assert ri["retriever_name"] == "vector_store"
    assert ri["source"] == "vector"
    assert ri["metadata_filter"] is None


@pytest.mark.unit
def test_retrieve_input_with_metadata_filter() -> None:
    ri: RetrieveInput = {
        "query": "q",
        "retriever_name": "es_retriever",
        "source": "es",
        "metadata_filter": {"lang": "en"},
    }
    assert ri["metadata_filter"] == {"lang": "en"}


@pytest.mark.unit
def test_retrieve_input_source_variants() -> None:
    for source in ("vector", "es", "web"):
        ri: RetrieveInput = {
            "query": "q",
            "retriever_name": f"{source}_retriever",
            "source": source,
            "metadata_filter": None,
        }
        assert ri["source"] == source
