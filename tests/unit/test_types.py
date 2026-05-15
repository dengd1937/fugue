"""tests/unit/test_types.py — 公开类型与异常的单元测试。"""

import dataclasses
from pathlib import Path

import pytest

from fugue.api.types import (
    Chunk,
    Document,
    FugueConfigError,
    FugueEmbeddingError,
    FugueError,
    FugueLLMError,
    FugueRegistryError,
    FugueRetrieverError,
    ParsedDocument,
    QueryResult,
    TransformResult,
)


# 测试 1：Document TypedDict 字段齐全
def test_document_typeddict_construction() -> None:
    doc: Document = Document(
        doc_id="d1",
        content="c",
        score=0.5,
        source="s",
        metadata={},
    )
    assert doc["doc_id"] == "d1"
    assert doc["score"] == 0.5


# 测试 2：QueryResult 是 frozen dataclass
def test_query_result_is_frozen() -> None:
    result = QueryResult(
        answer="test answer",
        ranked_documents=[],
        grade_score=0.8,
        grade_decision="sufficient",
        rewritten_queries=[],
        retrieval_rounds=1,
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        result.answer = "x"  # type: ignore[misc]


# 测试 3：异常继承关系
def test_exception_inheritance() -> None:
    assert issubclass(FugueLLMError, FugueError)
    assert issubclass(FugueConfigError, FugueError)
    assert issubclass(FugueRegistryError, FugueError)
    assert issubclass(FugueEmbeddingError, FugueError)
    assert issubclass(FugueRetrieverError, FugueError)


# 测试 4：异常可携带 message
def test_exception_message() -> None:
    e = FugueRegistryError("kg not found")
    assert str(e) == "kg not found"


# 测试 5：TransformResult 的 metadata_filter 默认 None
def test_transform_result_metadata_filter_default() -> None:
    tr = TransformResult(query="q")
    assert tr.metadata_filter is None


# 测试 6：Chunk 的 parent_id 允许 None
def test_chunk_parent_id_none() -> None:
    chunk = Chunk(chunk_id="c1", parent_id=None, content="x", metadata={})
    assert chunk.parent_id is None


# 测试 7：ParsedDocument 是 frozen dataclass
def test_parsed_document_is_frozen() -> None:
    doc = ParsedDocument(source_path=Path("/x"), content="y", metadata={})
    with pytest.raises(dataclasses.FrozenInstanceError):
        doc.content = "z"  # type: ignore[misc]


# 测试 8：FugueError 是 Exception 子类
def test_fugue_error_is_exception() -> None:
    assert issubclass(FugueError, Exception)
