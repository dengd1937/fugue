"""tests/test_state.py — rag.state 单元测试（TDD RED 阶段先写）"""

import pytest

from rag.state import merge_docs
from rag.types import Document

# ---------------------------------------------------------------------------
# 辅助工厂
# ---------------------------------------------------------------------------


def make_doc(
    doc_id: str,
    source: str = "vector",
    score: float = 0.9,
    content: str = "text",
) -> Document:
    return {
        "doc_id": doc_id,
        "content": content,
        "score": score,
        "source": source,
        "metadata": {},
    }


# ---------------------------------------------------------------------------
# merge_docs 基础行为
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_merge_empty_existing_with_new_doc() -> None:
    doc_a = make_doc("a")
    result = merge_docs([], [doc_a])
    assert result == [doc_a]


@pytest.mark.unit
def test_merge_with_empty_new_list() -> None:
    doc_a = make_doc("a")
    result = merge_docs([doc_a], [])
    assert result == [doc_a]


@pytest.mark.unit
def test_merge_both_empty() -> None:
    result = merge_docs([], [])
    assert result == []


@pytest.mark.unit
def test_merge_appends_different_doc_id() -> None:
    doc_a = make_doc("a")
    doc_b = make_doc("b")
    result = merge_docs([doc_a], [doc_b])
    assert result == [doc_a, doc_b]


# ---------------------------------------------------------------------------
# 去重行为（相同复合键）
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_merge_deduplicates_same_source_and_doc_id() -> None:
    doc_a = make_doc("a", source="vector")
    result = merge_docs([doc_a], [doc_a])
    assert result == [doc_a]
    assert len(result) == 1


@pytest.mark.unit
def test_merge_keeps_existing_when_duplicate() -> None:
    """相同 (source, doc_id) 重复时，保留 existing 中的版本（score 不同）。"""
    existing = make_doc("a", source="vector", score=0.9)
    new_version = make_doc("a", source="vector", score=0.5)
    result = merge_docs([existing], [new_version])
    assert len(result) == 1
    assert result[0]["score"] == 0.9  # 保留 existing


# ---------------------------------------------------------------------------
# 不同 source 同 doc_id — 都保留（复合键不同）
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_merge_keeps_same_doc_id_different_source() -> None:
    doc_vector = make_doc("a", source="vector")
    doc_es = make_doc("a", source="es")
    result = merge_docs([doc_vector], [doc_es])
    assert len(result) == 2
    sources = {d["source"] for d in result}
    assert sources == {"vector", "es"}


# ---------------------------------------------------------------------------
# 幂等性
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_merge_is_idempotent() -> None:
    doc_a = make_doc("a")
    doc_b = make_doc("b")
    result = merge_docs([doc_a, doc_b], [doc_a, doc_b])
    assert result == [doc_a, doc_b]


@pytest.mark.unit
def test_merge_self_is_idempotent() -> None:
    doc_a = make_doc("a")
    first = merge_docs([], [doc_a])
    second = merge_docs(first, first)
    assert second == first


# ---------------------------------------------------------------------------
# 顺序保证
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_merge_preserves_existing_order() -> None:
    docs = [make_doc(str(i)) for i in range(5)]
    result = merge_docs(docs, [])
    assert [d["doc_id"] for d in result] == [str(i) for i in range(5)]


@pytest.mark.unit
def test_merge_new_docs_appended_after_existing() -> None:
    doc_a = make_doc("a")
    doc_b = make_doc("b")
    result = merge_docs([doc_a], [doc_b])
    assert result[0]["doc_id"] == "a"
    assert result[1]["doc_id"] == "b"


# ---------------------------------------------------------------------------
# 多条新文档去重
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_merge_multiple_new_docs_with_partial_overlap() -> None:
    doc_a = make_doc("a")
    doc_b = make_doc("b")
    doc_c = make_doc("c")
    result = merge_docs([doc_a, doc_b], [doc_b, doc_c])
    assert len(result) == 3
    ids = [d["doc_id"] for d in result]
    assert ids == ["a", "b", "c"]


# ---------------------------------------------------------------------------
# 大数据量（性能边界）
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_merge_large_list_no_duplicates() -> None:
    existing = [make_doc(f"e{i}") for i in range(500)]
    new_docs = [make_doc(f"n{i}") for i in range(500)]
    result = merge_docs(existing, new_docs)
    assert len(result) == 1000


@pytest.mark.unit
def test_merge_large_list_all_duplicates() -> None:
    docs = [make_doc(f"d{i}") for i in range(500)]
    result = merge_docs(docs, docs)
    assert len(result) == 500
