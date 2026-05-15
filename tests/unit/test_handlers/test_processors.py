"""tests/unit/test_handlers/test_processors.py — Processor 处理器单元测试。"""

from unittest.mock import MagicMock

import pytest

from fugue.api.types import Document

# ===== Fixtures =====


@pytest.fixture
def clean_processor_registry():
    from fugue.registry import processor_registry

    saved = {n: processor_registry.get(n) for n in processor_registry.names()}
    for n in list(processor_registry.names()):
        processor_registry.unregister(n)
    yield processor_registry
    for n in list(processor_registry.names()):
        processor_registry.unregister(n)
    for n, fn in saved.items():
        processor_registry.register(n, fn)


@pytest.fixture
def mock_reranker():
    return MagicMock()


@pytest.fixture
def three_docs():
    """3 个跨 source 文档：vector:a, vector:b, bm25:a"""
    return [
        Document(doc_id="a", content="doc a vector", score=0.9, source="vector", metadata={}),
        Document(doc_id="b", content="doc b vector", score=0.7, source="vector", metadata={}),
        Document(doc_id="a", content="doc a bm25", score=2.5, source="bm25", metadata={}),
    ]


# ===== 测试 1: rrf 基础（手算对比）=====


def test_rrf_basic_scores(three_docs):
    """验证 rrf_score 按 RRF_K=60 的公式正确计算。"""
    from fugue.handlers.processors.rrf import RRF_K, rrf_fn

    result = rrf_fn(three_docs, "q", top_k=10)

    # 应返回 3 条：("vector","a"), ("vector","b"), ("bm25","a")
    assert len(result) == 3

    # 构建 (source, doc_id) → score 映射
    score_map: dict[tuple[str, str], float] = {
        (d["source"], d["doc_id"]): d["score"] for d in result
    }

    expected_vector_a = 1.0 / (RRF_K + 1)   # rank=1
    expected_vector_b = 1.0 / (RRF_K + 2)   # rank=2
    expected_bm25_a = 1.0 / (RRF_K + 1)     # rank=1（bm25 组内 rank=1）

    assert score_map[("vector", "a")] == pytest.approx(expected_vector_a)
    assert score_map[("vector", "b")] == pytest.approx(expected_vector_b)
    assert score_map[("bm25", "a")] == pytest.approx(expected_bm25_a)


# ===== 测试 2: rrf 加权 =====


def test_rrf_weighted(three_docs):
    """bm25 权重 0.5，相同 rank 下 bm25 分数是 vector 的一半。"""
    from fugue.handlers.processors.rrf import rrf_fn

    result = rrf_fn(
        three_docs, "q", top_k=10, retriever_weights={"vector": 1.0, "bm25": 0.5}
    )

    score_map: dict[tuple[str, str], float] = {
        (d["source"], d["doc_id"]): d["score"] for d in result
    }

    # vector rank=1 → 1.0/(60+1); bm25 rank=1 → 0.5/(60+1)
    vector_a_score = score_map[("vector", "a")]
    bm25_a_score = score_map[("bm25", "a")]
    assert bm25_a_score == pytest.approx(vector_a_score * 0.5)


# ===== 测试 3: rrf 跨 source 同 doc_id 保留两条 =====


def test_rrf_same_doc_id_different_source():
    """相同 doc_id 不同 source 应保留两条，不去重。"""
    from fugue.handlers.processors.rrf import rrf_fn

    docs = [
        Document(doc_id="a", content="from vector", score=0.9, source="vector", metadata={}),
        Document(doc_id="a", content="from bm25", score=2.5, source="bm25", metadata={}),
    ]
    result = rrf_fn(docs, "q", top_k=10)
    assert len(result) == 2


# ===== 测试 4: rrf 空输入 =====


def test_rrf_empty_input():
    from fugue.handlers.processors.rrf import rrf_fn

    assert rrf_fn([], "q", top_k=10) == []


# ===== 测试 5: rrf top_k 截断 =====


def test_rrf_top_k_truncation():
    """5 docs，top_k=2 只返回 2 个。"""
    from fugue.handlers.processors.rrf import rrf_fn

    docs = [
        Document(doc_id=f"d{i}", content=f"c{i}", score=float(i), source="vector", metadata={})
        for i in range(5)
    ]
    result = rrf_fn(docs, "q", top_k=2)
    assert len(result) == 2


# ===== 测试 6: rerank 基础 =====


def test_rerank_basic(mock_reranker):
    """验证 rerank 按 scored 排序返回正确文档。"""
    from fugue.handlers.processors.rerank import make_rerank

    docs = [
        Document(doc_id="a", content="doc a", score=0.5, source="vector", metadata={}),
        Document(doc_id="b", content="doc b", score=0.6, source="vector", metadata={}),
        Document(doc_id="c", content="doc c", score=0.7, source="vector", metadata={}),
    ]
    mock_reranker.rerank.return_value = [(2, 0.9), (0, 0.5)]

    rerank_fn = make_rerank(mock_reranker)
    result = rerank_fn(docs, "q", top_k=2)

    assert len(result) == 2
    # 第一个应是 docs[2] (doc_id="c")
    assert result[0]["doc_id"] == "c"
    assert result[0]["score"] == pytest.approx(0.9)
    assert result[0]["metadata"]["rerank_score"] == pytest.approx(0.9)
    # 第二个应是 docs[0] (doc_id="a")
    assert result[1]["doc_id"] == "a"
    assert result[1]["score"] == pytest.approx(0.5)


# ===== 测试 7: rerank 空输入 =====


def test_rerank_empty_input(mock_reranker):
    """空输入直接返回 []，不调用 reranker.rerank。"""
    from fugue.handlers.processors.rerank import make_rerank

    rerank_fn = make_rerank(mock_reranker)
    result = rerank_fn([], "q", top_k=10)

    assert result == []
    mock_reranker.rerank.assert_not_called()


# ===== 测试 8: rerank top_k 透传 =====


def test_rerank_top_k_passed_through(mock_reranker):
    """top_k 应原样传递给 reranker.rerank。"""
    from fugue.handlers.processors.rerank import make_rerank

    docs = [
        Document(doc_id="a", content="c", score=0.5, source="vector", metadata={}),
    ]
    mock_reranker.rerank.return_value = [(0, 0.8)]

    rerank_fn = make_rerank(mock_reranker)
    rerank_fn(docs, "q", top_k=2)

    mock_reranker.rerank.assert_called_once_with("q", ["c"], top_k=2)


# ===== 测试 9: register_processors 注册 =====


def test_register_processors_registers_all(mock_reranker, clean_processor_registry):
    """register_processors 后 rrf 和 rerank 都在 registry 中。"""
    from fugue.handlers.processors import register_processors

    register_processors(mock_reranker)

    assert clean_processor_registry.has("rrf")
    assert clean_processor_registry.has("rerank")


# ===== 测试 10: rrf rank 计算正确性 =====


def test_rrf_rank_order_same_source():
    """同 source 3 docs，score 降序排 rank，验证 rrf_score 对应 1/(60+rank)。"""
    from fugue.handlers.processors.rrf import RRF_K, rrf_fn

    docs = [
        Document(doc_id="d1", content="c1", score=0.9, source="vector", metadata={}),
        Document(doc_id="d2", content="c2", score=0.7, source="vector", metadata={}),
        Document(doc_id="d3", content="c3", score=0.5, source="vector", metadata={}),
    ]
    result = rrf_fn(docs, "q", top_k=10)

    # 结果按 score 降序，d1 排第一
    assert result[0]["doc_id"] == "d1"
    assert result[1]["doc_id"] == "d2"
    assert result[2]["doc_id"] == "d3"

    assert result[0]["score"] == pytest.approx(1.0 / (RRF_K + 1))
    assert result[1]["score"] == pytest.approx(1.0 / (RRF_K + 2))
    assert result[2]["score"] == pytest.approx(1.0 / (RRF_K + 3))
