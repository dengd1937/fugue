"""tests/test_graph.py — graph.py 集成测试（TDD RED 阶段）。"""

from __future__ import annotations

from dataclasses import asdict

import pytest

import rag.handlers  # noqa: F401 — 确保所有真实 handler 在 fixture 捕获状态前已注册
from rag.config import GraphConfig
from rag.registry import (
    generator_registry,
    grader_registry,
    processor_registry,
    retriever_registry,
    transform_registry,
)
from rag.types import Document

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def make_doc(doc_id: str, source: str, score: float = 0.9) -> Document:
    return {
        "doc_id": doc_id,
        "content": f"c_{doc_id}",
        "score": score,
        "source": source,
        "metadata": {},
    }


# ---------------------------------------------------------------------------
# fixture：stub 所有 registry
# ---------------------------------------------------------------------------


@pytest.fixture
def stub_registries():
    """注册 stub handler，yield 后复原。"""
    original = {
        "transform": dict(transform_registry._handlers),
        "retriever": dict(retriever_registry._handlers),
        "processor": dict(processor_registry._handlers),
        "grader": dict(grader_registry._handlers),
        "generator": dict(generator_registry._handlers),
    }

    transform_registry.register(
        "rewrite", lambda queries, n, **kw: [f"rewritten_{q}" for q in queries]
    )
    retriever_registry.register("vector", lambda query, **kw: [make_doc("d1", "vector")])
    retriever_registry.register("es", lambda query, **kw: [make_doc("d2", "es")])
    retriever_registry.register("web", lambda query, **kw: [make_doc("d3", "web")])
    processor_registry.register("rerank", lambda docs, **kw: docs)
    grader_registry.register("score", lambda docs, threshold, **kw: (0.9, "sufficient"))
    generator_registry.register("basic", lambda docs, query, **kw: f"answer for: {query}")

    yield

    transform_registry._handlers = original["transform"]
    retriever_registry._handlers = original["retriever"]
    processor_registry._handlers = original["processor"]
    grader_registry._handlers = original["grader"]
    generator_registry._handlers = original["generator"]


# ---------------------------------------------------------------------------
# 1. 基本流程：sufficient，无 fallback
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_basic_flow_sufficient(stub_registries):
    from rag.graph import build_rag_graph

    graph = build_rag_graph()
    cfg = GraphConfig(
        transforms=["rewrite"],
        n_rewrites=1,
        retrievers=["vector", "es"],
        top_k=2,
        fallback_chain=[],
    )
    result = graph.invoke(
        {"original_query": "LangGraph 如何实现动态路由？", "retry_count": 0, "source": "kb"},
        config={"configurable": asdict(cfg)},
    )
    assert result.get("answer"), "answer 应非空"
    assert "answer for" in result["answer"]
    assert len(result["ranked_documents"]) <= 2


# ---------------------------------------------------------------------------
# 2. Send fan-in 验证：多 retriever 的文档都被 grade 看到
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_send_fanin_both_retrievers_graded(stub_registries):
    from rag.graph import build_rag_graph

    graded_sources: list[str] = []

    def capturing_grader(docs, threshold, **kw):
        graded_sources.extend(d["source"] for d in docs)
        return (0.9, "sufficient")

    grader_registry.register("score", capturing_grader)

    graph = build_rag_graph()
    cfg = GraphConfig(
        transforms=["rewrite"],
        n_rewrites=1,
        retrievers=["vector", "es"],
        top_k=10,
        fallback_chain=[],
    )
    graph.invoke(
        {"original_query": "test query", "retry_count": 0, "source": "kb"},
        config={"configurable": asdict(cfg)},
    )
    # grade 应该见到来自 vector 和 es 的文档
    assert "vector" in graded_sources, "grade 应收到 vector 文档"
    assert "es" in graded_sources, "grade 应收到 es 文档"


# ---------------------------------------------------------------------------
# 3. max_queries 截断
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_max_queries_truncation(stub_registries):
    from rag.graph import build_rag_graph

    # rewrite 返回 3 个 rewritten，加上原始共 4 个，但 max_queries=2 → 截断
    transform_registry.register(
        "rewrite",
        lambda queries, n, **kw: [f"r{i}_{q}" for i in range(3) for q in queries],
    )

    graph = build_rag_graph()
    cfg = GraphConfig(
        transforms=["rewrite"],
        n_rewrites=3,
        retrievers=["vector"],
        max_queries=2,
        top_k=10,
        fallback_chain=[],
    )
    result = graph.invoke(
        {"original_query": "q", "retry_count": 0, "source": "kb"},
        config={"configurable": asdict(cfg)},
    )
    # rewritten_queries 长度应 <= max_queries
    assert len(result["rewritten_queries"]) <= 2
    assert result.get("answer"), "应有 answer"


# ---------------------------------------------------------------------------
# 4. fallback 终止性：insufficient + fallback_chain + max_retries
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_fallback_terminates(stub_registries):
    from rag.graph import build_rag_graph

    grader_registry.register("score", lambda docs, threshold, **kw: (0.1, "insufficient"))

    graph = build_rag_graph()
    cfg = GraphConfig(
        transforms=["rewrite"],
        n_rewrites=1,
        retrievers=["vector"],
        fallback_chain=["web"],
        max_retries=1,
        top_k=5,
    )
    result = graph.invoke(
        {"original_query": "fallback test", "retry_count": 0, "source": "kb"},
        config={"configurable": asdict(cfg)},
    )
    # 图必须终止，answer 存在
    assert result.get("answer"), "fallback 后仍应有 answer"
    # retry_count 不超过 max_retries
    assert result.get("retry_count", 0) <= 1


# ---------------------------------------------------------------------------
# 5. 无 fallback：insufficient 直接走 post_process
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_no_fallback_goes_to_post_process(stub_registries):
    from rag.graph import build_rag_graph

    grader_registry.register("score", lambda docs, threshold, **kw: (0.1, "insufficient"))

    graph = build_rag_graph()
    cfg = GraphConfig(
        transforms=["rewrite"],
        n_rewrites=1,
        retrievers=["vector"],
        fallback_chain=[],
        max_retries=1,
        top_k=3,
    )
    result = graph.invoke(
        {"original_query": "no fallback", "retry_count": 0, "source": "kb"},
        config={"configurable": asdict(cfg)},
    )
    assert result.get("answer"), "无 fallback 时仍应有 answer"


# ---------------------------------------------------------------------------
# 6. 图拓扑验证：节点名称存在
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_graph_has_expected_nodes(stub_registries):
    from rag.graph import build_rag_graph

    graph = build_rag_graph()
    node_names = set(graph.nodes.keys())
    expected = {
        "query_transform",
        "retrieve",
        "grade",
        "prepare_fallback",
        "post_process",
        "generate",
    }
    assert expected.issubset(node_names), f"缺少节点：{expected - node_names}"


# ---------------------------------------------------------------------------
# 7. ranked_documents top_k 限制
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_ranked_documents_respects_top_k(stub_registries):
    from rag.graph import build_rag_graph

    # 让两个 retriever 各返回 3 个文档
    retriever_registry.register(
        "vector",
        lambda query, **kw: [make_doc(f"v{i}", "vector") for i in range(3)],
    )
    retriever_registry.register(
        "es",
        lambda query, **kw: [make_doc(f"e{i}", "es") for i in range(3)],
    )

    graph = build_rag_graph()
    cfg = GraphConfig(
        transforms=[],
        n_rewrites=1,
        retrievers=["vector", "es"],
        top_k=2,
        fallback_chain=[],
    )
    result = graph.invoke(
        {"original_query": "top_k test", "retry_count": 0, "source": "kb"},
        config={"configurable": asdict(cfg)},
    )
    assert len(result["ranked_documents"]) <= 2
