"""tests/integration/test_engine_graph.py — 整图集成测试（mock providers）。"""

from collections.abc import Callable
from typing import Any

import pytest

from fugue.api.types import Document
from fugue.config import GraphConfig
from fugue.engine.graph import build_rag_graph
from fugue.registry import (
    generator_registry,
    grader_registry,
    processor_registry,
    retriever_registry,
    transform_registry,
)


def _doc(source: str, doc_id: str, score: float = 0.9, content: str = "x") -> Document:
    return Document(doc_id=doc_id, content=content, score=score, source=source, metadata={})


@pytest.fixture
def clean_registries():
    """清空所有 registry，yield 后恢复（避免污染其他测试）。"""
    saves: dict[str, dict[str, Callable[..., Any]]] = {}
    registries = {
        "transform": transform_registry,
        "retriever": retriever_registry,
        "processor": processor_registry,
        "grader": grader_registry,
        "generator": generator_registry,
    }
    for name, reg in registries.items():
        saves[name] = {n: reg.get(n) for n in reg.names()}
        for n in list(reg.names()):
            reg.unregister(n)
    yield
    for name, reg in registries.items():
        for n in list(reg.names()):
            reg.unregister(n)
        for n, fn in saves[name].items():
            reg.register(n, fn)


def _register_minimal(clean_registries: Any) -> None:
    """注册最小可用 handlers（mock 各 transform / retriever / grader / processor / generator）。"""
    transform_registry.register("rewrite", lambda q, n: [f"rewrite_{i}" for i in range(n)])
    retriever_registry.register(
        "vector", lambda query, metadata_filter=None: [_doc("vector", f"v_{query[:20]}", score=0.9)]
    )
    retriever_registry.register(
        "bm25", lambda query, metadata_filter=None: [_doc("bm25", f"b_{query[:20]}", score=15.0)]
    )
    processor_registry.register("identity", lambda docs, q, k, **kw: docs[:k])
    generator_registry.register("basic", lambda q, docs, t: f"answer: {len(docs)} docs")
    # grader: 通过 import 已注册（task 14 模块导入自动注册），需重新注册因清空了
    from fugue.handlers.graders.score import score_grader

    grader_registry.register("score", score_grader)


def _make_initial_state(query: str = "test query") -> dict[str, Any]:
    return {
        "original_query": query,
        "rewritten_queries": [],
        "documents": [],
        "grade_score": 0.0,
        "grade_decision": "insufficient",
        "source": "kb",
        "retry_count": 0,
        "retrieval_history": [],
        "ranked_documents": [],
        "answer": "",
    }


def _config(**overrides: Any) -> dict[str, Any]:
    defaults = GraphConfig(
        transforms=["rewrite"],
        n_rewrites=2,
        retrievers=["vector"],
        processors=["identity"],
        top_k=3,
        gen_mode="basic",
        grade_threshold=0.5,  # 低阈值容易 sufficient
        fallback_chain=[],
        max_retries=0,
    )
    cfg_dict = {
        "transforms": defaults.transforms,
        "n_rewrites": defaults.n_rewrites,
        "max_queries": defaults.max_queries,
        "retrievers": defaults.retrievers,
        "route_strategy": defaults.route_strategy,
        "retriever_weights": defaults.retriever_weights,
        "grade_threshold": defaults.grade_threshold,
        "grade_strategy": defaults.grade_strategy,
        "score_normalizers": defaults.score_normalizers,
        "fallback_chain": defaults.fallback_chain,
        "max_retries": defaults.max_retries,
        "processors": defaults.processors,
        "top_k": defaults.top_k,
        "gen_mode": defaults.gen_mode,
        "temperature": defaults.temperature,
    }
    cfg_dict.update(overrides)
    return {"configurable": cfg_dict}


# 1. 完整跑通 ----------------------------------------------------------


def test_graph_end_to_end_basic(clean_registries) -> None:
    """构建图 + 注册 mock handlers + invoke，验证返回 state 含 answer 非空。"""
    _register_minimal(clean_registries)
    graph = build_rag_graph()
    result = graph.invoke(_make_initial_state(), _config())
    assert result["answer"]
    assert "answer:" in result["answer"]


# 2. 多 retriever 合并 -----------------------------------------------


def test_multi_retrievers_merge(clean_registries) -> None:
    """retrievers=['vector','bm25'] 各返回 docs，grade 时 documents 含两组 source。"""
    _register_minimal(clean_registries)
    graph = build_rag_graph()
    result = graph.invoke(
        _make_initial_state(),
        _config(retrievers=["vector", "bm25"], score_normalizers={"bm25": 20.0}),
    )
    # ranked_documents 来自 retrieval_history[-1]（grade 把当前 docs 归档）；
    # 我们用 retrieval_history 验证合并行为
    history = result["retrieval_history"]
    assert len(history) >= 1
    all_docs = history[0]
    sources = {d["source"] for d in all_docs}
    assert "vector" in sources
    assert "bm25" in sources


# 3. fallback 闭环 ----------------------------------------------------


def test_fallback_loop(clean_registries) -> None:
    """grade 第一次 insufficient → prepare_fallback → 切到 web → 第二次 sufficient。"""
    _register_minimal(clean_registries)
    # 注册 web retriever
    retriever_registry.register("web", lambda query, metadata_filter=None: [_doc("web", f"w_{query[:20]}", score=0.95)])

    # 第一次 grader 返回 insufficient，第二次返回 sufficient
    call_count = {"n": 0}

    def stateful_grader(docs, query, threshold, **kw):
        call_count["n"] += 1
        if call_count["n"] == 1:
            return (0.2, "insufficient")
        return (0.9, "sufficient")

    grader_registry.unregister("score")
    grader_registry.register("score", stateful_grader)

    graph = build_rag_graph()
    result = graph.invoke(
        _make_initial_state(),
        _config(fallback_chain=["web"], max_retries=1),
    )
    # 第二轮成功后 retry_count = 1，retrieval_history 含两轮
    assert result["retry_count"] == 1
    assert len(result["retrieval_history"]) == 2
    # 最终 source 是 web（fallback 切换的）
    assert result["source"] == "web"


# 4. Overwrite([]) 重置语义 -------------------------------------------


def test_overwrite_resets_documents_across_fallback(clean_registries) -> None:
    """fallback 第二轮 retrieve 后 documents 不含第一轮（Overwrite([]) 工作）。"""
    _register_minimal(clean_registries)
    retriever_registry.register("web", lambda query, metadata_filter=None: [_doc("web", f"w_{query[:20]}", score=0.95)])

    call_count = {"n": 0}

    def stateful_grader(docs, query, threshold, **kw):
        call_count["n"] += 1
        if call_count["n"] == 1:
            return (0.2, "insufficient")
        return (0.9, "sufficient")

    grader_registry.unregister("score")
    grader_registry.register("score", stateful_grader)

    graph = build_rag_graph()
    result = graph.invoke(
        _make_initial_state(),
        _config(fallback_chain=["web"], max_retries=1),
    )
    # retrieval_history[-1] = 第二轮 documents（在第二次 grade 之前已 Overwrite 重置）
    second_round = result["retrieval_history"][-1]
    # 第二轮只应包含 web source（kb 的 vector 已被 Overwrite 清空）
    sources_second = {d["source"] for d in second_round}
    assert sources_second == {"web"}


# 5. 嵌套 transforms 端到端 -------------------------------------------


def test_nested_transforms_end_to_end(clean_registries) -> None:
    """transforms=['hyde', ['step_back', 'rewrite']] 端到端跑通。

    手算扇出: original(1) + hyde(2) + step_back(2)→rewrite(每个 2 = 总 4) = 7 queries
    × 1 retriever = 7 次 retrieve 调用（去重前；可能因 mock 实现而少）
    """
    _register_minimal(clean_registries)
    transform_registry.register("hyde", lambda q, n: [f"hyde_{i}" for i in range(n)])
    transform_registry.register("step_back", lambda q, n: [f"sb_{i}" for i in range(n)])

    retrieve_calls = {"n": 0}
    original_retriever = retriever_registry.get("vector")

    def counted_vector(query, metadata_filter=None):
        retrieve_calls["n"] += 1
        return original_retriever(query=query, metadata_filter=metadata_filter)

    retriever_registry.unregister("vector")
    retriever_registry.register("vector", counted_vector)

    graph = build_rag_graph()
    result = graph.invoke(
        _make_initial_state(),
        _config(
            transforms=["hyde", ["step_back", "rewrite"]],
            n_rewrites=2,
            retrievers=["vector"],
        ),
    )
    assert result["answer"]
    # 至少 5 次 retrieve 调用（具体数取决于 dedupe）
    assert retrieve_calls["n"] >= 5


# 6. Overwrite reducer 锁定语义（反面验证） -------------------------


def test_reducer_without_overwrite_accumulates(clean_registries) -> None:
    """反面对比：直接更新 documents（不通过 Overwrite）→ 累加而非覆盖。

    用 merge_docs 直接调用验证 reducer 在普通 list 输入下的累加行为。
    这是 reducer 语义锁定测试——未来 LangGraph 行为变化时此测试若失败需重审。
    """
    from fugue.engine.state import Overwrite, merge_docs

    d1 = _doc("vector", "1")
    d2 = _doc("vector", "2")

    # 普通 list：累加去重
    merged = merge_docs([d1], [d2])
    assert len(merged) == 2

    # Overwrite：覆盖
    merged_overwrite = merge_docs([d1], Overwrite([d2]))
    assert merged_overwrite == [d2]
