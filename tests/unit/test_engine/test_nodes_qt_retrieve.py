"""tests/unit/test_engine/test_nodes_qt_retrieve.py — query_transform 与 retrieve 节点测试。"""

import logging
from unittest.mock import MagicMock

from langgraph.types import Command, Send

from ragline.api.types import Document, TransformResult
from ragline.engine.nodes.query_transform import query_transform
from ragline.engine.nodes.retrieve import retrieve
from ragline.engine.state import Overwrite, RAGState
from ragline.registry import retriever_registry, transform_registry


def _initial_state(query: str = "原问题", source: str = "kb") -> RAGState:
    return RAGState(
        original_query=query,
        rewritten_queries=[],
        documents=[],
        grade_score=0.0,
        grade_decision="insufficient",
        source=source,
        retry_count=0,
        retrieval_history=[],
        ranked_documents=[],
        answer="",
    )


# ---------------------------------------------------------------------------
# query_transform 测试
# ---------------------------------------------------------------------------


def test_query_transform_basic_fanout(isolated_registries_fx) -> None:
    """transforms=['rewrite'] + n=2 + retrievers=['vector']
    → 3 queries × 1 retriever = 3 Send，原 query 在第 0 位。"""
    transform_registry.register("rewrite", lambda q, n: ["q1", "q2"])
    config = {
        "configurable": {
            "transforms": ["rewrite"],
            "n_rewrites": 2,
            "retrievers": ["vector"],
            "max_queries": 20,
        }
    }
    cmd = query_transform(_initial_state(), config)
    assert isinstance(cmd, Command)
    assert cmd.update is not None
    # 原 query 在第 0 位
    assert cmd.update["rewritten_queries"][0] == "原问题"
    assert cmd.update["rewritten_queries"] == ["原问题", "q1", "q2"]
    # documents 是 Overwrite([])
    assert isinstance(cmd.update["documents"], Overwrite)
    assert cmd.update["documents"].values == []
    # goto 是 3 个 Send
    sends = cmd.goto
    assert isinstance(sends, list)
    assert len(sends) == 3
    for s in sends:
        assert isinstance(s, Send)
        assert s.node == "retrieve"


def test_query_transform_dedup(isolated_registries_fx) -> None:
    """transforms 产出重复 query 被去重保留首次。"""
    transform_registry.register("rewrite", lambda q, n: ["原问题", "q1", "q1"])
    config = {
        "configurable": {
            "transforms": ["rewrite"],
            "n_rewrites": 3,
            "retrievers": ["vector"],
        }
    }
    cmd = query_transform(_initial_state(), config)
    # 原 query 在第 0，"原问题" 重复被去重
    assert cmd.update["rewritten_queries"] == ["原问题", "q1"]


def test_query_transform_max_queries_truncation(isolated_registries_fx) -> None:
    """all_queries 超过 max_queries 时截断。"""
    transform_registry.register("rewrite", lambda q, n: [f"q{i}" for i in range(20)])
    config = {
        "configurable": {
            "transforms": ["rewrite"],
            "n_rewrites": 20,
            "retrievers": ["vector"],
            "max_queries": 5,
        }
    }
    cmd = query_transform(_initial_state(), config)
    assert len(cmd.update["rewritten_queries"]) == 5


def test_query_transform_fallback_single_source(isolated_registries_fx) -> None:
    """state.source != 'kb' 时只用 [source] 作 retriever_names。"""
    transform_registry.register("rewrite", lambda q, n: ["q1"])
    config = {
        "configurable": {
            "transforms": ["rewrite"],
            "n_rewrites": 1,
            "retrievers": ["vector", "bm25"],
        }
    }
    cmd = query_transform(_initial_state(source="web"), config)
    # retriever_names 应该只有 "web"
    sends = cmd.goto
    for s in sends:
        assert s.arg["retriever_name"] == "web"


def test_query_transform_transform_result_with_metadata_filter(
    isolated_registries_fx,
) -> None:
    """TransformResult 携带的 metadata_filter 被写入对应 query 的 Send payload。"""
    transform_registry.register(
        "self_query",
        lambda q, n: [TransformResult(query="2024 论文", metadata_filter={"year": 2024})],
    )
    config = {
        "configurable": {
            "transforms": ["self_query"],
            "n_rewrites": 1,
            "retrievers": ["vector"],
        }
    }
    cmd = query_transform(_initial_state(), config)
    sends = cmd.goto
    # 找到那个 query=="2024 论文" 的 Send
    target = next(s for s in sends if s.arg["query"] == "2024 论文")
    assert target.arg["metadata_filter"] == {"year": 2024}
    # 原 query 的 Send 没有 filter
    original = next(s for s in sends if s.arg["query"] == "原问题")
    assert original.arg["metadata_filter"] is None


def test_query_transform_documents_overwrite_sentinel(isolated_registries_fx) -> None:
    """Command.update['documents'] 是 Overwrite([])。"""
    transform_registry.register("rewrite", lambda q, n: [])
    config = {"configurable": {"transforms": ["rewrite"], "n_rewrites": 1, "retrievers": ["vector"]}}
    cmd = query_transform(_initial_state(), config)
    docs = cmd.update["documents"]
    assert isinstance(docs, Overwrite)
    assert docs.values == []


# ---------------------------------------------------------------------------
# retrieve 测试
# ---------------------------------------------------------------------------


def test_retrieve_success_sets_source(isolated_registries_fx) -> None:
    mock_fn = MagicMock(
        return_value=[
            Document(doc_id="d1", content="x", score=0.9, source="raw", metadata={}),
            Document(doc_id="d2", content="y", score=0.8, source="raw", metadata={}),
        ]
    )
    retriever_registry.register("vector", mock_fn)
    result = retrieve(
        {
            "query": "q",
            "retriever_name": "vector",
            "source": "kb",
            "metadata_filter": None,
        }
    )
    assert len(result["documents"]) == 2
    assert all(d["source"] == "vector" for d in result["documents"])


def test_retrieve_exception_best_effort(isolated_registries_fx, caplog) -> None:
    """retriever 抛错时 best-effort 返回空 documents 并 log.error。"""
    mock_fn = MagicMock(side_effect=RuntimeError("simulated"))
    retriever_registry.register("vector", mock_fn)
    with caplog.at_level(logging.ERROR, logger="ragline.engine.nodes.retrieve"):
        result = retrieve(
            {
                "query": "q",
                "retriever_name": "vector",
                "source": "kb",
                "metadata_filter": None,
            }
        )
    assert result == {"documents": []}
    assert any("simulated" in r.getMessage() for r in caplog.records)


def test_retrieve_metadata_filter_passthrough(isolated_registries_fx) -> None:
    mock_fn = MagicMock(return_value=[])
    retriever_registry.register("vector", mock_fn)
    retrieve(
        {
            "query": "q",
            "retriever_name": "vector",
            "source": "kb",
            "metadata_filter": {"year": 2024},
        }
    )
    # 检查 fn 被以 metadata_filter={"year": 2024} 调用
    mock_fn.assert_called_once_with(query="q", metadata_filter={"year": 2024})
