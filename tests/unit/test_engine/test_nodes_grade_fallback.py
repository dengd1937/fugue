"""tests/unit/test_engine/test_nodes_grade_fallback.py — grade + fallback 节点测试。"""

import ragline.handlers.graders  # noqa: F401  — 触发 score grader 自动注册
from ragline.api.types import Document
from ragline.engine.nodes.grade import grade, route_after_grade
from ragline.engine.nodes.prepare_fallback import prepare_fallback
from ragline.engine.state import RAGState


def _doc(source: str, doc_id: str, score: float = 0.9) -> Document:
    return Document(doc_id=doc_id, content="x", score=score, source=source, metadata={})


def _state(
    *,
    docs: list[Document] | None = None,
    decision: str = "insufficient",
    retry_count: int = 0,
    history: list[list[Document]] | None = None,
) -> RAGState:
    return RAGState(
        original_query="q",
        rewritten_queries=[],
        documents=docs or [],
        grade_score=0.0,
        grade_decision=decision,  # type: ignore[typeddict-item]
        source="kb",
        retry_count=retry_count,
        retrieval_history=history or [],
        ranked_documents=[],
        answer="",
    )


# grade 测试 ----------------------------------------------------------


def test_grade_basic_sufficient() -> None:
    """grade 应返回 dict 含 grade_score / grade_decision / retrieval_history。"""
    docs = [
        _doc("vector", "1", score=0.8),
        _doc("vector", "2", score=0.9),
    ]
    config = {
        "configurable": {
            "grade_strategy": "score",
            "grade_threshold": 0.6,
            "score_normalizers": {},
        }
    }
    result = grade(_state(docs=docs), config)  # type: ignore[arg-type]
    assert "grade_score" in result
    assert "grade_decision" in result
    assert "retrieval_history" in result
    assert result["grade_decision"] == "sufficient"


def test_grade_appends_to_retrieval_history() -> None:
    """grade 应把当前 documents append 到 retrieval_history 末尾。"""
    docs1 = [_doc("vector", "1")]
    docs2 = [_doc("bm25", "2")]
    state = _state(docs=docs2, history=[docs1])
    config = {"configurable": {"grade_strategy": "score", "grade_threshold": 0.5}}
    result = grade(state, config)  # type: ignore[arg-type]
    assert len(result["retrieval_history"]) == 2
    assert result["retrieval_history"][0] == docs1
    assert result["retrieval_history"][1] == docs2


def test_grade_empty_documents() -> None:
    """空 docs 时 decision 为 insufficient，history 含一项空 list。"""
    config = {"configurable": {"grade_strategy": "score", "grade_threshold": 0.6}}
    result = grade(_state(docs=[]), config)  # type: ignore[arg-type]
    assert result["grade_decision"] == "insufficient"
    assert result["grade_score"] == 0.0
    assert result["retrieval_history"] == [[]]


# route_after_grade 测试 ----------------------------------------------


def test_route_sufficient_goes_to_post_process() -> None:
    config = {"configurable": {"fallback_chain": ["web"], "max_retries": 1}}
    assert route_after_grade(_state(decision="sufficient"), config) == "post_process"  # type: ignore[arg-type]


def test_route_insufficient_empty_fallback_goes_to_post_process() -> None:
    config = {"configurable": {"fallback_chain": [], "max_retries": 1}}
    assert route_after_grade(_state(decision="insufficient"), config) == "post_process"  # type: ignore[arg-type]


def test_route_insufficient_with_fallback_available() -> None:
    config = {"configurable": {"fallback_chain": ["web"], "max_retries": 1}}
    assert (
        route_after_grade(
            _state(decision="insufficient", retry_count=0),
            config,  # type: ignore[arg-type]
        )
        == "fallback_to_query_transform"
    )


def test_route_insufficient_fallback_exhausted_by_count() -> None:
    """retry_count 已等于 fallback_chain 长度，无下一个 source 可用。"""
    config = {"configurable": {"fallback_chain": ["web"], "max_retries": 5}}
    assert (
        route_after_grade(
            _state(decision="insufficient", retry_count=1),
            config,  # type: ignore[arg-type]
        )
        == "post_process"
    )


def test_route_insufficient_max_retries_reached() -> None:
    """retry_count 已达 max_retries 上限。"""
    config = {"configurable": {"fallback_chain": ["web", "kg"], "max_retries": 1}}
    assert (
        route_after_grade(
            _state(decision="insufficient", retry_count=1),
            config,  # type: ignore[arg-type]
        )
        == "post_process"
    )


# prepare_fallback 测试 -----------------------------------------------


def test_prepare_fallback_first_step() -> None:
    config = {"configurable": {"fallback_chain": ["web", "kg"]}}
    result = prepare_fallback(_state(retry_count=0), config)  # type: ignore[arg-type]
    assert result == {"source": "web", "retry_count": 1}


def test_prepare_fallback_second_step() -> None:
    config = {"configurable": {"fallback_chain": ["web", "kg"]}}
    result = prepare_fallback(_state(retry_count=1), config)  # type: ignore[arg-type]
    assert result == {"source": "kg", "retry_count": 2}
