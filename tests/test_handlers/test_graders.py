"""tests/test_handlers/test_graders.py — graders handler 测试。"""

from typing import Any

import pytest

from rag.types import Document

# ---------------------------------------------------------------------------
# FakeLLM stub
# ---------------------------------------------------------------------------


class FakeLLM:
    def __init__(self, response: str) -> None:
        self._response = response
        self.call_count = 0

    def invoke(self, prompt: str, temperature: float | None = None) -> str:
        self.call_count += 1
        return self._response


# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------


def _doc(
    doc_id: str = "d1",
    score: float = 0.5,
    source: str = "vector",
    content: str = "content",
    metadata: dict[str, Any] | None = None,
) -> Document:
    return Document(
        doc_id=doc_id,
        content=content,
        score=score,
        source=source,
        metadata=metadata or {},
    )


# ---------------------------------------------------------------------------
# _normalize_score 测试（graders 版本独立实现）
# ---------------------------------------------------------------------------


class TestNormalizeScoreInGraders:
    def test_normal_case(self) -> None:
        from rag.handlers.graders import _normalize_score

        doc = _doc(score=12.0, source="es")
        result = _normalize_score(doc, {"es": 20.0})
        assert result == pytest.approx(0.6)

    def test_max_zero_returns_zero(self) -> None:
        from rag.handlers.graders import _normalize_score

        doc = _doc(score=5.0, source="es")
        result = _normalize_score(doc, {"es": 0.0})
        assert result == 0.0

    def test_unregistered_source_default_max_one(self) -> None:
        from rag.handlers.graders import _normalize_score

        doc = _doc(score=0.7, source="unknown")
        result = _normalize_score(doc, {})
        assert result == pytest.approx(0.7)

    def test_score_clamped_to_one(self) -> None:
        from rag.handlers.graders import _normalize_score

        doc = _doc(score=30.0, source="es")
        result = _normalize_score(doc, {"es": 20.0})
        assert result == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# score_grader 测试
# ---------------------------------------------------------------------------


class TestScoreGrader:
    def test_empty_docs_returns_insufficient(self) -> None:
        from rag.handlers.graders import score_grader

        score, decision = score_grader([], threshold=0.6)
        assert score == 0.0
        assert decision == "insufficient"

    def test_single_doc_above_threshold(self) -> None:
        from rag.handlers.graders import score_grader

        doc = _doc(score=0.8)
        score, decision = score_grader([doc], threshold=0.6)
        assert score == pytest.approx(0.8)
        assert decision == "sufficient"

    def test_single_doc_below_threshold(self) -> None:
        from rag.handlers.graders import score_grader

        doc = _doc(score=0.4)
        score, decision = score_grader([doc], threshold=0.6)
        assert score == pytest.approx(0.4)
        assert decision == "insufficient"

    def test_single_doc_exactly_at_threshold(self) -> None:
        from rag.handlers.graders import score_grader

        doc = _doc(score=0.6)
        score, decision = score_grader([doc], threshold=0.6)
        assert score == pytest.approx(0.6)
        assert decision == "sufficient"

    def test_multiple_docs_uses_average(self) -> None:
        from rag.handlers.graders import score_grader

        docs = [_doc(score=0.8), _doc(score=0.4)]
        score, decision = score_grader(docs, threshold=0.6)
        # average = (0.8 + 0.4) / 2 = 0.6
        assert score == pytest.approx(0.6)
        assert decision == "sufficient"

    def test_score_normalizers_applied(self) -> None:
        from rag.handlers.graders import score_grader

        # ES 分数 12.0，normalizer=20.0 → normalized=0.6
        doc = _doc(score=12.0, source="es")
        score, decision = score_grader([doc], threshold=0.5, score_normalizers={"es": 20.0})
        assert score == pytest.approx(0.6)
        assert decision == "sufficient"

    def test_score_normalizers_multiple_sources(self) -> None:
        from rag.handlers.graders import score_grader

        # es: 12/20=0.6, vector: 0.8/1.0=0.8 → avg=0.7
        docs = [
            _doc(score=12.0, source="es"),
            _doc(score=0.8, source="vector"),
        ]
        score, decision = score_grader(docs, threshold=0.65, score_normalizers={"es": 20.0})
        assert score == pytest.approx(0.7)
        assert decision == "sufficient"

    def test_returns_tuple(self) -> None:
        from rag.handlers.graders import score_grader

        result = score_grader([_doc(score=0.5)], threshold=0.6)
        assert isinstance(result, tuple)
        assert len(result) == 2


# ---------------------------------------------------------------------------
# llm_grader 测试
# ---------------------------------------------------------------------------


class TestLlmGrader:
    def test_sufficient_response(self) -> None:
        from rag.handlers.graders import llm_grader

        llm = FakeLLM("sufficient")
        docs = [_doc(content="relevant content")]
        score, decision = llm_grader(docs, query="q", threshold=0.6, llm_client=llm)
        assert decision == "sufficient"

    def test_insufficient_response(self) -> None:
        from rag.handlers.graders import llm_grader

        llm = FakeLLM("insufficient")
        docs = [_doc(content="irrelevant content")]
        score, decision = llm_grader(docs, query="q", threshold=0.6, llm_client=llm)
        assert decision == "insufficient"

    def test_unparseable_response_fallback(self) -> None:
        from rag.handlers.graders import llm_grader

        llm = FakeLLM("some random text that is not yes/no")
        docs = [_doc()]
        score, decision = llm_grader(docs, query="q", threshold=0.6, llm_client=llm)
        # fallback: 返回 (0.0, "insufficient")
        assert decision == "insufficient"
        assert score == 0.0

    def test_llm_called_with_query(self) -> None:
        from rag.handlers.graders import llm_grader

        llm = FakeLLM("sufficient")
        docs = [_doc()]
        llm_grader(docs, query="test query", threshold=0.6, llm_client=llm)
        assert llm.call_count == 1

    def test_empty_docs_returns_insufficient(self) -> None:
        from rag.handlers.graders import llm_grader

        llm = FakeLLM("sufficient")
        score, decision = llm_grader([], query="q", threshold=0.6, llm_client=llm)
        assert decision == "insufficient"
        assert score == 0.0

    def test_returns_tuple(self) -> None:
        from rag.handlers.graders import llm_grader

        llm = FakeLLM("sufficient")
        result = llm_grader([_doc()], query="q", threshold=0.6, llm_client=llm)
        assert isinstance(result, tuple)
        assert len(result) == 2


# ---------------------------------------------------------------------------
# 注册验证
# ---------------------------------------------------------------------------


class TestGraderRegistration:
    def test_all_graders_registered(self) -> None:
        import rag.handlers.graders  # noqa: F401
        from rag.registry import grader_registry

        for name in ["score", "llm"]:
            assert grader_registry.has(name), f"'{name}' not registered"
