"""tests/test_handlers/test_processors.py — processors handler 测试。"""

import copy
from typing import Any

import pytest

from rag.types import Document

# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------


def _doc(
    doc_id: str,
    content: str = "content",
    score: float = 0.5,
    source: str = "vector",
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
# _normalize_score 测试
# ---------------------------------------------------------------------------


class TestNormalizeScore:
    def test_normal_case(self) -> None:
        from rag.handlers.processors import _normalize_score

        doc = _doc("d1", score=10.0, source="es")
        result = _normalize_score(doc, {"es": 20.0})
        assert result == pytest.approx(0.5)

    def test_max_zero_returns_zero(self) -> None:
        from rag.handlers.processors import _normalize_score

        doc = _doc("d1", score=5.0, source="es")
        result = _normalize_score(doc, {"es": 0.0})
        assert result == 0.0

    def test_unregistered_source_uses_default_max_one(self) -> None:
        from rag.handlers.processors import _normalize_score

        doc = _doc("d1", score=0.8, source="unknown_source")
        result = _normalize_score(doc, {})
        # 默认 max=1.0，score=0.8 → min(0.8/1.0, 1.0) = 0.8
        assert result == pytest.approx(0.8)

    def test_score_exceeds_max_clamped_to_one(self) -> None:
        from rag.handlers.processors import _normalize_score

        doc = _doc("d1", score=25.0, source="es")
        result = _normalize_score(doc, {"es": 20.0})
        assert result == pytest.approx(1.0)

    def test_zero_score(self) -> None:
        from rag.handlers.processors import _normalize_score

        doc = _doc("d1", score=0.0, source="vector")
        result = _normalize_score(doc, {"vector": 1.0})
        assert result == 0.0


# ---------------------------------------------------------------------------
# rrf_fn 测试（手算对拍）
# ---------------------------------------------------------------------------


class TestRrfFn:
    def test_single_source_ranking(self) -> None:
        from rag.handlers.processors import rrf_fn

        docs = [
            _doc("d1", score=0.9, source="vector"),
            _doc("d2", score=0.7, source="vector"),
        ]
        result = rrf_fn(docs, query="q", top_k=2)
        # d1 rank=1(0-indexed 0), d2 rank=2(0-indexed 1)
        # d1 rrf = 1/(60+1) ≈ 0.01639
        # d2 rrf = 1/(60+2) ≈ 0.01613
        assert result[0]["doc_id"] == "d1"
        assert result[1]["doc_id"] == "d2"

    def test_two_sources_equal_weight_hand_calc(self) -> None:
        from rag.handlers.processors import rrf_fn

        # source_a: doc_a rank=1, doc_b rank=2
        # source_b: doc_b rank=1, doc_a rank=2
        # hand calc (w=1.0, k=60):
        #   doc_a: 1/(60+1) + 1/(60+2) = 1/61 + 1/62 ≈ 0.016393 + 0.016129 ≈ 0.032522
        #   doc_b: 1/(60+2) + 1/(60+1) = 1/62 + 1/61 ≈ 0.016129 + 0.016393 ≈ 0.032522
        docs = [
            _doc("doc_a", score=0.9, source="source_a"),
            _doc("doc_b", score=0.7, source="source_a"),
            _doc("doc_b", score=0.95, source="source_b"),
            _doc("doc_a", score=0.5, source="source_b"),
        ]
        result = rrf_fn(docs, query="q", top_k=2)
        # 两个文档 rrf_score 相等，返回 top_k=2
        assert len(result) == 2
        doc_ids = {d["doc_id"] for d in result}
        assert doc_ids == {"doc_a", "doc_b"}

    def test_rrf_score_written_to_metadata(self) -> None:
        from rag.handlers.processors import rrf_fn

        docs = [_doc("d1", score=0.9, source="vector")]
        result = rrf_fn(docs, query="q", top_k=1)
        assert "rrf_score" in result[0]["metadata"]

    def test_top_k_limits_output(self) -> None:
        from rag.handlers.processors import rrf_fn

        docs = [_doc(str(i), score=float(i) / 10, source="vector") for i in range(10)]
        result = rrf_fn(docs, query="q", top_k=3)
        assert len(result) == 3

    def test_top_k_larger_than_docs_returns_all(self) -> None:
        from rag.handlers.processors import rrf_fn

        docs = [_doc("d1", score=0.5, source="v")]
        result = rrf_fn(docs, query="q", top_k=10)
        assert len(result) == 1

    def test_empty_docs_returns_empty(self) -> None:
        from rag.handlers.processors import rrf_fn

        result = rrf_fn([], query="q", top_k=3)
        assert result == []

    def test_retriever_weights_applied(self) -> None:
        from rag.handlers.processors import rrf_fn

        # source_a weight=2.0, source_b weight=1.0
        # doc_x only in source_a rank=1: rrf = 2.0/(60+1) ≈ 0.03279
        # doc_y only in source_b rank=1: rrf = 1.0/(60+1) ≈ 0.01639
        docs = [
            _doc("doc_x", score=0.5, source="source_a"),
            _doc("doc_y", score=0.5, source="source_b"),
        ]
        result = rrf_fn(
            docs, query="q", top_k=2, retriever_weights={"source_a": 2.0, "source_b": 1.0}
        )
        assert result[0]["doc_id"] == "doc_x"

    def test_stability_same_result_on_repeated_calls(self) -> None:
        from rag.handlers.processors import rrf_fn

        docs = [
            _doc("d1", score=0.9, source="v"),
            _doc("d2", score=0.5, source="v"),
            _doc("d3", score=0.3, source="es"),
        ]
        docs_copy = copy.deepcopy(docs)
        result1 = rrf_fn(docs, query="q", top_k=3)
        result2 = rrf_fn(docs_copy, query="q", top_k=3)
        assert [d["doc_id"] for d in result1] == [d["doc_id"] for d in result2]

    def test_does_not_mutate_input(self) -> None:
        from rag.handlers.processors import rrf_fn

        docs = [_doc("d1", score=0.9, source="v")]
        original_metadata = dict(docs[0]["metadata"])
        rrf_fn(docs, query="q", top_k=1)
        # 原始文档的 metadata 不应被修改（实现应创建新 doc）
        assert docs[0]["metadata"] == original_metadata

    def test_rrf_score_calculation_single_doc(self) -> None:
        """手算验证单文档 RRF 分数：rank=0 → w/(60+1) = 1/61。"""
        from rag.handlers.processors import rrf_fn

        docs = [_doc("d1", score=0.9, source="vector")]
        result = rrf_fn(docs, query="q", top_k=1)
        expected_rrf = 1.0 / (60 + 1)
        assert result[0]["metadata"]["rrf_score"] == pytest.approx(expected_rrf)

    def test_cross_source_same_doc_id_both_preserved(self) -> None:
        """回归：不同 source 下相同 doc_id 的文档不应相互合并或覆盖。"""
        from rag.handlers.processors import rrf_fn

        docs = [
            _doc("shared_id", score=0.9, source="vector"),
            _doc("shared_id", score=0.8, source="es"),
        ]
        result = rrf_fn(docs, query="q", top_k=10)
        assert len(result) == 2, "不同 source 的同名 doc_id 应各自保留"
        sources = {d["source"] for d in result}
        assert sources == {"vector", "es"}


# ---------------------------------------------------------------------------
# weighted_fusion_fn 测试
# ---------------------------------------------------------------------------


class TestWeightedFusionFn:
    def test_higher_weight_source_ranks_higher(self) -> None:
        from rag.handlers.processors import weighted_fusion_fn

        docs = [
            _doc("low_doc", score=0.3, source="low_weight_source"),
            _doc("high_doc", score=0.3, source="high_weight_source"),
        ]
        result = weighted_fusion_fn(
            docs,
            query="q",
            top_k=2,
            retriever_weights={"high_weight_source": 2.0, "low_weight_source": 0.5},
        )
        assert result[0]["doc_id"] == "high_doc"

    def test_returns_top_k(self) -> None:
        from rag.handlers.processors import weighted_fusion_fn

        docs = [_doc(str(i), score=float(i) / 10, source="v") for i in range(5)]
        result = weighted_fusion_fn(docs, query="q", top_k=2)
        assert len(result) == 2

    def test_empty_docs(self) -> None:
        from rag.handlers.processors import weighted_fusion_fn

        result = weighted_fusion_fn([], query="q", top_k=3)
        assert result == []

    def test_cross_source_same_doc_id_both_preserved(self) -> None:
        """回归：不同 source 下相同 doc_id 的文档不应相互覆盖。"""
        from rag.handlers.processors import weighted_fusion_fn

        docs = [
            _doc("shared_id", score=0.9, source="vector"),
            _doc("shared_id", score=0.8, source="es"),
        ]
        result = weighted_fusion_fn(docs, query="q", top_k=10)
        assert len(result) == 2, "不同 source 的同名 doc_id 应各自保留"
        sources = {d["source"] for d in result}
        assert sources == {"vector", "es"}


# ---------------------------------------------------------------------------
# rerank_fn 测试
# ---------------------------------------------------------------------------


class TestRerankFn:
    def test_rerank_with_fake_cross_encoder(self) -> None:
        from rag.handlers.processors import rerank_fn

        class FakeCrossEncoder:
            def predict(self, pairs: list[list[str]]) -> list[float]:
                # 返回固定分数：第二个文档得分更高
                return [0.3, 0.9]

        docs = [_doc("d1", score=0.5), _doc("d2", score=0.3)]
        result = rerank_fn(docs, query="q", top_k=2, cross_encoder=FakeCrossEncoder())
        assert result[0]["doc_id"] == "d2"

    def test_rerank_no_cross_encoder_returns_original_order(self) -> None:
        from rag.handlers.processors import rerank_fn

        docs = [_doc("d1", score=0.9), _doc("d2", score=0.5)]
        # 没有 cross_encoder 时，fallback 到原始顺序
        result = rerank_fn(docs, query="q", top_k=2)
        assert len(result) == 2

    def test_returns_top_k(self) -> None:
        from rag.handlers.processors import rerank_fn

        class FakeCrossEncoder:
            def predict(self, pairs: list[list[str]]) -> list[float]:
                return [float(i) for i in range(len(pairs))]

        docs = [_doc(str(i)) for i in range(5)]
        result = rerank_fn(docs, query="q", top_k=2, cross_encoder=FakeCrossEncoder())
        assert len(result) == 2


# ---------------------------------------------------------------------------
# filter_fn 测试
# ---------------------------------------------------------------------------


class TestFilterFn:
    def test_filter_by_score_threshold(self) -> None:
        from rag.handlers.processors import filter_fn

        docs = [
            _doc("d1", score=0.8),
            _doc("d2", score=0.3),
            _doc("d3", score=0.6),
        ]
        result = filter_fn(docs, score_threshold=0.5)
        doc_ids = {d["doc_id"] for d in result}
        assert "d1" in doc_ids
        assert "d3" in doc_ids
        assert "d2" not in doc_ids

    def test_no_threshold_returns_all(self) -> None:
        from rag.handlers.processors import filter_fn

        docs = [_doc("d1", score=0.1), _doc("d2", score=0.9)]
        result = filter_fn(docs)
        assert len(result) == 2

    def test_empty_docs(self) -> None:
        from rag.handlers.processors import filter_fn

        result = filter_fn([])
        assert result == []


# ---------------------------------------------------------------------------
# dedupe_fn 测试
# ---------------------------------------------------------------------------


class TestDedupeFn:
    def test_deduplicates_by_doc_id_and_source(self) -> None:
        from rag.handlers.processors import dedupe_fn

        docs = [
            _doc("d1", source="vector"),
            _doc("d1", source="vector"),  # 重复
            _doc("d1", source="es"),  # 不同 source，保留
        ]
        result = dedupe_fn(docs)
        assert len(result) == 2

    def test_idempotent(self) -> None:
        from rag.handlers.processors import dedupe_fn

        docs = [_doc("d1", source="v"), _doc("d2", source="v")]
        result1 = dedupe_fn(docs)
        result2 = dedupe_fn(result1)
        assert [d["doc_id"] for d in result1] == [d["doc_id"] for d in result2]

    def test_empty_docs(self) -> None:
        from rag.handlers.processors import dedupe_fn

        result = dedupe_fn([])
        assert result == []

    def test_preserves_order_of_first_occurrence(self) -> None:
        from rag.handlers.processors import dedupe_fn

        docs = [
            _doc("d2", source="v", score=0.5),
            _doc("d1", source="v", score=0.9),
            _doc("d2", source="v", score=0.5),  # 重复
        ]
        result = dedupe_fn(docs)
        assert result[0]["doc_id"] == "d2"
        assert result[1]["doc_id"] == "d1"


# ---------------------------------------------------------------------------
# compression_fn 测试
# ---------------------------------------------------------------------------


class TestCompressionFn:
    def test_compression_with_fake_compressor(self) -> None:
        from rag.handlers.processors import compression_fn

        class FakeCompressor:
            def compress(self, content: str, query: str) -> str:
                return f"compressed:{content}"

        docs = [_doc("d1", content="original content")]
        result = compression_fn(docs, query="q", compressor=FakeCompressor())
        assert result[0]["content"] == "compressed:original content"

    def test_no_compressor_returns_original(self) -> None:
        from rag.handlers.processors import compression_fn

        docs = [_doc("d1", content="original")]
        result = compression_fn(docs, query="q")
        assert result[0]["content"] == "original"

    def test_empty_docs(self) -> None:
        from rag.handlers.processors import compression_fn

        result = compression_fn([], query="q")
        assert result == []


# ---------------------------------------------------------------------------
# 注册验证
# ---------------------------------------------------------------------------


class TestProcessorRegistration:
    def test_all_processors_registered(self) -> None:
        import rag.handlers.processors  # noqa: F401
        from rag.registry import processor_registry

        for name in ["rrf", "weighted_fusion", "rerank", "filter", "dedupe", "compression"]:
            assert processor_registry.has(name), f"'{name}' not registered"
