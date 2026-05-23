"""tests/unit/test_handlers/test_graders.py — graders 处理器单元测试。"""

import pytest

from ragline.api.types import Document
from ragline.registry import grader_registry


def make_doc(doc_id: str, score: float, source: str) -> Document:
    return Document(doc_id=doc_id, content="x", score=score, source=source, metadata={})


# ===== 测试 1: normalize_score bm25 =====


def test_normalize_score_bm25():
    from ragline.handlers.graders.normalizer import normalize_score

    doc = make_doc("d1", score=15.0, source="bm25")
    assert normalize_score(doc, {"bm25": 20.0}) == pytest.approx(0.75)


# ===== 测试 2: normalize_score vector 默认 max=1 =====


def test_normalize_score_vector_default_max():
    from ragline.handlers.graders.normalizer import normalize_score

    doc = make_doc("d2", score=0.9, source="vector")
    assert normalize_score(doc, {}) == pytest.approx(0.9)


# ===== 测试 3: 截断到 1.0 =====


def test_normalize_score_clamp_to_1():
    from ragline.handlers.graders.normalizer import normalize_score

    doc = make_doc("d3", score=25.0, source="bm25")
    assert normalize_score(doc, {"bm25": 20.0}) == pytest.approx(1.0)


# ===== 测试 4: max_score=0 返回 0 =====


def test_normalize_score_zero_max():
    from ragline.handlers.graders.normalizer import normalize_score

    doc = make_doc("d4", score=10.0, source="bm25")
    assert normalize_score(doc, {"bm25": 0}) == pytest.approx(0.0)


# ===== 测试 5: score_grader 充分 =====


def test_score_grader_sufficient():
    from ragline.handlers.graders.score import score_grader

    docs = [
        make_doc("d1", score=0.6, source="vector"),
        make_doc("d2", score=0.7, source="vector"),
        make_doc("d3", score=0.8, source="vector"),
    ]
    avg, decision = score_grader(docs, "q", threshold=0.6)
    assert avg == pytest.approx(0.7)
    assert decision == "sufficient"


# ===== 测试 6: score_grader 不足 =====


def test_score_grader_insufficient():
    from ragline.handlers.graders.score import score_grader

    docs = [
        make_doc("d1", score=0.3, source="vector"),
        make_doc("d2", score=0.4, source="vector"),
        make_doc("d3", score=0.5, source="vector"),
    ]
    avg, decision = score_grader(docs, "q", threshold=0.6)
    assert avg == pytest.approx(0.4)
    assert decision == "insufficient"


# ===== 测试 7: score_grader 空 docs =====


def test_score_grader_empty_docs():
    from ragline.handlers.graders.score import score_grader

    avg, decision = score_grader([], "q", threshold=0.6)
    assert avg == pytest.approx(0.0)
    assert decision == "insufficient"


# ===== 测试 8: 跨 source 混合归一化 =====


def test_score_grader_mixed_sources():
    from ragline.handlers.graders.score import score_grader

    docs = [
        make_doc("d1", score=0.6, source="vector"),
        make_doc("d2", score=0.8, source="vector"),
        make_doc("d3", score=15.0, source="bm25"),
        make_doc("d4", score=18.0, source="bm25"),
    ]
    # 归一化后：0.6, 0.8, 0.75, 0.9 → avg = 0.7625
    avg, decision = score_grader(docs, "q", threshold=0.5, score_normalizers={"bm25": 20.0})
    assert avg == pytest.approx(0.7625)
    assert decision == "sufficient"


# ===== 测试 9: register_graders 注册 =====


def test_register_graders_registers_score(isolated_registries_fx):
    from ragline.handlers.graders import register_graders

    register_graders()
    assert grader_registry.has("score")


# ===== 测试 10: 边界 threshold 相等 =====


def test_score_grader_threshold_boundary_equal():
    from ragline.handlers.graders.score import score_grader

    docs = [
        make_doc("d1", score=0.6, source="vector"),
    ]
    avg, decision = score_grader(docs, "q", threshold=0.6)
    assert avg == pytest.approx(0.6)
    assert decision == "sufficient"


# ===== 测试 11: negative score 截断到 0 =====


def test_normalize_score_negative_clamp_to_0():
    from ragline.handlers.graders.normalizer import normalize_score

    doc = make_doc("d5", score=-5.0, source="bm25")
    assert normalize_score(doc, {"bm25": 20.0}) == pytest.approx(0.0)
