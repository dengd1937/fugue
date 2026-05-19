"""tests/unit/test_providers/test_bm25.py — BM25Provider 单元测试。"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed

import pytest

from ragline.api.types import Chunk
from ragline.providers.bm25 import BM25Provider


def _make_chunk(i: int, content: str) -> Chunk:
    return Chunk(chunk_id=f"c{i}", parent_id=None, content=content, metadata={})


# ──────────────────────────────────────────────
# 测试 1: rebuild + search 基础
# ──────────────────────────────────────────────
def test_rebuild_and_search_basic() -> None:
    chunks = [
        _make_chunk(0, "langchain is a framework for building llm applications"),
        _make_chunk(1, "langgraph extends langchain with graph based workflows"),
        _make_chunk(2, "chroma is a vector database for embeddings"),
        _make_chunk(3, "openai provides gpt models via api"),
        _make_chunk(4, "python is a popular programming language"),
    ]
    provider = BM25Provider()
    provider.rebuild(chunks)

    results = provider.search("langchain")
    assert len(results) > 0
    assert "langchain" in results[0]["content"]
    for doc in results:
        assert doc["source"] == "bm25"
        assert isinstance(doc["score"], float)


# ──────────────────────────────────────────────
# 测试 2: 空索引 search 不抛错
# ──────────────────────────────────────────────
def test_search_empty_index_returns_empty() -> None:
    provider = BM25Provider()
    # 未 rebuild，_bm25 is None
    result = provider.search("anything")
    assert result == []


# ──────────────────────────────────────────────
# 测试 3: rebuild 空列表后 search 返回空
# ──────────────────────────────────────────────
def test_rebuild_empty_list() -> None:
    chunks = [_make_chunk(0, "some content")]
    provider = BM25Provider()
    provider.rebuild(chunks)
    provider.rebuild([])
    result = provider.search("some")
    assert result == []


# ──────────────────────────────────────────────
# 测试 4: update 触发全量重建（增量追加）
# ──────────────────────────────────────────────
def test_update_appends_and_rebuilds() -> None:
    initial_chunks = [
        _make_chunk(0, "alpha is the first letter"),
        _make_chunk(1, "beta is the second letter"),
        _make_chunk(2, "gamma is the third letter"),
    ]
    new_chunks = [
        _make_chunk(3, "delta is the fourth letter"),
        _make_chunk(4, "epsilon is the fifth letter"),
    ]
    provider = BM25Provider()
    provider.rebuild(initial_chunks)
    provider.update(new_chunks)

    # 通过 search 大 k 验证 5 个 chunk 都在索引中
    results = provider.search("letter", k=10)
    assert len(results) == 5
    contents = {r["content"] for r in results}
    assert any("alpha" in c for c in contents)
    assert any("delta" in c for c in contents)
    assert any("epsilon" in c for c in contents)


# ──────────────────────────────────────────────
# 测试 5: k 截断
# ──────────────────────────────────────────────
def test_search_k_truncation() -> None:
    chunks = [
        _make_chunk(0, "keyword document one with keyword"),
        _make_chunk(1, "keyword document two with keyword"),
        _make_chunk(2, "keyword document three with keyword"),
        _make_chunk(3, "keyword document four with keyword"),
        _make_chunk(4, "keyword document five with keyword"),
    ]
    provider = BM25Provider()
    provider.rebuild(chunks)

    results = provider.search("keyword", k=2)
    assert len(results) <= 2


# ──────────────────────────────────────────────
# 测试 6: 并发 search 不报错
# ──────────────────────────────────────────────
def test_concurrent_search_thread_safe() -> None:
    chunks = [_make_chunk(i, f"concurrent test content topic {i} with search keyword") for i in range(10)]
    provider = BM25Provider()
    provider.rebuild(chunks)

    def do_search() -> int:
        results = provider.search("keyword", k=5)
        return len(results)

    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = [executor.submit(do_search) for _ in range(3)]
        lengths = [f.result() for f in as_completed(futures)]

    assert len(lengths) == 3
    for length in lengths:
        assert length > 0


# ──────────────────────────────────────────────
# 测试 7: 性能基线（slow marker）
# 普通 macOS/Linux dev 机器预期：rebuild 10k chunks 约 1-2 秒，远低于 5 秒阈值
# ──────────────────────────────────────────────
@pytest.mark.slow
def test_rebuild_performance_baseline() -> None:
    import time

    chunks = [
        Chunk(
            chunk_id=f"c{i}",
            parent_id=None,
            content=f"doc {i} content with keywords {i % 100}",
            metadata={},
        )
        for i in range(10_000)
    ]
    provider = BM25Provider()
    start = time.perf_counter()
    provider.rebuild(chunks)
    elapsed = time.perf_counter() - start
    assert elapsed < 5.0, f"rebuild 10k took {elapsed:.2f}s, expected < 5s"
    # 验证还能正常 search
    results = provider.search("keywords", k=10)
    assert len(results) > 0


# ──────────────────────────────────────────────
# 测试 8: 中文已知失效（xfail，MVP 空白分词不支持中文）
# P1 引入 jieba 后此测试将 XPASS，提醒升级
# ──────────────────────────────────────────────
@pytest.mark.xfail(reason="MVP 用空白分词，中文不支持；P1 加 jieba")
def test_chinese_tokenization_known_limitation() -> None:
    """已知限制：中文用空白分词查询失效。P1 引入 jieba 后此测试自动 XPASS 提醒。"""
    chunks = [
        Chunk(chunk_id=f"c{i}", parent_id=None, content=text, metadata={})
        for i, text in enumerate(
            [
                "向量数据库存储嵌入",
                "大语言模型生成回答",
                "检索增强生成是一种技术",
                "分块策略影响召回率",
                "重排序提升精度",
            ]
        )
    ]
    provider = BM25Provider()
    provider.rebuild(chunks)
    # 中文查询"向量数据库"——MVP 空白分词应找不到有效匹配（score > 0）
    results = provider.search("向量数据库", k=3)
    # 期望：有分数大于 0 的文档（中文字符级别匹配才算真正检索到）
    # MVP 空白分词把整句当单一 token，查询 token 与文档 token 不同，score=0，故 xfail
    positive_score_results = [r for r in results if r["score"] > 0.0]
    assert len(positive_score_results) > 0
    assert "向量数据库" in positive_score_results[0]["content"]


# ──────────────────────────────────────────────
# 测试 9: close 清理
# ──────────────────────────────────────────────
def test_close_clears_index() -> None:
    chunks = [_make_chunk(i, f"content topic {i}") for i in range(5)]
    provider = BM25Provider()
    provider.rebuild(chunks)

    # 确认 rebuild 后有结果
    assert len(provider.search("content")) > 0

    provider.close()
    # close 后搜索返回空
    assert provider.search("content") == []

    # 再次 close 不抛错
    provider.close()
    assert provider.search("content") == []
