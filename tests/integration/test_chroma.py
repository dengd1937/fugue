"""tests/integration/test_chroma.py — ChromaVectorStore 集成测试（真实 Chroma，无 mock）。"""

import logging

import pytest

from fugue.api.types import Chunk
from fugue.providers.vector_store import ChromaVectorStore

logging.getLogger("chromadb").setLevel(logging.WARNING)


def _make_chunk(i: int, metadata: dict | None = None, parent_id: str | None = None) -> Chunk:
    return Chunk(
        chunk_id=f"c{i}",
        parent_id=parent_id,
        content=f"text {i}",
        metadata=metadata or {},
    )


def _make_embedding(i: int, dim: int = 8) -> list[float]:
    """生成简单的 8 维 embedding，保证向量间有差异。"""
    return [float(i) * 0.1 + j * 0.01 for j in range(dim)]


# ---------- 测试 1: add + similarity_search 基础 ----------


def test_add_and_similarity_search_basic(tmp_path: pytest.TempPathFactory) -> None:
    store = ChromaVectorStore(persist_dir=str(tmp_path), collection_name="test_basic")

    chunks = [_make_chunk(i, {"year": 2024}) for i in range(1, 6)]
    embeddings = [_make_embedding(i) for i in range(1, 6)]

    store.add(chunks, embeddings)

    query_embedding = _make_embedding(3)
    results = store.similarity_search(query_embedding, k=3)

    assert len(results) == 3
    for doc in results:
        assert 0.0 <= doc["score"] <= 1.0
        assert doc["source"] == "vector"

    store.close()


# ---------- 测试 2: upsert 行为 ----------


def test_upsert_behavior(tmp_path: pytest.TempPathFactory) -> None:
    store = ChromaVectorStore(persist_dir=str(tmp_path), collection_name="test_upsert")

    # 首次写入 3 个 chunks
    chunks_v1 = [_make_chunk(i) for i in range(1, 4)]
    embeddings = [_make_embedding(i) for i in range(1, 4)]
    store.add(chunks_v1, embeddings)

    # 用相同 chunk_id 写入不同 content
    chunks_v2 = [Chunk(chunk_id=f"c{i}", parent_id=None, content=f"updated text {i}", metadata={}) for i in range(1, 4)]
    store.add(chunks_v2, embeddings)

    stats = store.stats()
    assert stats["num_chunks"] == 3  # upsert：不增加

    # 检索返回的内容应为新内容
    results = store.similarity_search(_make_embedding(2), k=3)
    contents = {r["content"] for r in results}
    assert all(c.startswith("updated text") for c in contents)

    store.close()


# ---------- 测试 3: metadata_filter ----------


def test_metadata_filter(tmp_path: pytest.TempPathFactory) -> None:
    store = ChromaVectorStore(persist_dir=str(tmp_path), collection_name="test_filter")

    chunks = [
        Chunk(chunk_id="c1", parent_id=None, content="text 1", metadata={"year": 2024}),
        Chunk(chunk_id="c2", parent_id=None, content="text 2", metadata={"year": 2023}),
        Chunk(chunk_id="c3", parent_id=None, content="text 3", metadata={"year": 2024}),
    ]
    embeddings = [_make_embedding(i) for i in range(1, 4)]
    store.add(chunks, embeddings)

    # 带 filter：仅返回 year=2024 的 2 个
    filtered = store.similarity_search(_make_embedding(2), k=10, metadata_filter={"year": 2024})
    assert len(filtered) == 2
    for doc in filtered:
        assert doc["metadata"]["year"] == 2024

    # 不带 filter：返回全部 3 个
    all_results = store.similarity_search(_make_embedding(2), k=10)
    assert len(all_results) == 3

    store.close()


# ---------- 测试 4: stats ----------


def test_stats(tmp_path: pytest.TempPathFactory) -> None:
    store = ChromaVectorStore(persist_dir=str(tmp_path), collection_name="test_stats")

    n = 7
    chunks = [_make_chunk(i) for i in range(n)]
    embeddings = [_make_embedding(i) for i in range(n)]
    store.add(chunks, embeddings)

    stats = store.stats()
    assert stats["num_chunks"] == n
    assert "collection_name" in stats
    assert stats["collection_name"] == "test_stats"

    store.close()


# ---------- 测试 5: iter_chunks 分批（小规模） ----------


def test_iter_chunks_batching(tmp_path: pytest.TempPathFactory) -> None:
    store = ChromaVectorStore(persist_dir=str(tmp_path), collection_name="test_iter")

    chunks = [_make_chunk(i) for i in range(5)]
    embeddings = [_make_embedding(i) for i in range(5)]
    store.add(chunks, embeddings)

    batches = list(store.iter_chunks(batch_size=2))

    assert len(batches) == 3  # 2 + 2 + 1
    total = sum(len(b) for b in batches)
    assert total == 5

    for batch in batches:
        for chunk in batch:
            assert isinstance(chunk, Chunk)

    store.close()


# ---------- 测试 6: iter_chunks 元数据保留 ----------


def test_iter_chunks_metadata_preserved(tmp_path: pytest.TempPathFactory) -> None:
    store = ChromaVectorStore(persist_dir=str(tmp_path), collection_name="test_iter_meta")

    chunks = [
        Chunk(chunk_id="c1", parent_id="p1", content="text with parent", metadata={"k": "v"}),
        Chunk(chunk_id="c2", parent_id=None, content="text without parent", metadata={"k": "w"}),
    ]
    embeddings = [_make_embedding(i) for i in range(1, 3)]
    store.add(chunks, embeddings)

    all_chunks: list[Chunk] = []
    for batch in store.iter_chunks(batch_size=10):
        all_chunks.extend(batch)

    assert len(all_chunks) == 2
    by_id = {c.chunk_id: c for c in all_chunks}

    assert by_id["c1"].parent_id == "p1"
    assert by_id["c1"].metadata["k"] == "v"
    assert by_id["c2"].parent_id is None
    assert by_id["c2"].metadata["k"] == "w"

    store.close()


# ---------- 测试 7: 空查询 ----------


def test_empty_collection_search(tmp_path: pytest.TempPathFactory) -> None:
    store = ChromaVectorStore(persist_dir=str(tmp_path), collection_name="test_empty")

    results = store.similarity_search(_make_embedding(1), k=5)
    assert results == []

    stats = store.stats()
    assert stats["num_chunks"] == 0

    store.close()


# ---------- 测试 8: delete_collection ----------


def test_delete_collection(tmp_path: pytest.TempPathFactory) -> None:
    store = ChromaVectorStore(persist_dir=str(tmp_path), collection_name="test_delete")

    chunks = [_make_chunk(i) for i in range(3)]
    embeddings = [_make_embedding(i) for i in range(3)]
    store.add(chunks, embeddings)

    assert store.stats()["num_chunks"] == 3

    store.delete_collection()
    assert store.stats()["num_chunks"] == 0

    # 再次 add 不应抛错
    store.add(chunks, embeddings)
    assert store.stats()["num_chunks"] == 3

    store.close()


# ---------- 测试 9: 混合 metadata 处理 ----------


def test_mixed_metadata_handling(tmp_path: pytest.TempPathFactory) -> None:
    """混合 metadata（空/非空）不应报错，且 iter_chunks 拿回的 metadata 与原始一致。"""
    store = ChromaVectorStore(persist_dir=str(tmp_path), collection_name="test_mixed_meta")

    chunks = [
        Chunk(chunk_id="c1", parent_id=None, content="no metadata", metadata={}),
        Chunk(chunk_id="c2", parent_id=None, content="has metadata k=v", metadata={"k": "v"}),
        Chunk(chunk_id="c3", parent_id=None, content="has metadata k=v2", metadata={"k": "v2"}),
    ]
    embeddings = [_make_embedding(i) for i in range(1, 4)]

    # 不应抛错
    store.add(chunks, embeddings)

    # similarity_search 能正常返回 3 个
    results = store.similarity_search(_make_embedding(2), k=10)
    assert len(results) == 3

    # iter_chunks 拿回的 metadata 与原始一致
    all_chunks: list[Chunk] = []
    for batch in store.iter_chunks(batch_size=10):
        all_chunks.extend(batch)

    assert len(all_chunks) == 3
    by_id = {c.chunk_id: c for c in all_chunks}

    # 空 metadata 仍是空 dict，不带 _empty 内部 key
    assert by_id["c1"].metadata == {}
    assert "_empty" not in by_id["c1"].metadata

    # 非空 metadata 保持原值
    assert by_id["c2"].metadata == {"k": "v"}
    assert by_id["c3"].metadata == {"k": "v2"}

    store.close()
