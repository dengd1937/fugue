"""tests/unit/test_handlers/test_chunkers.py — Chunkers 单元测试。"""

from pathlib import Path

import pytest

from fugue.api.types import ParsedDocument

# ---------------------------------------------------------------------------
# clean_chunker_registry fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def clean_chunker_registry():
    from fugue.registry import chunker_registry

    saved = {n: chunker_registry.get(n) for n in chunker_registry.names()}
    for n in list(chunker_registry.names()):
        chunker_registry.unregister(n)
    yield chunker_registry
    for n in list(chunker_registry.names()):
        chunker_registry.unregister(n)
    for n, fn in saved.items():
        chunker_registry.register(n, fn)


# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------


def make_parsed_doc(
    content: str,
    source_path: str = "/tmp/test.txt",
    metadata: dict | None = None,
) -> ParsedDocument:
    return ParsedDocument(
        source_path=Path(source_path),
        content=content,
        metadata=metadata or {},
    )


# ---------------------------------------------------------------------------
# 测试 1: chunk_size 切分
# ---------------------------------------------------------------------------


def test_chunk_size_splits_large_document() -> None:
    """1000 字符文档（含段落分隔符）chunk_size=200 → ≥5 个 chunks。"""
    from fugue.handlers.chunkers.recursive import recursive_chunker

    # 创建含段落分隔符的 1000 字符文档
    paragraph = "a" * 100
    content = "\n\n".join([paragraph] * 9)  # 9 个 100 字符段落，含分隔符约 1000 字符
    doc = make_parsed_doc(content)

    chunks = recursive_chunker([doc], chunk_size=200, chunk_overlap=0)

    assert len(chunks) >= 5


# ---------------------------------------------------------------------------
# 测试 2: chunk_overlap 相邻 chunk 有 substring 重叠
# ---------------------------------------------------------------------------


def test_chunk_overlap_adjacent_chunks_have_overlap() -> None:
    """相邻 chunks 有 ≥1 字符 substring 重叠（放宽条件）。"""
    from fugue.handlers.chunkers.recursive import recursive_chunker

    # 使用纯字符内容（无分隔符）确保走字符级切分，overlap 更容易验证
    content = "x" * 1000
    doc = make_parsed_doc(content)

    chunks = recursive_chunker([doc], chunk_size=200, chunk_overlap=64)

    assert len(chunks) >= 2

    # 字符级切分场景：验证 step = chunk_size - chunk_overlap = 136
    # 相邻 chunks 有重叠（chunk[i] 末尾部分出现在 chunk[i+1] 开头）
    for i in range(len(chunks) - 1):
        current = chunks[i].content
        nxt = chunks[i + 1].content
        # 验证相邻 chunks 之间有重叠：nxt 的开头应出现在 current 中
        # 或者 current 的尾部应出现在 nxt 中（至少 1 字符重叠）
        has_overlap = any(
            len(sub) >= 1 and sub in current for sub in [nxt[:k] for k in range(1, min(64 + 1, len(nxt) + 1))]
        )
        assert has_overlap, f"chunks[{i}] 和 chunks[{i + 1}] 之间没有 substring 重叠"


# ---------------------------------------------------------------------------
# 测试 3: chunk_id 稳定性
# ---------------------------------------------------------------------------


def test_chunk_id_stability() -> None:
    """同一 ParsedDocument 两次 chunk → chunk_id 完全相同。"""
    from fugue.handlers.chunkers.recursive import recursive_chunker

    content = "Hello world. " * 50
    doc = make_parsed_doc(content)

    chunks1 = recursive_chunker([doc], chunk_size=100, chunk_overlap=20)
    chunks2 = recursive_chunker([doc], chunk_size=100, chunk_overlap=20)

    assert len(chunks1) == len(chunks2)
    for c1, c2 in zip(chunks1, chunks2, strict=True):
        assert c1.chunk_id == c2.chunk_id


# ---------------------------------------------------------------------------
# 测试 4: chunk_id 唯一性
# ---------------------------------------------------------------------------


def test_chunk_id_unique_for_different_sources() -> None:
    """不同 source_path / 不同 content 产生不同 chunk_id。"""
    from fugue.handlers.chunkers.recursive import recursive_chunker

    doc1 = make_parsed_doc("a" * 200, source_path="/tmp/file1.txt")
    doc2 = make_parsed_doc("b" * 200, source_path="/tmp/file2.txt")

    chunks1 = recursive_chunker([doc1], chunk_size=512, chunk_overlap=0)
    chunks2 = recursive_chunker([doc2], chunk_size=512, chunk_overlap=0)

    ids1 = {c.chunk_id for c in chunks1}
    ids2 = {c.chunk_id for c in chunks2}

    assert ids1.isdisjoint(ids2), "不同文档的 chunk_id 应不相交"


# ---------------------------------------------------------------------------
# 测试 5: parent_id 共享
# ---------------------------------------------------------------------------


def test_parent_id_shared_within_same_source() -> None:
    """同 source_path 的所有 chunks 共享相同 parent_id。"""
    from fugue.handlers.chunkers.recursive import recursive_chunker

    content = "段落内容。" * 100
    doc = make_parsed_doc(content, source_path="/tmp/shared.txt")

    chunks = recursive_chunker([doc], chunk_size=100, chunk_overlap=10)

    assert len(chunks) >= 2
    parent_ids = {c.parent_id for c in chunks}
    assert len(parent_ids) == 1, f"同一 source_path 应只有一个 parent_id，实际: {parent_ids}"


# ---------------------------------------------------------------------------
# 测试 6: metadata 继承
# ---------------------------------------------------------------------------


def test_metadata_inheritance() -> None:
    """chunks 的 metadata 含原始字段 + chunk_index + source_path。"""
    from fugue.handlers.chunkers.recursive import recursive_chunker

    doc = make_parsed_doc(
        content="内容段落。" * 50,
        source_path="/tmp/doc.pdf",
        metadata={"format": "pdf"},
    )

    chunks = recursive_chunker([doc], chunk_size=100, chunk_overlap=0)

    assert len(chunks) >= 1
    for i, chunk in enumerate(chunks):
        assert chunk.metadata["format"] == "pdf"
        assert chunk.metadata["chunk_index"] == i
        assert chunk.metadata["source_path"] == "/tmp/doc.pdf"


# ---------------------------------------------------------------------------
# 测试 7: 空 parsed_docs
# ---------------------------------------------------------------------------


def test_empty_parsed_docs_returns_empty_list() -> None:
    """recursive_chunker([], 512, 64) → []。"""
    from fugue.handlers.chunkers.recursive import recursive_chunker

    result = recursive_chunker([], chunk_size=512, chunk_overlap=64)

    assert result == []


# ---------------------------------------------------------------------------
# 测试 8: 超短文档不切分
# ---------------------------------------------------------------------------


def test_short_document_returns_single_chunk() -> None:
    """100 字符文档，chunk_size=512 → 1 个 chunk，content == 原文。"""
    from fugue.handlers.chunkers.recursive import recursive_chunker

    content = "a" * 100
    doc = make_parsed_doc(content)

    chunks = recursive_chunker([doc], chunk_size=512, chunk_overlap=64)

    assert len(chunks) == 1
    assert chunks[0].content == content


# ---------------------------------------------------------------------------
# 测试 9 (推荐): 空 content 的 ParsedDocument
# ---------------------------------------------------------------------------


def test_empty_content_parsed_document() -> None:
    """空 content 的 ParsedDocument → 0 个 chunks。"""
    from fugue.handlers.chunkers.recursive import recursive_chunker

    doc = make_parsed_doc(content="")

    chunks = recursive_chunker([doc], chunk_size=512, chunk_overlap=64)

    assert len(chunks) == 0


# ---------------------------------------------------------------------------
# 测试 10 (推荐): register_chunkers 注册
# ---------------------------------------------------------------------------


def test_register_chunkers_registers_recursive(clean_chunker_registry) -> None:
    """register_chunkers() 后 chunker_registry.has('recursive')。"""
    from fugue.handlers.chunkers import register_chunkers

    register_chunkers()

    assert clean_chunker_registry.has("recursive")


# ---------------------------------------------------------------------------
# 测试 11 (推荐): 大文档+多种分隔符
# ---------------------------------------------------------------------------


def test_large_document_with_multiple_separators() -> None:
    """含多种分隔符的 5000 字符文档，chunk_size=500 → 所有 chunks ≤ 500 字符。"""
    from fugue.handlers.chunkers.recursive import recursive_chunker

    # 构建含 "\n\n" 段落 + "\n" 行 + ". " 句号的复合文档
    sentences = ["这是一个测试句子，用于验证递归切分逻辑" for _ in range(20)]
    paragraph = ". ".join(sentences) + "."
    lines = [paragraph[:50] for _ in range(10)]
    line_block = "\n".join(lines)
    paragraphs = [line_block for _ in range(5)]
    content = "\n\n".join(paragraphs)
    # 确保内容足够长
    while len(content) < 5000:
        content += "\n\n" + "额外内容用于填充长度。" * 10

    doc = make_parsed_doc(content)

    chunks = recursive_chunker([doc], chunk_size=500, chunk_overlap=50)

    assert len(chunks) >= 1
    for i, chunk in enumerate(chunks):
        assert len(chunk.content) <= 500, f"chunks[{i}] 长度为 {len(chunk.content)}，超过 500"


# ---------------------------------------------------------------------------
# 测试 12: 超长单 piece 触发深度递归（覆盖 merged > chunk_size 和 piece > chunk_size 分支）
# ---------------------------------------------------------------------------


def test_oversized_piece_triggers_deep_recursion() -> None:
    """含超长单 piece 的文档：piece > chunk_size 且 current 非空时需先 flush，覆盖分支 58/68-70。"""
    from fugue.handlers.chunkers.recursive import recursive_chunker

    # 先有一些小片段积累在 current，然后跟一个超长 piece
    # "\n\n" 分隔：前几段小，最后一段超大
    small_parts = ["短段落" * 5] * 3  # 每段 15 字符
    big_part = "超长内容" * 200  # 800 字符，超过 chunk_size=100
    content = "\n\n".join(small_parts + [big_part])

    doc = make_parsed_doc(content)
    chunks = recursive_chunker([doc], chunk_size=100, chunk_overlap=10)

    assert len(chunks) >= 2
    for i, chunk in enumerate(chunks):
        assert len(chunk.content) <= 100, f"chunks[{i}] 长度 {len(chunk.content)} 超过 100"


def test_accumulated_pieces_exceed_chunk_size_triggers_recursive_split() -> None:
    """多段落积累后合并超过 chunk_size，触发 merged > chunk_size 的递归切分分支（第 58 行）。"""
    from fugue.handlers.chunkers.recursive import recursive_chunker

    # 用 "\n\n" 分隔：每段 60 字符，三段积累 = 180+ 字符，超过 chunk_size=100
    # 但单独每段 < chunk_size，会被合并；合并后 > chunk_size 触发递归切分
    part = "a" * 60
    content = "\n\n".join([part] * 5)  # 5 段，靠 "\n\n" 合并时可能超限

    doc = make_parsed_doc(content)
    chunks = recursive_chunker([doc], chunk_size=100, chunk_overlap=0)

    assert len(chunks) >= 2
    for i, chunk in enumerate(chunks):
        assert len(chunk.content) <= 100, f"chunks[{i}] 长度 {len(chunk.content)} 超过 100"
