"""tests/integration/test_ingest_pipeline.py — IngestPipeline 集成测试。"""

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pypdf
import pytest

from ragline.api.ingest import IngestPipeline
from ragline.config import FugueConfig, GraphConfig, IngestConfig
from ragline.providers.bm25 import BM25Provider
from ragline.providers.vector_store.chroma import ChromaVectorStore

FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"


@pytest.fixture
def sample_pdf(tmp_path_factory):
    """动态生成 2 页 PDF（避免 commit 二进制）。"""
    pdf_path = tmp_path_factory.mktemp("pdf") / "sample.pdf"
    writer = pypdf.PdfWriter()
    writer.add_blank_page(width=200, height=200)
    writer.add_blank_page(width=200, height=200)
    with open(pdf_path, "wb") as f:
        writer.write(f)
    return pdf_path


@pytest.fixture
def mock_embedding():
    """Mock EmbeddingClient，返回 8 维 0.1 向量。"""
    client = MagicMock()

    def fake_embed(texts):
        return [[0.1] * 8 for _ in texts]

    client.embed.side_effect = fake_embed
    return client


@pytest.fixture
def real_chroma(tmp_path):
    """真实 Chroma store on tmp_path。"""
    return ChromaVectorStore(persist_dir=str(tmp_path / "chroma"))


@pytest.fixture
def real_bm25():
    return BM25Provider()


def _config(*, retrievers: list[str] | None = None, **ingest_overrides: Any) -> FugueConfig:
    return FugueConfig(
        graph=GraphConfig(retrievers=retrievers or ["vector"]),
        ingest=IngestConfig(
            chunk_size=200,
            chunk_overlap=20,
            collection_name="test",
            persist_dir=".",
            **ingest_overrides,
        ),
    )


# 1. 端到端 ingest（3 fixtures: md + txt + pdf） ---------------------


def test_end_to_end_ingest(mock_embedding, real_chroma, real_bm25, sample_pdf) -> None:
    """3 文件全部摄取，IngestResult.num_chunks > 0，Chroma stats 相符。"""
    pipeline = IngestPipeline(
        config=_config(retrievers=["vector", "bm25"]),
        embedding_client=mock_embedding,
        vector_store=real_chroma,
        bm25_provider=real_bm25,
    )
    sources = [
        FIXTURES_DIR / "sample.md",
        FIXTURES_DIR / "sample.txt",
        sample_pdf,
    ]
    result = pipeline.run(sources, show_progress=False)
    assert result.num_documents == 3
    assert result.num_chunks > 0
    assert real_chroma.stats()["num_chunks"] == result.num_chunks


# 2. glob 展开 ------------------------------------------------------


def test_glob_expansion(mock_embedding, real_chroma, real_bm25) -> None:
    """sources='tests/fixtures/*.md' 展开正确。"""
    pipeline = IngestPipeline(
        config=_config(),
        embedding_client=mock_embedding,
        vector_store=real_chroma,
        bm25_provider=real_bm25,
    )
    pattern = str(FIXTURES_DIR / "*.md")
    result = pipeline.run(pattern, show_progress=False)
    # FIXTURES_DIR 有 sample.md + sample_empty.md
    assert result.num_documents >= 1


# 3. bm25 update（retrievers 含 bm25） ------------------------------


def test_bm25_update_when_retrievers_include_bm25(mock_embedding, real_chroma) -> None:
    """retrievers=['bm25'] 时 BM25 索引能 search 到内容。"""
    bm25 = BM25Provider()
    pipeline = IngestPipeline(
        config=_config(retrievers=["bm25"]),
        embedding_client=mock_embedding,
        vector_store=real_chroma,
        bm25_provider=bm25,
    )
    pipeline.run([FIXTURES_DIR / "sample.md"], show_progress=False)
    # BM25 应该能 search 到（sample.md 含 "Markdown"）
    results = bm25.search("markdown", k=5)
    assert len(results) >= 1


# 4. bm25 skip（retrievers 不含 bm25） ------------------------------


def test_bm25_skip_when_retrievers_excludes_bm25(mock_embedding, real_chroma) -> None:
    """retrievers=['vector'] 时 bm25_provider.update 不被调用。"""
    bm25_spy = MagicMock(spec=BM25Provider)
    pipeline = IngestPipeline(
        config=_config(retrievers=["vector"]),
        embedding_client=mock_embedding,
        vector_store=real_chroma,
        bm25_provider=bm25_spy,
    )
    pipeline.run([FIXTURES_DIR / "sample.md"], show_progress=False)
    bm25_spy.update.assert_not_called()


# 5. chunk_id 稳定（upsert 去重） -----------------------------------


def test_idempotent_ingest(mock_embedding, real_chroma, real_bm25) -> None:
    """同一文件二次 ingest 后 num_chunks 不变。"""
    pipeline = IngestPipeline(
        config=_config(),
        embedding_client=mock_embedding,
        vector_store=real_chroma,
        bm25_provider=real_bm25,
    )
    pipeline.run([FIXTURES_DIR / "sample.md"], show_progress=False)
    first_count = real_chroma.stats()["num_chunks"]
    pipeline.run([FIXTURES_DIR / "sample.md"], show_progress=False)
    second_count = real_chroma.stats()["num_chunks"]
    assert first_count == second_count


# 6. 空 sources ----------------------------------------------------


def test_empty_sources(mock_embedding, real_chroma, real_bm25) -> None:
    """sources=[] 返回 num_documents=0, num_chunks=0。"""
    pipeline = IngestPipeline(
        config=_config(),
        embedding_client=mock_embedding,
        vector_store=real_chroma,
        bm25_provider=real_bm25,
    )
    result = pipeline.run([], show_progress=False)
    assert result.num_documents == 0
    assert result.num_chunks == 0


# 7. 未知扩展名 ----------------------------------------------------


def test_unknown_extension_raises(mock_embedding, real_chroma, real_bm25, tmp_path) -> None:
    """未知扩展抛 ValueError（auto_parser 错误透传）。"""
    unknown = tmp_path / "test.xyz"
    unknown.write_text("content")
    pipeline = IngestPipeline(
        config=_config(),
        embedding_client=mock_embedding,
        vector_store=real_chroma,
        bm25_provider=real_bm25,
    )
    with pytest.raises(ValueError, match="unsupported extension"):
        pipeline.run([unknown], show_progress=False)


# 8. single source 字符串 -----------------------------------------


def test_single_source_string(mock_embedding, real_chroma, real_bm25) -> None:
    """sources 是单个字符串路径也接受。"""
    pipeline = IngestPipeline(
        config=_config(),
        embedding_client=mock_embedding,
        vector_store=real_chroma,
        bm25_provider=real_bm25,
    )
    result = pipeline.run(str(FIXTURES_DIR / "sample.md"), show_progress=False)
    assert result.num_documents == 1
