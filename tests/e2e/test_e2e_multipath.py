"""tests/e2e/test_e2e_multipath.py — 多路 + RRF + rerank + citation。

需要 OPENAI_API_KEY + 已下载 BGE Reranker 模型（首次约 600MB）。
"""

from pathlib import Path

from ragline import RAG, FugueConfig, GraphConfig, IngestConfig, ProviderConfig


def test_e2e_multipath_with_citation(
    e2e_provider: ProviderConfig,
    e2e_fixtures_dir: Path,
    tmp_path: Path,
) -> None:
    """transforms + 多路 retriever + rrf + rerank + citation mode 端到端。"""
    cfg = FugueConfig(
        graph=GraphConfig(
            transforms=["rewrite"],
            n_rewrites=2,
            retrievers=["vector", "bm25"],
            processors=["rrf", "rerank"],
            top_k=3,
            gen_mode="citation",
            grade_threshold=0.0,
            score_normalizers={"bm25": 20.0},
        ),
        ingest=IngestConfig(
            chunk_size=400,
            chunk_overlap=40,
            persist_dir=str(tmp_path / "chroma"),
            collection_name="e2e_multi",
        ),
        providers=e2e_provider,
    )
    with RAG(cfg) as rag:
        rag.ingest(
            [
                e2e_fixtures_dir / "doc1.md",
                e2e_fixtures_dir / "doc2.md",
                e2e_fixtures_dir / "doc3.md",
            ],
            show_progress=False,
        )
        result = rag.query("Summarize the documents using citations.")
        assert result.answer
        # citation 模式应在 answer 中含 [N] 编号
        assert "[1]" in result.answer or "[2]" in result.answer
        # 多 transforms 应产生 > 1 个 rewritten query
        assert len(result.rewritten_queries) > 1


def test_e2e_chinese_query_basic(
    e2e_provider: ProviderConfig,
    e2e_fixtures_dir: Path,
    tmp_path: Path,
) -> None:
    """中文场景：BM25 召回为 0 是已知限制，但向量路径应仍能工作。"""
    cfg = FugueConfig(
        graph=GraphConfig(
            transforms=[],  # 不改写，纯检索
            retrievers=["vector"],  # 只用向量避开 BM25 中文问题
            processors=[],
            top_k=3,
            grade_threshold=0.0,
        ),
        ingest=IngestConfig(
            chunk_size=300,
            chunk_overlap=30,
            persist_dir=str(tmp_path / "chroma"),
            collection_name="e2e_zh",
        ),
        providers=e2e_provider,
    )
    with RAG(cfg) as rag:
        rag.ingest(
            [e2e_fixtures_dir / "doc_chinese.md"],
            show_progress=False,
        )
        result = rag.query("文档讲了什么？")
        assert result.answer  # 中文 answer 非空，验证端到端不破
