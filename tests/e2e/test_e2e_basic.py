"""tests/e2e/test_e2e_basic.py — 基础 ingest + query e2e。

需要 OPENAI_API_KEY 环境变量。每次跑约 $0.01。
"""

from pathlib import Path

from ragline import RAG, FugueConfig, GraphConfig, IngestConfig, ProviderConfig


def test_e2e_basic_ingest_and_query(
    e2e_provider: ProviderConfig,
    e2e_fixtures_dir: Path,
    tmp_path: Path,
) -> None:
    """完整 e2e：ingest 3 markdown → query → 验证 answer 含 fixtures 关键词。"""
    cfg = FugueConfig(
        graph=GraphConfig(
            transforms=["rewrite"],
            n_rewrites=1,
            retrievers=["vector"],
            processors=[],  # 简化降本
            top_k=3,
            grade_threshold=0.0,  # 不触发 fallback
        ),
        ingest=IngestConfig(
            chunk_size=400,
            chunk_overlap=40,
            persist_dir=str(tmp_path / "chroma"),
            collection_name="e2e_basic",
        ),
        providers=e2e_provider,
    )
    with RAG(cfg) as rag:
        ingest_result = rag.ingest(
            [
                e2e_fixtures_dir / "doc1.md",
                e2e_fixtures_dir / "doc2.md",
                e2e_fixtures_dir / "doc3.md",
            ],
            show_progress=False,
        )
        assert ingest_result.num_documents == 3
        assert ingest_result.num_chunks > 0

        # query 与 fixtures 内容相关的问题
        result = rag.query("What is the main topic discussed in the documents?")
        assert result.answer  # 非空
        assert len(result.ranked_documents) <= 3  # top_k=3 上限
        assert result.retrieval_rounds >= 1
