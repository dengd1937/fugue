"""examples/quickstart/consumer_minimal.py — Ragline 最小可运行示例。

演示：用 mock providers（无需真实 API key）ingest 本地 docs/ 目录，
然后对 RAG 发起一次查询，输出 ingest 统计和答案。
"""

import tempfile
from pathlib import Path

from ragline import RAG, GraphConfig, IngestConfig, RaglineConfig
from ragline.testing import isolated_registries, mock_rag_providers


def main() -> None:
    docs_dir = Path(__file__).parent / "docs"

    with tempfile.TemporaryDirectory() as tmp_dir:
        cfg = RaglineConfig(
            graph=GraphConfig(
                retrievers=["vector"],
                processors=[],
                grade_threshold=0.01,
                transforms=[],
                fallback_chain=[],
            ),
            ingest=IngestConfig(
                persist_dir=tmp_dir,
                collection_name="quickstart-demo",
            ),
        )
        cfg.providers.llm_api_key = "fake-key-for-demo"

        with isolated_registries(), mock_rag_providers() as (llm, _):
            llm.answer = "Ragline is a config-driven RAG library."

            with RAG(cfg) as rag:
                result = rag.ingest(docs_dir / "*.md", show_progress=False)
                print(f"INGESTED: {result.num_chunks} chunks")

                query_result = rag.query("What is Ragline?")
                print(f"ANSWER: {query_result.answer}")


if __name__ == "__main__":
    main()
