"""tests/e2e/test_e2e_server.py — FastAPI server e2e。

需要 OPENAI_API_KEY。
"""

from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient

from fugue.api.rag import RAG
from fugue.config import FugueConfig, GraphConfig, IngestConfig, ProviderConfig
from fugue.server.endpoints import create_endpoints


def test_e2e_server_endpoints(
    openai_key: str,
    e2e_fixtures_dir: Path,
    tmp_path: Path,
) -> None:
    """端到端：构建 FastAPI app → ingest + query + health 三 endpoint。"""
    cfg = FugueConfig(
        graph=GraphConfig(
            transforms=[],
            retrievers=["vector"],
            processors=[],
            top_k=3,
            grade_threshold=0.0,
        ),
        ingest=IngestConfig(
            chunk_size=400,
            chunk_overlap=40,
            persist_dir=str(tmp_path / "chroma"),
            collection_name="e2e_server",
        ),
        providers=ProviderConfig(llm_api_key=openai_key),
    )
    with RAG(cfg) as rag:
        app = FastAPI(title="Fugue E2E")
        create_endpoints(app, rag)
        client = TestClient(app)

        # /health
        resp_health = client.get("/health")
        assert resp_health.status_code == 200
        assert resp_health.json() == {"status": "ok"}

        # /ingest
        resp_ingest = client.post(
            "/ingest",
            json={"paths": [str(e2e_fixtures_dir / "doc1.md")]},
        )
        assert resp_ingest.status_code == 200
        ingest_data = resp_ingest.json()
        assert ingest_data["num_chunks"] > 0

        # /query
        resp_query = client.post(
            "/query",
            json={"question": "What is in the document?"},
        )
        assert resp_query.status_code == 200
        query_data = resp_query.json()
        assert query_data["answer"]
