"""tests/integration/test_server.py — FastAPI 集成测试（mock RAG）。"""

from unittest.mock import MagicMock

from fastapi import FastAPI
from fastapi.testclient import TestClient

from ragline.api.types import Document, FugueError, IngestResult, QueryResult
from ragline.server.endpoints import create_endpoints


def _mock_rag() -> MagicMock:
    """构造 mock RAG 实例。"""
    rag = MagicMock()
    rag.query.return_value = QueryResult(
        answer="mocked answer",
        ranked_documents=[
            Document(doc_id="d1", content="content 1", score=0.9, source="vector", metadata={}),
        ],
        grade_score=0.8,
        grade_decision="sufficient",
        rewritten_queries=["original", "rewrite1"],
        retrieval_rounds=1,
    )
    rag.ingest.return_value = IngestResult(
        num_documents=3,
        num_chunks=12,
        collection_name="test",
        duration_seconds=1.5,
    )
    return rag


def _make_app(rag: MagicMock) -> FastAPI:
    app = FastAPI()
    create_endpoints(app, rag)
    return app


# /query 测试 ----------------------------------------------------


def test_query_endpoint_returns_answer() -> None:
    rag = _mock_rag()
    client = TestClient(_make_app(rag))
    resp = client.post("/query", json={"question": "test query"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["answer"] == "mocked answer"
    assert data["grade_score"] == 0.8
    assert data["retrieval_rounds"] == 1
    assert len(data["ranked_documents"]) == 1
    rag.query.assert_called_once_with("test query")


def test_query_endpoint_handles_exception() -> None:
    rag = _mock_rag()
    rag.query.side_effect = FugueError("LLM unavailable")
    client = TestClient(_make_app(rag))
    resp = client.post("/query", json={"question": "test"})
    assert resp.status_code == 500
    detail = resp.json()["detail"]
    assert "LLM unavailable" in detail


def test_query_endpoint_missing_question_returns_422() -> None:
    """空 body 应被 Pydantic 验证拦截返回 422。"""
    rag = _mock_rag()
    client = TestClient(_make_app(rag))
    resp = client.post("/query", json={})
    assert resp.status_code == 422


def test_query_endpoint_invalid_body_type_returns_422() -> None:
    rag = _mock_rag()
    client = TestClient(_make_app(rag))
    resp = client.post("/query", json={"question": 123})  # int 不是 str
    assert resp.status_code == 422


# /ingest 测试 ---------------------------------------------------


def test_ingest_endpoint_returns_result() -> None:
    rag = _mock_rag()
    client = TestClient(_make_app(rag))
    resp = client.post("/ingest", json={"paths": ["./docs/a.md", "./docs/b.pdf"]})
    assert resp.status_code == 200
    data = resp.json()
    assert data["num_documents"] == 3
    assert data["num_chunks"] == 12
    rag.ingest.assert_called_once_with(["./docs/a.md", "./docs/b.pdf"])


def test_ingest_endpoint_handles_exception() -> None:
    rag = _mock_rag()
    rag.ingest.side_effect = FugueError("vector store failed")
    client = TestClient(_make_app(rag))
    resp = client.post("/ingest", json={"paths": ["./x.md"]})
    assert resp.status_code == 500
    assert "vector store failed" in resp.json()["detail"]


def test_ingest_endpoint_missing_paths_returns_422() -> None:
    rag = _mock_rag()
    client = TestClient(_make_app(rag))
    resp = client.post("/ingest", json={})
    assert resp.status_code == 422


# /health 测试 --------------------------------------------------


def test_health_endpoint() -> None:
    rag = _mock_rag()
    client = TestClient(_make_app(rag))
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


# create_app 集成 ------------------------------------------------


def test_create_app_from_yaml(tmp_path, monkeypatch) -> None:
    """create_app(yaml_path) 实例化 RAG 并装配三 endpoint。"""
    from unittest.mock import patch

    monkeypatch.setenv("OPENAI_API_KEY", "fake-key")
    yaml_path = tmp_path / "config.yaml"
    yaml_path.write_text(f"""
ingest:
  persist_dir: "{tmp_path / "chroma"}"
""")
    # mock RAG.from_yaml 避免实际初始化 providers
    with patch("ragline.server.app.RAG.from_yaml") as mock_from_yaml:
        mock_rag = _mock_rag()
        mock_from_yaml.return_value = mock_rag
        from ragline.server.app import create_app

        app = create_app(yaml_path)
        assert app is not None
        # 验证三 endpoint 已注册
        client = TestClient(app)
        assert client.get("/health").status_code == 200
        assert client.post("/query", json={"question": "x"}).status_code == 200
        assert client.post("/ingest", json={"paths": ["x"]}).status_code == 200
        mock_from_yaml.assert_called_once_with(yaml_path)
