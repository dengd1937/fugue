"""src/ragline/server/endpoints.py — Ragline HTTP API endpoints。"""

import logging
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from ragline.api.rag import RAG

logger = logging.getLogger(__name__)


class QueryRequest(BaseModel):
    """POST /query 请求体。"""

    question: str


class IngestRequest(BaseModel):
    """POST /ingest 请求体。"""

    paths: list[str]


def _query_result_to_dict(result: Any) -> dict[str, Any]:
    """QueryResult dataclass → dict（递归处理嵌套 Document）。"""
    from dataclasses import asdict

    return asdict(result)


def _ingest_result_to_dict(result: Any) -> dict[str, Any]:
    """IngestResult dataclass → dict。"""
    from dataclasses import asdict

    return asdict(result)


def create_endpoints(app: FastAPI, rag: RAG) -> None:
    """在 FastAPI app 上注册 /query / /ingest / /health 三个 endpoint。"""

    @app.post("/query")
    def query(req: QueryRequest) -> dict[str, Any]:
        try:
            result = rag.query(req.question)
            return _query_result_to_dict(result)
        except HTTPException:
            raise
        except Exception as e:
            logger.exception("Query failed for question: %s", req.question[:100])
            raise HTTPException(status_code=500, detail=str(e)) from e

    @app.post("/ingest")
    def ingest(req: IngestRequest) -> dict[str, Any]:
        try:
            result = rag.ingest(req.paths)
            return _ingest_result_to_dict(result)
        except HTTPException:
            raise
        except Exception as e:
            logger.exception("Ingest failed for paths: %s", req.paths)
            raise HTTPException(status_code=500, detail=str(e)) from e

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}
