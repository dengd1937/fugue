"""src/fugue/api/types.py — Fugue 公开类型与异常定义。"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, TypedDict


class Document(TypedDict):
    doc_id: str
    content: str
    score: float
    source: str
    metadata: dict[str, Any]


@dataclass(frozen=True)
class ParsedDocument:
    source_path: Path
    content: str
    metadata: dict[str, Any]


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    parent_id: str | None
    content: str
    metadata: dict[str, Any]


@dataclass(frozen=True)
class TransformResult:
    query: str
    metadata_filter: dict[str, Any] | None = None


@dataclass(frozen=True)
class QueryResult:
    answer: str
    ranked_documents: list[Document]
    grade_score: float
    grade_decision: Literal["sufficient", "insufficient"]
    rewritten_queries: list[str]
    retrieval_rounds: int


@dataclass(frozen=True)
class IngestResult:
    num_documents: int
    num_chunks: int
    collection_name: str
    duration_seconds: float


# 异常体系
class FugueError(Exception):
    """Fugue 异常基类。"""


class FugueConfigError(FugueError): ...


class FugueRegistryError(FugueError): ...


class FugueLLMError(FugueError): ...


class FugueEmbeddingError(FugueError): ...


class FugueRetrieverError(FugueError): ...
