"""src/fugue/__init__.py — 顶层 re-export。"""

from fugue.api.rag import RAG
from fugue.api.types import (
    Chunk,
    Document,
    FugueConfigError,
    FugueEmbeddingError,
    FugueError,
    FugueLLMError,
    FugueRegistryError,
    FugueRetrieverError,
    IngestResult,
    ParsedDocument,
    QueryResult,
    TransformResult,
)
from fugue.config import FugueConfig, GraphConfig, IngestConfig, ProviderConfig

__all__ = [
    "Chunk",
    "Document",
    "FugueConfig",
    "FugueConfigError",
    "FugueEmbeddingError",
    "FugueError",
    "FugueLLMError",
    "FugueRegistryError",
    "FugueRetrieverError",
    "GraphConfig",
    "IngestConfig",
    "IngestResult",
    "ParsedDocument",
    "ProviderConfig",
    "QueryResult",
    "RAG",
    "TransformResult",
]
