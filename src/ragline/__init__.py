"""src/ragline/__init__.py — 顶层 re-export。"""

from ragline.api.rag import RAG
from ragline.api.types import (
    Chunk,
    Document,
    IngestResult,
    ParsedDocument,
    QueryResult,
    RaglineConfigError,
    RaglineEmbeddingError,
    RaglineError,
    RaglineLLMError,
    RaglineRegistryError,
    RaglineRetrieverError,
    TransformResult,
)
from ragline.config import GraphConfig, IngestConfig, ProviderConfig, RaglineConfig

__all__ = [
    "RAG",
    "Chunk",
    "Document",
    "GraphConfig",
    "IngestConfig",
    "IngestResult",
    "ParsedDocument",
    "ProviderConfig",
    "QueryResult",
    "RaglineConfig",
    "RaglineConfigError",
    "RaglineEmbeddingError",
    "RaglineError",
    "RaglineLLMError",
    "RaglineRegistryError",
    "RaglineRetrieverError",
    "TransformResult",
]
