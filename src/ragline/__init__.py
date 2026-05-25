"""src/ragline/__init__.py — 顶层 re-export。"""

from importlib.metadata import PackageNotFoundError as _PackageNotFoundError
from importlib.metadata import version as _pkg_version

try:
    __version__: str = _pkg_version("ragline")
except _PackageNotFoundError:  # pragma: no cover
    __version__ = "0.0.0+unknown"

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
