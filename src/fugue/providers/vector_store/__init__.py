"""src/fugue/providers/vector_store/__init__.py — re-export。"""

from fugue.providers.vector_store.base import VectorStore
from fugue.providers.vector_store.chroma import ChromaVectorStore

__all__ = ["ChromaVectorStore", "VectorStore"]
