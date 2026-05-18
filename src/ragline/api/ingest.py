"""src/ragline/api/ingest.py — IngestPipeline 端到端摄取流水线。"""

import logging
import time
from collections.abc import Iterable
from glob import glob
from pathlib import Path

import ragline.handlers.chunkers  # noqa: F401  — 触发 chunker_registry 注册副作用
import ragline.handlers.parsers  # noqa: F401  — 触发 parser_registry 注册副作用
from ragline.api.types import IngestResult, ParsedDocument
from ragline.config import RaglineConfig
from ragline.providers.bm25 import BM25Provider
from ragline.providers.embedding import EmbeddingClient
from ragline.providers.vector_store.base import VectorStore
from ragline.registry import chunker_registry, parser_registry

logger = logging.getLogger(__name__)


class IngestPipeline:
    """端到端摄取流水线: parse → chunk → embed → upsert → bm25 update。

    bm25_provider.update 仅当 retrievers 包含 'bm25' 时调用。
    """

    def __init__(
        self,
        config: RaglineConfig,
        embedding_client: EmbeddingClient,
        vector_store: VectorStore,
        bm25_provider: BM25Provider,
    ) -> None:
        self._config = config
        self._embedding_client = embedding_client
        self._vector_store = vector_store
        self._bm25_provider = bm25_provider

    def run(
        self,
        sources: str | Path | Iterable[str | Path],
        *,
        show_progress: bool = True,
    ) -> IngestResult:
        """执行摄取流水线。返回 IngestResult。"""
        start = time.perf_counter()

        # 1. 展开 sources 到 list[Path]
        paths = self._expand_sources(sources)

        if not paths:
            return IngestResult(
                num_documents=0,
                num_chunks=0,
                collection_name=self._config.ingest.collection_name,
                duration_seconds=time.perf_counter() - start,
            )

        # 2. parse 每个文件
        all_parsed: list[ParsedDocument] = []
        for path in paths:
            parser_name = self._select_parser(path)
            parser_fn = parser_registry.get(parser_name)
            parsed = parser_fn(path)
            all_parsed.extend(parsed)
            if show_progress:
                logger.info("Parsed %s → %d documents", path, len(parsed))

        # 3. chunk
        chunker_fn = chunker_registry.get(self._config.ingest.chunker)
        chunks = chunker_fn(
            all_parsed,
            chunk_size=self._config.ingest.chunk_size,
            chunk_overlap=self._config.ingest.chunk_overlap,
        )

        if not chunks:
            return IngestResult(
                num_documents=len(paths),
                num_chunks=0,
                collection_name=self._config.ingest.collection_name,
                duration_seconds=time.perf_counter() - start,
            )

        # 4. embed
        embeddings = self._embedding_client.embed([c.content for c in chunks])

        # 5. vector store upsert
        self._vector_store.add(chunks, embeddings)

        # 6. bm25 update（仅当 retrievers 包含 'bm25'）
        if "bm25" in self._config.graph.retrievers:
            self._bm25_provider.update(chunks)

        return IngestResult(
            num_documents=len(paths),
            num_chunks=len(chunks),
            collection_name=self._config.ingest.collection_name,
            duration_seconds=time.perf_counter() - start,
        )

    def _expand_sources(self, sources: str | Path | Iterable[str | Path]) -> list[Path]:
        """glob 展开 sources 为有序 Path 列表（去重）。"""
        if isinstance(sources, (str, Path)):
            sources_list: list[str | Path] = [sources]
        else:
            sources_list = list(sources)

        seen: set[Path] = set()
        result: list[Path] = []
        for src in sources_list:
            src_str = str(src)
            if any(c in src_str for c in ["*", "?", "["]):
                # glob 模式
                for matched in sorted(glob(src_str, recursive=True)):
                    p = Path(matched).resolve()
                    if p not in seen and p.is_file():
                        seen.add(p)
                        result.append(p)
            else:
                p = Path(src_str).resolve()
                if p not in seen and p.is_file():
                    seen.add(p)
                    result.append(p)
        return result

    def _select_parser(self, path: Path) -> str:
        """按配置或扩展名选择 parser 名。"""
        configured = self._config.ingest.parser
        if configured != "auto":
            return configured
        return "auto"  # auto_parser 自身会按 suffix 分派
