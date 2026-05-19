"""src/ragline/providers/vector_store/chroma.py — Chroma 实现。"""

from collections.abc import Iterator, Sequence
from typing import Any

import chromadb
from chromadb.api.types import Metadata

from ragline.api.types import Chunk, Document

# Chroma Metadata value 类型别名（str | int | float | bool | None）
_MetadataValue = str | int | float | bool | None


class ChromaVectorStore:
    """Chroma PersistentClient 实现 VectorStore Protocol。"""

    def __init__(self, persist_dir: str, collection_name: str = "default") -> None:
        self._client = chromadb.PersistentClient(path=persist_dir)
        self._collection = self._client.get_or_create_collection(
            collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        self._name = collection_name
        self._persist_dir = persist_dir

    def add(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        """upsert chunks + embeddings。ids = chunk_ids 避免重复 ingest 报错。"""
        if not chunks:
            return
        if len(chunks) != len(embeddings):
            raise ValueError(f"chunks ({len(chunks)}) and embeddings ({len(embeddings)}) length mismatch")
        ids = [c.chunk_id for c in chunks]
        documents = [c.content for c in chunks]
        # metadata: 合并 chunk.metadata + parent_id（Chroma 不允许 None 值，也不允许空 dict）
        # 先用普通 dict 构建，最后转为 Metadata 兼容类型
        metadatas: list[Metadata] = []
        any_non_empty = False
        for c in chunks:
            raw: dict[str, _MetadataValue] = {k: v for k, v in c.metadata.items()}
            if c.parent_id is not None:
                raw["_parent_id"] = c.parent_id
            if raw:
                any_non_empty = True
                metadatas.append(raw)
            else:
                # Chroma 拒绝空 dict，用占位 key；iter_chunks 读出时会清理
                metadatas.append({"_empty": True})
        chroma_embeddings: list[Sequence[float]] = list(embeddings)
        self._collection.upsert(
            ids=ids,
            embeddings=chroma_embeddings,
            documents=documents,
            metadatas=metadatas if any_non_empty else None,
        )

    def similarity_search(
        self,
        query_embedding: list[float],
        k: int = 20,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[Document]:
        """Chroma 查询；distance → score = 1 - distance (cosine 归一化)；source='vector'。"""
        # 若 collection 为空直接返回空列表，避免 Chroma 报错
        count = self._collection.count()
        if count == 0:
            return []
        # k 不能超过实际 chunk 数量
        actual_k = min(k, count)
        query_emb: list[Sequence[float]] = [query_embedding]
        result = self._collection.query(
            query_embeddings=query_emb,
            n_results=actual_k,
            where=metadata_filter,
        )
        documents: list[Document] = []
        raw_ids = result.get("ids") or [[]]
        raw_contents = result.get("documents") or [[]]
        raw_distances = result.get("distances") or [[]]
        raw_metadatas = result.get("metadatas") or [[]]
        ids: list[str] = raw_ids[0] if raw_ids else []
        contents: list[str] = list(raw_contents[0]) if raw_contents and raw_contents[0] else []
        distances: list[float] = list(raw_distances[0]) if raw_distances and raw_distances[0] else []
        mds = list(raw_metadatas[0]) if raw_metadatas and raw_metadatas[0] else []
        for i, doc_id in enumerate(ids):
            content = contents[i] if i < len(contents) else ""
            distance = distances[i] if i < len(distances) else 0.0
            md_raw = mds[i] if i < len(mds) else {}
            md_clean: dict[str, Any] = dict(md_raw) if md_raw else {}
            score = max(0.0, 1.0 - float(distance))  # cosine: distance ∈ [0,2], score ∈ [0,1]
            documents.append(
                Document(
                    doc_id=doc_id,
                    content=content,
                    score=score,
                    source="vector",
                    metadata=md_clean,
                )
            )
        return documents

    def delete_collection(self) -> None:
        """删除 collection 并重建一个空的。"""
        self._client.delete_collection(self._name)
        self._collection = self._client.get_or_create_collection(
            self._name,
            metadata={"hnsw:space": "cosine"},
        )

    def stats(self) -> dict[str, Any]:
        return {
            "num_chunks": self._collection.count(),
            "collection_name": self._name,
            "persist_dir": self._persist_dir,
        }

    def iter_chunks(self, batch_size: int = 1000) -> Iterator[list[Chunk]]:
        """Chroma get(limit, offset) 分批拉取所有 chunks。"""
        offset = 0
        while True:
            batch = self._collection.get(
                limit=batch_size,
                offset=offset,
                include=["documents", "metadatas"],
            )
            ids: list[str] = batch.get("ids") or []
            if not ids:
                break
            raw_contents = batch.get("documents") or []
            raw_metadatas = batch.get("metadatas") or []
            chunks: list[Chunk] = []
            for i, cid in enumerate(ids):
                content = str(raw_contents[i]) if i < len(raw_contents) and raw_contents[i] is not None else ""
                md: dict[str, Any] = dict(raw_metadatas[i]) if i < len(raw_metadatas) and raw_metadatas[i] else {}
                md.pop("_empty", None)  # 清理 add() 写入的空 dict 占位 key
                parent_id_raw = md.pop("_parent_id", None)
                parent_id: str | None = str(parent_id_raw) if parent_id_raw is not None else None
                chunks.append(
                    Chunk(
                        chunk_id=cid,
                        parent_id=parent_id,
                        content=content,
                        metadata=md,
                    )
                )
            yield chunks
            if len(ids) < batch_size:
                break
            offset += batch_size

    def close(self) -> None:
        """Chroma PersistentClient 无显式 close，noop。"""
        pass
