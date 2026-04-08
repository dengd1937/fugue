"""rag/handlers/processors.py — 后处理 handler 实现。"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from rag.handlers._utils import normalize_score as _normalize_score
from rag.registry import processor_registry
from rag.types import Document

# ---------------------------------------------------------------------------
# RRF 常数
# ---------------------------------------------------------------------------

_RRF_K = 60


def _with_metadata(doc: Document, extra: dict[str, Any]) -> Document:
    """返回带额外 metadata 字段的新 Document（不修改原始对象）。"""
    return Document(
        doc_id=doc["doc_id"],
        content=doc["content"],
        score=doc["score"],
        source=doc["source"],
        metadata={**doc["metadata"], **extra},
    )


# ---------------------------------------------------------------------------
# rrf_fn
# ---------------------------------------------------------------------------


def rrf_fn(
    docs: list[Document],
    query: str,
    top_k: int,
    **kwargs: Any,
) -> list[Document]:
    """Reciprocal Rank Fusion（RRF）多路融合。

    - 按 (source, doc_id) 复合键聚合：source 分组 → 组内按 score 降序 → rank
    - retriever_weights：未设置默认 1.0
    - 结果写入 metadata["rrf_score"]
    """
    retriever_weights: dict[str, float] = kwargs.get("retriever_weights", {})

    # 按 source 分组
    by_source: dict[str, list[Document]] = defaultdict(list)
    for doc in docs:
        by_source[doc["source"]].append(doc)

    # 每组按 score 降序，计算 rrf 贡献，用 (source, doc_id) 复合键保证跨源不混淆
    rrf_scores: dict[tuple[str, str], float] = defaultdict(float)
    seen_docs: dict[tuple[str, str], Document] = {}

    for source, source_docs in by_source.items():
        w = retriever_weights.get(source, 1.0)
        sorted_docs = sorted(source_docs, key=lambda d: d["score"], reverse=True)
        for rank, doc in enumerate(sorted_docs):
            key = (doc["source"], doc["doc_id"])
            rrf_scores[key] += w / (_RRF_K + rank + 1)
            if key not in seen_docs:
                seen_docs[key] = doc

    # 按 rrf_score 降序排列，写入 metadata
    ranked = sorted(seen_docs.keys(), key=lambda k: rrf_scores[k], reverse=True)
    result: list[Document] = []
    for key in ranked[:top_k]:
        doc = seen_docs[key]
        result.append(_with_metadata(doc, {"rrf_score": rrf_scores[key]}))

    return result


# ---------------------------------------------------------------------------
# weighted_fusion_fn
# ---------------------------------------------------------------------------


def weighted_fusion_fn(
    docs: list[Document],
    query: str,
    top_k: int,
    **kwargs: Any,
) -> list[Document]:
    """加权分数融合：normalized_score * weight，按 doc_id 聚合取最高。"""
    retriever_weights: dict[str, float] = kwargs.get("retriever_weights", {})
    score_normalizers: dict[str, float] = kwargs.get("score_normalizers", {})

    if not docs:
        return []

    # 按 (source, doc_id) 复合键取最高加权分数，避免跨源同名 doc_id 相互覆盖
    best_score: dict[tuple[str, str], float] = {}
    best_doc: dict[tuple[str, str], Document] = {}

    for doc in docs:
        norm = _normalize_score(doc, score_normalizers)
        w = retriever_weights.get(doc["source"], 1.0)
        weighted = norm * w
        key = (doc["source"], doc["doc_id"])
        if key not in best_score or weighted > best_score[key]:
            best_score[key] = weighted
            best_doc[key] = doc

    ranked = sorted(best_doc.keys(), key=lambda k: best_score[k], reverse=True)
    return [best_doc[k] for k in ranked[:top_k]]


# ---------------------------------------------------------------------------
# rerank_fn
# ---------------------------------------------------------------------------


def rerank_fn(
    docs: list[Document],
    query: str,
    top_k: int,
    **kwargs: Any,
) -> list[Document]:
    """交叉编码器精排。cross_encoder 通过 kwargs 注入，缺失时 fallback 到原始顺序。"""
    cross_encoder = kwargs.get("cross_encoder")
    if not docs:
        return []
    if cross_encoder is None:
        return docs[:top_k]

    pairs = [[query, doc["content"]] for doc in docs]
    scores: list[float] = cross_encoder.predict(pairs)
    ranked = sorted(zip(scores, docs, strict=True), key=lambda x: x[0], reverse=True)
    return [doc for _, doc in ranked[:top_k]]


# ---------------------------------------------------------------------------
# filter_fn
# ---------------------------------------------------------------------------


def filter_fn(
    docs: list[Document],
    **kwargs: Any,
) -> list[Document]:
    """按 score_threshold 过滤低质量文档。"""
    threshold: float | None = kwargs.get("score_threshold")
    if threshold is None:
        return list(docs)
    return [doc for doc in docs if doc["score"] >= threshold]


# ---------------------------------------------------------------------------
# dedupe_fn
# ---------------------------------------------------------------------------


def dedupe_fn(
    docs: list[Document],
    **kwargs: Any,
) -> list[Document]:
    """按 (source, doc_id) 复合键去重，保留首次出现的文档。"""
    seen: set[tuple[str, str]] = set()
    result: list[Document] = []
    for doc in docs:
        key = (doc["source"], doc["doc_id"])
        if key not in seen:
            seen.add(key)
            result.append(doc)
    return result


# ---------------------------------------------------------------------------
# compression_fn
# ---------------------------------------------------------------------------


def compression_fn(
    docs: list[Document],
    query: str = "",
    **kwargs: Any,
) -> list[Document]:
    """压缩文档内容，仅保留与 query 相关的片段。compressor 通过 kwargs 注入。"""
    compressor = kwargs.get("compressor")
    if not docs:
        return []
    if compressor is None:
        return list(docs)
    result: list[Document] = []
    for doc in docs:
        compressed_content = compressor.compress(doc["content"], query)
        result.append(
            Document(
                doc_id=doc["doc_id"],
                content=compressed_content,
                score=doc["score"],
                source=doc["source"],
                metadata=dict(doc["metadata"]),
            )
        )
    return result


# ---------------------------------------------------------------------------
# 注册
# ---------------------------------------------------------------------------

processor_registry.register("rrf", rrf_fn)
processor_registry.register("weighted_fusion", weighted_fusion_fn)
processor_registry.register("rerank", rerank_fn)
processor_registry.register("filter", filter_fn)
processor_registry.register("dedupe", dedupe_fn)
processor_registry.register("compression", compression_fn)
