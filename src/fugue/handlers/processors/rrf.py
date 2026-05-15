"""src/fugue/handlers/processors/rrf.py — Reciprocal Rank Fusion 处理器。"""

from typing import Any

from fugue.api.types import Document

RRF_K = 60  # RRF 常数（防止排名 1 主导）


def rrf_fn(
    docs: list[Document],
    query: str,  # noqa: ARG001 - 签名统一保留，rrf 不依赖 query
    top_k: int,
    *,
    retriever_weights: dict[str, float] | None = None,
    **kwargs: Any,  # noqa: ARG001 - 接受额外配置参数
) -> list[Document]:
    """Reciprocal Rank Fusion: RRF_score = Σ (w_source / (RRF_K + rank))

    用 (source, doc_id) 复合键标识，按 source 分组，组内按原始 score 降序排 rank。
    rank 从 1 开始（第 1 名 rank=1）。
    rrf_score 写入 metadata["rrf_score"]，同时覆盖 score 字段让下游使用统一接口。
    返回按 score 降序排列的列表，top_k 截断。

    Args:
        docs: 待融合文档列表
        query: 查询字符串（rrf 不使用，签名统一保留）
        top_k: 截断保留前 K 个
        retriever_weights: source → weight 映射，未指定 source 默认权重 1.0
    """
    if not docs:
        return []

    weights = retriever_weights or {}

    # 按 source 分组，组内按原始 score 降序，确定每个 doc 的 rank
    by_source: dict[str, list[Document]] = {}
    for d in docs:
        by_source.setdefault(d["source"], []).append(d)

    # 复合键 (source, doc_id) → 累积 rrf_score
    accumulated: dict[tuple[str, str], float] = {}
    # 复合键 → 代表 Document（保留首次见到的）
    representative: dict[tuple[str, str], Document] = {}

    for source, group in by_source.items():
        sorted_group = sorted(group, key=lambda d: d["score"], reverse=True)
        weight = weights.get(source, 1.0)
        for rank, doc in enumerate(sorted_group, start=1):
            key = (source, doc["doc_id"])
            accumulated[key] = accumulated.get(key, 0.0) + weight / (RRF_K + rank)
            if key not in representative:
                representative[key] = doc

    # 构造结果：rrf_score 写入 metadata，同时覆盖 score 字段
    result: list[Document] = []
    for key, rrf_score in accumulated.items():
        orig = representative[key]
        md = dict(orig.get("metadata", {}))
        md["rrf_score"] = rrf_score
        new_doc = Document(
            doc_id=orig["doc_id"],
            content=orig["content"],
            score=rrf_score,  # 用 rrf_score 替换原 score，让下游 sort 用统一字段
            source=orig["source"],
            metadata=md,
        )
        result.append(new_doc)

    result.sort(key=lambda d: d["score"], reverse=True)
    return result[:top_k]
