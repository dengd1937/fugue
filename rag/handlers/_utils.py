"""rag/handlers/_utils.py — handler 共用工具函数。"""

from __future__ import annotations

from rag.types import Document


def normalize_score(doc: Document, score_normalizers: dict[str, float]) -> float:
    """将文档分数归一化到 [0, 1]。

    - 未注册 source 时默认 max=1.0
    - max <= 0 时返回 0.0
    """
    max_score = score_normalizers.get(doc["source"], 1.0)
    if max_score <= 0:
        return 0.0
    return min(doc["score"] / max_score, 1.0)
