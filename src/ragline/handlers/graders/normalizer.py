"""src/ragline/handlers/graders/normalizer.py — 按 source 归一化分数。"""

from ragline.api.types import Document


def normalize_score(doc: Document, score_normalizers: dict[str, float]) -> float:
    """按 source 归一化分数到 0~1。

    score_normalizers={"bm25": 20.0} 表示 bm25 原始分数上界 20.0，
    未列出的 source 默认上界 1.0（适用于 vector cosine 已归一化的场景）。

    若 max_score <= 0 返回 0.0；raw / max_score 截断到 [0, 1]。
    """
    raw = doc.get("score", 0.0)
    source = doc.get("source", "")
    max_score = score_normalizers.get(source, 1.0)
    if max_score <= 0:
        return 0.0
    normalized = raw / max_score
    return max(0.0, min(normalized, 1.0))
