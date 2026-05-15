"""src/fugue/handlers/graders/score.py — 基于归一化平均分的 grader。"""

from typing import Any, Literal

from fugue.api.types import Document
from fugue.handlers.graders.normalizer import normalize_score


def score_grader(
    docs: list[Document],
    query: str,  # noqa: ARG001 - 签名统一，score grader 不依赖 query
    threshold: float,
    *,
    score_normalizers: dict[str, float] | None = None,
    **kwargs: Any,  # noqa: ARG001
) -> tuple[float, Literal["sufficient", "insufficient"]]:
    """基于归一化分数均值判定 sufficient/insufficient。

    空 docs 返回 (0.0, "insufficient")。
    """
    if not docs:
        return 0.0, "insufficient"
    normalizers = score_normalizers or {}
    normalized = [normalize_score(d, normalizers) for d in docs]
    avg = sum(normalized) / len(normalized)
    decision: Literal["sufficient", "insufficient"] = (
        "sufficient" if avg >= threshold else "insufficient"
    )
    return avg, decision
