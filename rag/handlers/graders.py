"""rag/handlers/graders.py — 检索质量评估 handler 实现。"""

from __future__ import annotations

from typing import Any, Literal

from rag.handlers._utils import normalize_score as _normalize_score
from rag.handlers.transforms import _default_llm
from rag.registry import grader_registry
from rag.types import Document

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

PROMPT_LLM_GRADE = (
    "请判断以下检索到的文档是否足以回答给定问题。\n"
    "如果文档质量足够，回复 'sufficient'；如果不够，回复 'insufficient'。\n"
    "只回复这两个词之一，不要其他内容。\n\n"
    "问题：{query}\n\n"
    "文档内容：\n{context}"
)


# ---------------------------------------------------------------------------
# score_grader
# ---------------------------------------------------------------------------


def score_grader(
    docs: list[Document],
    threshold: float,
    **kwargs: Any,
) -> tuple[float, Literal["sufficient", "insufficient"]]:
    """基于文档分数的评估器。

    - 空文档返回 (0.0, "insufficient")
    - 多文档取归一化分数的平均值
    - 平均分 >= threshold → "sufficient"
    """
    if not docs:
        return (0.0, "insufficient")

    score_normalizers: dict[str, float] = kwargs.get("score_normalizers", {})
    normalized_scores = [_normalize_score(doc, score_normalizers) for doc in docs]
    avg_score = sum(normalized_scores) / len(normalized_scores)

    decision: Literal["sufficient", "insufficient"] = (
        "sufficient" if avg_score >= threshold else "insufficient"
    )
    return (avg_score, decision)


# ---------------------------------------------------------------------------
# llm_grader
# ---------------------------------------------------------------------------


def llm_grader(
    docs: list[Document],
    query: str,
    threshold: float,
    **kwargs: Any,
) -> tuple[float, Literal["sufficient", "insufficient"]]:
    """基于 LLM 判断的评估器。

    - 空文档直接返回 (0.0, "insufficient")
    - LLM 返回 "sufficient"/"insufficient"，无法解析时 fallback (0.0, "insufficient")
    """
    if not docs:
        return (0.0, "insufficient")

    llm_client: Any = kwargs.get("llm_client", _default_llm)
    context = "\n\n".join(f"[文档 {i + 1}] {doc['content']}" for i, doc in enumerate(docs))
    prompt = PROMPT_LLM_GRADE.format(query=query, context=context)
    response = llm_client.invoke(prompt).strip().lower()

    if response == "sufficient":
        return (1.0, "sufficient")
    elif response == "insufficient":
        return (0.0, "insufficient")
    else:
        # fallback
        return (0.0, "insufficient")


# ---------------------------------------------------------------------------
# hybrid_grader
# ---------------------------------------------------------------------------


def hybrid_grader(
    docs: list[Document],
    query: str,
    threshold: float,
    **kwargs: Any,
) -> tuple[float, Literal["sufficient", "insufficient"]]:
    """混合评估器：score + LLM 双重校验。

    先用 score_grader 快速判断，只有 score 通过才启动 LLM 二次确认。
    任一步判断为 insufficient 则返回 insufficient。
    """
    score, score_decision = score_grader(docs=docs, threshold=threshold, **kwargs)
    if score_decision == "insufficient":
        return (score, "insufficient")
    _, llm_decision = llm_grader(docs=docs, query=query, threshold=threshold, **kwargs)
    return (score, llm_decision)


# ---------------------------------------------------------------------------
# 注册
# ---------------------------------------------------------------------------

grader_registry.register("score", score_grader)
grader_registry.register("llm", llm_grader)
grader_registry.register("hybrid", hybrid_grader)
