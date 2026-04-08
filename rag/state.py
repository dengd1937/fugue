"""rag/state.py — LangGraph RAG 图状态定义。"""

from typing import Annotated, Literal

from typing_extensions import TypedDict

from rag.types import Document


def merge_docs(existing: list[Document], new: list[Document]) -> list[Document]:
    """自定义 reducer：按 (source, doc_id) 复合键去重合并。

    - 已存在的文档保持不变（existing 优先）
    - new 中与 existing 复合键相同的文档被丢弃
    """
    seen: set[tuple[str, str]] = {(d["source"], d["doc_id"]) for d in existing}
    deduped_new = [d for d in new if (d["source"], d["doc_id"]) not in seen]
    return existing + deduped_new


class RAGState(TypedDict):
    """RAG 图的完整状态。"""

    original_query: str
    rewritten_queries: list[str]
    documents: Annotated[list[Document], merge_docs]
    grade_score: float
    grade_decision: Literal["sufficient", "insufficient"]
    source: Literal["kb", "web", "kg"]
    retry_count: int
    retrieval_history: list[list[Document]]
    ranked_documents: list[Document]
    answer: str
