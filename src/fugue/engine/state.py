"""src/fugue/engine/state.py — RAGState TypedDict + merge_docs reducer。"""

from typing import Annotated, Any, Literal, TypedDict

from fugue.api.types import Document


def merge_docs(existing: list[Document], new: list[Document]) -> list[Document]:
    """按 (source, doc_id) 复合键去重合并 Document 列表。

    顺序：existing 在前，new 中未在 existing 中出现的按顺序追加。
    """
    seen = {(d["source"], d["doc_id"]) for d in existing}
    return existing + [d for d in new if (d["source"], d["doc_id"]) not in seen]


class RAGState(TypedDict):
    """LangGraph state schema for the RAG query path."""

    original_query: str
    rewritten_queries: list[str]
    documents: Annotated[list[Document], merge_docs]
    grade_score: float
    grade_decision: Literal["sufficient", "insufficient"]
    source: str  # "kb" / "web" / 任意自定义 fallback 源
    retry_count: int
    retrieval_history: list[list[Document]]
    ranked_documents: list[Document]
    answer: str


class RetrieveInput(TypedDict):
    """Send() 发送给 retrieve 节点的负载。"""

    query: str
    retriever_name: str
    source: str
    metadata_filter: dict[str, Any] | None
