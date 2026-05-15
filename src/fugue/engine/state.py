"""src/fugue/engine/state.py — RAGState TypedDict + merge_docs reducer + Overwrite。"""

from dataclasses import dataclass
from typing import Annotated, Any, Literal, TypedDict

from fugue.api.types import Document


@dataclass(frozen=True)
class Overwrite:
    """Sentinel：让 merge_docs reducer 跳过合并并覆盖 documents 列表。

    LangGraph 0.3.x 不提供官方 Overwrite 类型，自实现 sentinel 让节点能显式
    重置累积的 documents 列表（如 query_transform 节点开始新一轮检索时）。
    LangGraph 1.x 引入官方 Overwrite 后可替换为官方实现。
    """

    values: list[Document]


def merge_docs(
    existing: list[Document],
    new: list[Document] | Overwrite,
) -> list[Document]:
    """按 (source, doc_id) 复合键去重合并。

    若 new 是 Overwrite sentinel，直接返回 sentinel.values（覆盖语义）。
    否则 existing 在前，new 中未重复项按原顺序追加。
    """
    if isinstance(new, Overwrite):
        return list(new.values)
    seen = {(d["source"], d["doc_id"]) for d in existing}
    return existing + [d for d in new if (d["source"], d["doc_id"]) not in seen]


class RAGState(TypedDict):
    """LangGraph state schema for the RAG query path."""

    original_query: str
    rewritten_queries: list[str]
    documents: Annotated[list[Document], merge_docs]
    grade_score: float
    grade_decision: Literal["sufficient", "insufficient"]
    source: str
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
