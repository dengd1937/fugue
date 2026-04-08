"""rag/types.py — 核心数据类型定义。"""

from dataclasses import dataclass
from typing import Any

from typing_extensions import TypedDict


class Document(TypedDict):
    """检索结果文档。"""

    doc_id: str
    content: str
    score: float
    source: str  # retriever 名称，如 "vector"/"es"/"web"
    metadata: dict[str, Any]


@dataclass(frozen=True)
class TransformResult:
    """查询转换结果（不可变）。"""

    query: str
    metadata_filter: dict[str, Any] | None = None


class RetrieveInput(TypedDict):
    """发送给单个 retriever 的输入。"""

    query: str
    retriever_name: str
    source: str
    metadata_filter: dict[str, Any] | None
