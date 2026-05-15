"""src/fugue/handlers/processors/rerank.py — Rerank 处理器工厂。"""

from collections.abc import Callable
from typing import Any

from fugue.api.types import Document
from fugue.providers.reranker.base import Reranker

ProcessorFn = Callable[..., list[Document]]


def make_rerank(reranker: Reranker) -> ProcessorFn:
    """返回 rerank 处理器闭包，闭包绑定 reranker。

    使用 reranker.rerank(query, [doc.content], top_k) 返回 [(idx, score)] 降序，
    重排 docs 并将 rerank_score 写入 metadata["rerank_score"]，
    同时覆盖 score 字段为 rerank_score，让下游使用统一接口。
    """

    def rerank_fn(
        docs: list[Document],
        query: str,
        top_k: int,
        **kwargs: Any,  # noqa: ARG001
    ) -> list[Document]:
        if not docs:
            return []
        contents = [d["content"] for d in docs]
        scored = reranker.rerank(query, contents, top_k=top_k)
        # scored = [(原 idx, new score), ...] 已降序
        result: list[Document] = []
        for idx, score in scored:
            original = docs[idx]
            md = dict(original.get("metadata", {}))
            md["rerank_score"] = score
            new_doc = Document(
                doc_id=original["doc_id"],
                content=original["content"],
                score=score,  # 覆盖 score 为 rerank_score
                source=original["source"],
                metadata=md,
            )
            result.append(new_doc)
        return result

    return rerank_fn
