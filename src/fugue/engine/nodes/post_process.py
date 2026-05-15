"""src/fugue/engine/nodes/post_process.py — 跨轮合并去重 + 链式 processors。"""

from typing import Any

from langchain_core.runnables import RunnableConfig

from fugue.api.types import Document
from fugue.engine.runtime import get_config
from fugue.engine.state import RAGState
from fugue.registry import processor_registry

MAX_DOCS_BEFORE_PROCESS = 1000


def _merge_dedupe(*lists: list[Document]) -> list[Document]:
    """按 (source, doc_id) 复合键去重，保留首次出现。"""
    seen: set[tuple[str, str]] = set()
    result: list[Document] = []
    for docs in lists:
        for d in docs:
            key = (d["source"], d["doc_id"])
            if key not in seen:
                seen.add(key)
                result.append(d)
    return result


def post_process(state: RAGState, config: RunnableConfig) -> dict[str, Any]:
    """1. 合并 state.documents + retrieval_history[:-1]（最后一项是当前轮，已在 documents）
    2. 按 (source, doc_id) 去重
    3. 防御性上界：> 1000 docs 时按 score 降序截断到 1000
    4. 链式 processors（按 cfg.processors 顺序）
    5. top_k 截断
    6. 返回 {ranked_documents}
    """
    cfg = get_config(config)
    current_docs = list(state.get("documents", []))
    history = state.get("retrieval_history", [])
    # history[:-1] 是历史轮的 documents（最后一项是当前轮，已在 state.documents）
    previous_rounds = history[:-1] if history else []

    # 1. 合并 + 2. 去重
    merged = _merge_dedupe(current_docs, *previous_rounds)

    # 3. 防御性上界
    if len(merged) > MAX_DOCS_BEFORE_PROCESS:
        merged = sorted(merged, key=lambda d: d["score"], reverse=True)[:MAX_DOCS_BEFORE_PROCESS]

    # 4. 链式 processors
    docs: list[Document] = merged
    query = state["original_query"]
    for processor_name in cfg.processors:
        processor_fn = processor_registry.get(processor_name)
        docs = processor_fn(
            docs,
            query,
            cfg.top_k,
            retriever_weights=cfg.retriever_weights,
            score_normalizers=cfg.score_normalizers,
        )

    # 5. top_k 截断（processor 可能未截断或返回多于 top_k）
    docs = docs[: cfg.top_k]

    # 6. 返回
    return {"ranked_documents": docs}
