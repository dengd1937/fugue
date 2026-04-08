"""rag/nodes.py — LangGraph RAG 图节点实现。"""

from __future__ import annotations

from typing import Any, Literal, cast

from langchain_core.runnables import RunnableConfig
from langgraph.types import Command, Overwrite, Send

from rag.config import _GRAPH_CONFIG_FIELDS, GraphConfig
from rag.handlers.intent_router import intent_router
from rag.registry import (
    generator_registry,
    grader_registry,
    processor_registry,
    retriever_registry,
    transform_registry,
)
from rag.state import RAGState
from rag.types import Document, RetrieveInput, TransformResult

# ---------------------------------------------------------------------------
# 配置提取
# ---------------------------------------------------------------------------


def get_config(config: RunnableConfig) -> GraphConfig:
    """从 RunnableConfig 中提取业务配置，过滤框架注入的键。"""
    raw = config.get("configurable", {})
    filtered = {k: v for k, v in raw.items() if k in _GRAPH_CONFIG_FIELDS}
    return GraphConfig(**filtered)


# ---------------------------------------------------------------------------
# transform 辅助
# ---------------------------------------------------------------------------


def _run_transform_branch(
    branch: str | list[str],
    queries: list[str],
    n: int,
) -> list[str | TransformResult]:
    """执行单个 transform 分支（str=原子，list[str]=管道链）。"""
    if isinstance(branch, str):
        fn = transform_registry.get(branch)
        return cast(list[str | TransformResult], fn(queries=queries, n=n))

    # pipeline：链式执行
    current: list[str] = queries
    result: list[str | TransformResult] = []
    for name in branch:
        fn = transform_registry.get(name)
        result = cast(list[str | TransformResult], fn(queries=current, n=n))
        if result and isinstance(result[0], TransformResult):
            current = [r.query for r in cast(list[TransformResult], result)]
        else:
            current = cast(list[str], result)
    return result


# ---------------------------------------------------------------------------
# 节点：query_transform
# ---------------------------------------------------------------------------


def query_transform(state: RAGState, config: RunnableConfig) -> Command[Literal["retrieve"]]:
    """扇出节点：生成 rewritten queries，Send 给 retrieve。"""
    cfg = get_config(config)
    query = state["original_query"]
    all_queries: list[str] = [query]
    query_filters: dict[str, dict[str, Any] | None] = {}

    for branch in cfg.transforms:
        results = _run_transform_branch(branch, [query], cfg.n_rewrites)
        for r in results:
            if isinstance(r, TransformResult):
                all_queries.append(r.query)
                if r.metadata_filter:
                    query_filters[r.query] = r.metadata_filter
            else:
                all_queries.append(str(r))

    # 去重保序 + max_queries 截断
    all_queries = list(dict.fromkeys(all_queries))[: cfg.max_queries]

    # fallback 时只用指定 source 的 retriever
    source = state.get("source", "kb")
    retriever_names: list[str] = [source] if source != "kb" else cfg.retrievers

    sends: list[Send] = []
    for q in all_queries:
        if cfg.route_strategy == "intent" and source == "kb":
            targets = intent_router(q, retriever_names)
        else:
            targets = retriever_names

        for retriever_name in targets:
            sends.append(
                Send(
                    "retrieve",
                    {
                        "query": q,
                        "retriever_name": retriever_name,
                        "source": source,
                        "metadata_filter": query_filters.get(q),
                    },
                )
            )

    return Command(
        update={"rewritten_queries": all_queries, "documents": Overwrite([])},
        goto=sends,
    )


# ---------------------------------------------------------------------------
# 节点：retrieve
# ---------------------------------------------------------------------------


def retrieve(state: RetrieveInput) -> dict[str, Any]:
    """单个检索任务，由 Send 调度，接收 RetrieveInput（非 RAGState）。"""
    query = state["query"]
    retriever_name = state["retriever_name"]
    metadata_filter = state.get("metadata_filter")

    fn = retriever_registry.get(retriever_name)
    raw_docs: list[Document] = fn(query=query, metadata_filter=metadata_filter)
    docs = [{**d, "source": retriever_name} for d in raw_docs]

    return {"documents": docs}


# ---------------------------------------------------------------------------
# 节点：grade
# ---------------------------------------------------------------------------


def grade(state: RAGState, config: RunnableConfig) -> dict[str, Any]:
    """质量评估，归档当前轮文档到 retrieval_history。"""
    cfg = get_config(config)
    docs = state["documents"]
    grader_fn = grader_registry.get(cfg.grade_strategy)

    score, decision = grader_fn(
        docs=docs,
        query=state["original_query"],
        threshold=cfg.grade_threshold,
        score_normalizers=cfg.score_normalizers,
    )

    history: list[list[Document]] = list(state.get("retrieval_history", []))
    history.append(docs)

    return {
        "grade_score": score,
        "grade_decision": decision,
        "retrieval_history": history,
    }


# ---------------------------------------------------------------------------
# 条件边：route_after_grade
# ---------------------------------------------------------------------------


def route_after_grade(state: RAGState, config: RunnableConfig) -> str:
    """条件边路由函数。"""
    cfg = get_config(config)

    if state["grade_decision"] == "sufficient":
        return "post_process"

    if not cfg.fallback_chain:
        return "post_process"

    retry_count = state.get("retry_count", 0)

    if retry_count >= len(cfg.fallback_chain):
        return "post_process"

    if retry_count >= cfg.max_retries:
        return "post_process"

    return "fallback_to_query_transform"


# ---------------------------------------------------------------------------
# 节点：prepare_fallback
# ---------------------------------------------------------------------------


def prepare_fallback(state: RAGState, config: RunnableConfig) -> dict[str, Any]:
    """回退前改写 source 和 retry_count。"""
    cfg = get_config(config)
    retry_count = state.get("retry_count", 0)
    next_source = cfg.fallback_chain[retry_count]
    return {
        "source": next_source,
        "retry_count": retry_count + 1,
    }


# ---------------------------------------------------------------------------
# 节点：post_process
# ---------------------------------------------------------------------------


def post_process(state: RAGState, config: RunnableConfig) -> dict[str, Any]:
    """链式后处理：合并所有轮次文档 → processor 链 → top_k 截断。"""
    cfg = get_config(config)

    docs: list[Document] = list(state["documents"])
    history: list[list[Document]] = list(state.get("retrieval_history", []))
    for round_docs in history[:-1]:  # 最后一轮已在 documents 中
        docs.extend(round_docs)

    # (source, doc_id) 去重，保序
    seen: set[tuple[str, str]] = set()
    unique_docs: list[Document] = []
    for d in docs:
        key = (d["source"], d["doc_id"])
        if key not in seen:
            seen.add(key)
            unique_docs.append(d)
    docs = unique_docs

    for proc_name in cfg.processors:
        fn = processor_registry.get(proc_name)
        docs = fn(
            docs=docs,
            query=state["original_query"],
            top_k=cfg.top_k,
            retriever_weights=cfg.retriever_weights,
            score_normalizers=cfg.score_normalizers,
        )

    docs = docs[: cfg.top_k]
    return {"ranked_documents": docs}


# ---------------------------------------------------------------------------
# 节点：generate
# ---------------------------------------------------------------------------


def generate(state: RAGState, config: RunnableConfig) -> dict[str, Any]:
    """基于精排文档生成最终回答。"""
    cfg = get_config(config)
    gen_fn = generator_registry.get(cfg.gen_mode)
    answer = gen_fn(
        docs=state["ranked_documents"],
        query=state["original_query"],
        temperature=cfg.temperature,
    )
    return {"answer": answer}
