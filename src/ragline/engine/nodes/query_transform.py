"""src/ragline/engine/nodes/query_transform.py — query 改写 + 扇出 Send 节点。"""

from typing import Any, cast

from langchain_core.runnables import RunnableConfig
from langgraph.types import Command, Send

from ragline.api.types import TransformResult
from ragline.engine.runtime import get_config
from ragline.engine.state import Overwrite, RAGState, RetrieveInput
from ragline.handlers.transforms.pipeline import run_transform_branch
from ragline.registry import transform_registry


def _dedupe_keep_order(items: list[str]) -> list[str]:
    """按顺序去重保留首次出现。"""
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result


def query_transform(state: RAGState, config: RunnableConfig) -> Command[Any]:
    """1. 原始 query 始终在第 0 位
    2. 遍历 cfg.transforms 顶层分支，调用 run_transform_branch
    3. TransformResult 提取在此节点进行
    4. 去重 + 截断 cfg.max_queries
    5. 按 source 决定 retriever_names
    6. 构建 Send 列表（笛卡尔积 queries × retriever_names）
    7. Command(update={'rewritten_queries': all_queries,
                       'documents': Overwrite([])},
               goto=sends)
    """
    cfg = get_config(config)
    original_query = state["original_query"]
    source = state.get("source", "kb")

    # 1. 始终将原始 query 作为第 0 个
    all_queries: list[str] = [original_query]
    query_filters: dict[str, dict[str, Any]] = {}

    # 2-3. 遍历 transforms 分支，TransformResult 提取在此节点
    for branch in cfg.transforms:
        results = run_transform_branch(
            branch,
            [original_query],
            cfg.n_rewrites,
            transform_registry,
        )
        for r in results:
            if isinstance(r, TransformResult):
                all_queries.append(r.query)
                if r.metadata_filter:
                    query_filters[r.query] = r.metadata_filter
            else:
                # r is str
                all_queries.append(r)

    # 4. 去重保持顺序，截断 max_queries
    all_queries = _dedupe_keep_order(all_queries)[: cfg.max_queries]

    # 5. 按 source 决定 retrievers（非 kb 时 fallback 单源）
    retriever_names = [source] if source != "kb" else list(cfg.retrievers)

    # 6. 构建 Send 列表（笛卡尔积）
    sends: list[Send] = []
    for q in all_queries:
        for r_name in retriever_names:
            payload: RetrieveInput = RetrieveInput(
                query=q,
                retriever_name=r_name,
                source=source,
                metadata_filter=query_filters.get(q),
            )
            sends.append(Send("retrieve", payload))

    # 7. Command：重置 documents 列表（Overwrite sentinel），扇出 retrieve
    return Command(
        update={
            "rewritten_queries": all_queries,
            "documents": cast(Any, Overwrite([])),
        },
        goto=sends,
    )
