"""rag/graph.py — 构建并编译 RAG LangGraph 图。"""

from typing import Any

from langgraph.graph import END, StateGraph

import rag.handlers  # noqa: F401 — 触发所有内置 handler 的注册副作用
from rag.nodes import (
    generate,
    grade,
    post_process,
    prepare_fallback,
    query_transform,
    retrieve,
    route_after_grade,
)
from rag.state import RAGState


def build_rag_graph() -> Any:
    """构建并编译 RAG 图。业务配置不在此传入，运行时通过 RunnableConfig 注入。"""
    graph = StateGraph(RAGState)

    graph.add_node("query_transform", query_transform)
    graph.add_node("retrieve", retrieve)
    graph.add_node("grade", grade)
    graph.add_node("prepare_fallback", prepare_fallback)
    graph.add_node("post_process", post_process)
    graph.add_node("generate", generate)

    graph.set_entry_point("query_transform")
    graph.add_edge("retrieve", "grade")

    graph.add_conditional_edges(
        "grade",
        route_after_grade,
        {
            "post_process": "post_process",
            "fallback_to_query_transform": "prepare_fallback",
        },
    )
    graph.add_edge("prepare_fallback", "query_transform")
    graph.add_edge("post_process", "generate")
    graph.add_edge("generate", END)

    return graph.compile()
