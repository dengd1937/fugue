"""src/ragline/engine/graph.py — LangGraph 整图组装。"""

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from ragline.engine.nodes.generate import generate
from ragline.engine.nodes.grade import grade, route_after_grade
from ragline.engine.nodes.post_process import post_process
from ragline.engine.nodes.prepare_fallback import prepare_fallback
from ragline.engine.nodes.query_transform import query_transform
from ragline.engine.nodes.retrieve import retrieve
from ragline.engine.state import RAGState


def build_rag_graph() -> CompiledStateGraph:
    """组装 7 节点 RAG 图：

    - 节点：query_transform / retrieve / grade / prepare_fallback /
            post_process / generate
    - 边：
        START → query_transform
        query_transform → (Send 扇出) retrieve
        retrieve → grade
        grade → conditional → {post_process, fallback_to_query_transform}
        fallback_to_query_transform → prepare_fallback → query_transform
        post_process → generate → END
    """
    graph: StateGraph = StateGraph(RAGState)

    # 添加节点
    graph.add_node("query_transform", query_transform)
    graph.add_node("retrieve", retrieve)
    graph.add_node("grade", grade)
    graph.add_node("prepare_fallback", prepare_fallback)
    graph.add_node("post_process", post_process)
    graph.add_node("generate", generate)

    # 起点
    graph.add_edge(START, "query_transform")

    # query_transform 通过 Command(goto=[Send,...]) 扇出到 retrieve
    # （LangGraph 自动处理 Send 路由，无需显式 edge）

    # retrieve → grade（Send 完成后自动汇聚）
    graph.add_edge("retrieve", "grade")

    # grade conditional edge
    graph.add_conditional_edges(
        "grade",
        route_after_grade,
        {
            "post_process": "post_process",
            "fallback_to_query_transform": "prepare_fallback",
        },
    )

    # prepare_fallback → query_transform（再次进入循环）
    graph.add_edge("prepare_fallback", "query_transform")

    # post_process → generate → END
    graph.add_edge("post_process", "generate")
    graph.add_edge("generate", END)

    return graph.compile()
