"""examples/configs.py — fuge_plan.md Section 10 的 8 个配置场景示例。"""

from rag.config import GraphConfig

# 10.1 经典 Vector + BM25 混合检索（RRF 融合）
config_vector_bm25 = GraphConfig(
    transforms=["rewrite"],
    n_rewrites=3,
    retrievers=["vector", "es"],
    processors=["rrf", "rerank"],
    retriever_weights={"vector": 1.0, "es": 0.8},
    top_k=5,
)

# 10.2 Vector + ES + Knowledge Graph 三路召回
config_three_way = GraphConfig(
    retrievers=["vector", "es", "kg"],
    processors=["rrf", "rerank"],
    retriever_weights={"vector": 1.0, "es": 0.7, "kg": 1.2},
    top_k=5,
)

# 10.3 意图路由（SQL 问题只走 SQL）
config_intent_routing = GraphConfig(
    retrievers=["vector", "es", "sql"],
    route_strategy="intent",
    processors=["rerank"],
    top_k=5,
)

# 10.4 Parent-Child 分块（小块检索，大块返回）
config_parent_child = GraphConfig(
    retrievers=["vector_parent_child", "es"],
    processors=["rrf", "rerank"],
    top_k=3,
)

# 10.5 Sentence Window 检索
config_sentence_window = GraphConfig(
    retrievers=["vector_sentence_window", "es"],
    processors=["rerank"],
    top_k=5,
)

# 10.6 Self-Query（自动提取 metadata 过滤）
config_self_query = GraphConfig(
    transforms=["self_query", "rewrite"],
    retrievers=["vector", "es"],
    processors=["rrf", "rerank"],
    top_k=5,
)

# 10.7 HyDE + Step-Back 混合改写 + 多路检索 + 全栈后处理
config_full_stack = GraphConfig(
    transforms=["hyde", ["step_back", "rewrite"]],
    n_rewrites=3,
    retrievers=["vector_parent_child", "es", "kg"],
    route_strategy="all",
    processors=["rrf", "compression", "rerank"],
    retriever_weights={"vector_parent_child": 1.0, "es": 0.7, "kg": 1.5},
    grade_threshold=0.6,
    fallback_chain=["web"],
    max_retries=1,
    gen_mode="citation",
    top_k=5,
)

# 10.8 KB 优先 → Web 兜底（带质量阈值）
config_kb_with_web_fallback = GraphConfig(
    transforms=[["step_back", "rewrite"]],
    retrievers=["vector", "es"],
    grade_threshold=0.7,
    grade_strategy="llm",
    fallback_chain=["web"],
    max_retries=1,
    processors=["rrf", "rerank"],
    top_k=5,
)

ALL_CONFIGS = {
    "vector_bm25": config_vector_bm25,
    "three_way": config_three_way,
    "intent_routing": config_intent_routing,
    "parent_child": config_parent_child,
    "sentence_window": config_sentence_window,
    "self_query": config_self_query,
    "full_stack": config_full_stack,
    "kb_with_web_fallback": config_kb_with_web_fallback,
}
